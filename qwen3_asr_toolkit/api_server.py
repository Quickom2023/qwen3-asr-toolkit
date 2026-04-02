import concurrent.futures
import os
import shutil
import tempfile
from collections import Counter
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import srt
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel
from silero_vad import load_silero_vad
try:
    from dotenv import find_dotenv, load_dotenv  # type: ignore
except Exception:
    find_dotenv = None
    load_dotenv = None

from qwen3_asr_toolkit.audio_tools import (
    WAV_SAMPLE_RATE,
    load_audio,
    process_vad,
    save_audio_file,
)
from qwen3_asr_toolkit.qwen3asr import QwenASR


DEFAULT_CONTEXT = "Transcribe with punctuation. Preserve sentence meaning across pauses."
DEFAULT_TMP_DIR = os.path.join(os.path.expanduser("~"), "qwen3-asr-cache")

if load_dotenv and find_dotenv:
    load_dotenv(find_dotenv(usecwd=True), override=False)


def _is_true_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _get_default_api_url() -> str:
    return os.getenv(
        "QWEN3_ASR_API_URL",
        "http://localhost:8000/v1/audio/transcriptions",
    )


def _get_openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _get_default_summary_model() -> str:
    return os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-mini")


app = FastAPI(title="Qwen3-ASR Toolkit API", version="1.0.0")


def _verify_api_key(x_api_key: Optional[str]) -> None:
    expected_api_key = os.getenv("QWEN3_ASR_API_KEY")
    if not expected_api_key:
        return
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Api-Key header")


def _try_cleanup_cache_root(tmp_dir: str) -> None:
    upload_root = os.path.join(tmp_dir, "uploads")
    try:
        if os.path.isdir(upload_root) and not os.listdir(upload_root):
            os.rmdir(upload_root)
    except OSError:
        pass

    if not _is_true_env("QWEN3_ASR_AUTO_CLEAN_CACHE"):
        return

    try:
        if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)
    except OSError:
        pass


class TranscribeRequest(BaseModel):
    input_file: str
    context: str = DEFAULT_CONTEXT
    model: Optional[str] = "Qwen/Qwen3-ASR-1.7B"
    api_timeout: int = 300
    temperature: float = 0.2
    skip_failed: bool = False
    max_retries: int = 10
    num_threads: int = 4
    vad_segment_threshold: int = 120
    max_segment_seconds: int = 180
    vad_trigger_seconds: int = 180
    tmp_dir: str = DEFAULT_TMP_DIR
    save_srt: bool = False
    include_srt: bool = True
    include_text: bool = False


class SummarizeTextRequest(BaseModel):
    text: str
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1000   


def _validate_input_file(input_file: str) -> None:
    if input_file.startswith(("http://", "https://")):
        try:
            response = requests.head(input_file, allow_redirects=True, timeout=5)
            if response.status_code >= 400:
                raise FileNotFoundError("returned status code %s" % response.status_code)
        except Exception as exc:
            raise FileNotFoundError(
                "HTTP link %s does not exist or is inaccessible: %s" % (input_file, exc)
            )
        return

    if not os.path.exists(input_file):
        raise FileNotFoundError('Input file "%s" does not exist!' % input_file)


def _build_output_path(input_file: str) -> str:
    if os.path.exists(input_file):
        return os.path.splitext(input_file)[0] + ".txt"

    parsed_path = urlparse(input_file).path
    output_name = os.path.splitext(parsed_path)[0].split("/")[-1] or "transcription"
    return output_name + ".txt"


def _serialize_segments(
    wav_list: List[Tuple[int, int, object]],
    results: List[Tuple[int, str]],
) -> List[Dict[str, object]]:
    ordered_results = dict(results)
    serialized = []
    for idx, (start_sample, end_sample, _) in enumerate(wav_list):
        serialized.append(
            {
                "index": idx,
                "start_seconds": round(start_sample / WAV_SAMPLE_RATE, 3),
                "end_seconds": round(end_sample / WAV_SAMPLE_RATE, 3),
                "text": ordered_results.get(idx, ""),
            }
        )
    return serialized


def _compose_srt_content(
    wav_list: List[Tuple[int, int, object]],
    results: List[Tuple[int, str]],
) -> str:
    ordered_results = dict(results)
    blocks = []
    for idx, (start_sample, end_sample, _) in enumerate(wav_list):
        start_time = timedelta(seconds=start_sample / WAV_SAMPLE_RATE)
        end_time = timedelta(seconds=end_sample / WAV_SAMPLE_RATE)
        content = ordered_results.get(idx, "")
        blocks.append(
            f"{srt.timedelta_to_srt_timestamp(start_time)} --> "
            f"{srt.timedelta_to_srt_timestamp(end_time)}\n{content}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _save_srt_file(save_file: str, srt_content: str) -> str:
    srt_path = os.path.splitext(save_file)[0] + ".srt"
    with open(srt_path, "w", encoding="utf-8") as handle:
        handle.write(srt_content)
    return srt_path


def _extract_openai_message_content(response_json: Dict[str, object]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("OpenAI response has invalid choice format")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenAI response missing message")

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text", "")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "\n".join(part for part in text_parts if part).strip()

    return str(content).strip()


def _summarize_text_with_openai(
    text: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> Dict[str, object]:
    if not text or not text.strip():
        raise ValueError("Field 'text' must not be empty.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    model_name = model or _get_default_summary_model()
    endpoint = _get_openai_base_url() + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You summarize text clearly and concisely. Preserve important facts, names, actions, and conclusions.\n\n"
                    "When summarizing a meeting or call transcript, follow this structure:\n\n"
                    "1. **Language Detection** — Identify the language of the input and respond entirely in that same language\n\n"
                    "2. **Meeting Summary by Chapter** — Divide the meeting into logical sections or topics. Write a short, clear summary for each chapter/segment.\n\n"
                    "3. **Key Insights & Takeaways** — Capture the most important conclusions, decisions, observations, or themes from the call.\n\n"
                    "Return the output as a clean, structured report. Do not return raw markdown symbols.\n"
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise ValueError(f"OpenAI API request failed: {detail}") from exc
    except requests.RequestException as exc:
        raise ValueError(f"OpenAI API request failed: {exc}") from exc

    response_json = response.json()
    summary = _extract_openai_message_content(response_json)
    return {
        # "model": model_name,
        "summary": summary,
    }


def _transcribe(
    input_file: str,
    context: str,
    api_url: str,
    model: Optional[str],
    api_timeout: int,
    temperature: float,
    skip_failed: bool,
    max_retries: int,
    num_threads: int,
    vad_segment_threshold: int,
    max_segment_seconds: int,
    vad_trigger_seconds: int,
    tmp_dir: str,
    save_srt: bool,
    include_srt: bool,
    include_text: bool,
) -> Dict[str, object]:
    _validate_input_file(input_file)
    os.makedirs(tmp_dir, exist_ok=True)

    qwen3asr = QwenASR(
        api_url=api_url,
        model=model,
        timeout_s=api_timeout,
        temperature=temperature,
        max_retries=max_retries,
    )

    wav = load_audio(input_file)
    wav_duration_seconds = len(wav) / WAV_SAMPLE_RATE

    if wav_duration_seconds >= vad_trigger_seconds:
        worker_vad_model = load_silero_vad(onnx=True)
        wav_list = process_vad(
            wav,
            worker_vad_model,
            segment_threshold_s=vad_segment_threshold,
            max_segment_threshold_s=max_segment_seconds,
        )
    else:
        wav_list = [(0, len(wav), wav)]

    source_name = os.path.basename(urlparse(input_file).path) if input_file.startswith(("http://", "https://")) else os.path.basename(input_file)
    source_name = source_name or "input_audio"
    source_stem = os.path.splitext(source_name)[0]
    save_dir = tempfile.mkdtemp(prefix=source_stem + "_", dir=tmp_dir)

    wav_path_list = []
    for idx, (_, _, wav_data) in enumerate(wav_list):
        wav_path = os.path.join(save_dir, "%s_%s.wav" % (source_stem, idx))
        save_audio_file(wav_data, wav_path)
        wav_path_list.append(wav_path)

    results = []
    languages = []
    failed_segments = []

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_dict = {
                executor.submit(qwen3asr.asr, wav_path, context): idx
                for idx, wav_path in enumerate(wav_path_list)
            }

            for future in concurrent.futures.as_completed(future_dict):
                idx = future_dict[future]
                try:
                    language, recog_text = future.result()
                    results.append((idx, recog_text))
                    languages.append(language)
                except Exception as exc:
                    if not skip_failed:
                        raise exc
                    failed_segments.append(
                        {"index": idx, "error": "%s: %s" % (exc.__class__.__name__, exc)}
                    )
                    results.append((idx, ""))
                    languages.append("Unknown")
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
        _try_cleanup_cache_root(tmp_dir)

    results.sort(key=lambda item: item[0])
    full_text = " ".join(text for _, text in results).strip()
    language = Counter(languages).most_common(1)[0][0] if languages else "Unknown"

    save_file = _build_output_path(input_file)
    with open(save_file, "w", encoding="utf-8") as handle:
        handle.write(language + "\n")
        handle.write(full_text + "\n")

    srt_content = None
    if include_srt or save_srt:
        srt_content = _compose_srt_content(wav_list, results)

    srt_path = None
    if save_srt:
        srt_path = _save_srt_file(save_file, srt_content or "")

    response = {
        # "input_file": input_file,
        # "detected_language": language,
        "duration_seconds": round(wav_duration_seconds, 3),
        "segment_count": len(wav_list),
        # "segments": _serialize_segments(wav_list, results),
        "failed_segments": failed_segments,
        # "text_file": os.path.abspath(save_file),
        # "srt_file": os.path.abspath(srt_path) if srt_path else None,
    }
    if include_srt:
        response["srt_content"] = srt_content
    if include_text:
        response["full_text"] = full_text
    return response


def _raise_as_http_error(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize")
def summarize_text(
    request: SummarizeTextRequest,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> Dict[str, object]:
    _verify_api_key(x_api_key)
    try:
        return _summarize_text_with_openai(
            text=request.text,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        _raise_as_http_error(exc)


@app.post("/transcribe-cmd")
def transcribe(
    request: TranscribeRequest,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> Dict[str, object]:
    _verify_api_key(x_api_key)
    try:
        return _transcribe(
            input_file=request.input_file,
            context=request.context,
            api_url=_get_default_api_url(),
            model=request.model,
            api_timeout=request.api_timeout,
            temperature=request.temperature,
            skip_failed=request.skip_failed,
            max_retries=request.max_retries,
            num_threads=request.num_threads,
            vad_segment_threshold=request.vad_segment_threshold,
            max_segment_seconds=request.max_segment_seconds,
            vad_trigger_seconds=request.vad_trigger_seconds,
            tmp_dir=request.tmp_dir,
            save_srt=request.save_srt,
            include_srt=request.include_srt,
            include_text=request.include_text,
        )
    except Exception as exc:
        _raise_as_http_error(exc)


@app.post("/transcribe")
def transcribe_upload(
    file: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    context: str = Form(DEFAULT_CONTEXT),
    model: Optional[str] = Form(None),
    api_timeout: int = Form(300),
    temperature: float = Form(0.2),
    skip_failed: bool = Form(False),
    max_retries: int = Form(10),
    num_threads: int = Form(8),
    vad_segment_threshold: int = Form(60),
    max_segment_seconds: int = Form(120),
    vad_trigger_seconds: int = Form(70),
    tmp_dir: str = Form(DEFAULT_TMP_DIR),
    save_srt: bool = Form(False),
    include_srt: bool = Form(True),
    include_text: bool = Form(False),
) -> Dict[str, object]:
    _verify_api_key(x_api_key)
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Missing file in request body field 'file'.")
    os.makedirs(tmp_dir, exist_ok=True)
    upload_root = os.path.join(tmp_dir, "uploads")
    os.makedirs(upload_root, exist_ok=True)

    original_name = os.path.basename(file.filename or "upload.wav")
    upload_dir = tempfile.mkdtemp(prefix="upload_", dir=upload_root)
    upload_path = os.path.join(upload_dir, original_name)

    try:
        with open(upload_path, "wb") as handle:
            shutil.copyfileobj(file.file, handle)

        result = _transcribe(
            input_file=upload_path,
            context=context,
            api_url=_get_default_api_url(),
            model=model,
            api_timeout=api_timeout,
            temperature=temperature,
            skip_failed=skip_failed,
            max_retries=max_retries,
            num_threads=num_threads,
            vad_segment_threshold=vad_segment_threshold,
            max_segment_seconds=max_segment_seconds,
            vad_trigger_seconds=vad_trigger_seconds,
            tmp_dir=tmp_dir,
            save_srt=save_srt,
            include_srt=include_srt,
            include_text=include_text,
        )
        result["uploaded_filename"] = original_name
        return result
    except Exception as exc:
        _raise_as_http_error(exc)
    finally:
        file.file.close()
        shutil.rmtree(upload_dir, ignore_errors=True)
        _try_cleanup_cache_root(tmp_dir)


def run() -> None:
    host = os.getenv("QWEN3_ASR_HOST", "0.0.0.0")
    port = int(os.getenv("QWEN3_ASR_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
