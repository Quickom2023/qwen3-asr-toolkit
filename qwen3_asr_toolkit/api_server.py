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
    tmp_dir: str = DEFAULT_TMP_DIR
    save_srt: bool = False


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


def _save_srt_file(
    wav_list: List[Tuple[int, int, object]],
    results: List[Tuple[int, str]],
    save_file: str,
) -> str:
    ordered_results = dict(results)
    subtitles = []
    for idx, (start_sample, end_sample, _) in enumerate(wav_list):
        subtitles.append(
            srt.Subtitle(
                index=idx,
                start=timedelta(seconds=start_sample / WAV_SAMPLE_RATE),
                end=timedelta(seconds=end_sample / WAV_SAMPLE_RATE),
                content=ordered_results.get(idx, ""),
            )
        )

    srt_path = os.path.splitext(save_file)[0] + ".srt"
    with open(srt_path, "w", encoding="utf-8") as handle:
        handle.write(srt.compose(subtitles))
    return srt_path


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
    tmp_dir: str,
    save_srt: bool,
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

    if wav_duration_seconds >= 180:
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

    srt_path = None
    if save_srt:
        srt_path = _save_srt_file(wav_list, results, save_file)

    return {
        # "input_file": input_file,
        # "detected_language": language,
        "full_text": full_text,
        "duration_seconds": round(wav_duration_seconds, 3),
        "segment_count": len(wav_list),
        # "segments": _serialize_segments(wav_list, results),
        "failed_segments": failed_segments,
        # "text_file": os.path.abspath(save_file),
        # "srt_file": os.path.abspath(srt_path) if srt_path else None,
    }


def _raise_as_http_error(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


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
            tmp_dir=request.tmp_dir,
            save_srt=request.save_srt,
        )
    except Exception as exc:
        _raise_as_http_error(exc)


@app.post("/transcribe")
def transcribe_upload(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    context: str = Form(DEFAULT_CONTEXT),
    model: Optional[str] = Form(None),
    api_timeout: int = Form(300),
    temperature: float = Form(0.2),
    skip_failed: bool = Form(False),
    max_retries: int = Form(10),
    num_threads: int = Form(4),
    vad_segment_threshold: int = Form(120),
    max_segment_seconds: int = Form(180),
    tmp_dir: str = Form(DEFAULT_TMP_DIR),
    save_srt: bool = Form(False),
) -> Dict[str, object]:
    _verify_api_key(x_api_key)
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
            tmp_dir=tmp_dir,
            save_srt=save_srt,
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
