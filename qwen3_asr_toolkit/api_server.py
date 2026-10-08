import concurrent.futures
import os
import re
import shutil
import tempfile
import threading
from collections import Counter
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import srt
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware
from silero_vad import load_silero_vad
try:
    from dotenv import find_dotenv, load_dotenv  # type: ignore
except Exception:
    find_dotenv = None
    load_dotenv = None

from qwen3_asr_toolkit.audio_tools import (
    WAV_SAMPLE_RATE,
    has_speech,
    load_audio,
    process_vad,
    save_audio_file,
)
from qwen3_asr_toolkit.content_generation import (
    build_each_person_prompt,
    build_minutes_conclusion_prompt,
    generate_each_person_from_transcript,
    generate_conclusions_from_summaries,
)
from qwen3_asr_toolkit.live_summary import (
    LiveSummaryLLMError,
    generate_live_summary,
)
from qwen3_asr_toolkit.meeting_summary import generate_meeting_summary
from qwen3_asr_toolkit.report_generation import (
    GroupTranscript,
    generate_report_from_groups,
)
from qwen3_asr_toolkit.qwen3asr import QwenASR
from qwen3_asr_toolkit.speaker_attribution import (
    attribute_speakers_as_srt,
    client_from_env,
)
from qwen3_asr_toolkit.task_generation import (
    ActionItemsLLMError,
    generate_action_items,
)


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


app = FastAPI(
    title="Qwen3-ASR Toolkit API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Routes whose responses are gzipped for a client that sends
# "Accept-Encoding: gzip": the live summary goes out every couple of minutes
# to every delegate's device.
GZIP_PATHS = frozenset({"/summarize/live"})
# Bytes below which a response is sent as it is: compressing it gains nothing.
GZIP_MINIMUM_SIZE = 500


class _GZipSomePaths:
    """GZipMiddleware for the paths in GZIP_PATHS only."""

    def __init__(self, app, minimum_size: int) -> None:
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=minimum_size)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["path"] in GZIP_PATHS:
            await self.gzip(scope, receive, send)
        else:
            await self.app(scope, receive, send)


app.add_middleware(_GZipSomePaths, minimum_size=GZIP_MINIMUM_SIZE)
_shared_vad_model = None
_shared_vad_model_lock = threading.Lock()


def _verify_api_key(x_api_key: Optional[str]) -> None:
    expected_api_key = os.getenv("QWEN3_ASR_API_KEY")
    if not expected_api_key:
        return
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Api-Key header")




def _get_shared_vad_model():
    global _shared_vad_model
    if _shared_vad_model is None:
        with _shared_vad_model_lock:
            if _shared_vad_model is None:
                _shared_vad_model = load_silero_vad(onnx=True)
    return _shared_vad_model


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
    locale: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4096
    # Adds "Nội dung chất vấn và giải đáp" and "Kết luận và phân công nhiệm vụ".
    include_details: bool = False


class GenerateEachPersonRequest(BaseModel):
    transcript: str
    model: Optional[str] = None
    locale: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 3000
    include_prompt: bool = False


class GenerateConclusionsRequest(BaseModel):
    transcript: str
    model: Optional[str] = None
    locale: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 10000
    include_prompt: bool = False


class AttributeSpeakersRequest(BaseModel):
    srt_content: str
    model: Optional[str] = None
    roster: Optional[List[str]] = None

class LiveSummaryRequest(BaseModel):
    srt_content: str
    speaker_roles: Optional[Dict[str, str]] = None
    agenda_title: Optional[str] = None
    meeting_id: Optional[str] = None
    # ISO 8601 date or datetime of the meeting; deadlines are dated from it.
    meeting_date: Optional[str] = None


class ActionItemsRequest(BaseModel):
    transcript: str
    speaker_roles: Optional[Dict[str, str]] = None
    agenda_title: Optional[str] = None
    meeting_id: Optional[str] = None
    # ISO 8601 date or datetime of the meeting; deadlines are dated from it.
    meeting_date: Optional[str] = None


def _split_markdown_sections(markdown_text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for raw_line in markdown_text.splitlines():
        heading_match = re.match(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$", raw_line)
        if heading_match:
            if current_title is not None:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append((current_title, content))
            current_title = heading_match.group(2).strip()
            current_lines = []
            continue

        if current_title is not None:
            current_lines.append(raw_line)

    if current_title is not None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_title, content))

    return sections


def _extract_summary_title(summary_content: str, fallback_title: str) -> str:
    for line in summary_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback_title
    return fallback_title


def _strip_summary_title_line(summary_content: str) -> str:
    lines = summary_content.splitlines()
    if lines and lines[0].strip().startswith("# "):
        return "\n".join(lines[1:]).strip()
    return summary_content.strip()


def _has_title_keyword(title: str, keywords: Tuple[str, ...]) -> bool:
    normalized_title = title.casefold()
    return any(keyword in normalized_title for keyword in keywords)


def _is_minutes_conclusion_title(title: str) -> bool:
    return _has_title_keyword(title, ("kết luận", "ket luan"))


# For meeting minutes, if the title contains keywords like "conclusion", use a prompt that focuses on extracting conclusions.
def _get_prompt_for_minutes_title(title: str) -> str:
    if _is_minutes_conclusion_title(title):
        return build_minutes_conclusion_prompt()

    if _has_title_keyword(
        title,
        ("chỉ đạo", "chi dao"),
        # ("bí thư", "chủ tịch", "chỉ đạo", "bi thu", "chu tich", "chi dao"),
    ):
        return build_each_person_prompt(
            max_bullet_points=10,
            compression_percent=60,
            transcript_sentence_range="2-3",
        )
    return build_each_person_prompt()


def _renumber_minutes_conclusion_titles(content: str, section_index: int) -> str:
    def _replace(match: re.Match[str]) -> str:
        leading_ws = match.group(1)
        local_index = match.group(2)
        title_text = match.group(3).strip()
        return f"{leading_ws}**{section_index}.{local_index}. {title_text}**"

    return re.sub(r"^(\s*)(?:\*\*)?(\d+)\.\s+(.*?)(?:\*\*)?$", _replace, content, flags=re.MULTILINE)


def _build_minutes_markdown(sections: List[Tuple[str, str, bool]]) -> str:
    lines = ["# I. THÀNH PHẦN THAM DỰ"]
    for title, _, _ in sections:
        lines.append(f"- {title}.")
    lines.append("")
    lines.append("# II. DIỄN BIẾN CUỘC HỌP")
    for index, (title, content, is_minutes_conclusion) in enumerate(sections, start=1):
        rendered_content = (
            _renumber_minutes_conclusion_titles(content.strip(), index)
            if is_minutes_conclusion
            else content.strip()
        )
        lines.append(f"**{index}. {title}**")
        lines.append(rendered_content)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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


def _uppercase_first_word(text: str) -> str:
    if not text:
        return text

    chars = list(text)
    sentence_end_chars = {".", "?", "!"}
    capitalize_next = True
    for idx, char in enumerate(chars):
        if char in sentence_end_chars:
            capitalize_next = True
            continue

        if capitalize_next and char.isalpha():
            chars[idx] = char.upper()
            capitalize_next = False

    return "".join(chars)


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
    transcription_result = _transcribe_internal(
        input_file=input_file,
        context=context,
        api_url=api_url,
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
    )
    return _build_transcribe_response(
        transcription_result=transcription_result,
        include_srt=include_srt,
        include_text=include_text,
    )


def _transcribe_internal(
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

    vad_model = _get_shared_vad_model()
    source_name = os.path.basename(urlparse(input_file).path) if input_file.startswith(("http://", "https://")) else os.path.basename(input_file)
    source_name = source_name or "input_audio"
    source_stem = os.path.splitext(source_name)[0]
    save_file = _build_output_path(input_file)

    if not has_speech(wav, vad_model):
        with open(save_file, "w", encoding="utf-8") as handle:
            handle.write("Unknown\n\n")

        srt_path = None
        if save_srt:
            srt_path = _save_srt_file(save_file, "")

        return {
            "duration_seconds": round(wav_duration_seconds, 3),
            "segment_count": 0,
            "failed_segments": [],
            "detected_language": "Unknown",
            "full_text": "",
            "segments": [],
            "srt_content": "",
            "text_file": os.path.abspath(save_file),
            "srt_file": os.path.abspath(srt_path) if srt_path else None,
        }

    if wav_duration_seconds >= vad_trigger_seconds:
        wav_list = process_vad(
            wav,
            vad_model,
            segment_threshold_s=vad_segment_threshold,
            max_segment_threshold_s=max_segment_seconds,
        )
    else:
        wav_list = [(0, len(wav), wav)]

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
                    results.append((idx, _uppercase_first_word(recog_text)))
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

    with open(save_file, "w", encoding="utf-8") as handle:
        handle.write(language + "\n")
        handle.write(full_text + "\n")

    srt_content = _compose_srt_content(wav_list, results)

    srt_path = None
    if save_srt:
        srt_path = _save_srt_file(save_file, srt_content or "")

    return {
        "duration_seconds": round(wav_duration_seconds, 3),
        "segment_count": len(wav_list),
        "failed_segments": failed_segments,
        "detected_language": language,
        "full_text": full_text,
        "segments": _serialize_segments(wav_list, results),
        "srt_content": srt_content,
        "text_file": os.path.abspath(save_file),
        "srt_file": os.path.abspath(srt_path) if srt_path else None,
    }


def _build_transcribe_response(
    transcription_result: Dict[str, object],
    include_srt: bool,
    include_text: bool,
) -> Dict[str, object]:
    response = {
        "duration_seconds": transcription_result["duration_seconds"],
        "segment_count": transcription_result["segment_count"],
        "failed_segments": transcription_result["failed_segments"],
    }
    if include_srt:
        response["srt_content"] = transcription_result["srt_content"]
    if include_text:
        response["full_text"] = transcription_result["full_text"]
    return response


def _transcribe_uploaded_file(
    file: Optional[UploadFile],
    *,
    context: str,
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
) -> Tuple[str, Dict[str, object]]:
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

        transcription_result = _transcribe_internal(
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
        )
        return original_name, transcription_result
    finally:
        file.file.close()
        shutil.rmtree(upload_dir, ignore_errors=True)
        _try_cleanup_cache_root(tmp_dir)


def _raise_as_http_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
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
        return generate_meeting_summary(
            text=request.text,
            model=request.model,
            locale=request.locale,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_details=request.include_details,
        )
    except Exception as exc:
        _raise_as_http_error(exc)


@app.post("/summarize/minutes")
async def summarize_minutes(
    file: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    # model: Optional[str] = Form(None),
    # locale: Optional[str] = Form(None),
    # temperature: float = Form(0.2),
    # max_tokens: int = Form(2000),
    # include_prompt: bool = Form(False),
) -> Dict[str, object]:
    _verify_api_key(x_api_key)
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Missing file in request body field 'file'.")

    try:
        markdown_text = (await file.read()).decode("utf-8-sig")
        sections = _split_markdown_sections(markdown_text)
        if not sections:
            raise ValueError("Markdown file must contain at least one heading with transcript content below it.")

        summarized_sections: List[Optional[Tuple[str, str, bool]]] = [None] * len(sections)

        def _summarize_section(index: int, original_title: str, transcript: str) -> Tuple[int, str, str, bool]:
            is_minutes_conclusion = _is_minutes_conclusion_title(original_title)
            system_prompt = _get_prompt_for_minutes_title(original_title)
            result = generate_each_person_from_transcript(
                transcript=f"{original_title}\n\n{transcript}",
                system_prompt=system_prompt,
                # model=model,
                # locale=locale,
                # temperature=temperature,
                # max_tokens=max_tokens,
                # include_prompt=include_prompt,
            )
            content = str(result["content"]).strip()
            display_title = _extract_summary_title(content, original_title)
            return index, display_title, _strip_summary_title_line(content), is_minutes_conclusion

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(_summarize_section, index, original_title, transcript)
                for index, (original_title, transcript) in enumerate(sections)
            ]
            for future in concurrent.futures.as_completed(futures):
                index, display_title, content, is_minutes_conclusion = future.result()
                summarized_sections[index] = (display_title, content, is_minutes_conclusion)

        ordered_sections = [item for item in summarized_sections if item is not None]
        if len(ordered_sections) != len(sections):
            raise ValueError("Failed to summarize all sections.")

        output_markdown = _build_minutes_markdown(ordered_sections)
        return {"content": output_markdown}
    except Exception as exc:
        _raise_as_http_error(exc)
    finally:
        file.file.close()


@app.post("/summarize/report")
async def summarize_report(
    file: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> Dict[str, object]:
    _verify_api_key(x_api_key)
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Missing file in request body field 'file'.")

    try:
        markdown_text = (await file.read()).decode("utf-8-sig")
        sections = _split_markdown_sections(markdown_text)
        if not sections:
            raise ValueError(
                "File markdown phải có ít nhất một tiêu đề, mỗi tiêu đề là một tổ "
                "và nội dung bên dưới là biên bản của tổ đó."
            )

        groups = [
            GroupTranscript(group_id=title, transcript=transcript)
            for title, transcript in sections
        ]
        return generate_report_from_groups(groups)
    except Exception as exc:
        _raise_as_http_error(exc)
    finally:
        file.file.close()


@app.post("/summarize/conclusions")
def summarize_conclusion(
    request: GenerateConclusionsRequest,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> Dict[str, object]:
    _verify_api_key(x_api_key)
    try:
        return generate_conclusions_from_summaries(
            transcript=request.transcript,
            # model=request.model,
            # locale=request.locale,
            # temperature=request.temperature,
            # max_tokens=request.max_tokens,
            # include_prompt=request.include_prompt,
        )
    except Exception as exc:
        _raise_as_http_error(exc)

@app.post("/summarize/live")
def summarize_live(
    request: LiveSummaryRequest,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> Dict[str, object]:
    _verify_api_key(x_api_key)
    try:
        return generate_live_summary(
            request.srt_content,
            agenda_title=request.agenda_title,
            speaker_roles=request.speaker_roles,
            meeting_id=request.meeting_id,
            meeting_date=request.meeting_date,
        )
    except LiveSummaryLLMError as exc:
        # The model failed, not the request: the conference resends these blocks.
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        _raise_as_http_error(exc)


@app.post("/summarize/tasks")
def summarize_tasks(
    request: ActionItemsRequest,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> Dict[str, object]:
    _verify_api_key(x_api_key)
    try:
        return generate_action_items(
            request.transcript,
            agenda_title=request.agenda_title,
            speaker_roles=request.speaker_roles,
            meeting_id=request.meeting_id,
            meeting_date=request.meeting_date,
        )
    except ActionItemsLLMError as exc:
        # The model failed, not the request: the same request can be sent again.
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        _raise_as_http_error(exc)


@app.post("/attribute-speakers")
def attribute_speakers_srt(
    request: AttributeSpeakersRequest,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
) -> List[Dict[str, str]]:
    _verify_api_key(x_api_key)
    try:
        if not request.srt_content or not request.srt_content.strip():
            raise ValueError("Field 'srt_content' must not be empty.")
        return attribute_speakers_as_srt(
            request.srt_content,
            client=client_from_env(request.model),
            roster=request.roster,
        )
    except Exception as exc:
        _raise_as_http_error(exc)


# @app.post("/transcribe-cmd")
# def transcribe(
#     request: TranscribeRequest,
#     x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
# ) -> Dict[str, object]:
#     _verify_api_key(x_api_key)
#     try:
#         return _transcribe(
#             input_file=request.input_file,
#             context=request.context,
#             api_url=_get_default_api_url(),
#             model=request.model,
#             api_timeout=request.api_timeout,
#             temperature=request.temperature,
#             skip_failed=request.skip_failed,
#             max_retries=request.max_retries,
#             num_threads=request.num_threads,
#             vad_segment_threshold=request.vad_segment_threshold,
#             max_segment_seconds=request.max_segment_seconds,
#             vad_trigger_seconds=request.vad_trigger_seconds,
#             tmp_dir=request.tmp_dir,
#             save_srt=request.save_srt,
#             include_srt=request.include_srt,
#             include_text=request.include_text,
#         )
#     except Exception as exc:
#         _raise_as_http_error(exc)


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
    try:
        original_name, transcription_result = _transcribe_uploaded_file(
            file=file,
            context=context,
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
        )
        result = _build_transcribe_response(
            transcription_result=transcription_result,
            include_srt=include_srt,
            include_text=include_text,
        )
        result["uploaded_filename"] = original_name
        return result
    except Exception as exc:
        _raise_as_http_error(exc)


@app.post("/v1/audio/transcriptions")
def transcribe_openai(
    file: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    model: Optional[str] = Form(None),
    prompt: str = Form(DEFAULT_CONTEXT),
    temperature: float = Form(0.2),
    response_format: str = Form("json"),
    language: Optional[str] = Form(None),
    api_timeout: int = Form(300),
    skip_failed: bool = Form(False),
    max_retries: int = Form(10),
    num_threads: int = Form(8),
    vad_segment_threshold: int = Form(60),
    max_segment_seconds: int = Form(120),
    vad_trigger_seconds: int = Form(70),
    tmp_dir: str = Form(DEFAULT_TMP_DIR),
    save_srt: bool = Form(False),
) -> Dict[str, object]:
    del language
    # _verify_api_key(x_api_key)

    try:
        _, transcription_result = _transcribe_uploaded_file(
            file=file,
            context=prompt,
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
        )

        normalized_format = response_format.strip().lower()
        if normalized_format == "json":
            return {
                "text": transcription_result["full_text"],
                "language": transcription_result["detected_language"],
            }
        if normalized_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": transcription_result["detected_language"],
                "duration": transcription_result["duration_seconds"],
                "text": transcription_result["full_text"],
                "segments": transcription_result["segments"],
                "failed_segments": transcription_result["failed_segments"],
            }
        raise HTTPException(
            status_code=400,
            detail="Unsupported response_format. Use 'json' or 'verbose_json'.",
        )
    except Exception as exc:
        _raise_as_http_error(exc)
        
@app.post("/api/v1/audio/transcriptions")
def transcribe_openai(
    file: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    model: Optional[str] = Form("Qwen/Qwen3-ASR-1.7B"),
    prompt: str = Form(DEFAULT_CONTEXT),
    temperature: float = Form(0.2),
    response_format: str = Form("json"),
    language: Optional[str] = Form(None),
    api_timeout: int = Form(300),
    skip_failed: bool = Form(False),
    max_retries: int = Form(10),
    num_threads: int = Form(8),
    vad_segment_threshold: int = Form(60),
    max_segment_seconds: int = Form(120),
    vad_trigger_seconds: int = Form(70),
    tmp_dir: str = Form(DEFAULT_TMP_DIR),
    save_srt: bool = Form(False),
) -> Dict[str, object]:
    del language
    _verify_api_key(x_api_key)

    try:
        _, transcription_result = _transcribe_uploaded_file(
            file=file,
            context=prompt,
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
        )

        normalized_format = response_format.strip().lower()
        if normalized_format == "json":
            return {
                "status": "ok",
                "transcript": transcription_result["full_text"],
            }
        raise HTTPException(
            status_code=400,
            detail="Unsupported response_format. Use 'json'.",
        )
    except Exception as exc:
        _raise_as_http_error(exc)


def run() -> None:
    host = os.getenv("QWEN3_ASR_HOST", "0.0.0.0")
    port = int(os.getenv("QWEN3_ASR_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()

