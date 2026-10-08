"""Action items assigned or accepted during a whole meeting.

The conference backend sends the full transcript once, after the meeting. It
is cut into chunks of whole lines, up to CHUNK_MAX_CHARS each, and every chunk
is read by one model call with the action-item rules the live summary uses. The lines
just before a chunk come along as context, so a "Đồng ý, Sở A làm..." that
opens a chunk can be read against the request it answers. Chunks are
independent, so several calls run at once (llm_workers).

Each item must cite a line of its own chunk; code checks that and drops what
fails. Items are returned in transcript order, as each chunk found them: a task
restated in the closing remarks appears again, and a deadline changed later in
the meeting does not update the earlier item.
"""

import json
import logging
import re
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

from qwen3_asr_toolkit.action_items import (
    ACTION_ITEM_RULES,
    ACTION_ITEM_SCHEMA,
    TRANSCRIPT_FORMAT,
    chair_labels,
    clean_action_items,
    public_action_item,
    verify_action_items,
)
from qwen3_asr_toolkit.llm_inference import InferenceClient, InvalidJSONError, client_from_env
from qwen3_asr_toolkit.srt_parser import Line, _clean, build_lines, parse_transcript


logger = logging.getLogger(__name__)

# The model server and model come from the LIVE_SUMMARY_* variables, shared
# with the live summary. This route has no settings of its own: the
# constants below are the whole configuration.
LLM_ENV_PREFIX = "LIVE_SUMMARY"

# About 5-8 minutes of speech: the two 1-hour meetings in tests/fixtures run
# at 750-1100 characters a minute.
CHUNK_MAX_CHARS = 6000
# A longer line is split at sentence ends, so that no chunk grows past it.
LONG_LINE_CHARS = 4000
# Lines shown before a chunk as context.
CONTEXT_LINES = 2
# Model calls in flight at once: Ollama is kept to one at a time, an
# OpenAI-compatible server takes several.
OLLAMA_WORKERS = 1
OPENAI_WORKERS = 4
LLM_TEMPERATURE = 0.1
# Stops a looping model; not a cap on entries, of which there is none.
LLM_MAX_TOKENS = 4096

_SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?…])\s+")

_T = TypeVar("_T")


class ActionItemsLLMError(Exception):
    """The model produced no usable reply: server down, timeout, bad JSON, or a
    reply cut off at its token limit.

    Kept apart from ValueError so the route can answer 502 instead of 400.
    """


# ------------------------------------------------------------------ lines

def _split_text(text: str, room: int) -> List[str]:
    pieces: List[str] = []
    current = ""
    for sentence in _SENTENCE_END_PATTERN.split(text):
        while len(sentence) > room:
            # One sentence longer than a whole piece: cut at the last space.
            if current:
                pieces.append(current)
                current = ""
            cut = sentence.rfind(" ", 0, room)
            cut = cut if cut > 0 else room
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        candidate = (current + " " + sentence).strip()
        if current and len(candidate) > room:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def split_long_lines(lines: List[Line], max_chars: int = LONG_LINE_CHARS) -> List[Line]:
    """Split each line whose rendering exceeds max_chars at sentence ends.

    Every piece keeps the line's time and speaker, so a citation of that time
    stays valid whichever piece the model read it in.
    """
    result: List[Line] = []
    for line in lines:
        rendered = line.render()
        if len(rendered) <= max_chars:
            result.append(line)
            continue
        # A speaker label that fills the line by itself leaves no room; the
        # pieces then run over max_chars rather than never ending.
        room = max(max_chars - (len(rendered) - len(line.text)), max_chars // 4, 1)
        for piece in _split_text(line.text, room):
            result.append(Line(time=line.time, speaker=line.speaker, text=piece))
    return result


def chunk_lines(lines: List[Line], max_chars: int) -> List[Tuple[int, int]]:
    """[start, end) ranges of whole lines whose renderings, newline-joined, fit
    in max_chars. A line over budget by itself is a chunk of its own."""
    chunks: List[Tuple[int, int]] = []
    start, total = 0, 0
    for index, line in enumerate(lines):
        size = len(line.render())
        if index > start and total + 1 + size > max_chars:
            chunks.append((start, index))
            start, total = index, size
        else:
            total += size + (1 if index > start else 0)
    if lines:
        chunks.append((start, len(lines)))
    return chunks


# ----------------------------------------------------------------- prompt

# Shown in the prompt. Some servers (Ollama with thinking off) do not enforce
# the schema, and a model left to guess the shape drops source_time, which
# costs every task it found. Units and times are not from a real meeting, so
# the model has nothing to copy.
EXAMPLE: Dict[str, object] = {
    "action_items": [
        {
            "name": ["Sở Giáo dục và Đào tạo", "Sở Xây dựng"],
            "task": "Rà soát nhu cầu phòng học",
            "deadline": "trước ngày 30 tháng 11",
            "source_time": "14:05:10",
        },
        {
            "name": ["Sở Du lịch"],
            "task": "Chuẩn bị phương án đón khách",
            "deadline": "",
            "source_time": "14:05:10",
        },
    ]
}

SYSTEM_PROMPT = (
    "Bạn là thư ký cuộc họp. Mỗi lần bạn nhận một đoạn TRANSCRIPT của cuộc họp, có thể kèm "
    "NGỮ CẢNH là vài dòng nói ngay trước đoạn đó.\n\n"
    + TRANSCRIPT_FORMAT + "\n\n"
    "Liệt kê action_items của TRANSCRIPT: " + ACTION_ITEM_RULES + " NGỮ CẢNH chỉ để hiểu; "
    "KHÔNG lấy việc từ NGỮ CẢNH.\n\n"
    "Đoạn không có việc được giao hay được nhận thì trả về action_items rỗng, KHÔNG tự thêm.\n\n"
    "Viết bằng tiếng Việt. Chỉ trả về JSON theo schema. Mẫu (chỉ minh hoạ hình thức, KHÔNG "
    "chép nội dung):\n" + json.dumps(EXAMPLE, ensure_ascii=False)
)

RESPONSE_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {"action_items": {"type": "array", "items": ACTION_ITEM_SCHEMA}},
    "required": ["action_items"],
    "additionalProperties": False,
}


def chunk_prompt(
    lines: List[Line], start: int, end: int, *, agenda: str, chairs: List[str]
) -> str:
    parts: List[str] = []
    if agenda:
        parts.append("MỤC NGHỊ SỰ: " + agenda)
    if chairs:
        parts.append("CHỦ TOẠ: " + "; ".join(chairs))
    context = lines[max(0, start - CONTEXT_LINES):start]
    if context:
        parts.append("NGỮ CẢNH:\n" + "\n".join(line.render() for line in context))
    parts.append("TRANSCRIPT:\n" + "\n".join(line.render() for line in lines[start:end]))
    parts.append("Hãy liệt kê các việc được giao hoặc được nhận trong TRANSCRIPT.")
    return "\n\n".join(parts)


# ------------------------------------------------------------ read chunks

def _call(client: InferenceClient, user_prompt: str, where: str) -> Optional[Dict[str, object]]:
    """The model's reply, or None when it was not JSON twice: the caller skips
    that chunk (where, as "chunk 08:10:00"), so one bad reply does not cost
    the whole meeting."""
    for attempt in (1, 2):
        try:
            return client.complete_json_strict(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=RESPONSE_SCHEMA,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
        except InvalidJSONError as exc:
            if attempt == 2:
                logger.warning("action items: %s skipped after two invalid JSON replies: %s",
                               where, exc)
        except ValueError as exc:
            # A failed request, or ContextOverflowError: a reply cut off at
            # its limit, which asking again would only repeat.
            raise ActionItemsLLMError(str(exc)) from exc
    return None


def read_chunk(
    client: InferenceClient,
    lines: List[Line],
    start: int,
    end: int,
    *,
    agenda: str,
    chairs: List[str],
) -> Optional[List[Dict[str, object]]]:
    """The chunk's action items that cite one of its own lines, or None when
    its replies were not JSON."""
    reply = _call(client, chunk_prompt(lines, start, end, agenda=agenda, chairs=chairs),
                  "chunk %s" % lines[start].time)
    if reply is None:
        return None
    return verify_action_items(clean_action_items(reply.get("action_items")), lines[start:end])


# ----------------------------------------------------------- orchestrator

def llm_workers(client: InferenceClient) -> int:
    """Calls to run at once: one for Ollama, several for an OpenAI-compatible
    server. A client without a config, such as a test fake, counts as Ollama."""
    provider = getattr(getattr(client, "config", None), "provider", "")
    return OPENAI_WORKERS if provider == "openai" else OLLAMA_WORKERS


def _map(
    pool: ThreadPoolExecutor, function: Callable[[Tuple[int, int]], _T],
    chunks: List[Tuple[int, int]],
) -> List[_T]:
    """pool.map, except that a failed call cancels the calls still queued: a
    request that will answer 502 should not first spend minutes on them.

    Waits for the first failure, not in order: behind a slow call, the other
    workers would drain the queue before a later failure was seen.
    """
    futures = [pool.submit(function, chunk) for chunk in chunks]
    try:
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for future in futures:
            if future in done and future.exception() is not None:
                raise future.exception()
        return [future.result() for future in futures]
    except BaseException:
        for future in futures:
            future.cancel()
        raise


def find_action_items(
    srt_content: str,
    *,
    agenda_title: Optional[str] = None,
    speaker_roles: Optional[Dict[str, str]] = None,
    meeting_id: Optional[str] = None,
    client: Optional[InferenceClient] = None,
) -> List[Dict[str, object]]:
    """Every verified action item of the meeting, with its source_time, in
    transcript order."""
    lines = split_long_lines(build_lines(parse_transcript(srt_content), speaker_roles))
    if not lines:
        raise ValueError("Field 'srt_content' contains no spoken text.")
    chunks = chunk_lines(lines, CHUNK_MAX_CHARS)
    agenda = _clean(agenda_title)
    chairs = chair_labels(speaker_roles)
    resolved_client = client if client is not None else client_from_env(LLM_ENV_PREFIX)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=llm_workers(resolved_client)) as pool:
        results = _map(
            pool,
            lambda chunk: read_chunk(resolved_client, lines, chunk[0], chunk[1],
                                     agenda=agenda, chairs=chairs),
            chunks,
        )
    if all(found is None for found in results):
        # A model that never answers in JSON would otherwise look like a
        # meeting that assigned nothing.
        raise ActionItemsLLMError("Every chunk reply was invalid JSON.")
    items = [item for found in results if found for item in found]
    logger.info(
        "action items meeting_id=%s lines=%d chunks=%d skipped=%d items=%d seconds=%.1f",
        meeting_id or "-",
        len(lines),
        len(chunks),
        sum(1 for found in results if found is None),
        len(items),
        time.monotonic() - started,
    )
    return items


def generate_action_items(
    srt_content: str,
    *,
    agenda_title: Optional[str] = None,
    speaker_roles: Optional[Dict[str, str]] = None,
    meeting_id: Optional[str] = None,
    client: Optional[InferenceClient] = None,
) -> Dict[str, object]:
    items = find_action_items(
        srt_content,
        agenda_title=agenda_title,
        speaker_roles=speaker_roles,
        meeting_id=meeting_id,
        client=client,
    )
    return {"action_items": [public_action_item(item) for item in items]}
