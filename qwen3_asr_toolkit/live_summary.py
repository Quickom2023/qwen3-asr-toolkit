"""Live summary of the last couple of minutes of a meeting.

The conference backend calls this about every 2 minutes with only the
transcript that arrived since its last successful call. The server keeps no
state; each response covers only its own chunk, and the conference appends it
to what is already on screen.

One LLM call per request. The model writes JSON only. Every action item and
table must name the transcript line it came from, and every number in a table
must appear in the transcript; code checks both and drops what fails. The
markdown shown on screen is rendered here, so its layout never depends on the
model.
"""

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from qwen3_asr_toolkit.llm_inference import InferenceClient, client_from_env


logger = logging.getLogger(__name__)

ENV_PREFIX = "LIVE_SUMMARY"
# A 2-minute chunk is ~2k chars; this only guards against a caller that stops chunking.
DEFAULT_MAX_TRANSCRIPT_CHARS = 40000
DEFAULT_SPEAKER = "Không rõ"
TRUNCATION_MARKER = "[... phần đầu đã lược bỏ ...]"

MIN_TABLE_COLUMNS = 2
MIN_TABLE_ROWS = 2

LLM_TEMPERATURE = 0.1
# Stops a looping model; not a cap on entries, of which there is none.
LLM_MAX_TOKENS = 4096

# Hidden for now; verification still uses them. Set True to return them again.
SHOW_SOURCE_TIMES = False

KEY_POINTS_HEADING = "## Ý chính"
ACTION_ITEMS_HEADING = "## Việc cần làm"
TABLES_HEADING = "## Bảng số liệu"

# How speaker_roles marks the chair, after casefolding.
CHAIR_MARKS = ("chủ toạ", "chủ tọa")

DAY_SECONDS = 24 * 3600
HALF_DAY_SECONDS = 12 * 3600

_WHITESPACE_PATTERN = re.compile(r"\s+")
_CLOCK_PATTERN = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s*([AaPp][Mm])?\s*$")
# One block header: "[3:49:30 PM -> 3:49:31 PM] Đình Nam Nhữ:". The speaker is
# everything up to the colon that ends the line, so a name may contain colons.
_BLOCK_HEADER_PATTERN = re.compile(
    r"^[ \t]*\[([^\]\n]+?)[ \t]*-{1,2}>[ \t]*([^\]\n]+?)[ \t]*\][ \t]*(.*?)[ \t]*:[ \t]*$",
    re.MULTILINE,
)
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")
_NUMBER_SEPARATOR_PATTERN = re.compile(r"[.,]")


class LiveSummaryLLMError(Exception):
    """The model produced no usable reply: server down, timeout or bad JSON.

    Kept apart from ValueError so the route can answer 502 (retry next round)
    instead of 400 (the request itself is wrong).
    """


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float  # seconds since midnight of the day the transcript starts
    text: str


@dataclass(frozen=True)
class Line:
    time: str  # "HH:MM:SS", the key action items and tables cite
    speaker: str  # name, plus " - role" when the caller supplied one
    text: str

    def render(self) -> str:
        return "[%s] %s: %s" % (self.time, self.speaker, self.text)


def _clean(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return ""
    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def _as_list(value: object) -> List[object]:
    return value if isinstance(value, list) else []


def _name_key(value: object) -> str:
    # NFC, so a speaker_roles key typed in NFD still matches the transcript.
    return unicodedata.normalize("NFC", _clean(value))


# ------------------------------------------------------------- transcript

def parse_clock(value: str) -> float:
    """Seconds since midnight for "3:49:30 PM", "3:49:30 pm" or "15:49:30"."""
    match = _CLOCK_PATTERN.match(value)
    if not match:
        raise ValueError("Unrecognised time %r in srt_content." % value)
    hours, minutes, seconds = (int(match.group(n)) for n in (1, 2, 3))
    meridiem = (match.group(4) or "").upper()
    if meridiem:
        if not 1 <= hours <= 12:
            raise ValueError("Unrecognised time %r in srt_content." % value)
        hours = hours % 12 + (12 if meridiem == "PM" else 0)
    if hours > 23 or minutes > 59 or seconds > 59:
        raise ValueError("Unrecognised time %r in srt_content." % value)
    return float(hours * 3600 + minutes * 60 + seconds)


def format_clock(seconds: float) -> str:
    total = int(seconds) % DAY_SECONDS
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return "%02d:%02d:%02d" % (hours, minutes, secs)


def parse_transcript(raw: str) -> List[Segment]:
    """Split the conference's transcript format into segments.

    A block is a "[start -> end] Speaker:" header followed by its text, which
    runs until the next header and may span several lines. Only the start time
    is used. A start more than 12 hours earlier than the one before means the
    meeting crossed midnight.
    """
    text = raw.replace("\r\n", "\n")
    headers = list(_BLOCK_HEADER_PATTERN.finditer(text))
    if not headers:
        raise ValueError(
            "Field 'srt_content' contains no '[start -> end] Speaker:' blocks."
        )
    segments: List[Segment] = []
    day_offset = 0.0
    last_start: Optional[float] = None
    for index, header in enumerate(headers):
        body_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        start = parse_clock(header.group(1)) + day_offset
        if last_start is not None and start < last_start - HALF_DAY_SECONDS:
            day_offset += DAY_SECONDS
            start += DAY_SECONDS
        last_start = start
        segments.append(
            Segment(
                speaker=_clean(header.group(3)),
                start=start,
                text=_clean(text[header.end():body_end]),
            )
        )
    return segments


def build_lines(
    segments: List[Segment], speaker_roles: Optional[Dict[str, str]] = None
) -> List[Line]:
    """One line per non-empty segment, oldest first, with roles attached."""
    roles = dict(
        (_name_key(name), _clean(role)) for name, role in (speaker_roles or {}).items()
    )
    lines: List[Line] = []
    # sorted() is stable, so segments sharing a start keep the caller's order.
    for segment in sorted(segments, key=lambda item: item.start):
        if not segment.text:
            continue
        speaker = segment.speaker or DEFAULT_SPEAKER
        role = roles.get(_name_key(segment.speaker))
        if role:
            speaker = "%s - %s" % (speaker, role)
        lines.append(Line(time=format_clock(segment.start), speaker=speaker, text=segment.text))
    return lines


def truncate_lines(lines: List[str], max_chars: int) -> Tuple[List[str], bool]:
    """Drop the oldest lines until the rest, newline-joined, fit in max_chars.

    The newest line is always kept, even alone over budget: the most recent
    speech is what this update exists to reflect.
    """
    kept: List[str] = []
    total = 0
    for line in reversed(lines):
        cost = len(line) + (1 if kept else 0)
        if kept and total + cost > max_chars:
            break
        kept.append(line)
        total += cost
    kept.reverse()
    return kept, len(kept) < len(lines)


# ---------------------------------------------------------------- summary

def _clean_time(value: object) -> str:
    return _clean(value).strip("[] ")


def _clean_table(item: object) -> Optional[Dict[str, object]]:
    if not isinstance(item, dict):
        return None
    columns = [_clean(column) for column in _as_list(item.get("columns"))]
    if len(columns) < MIN_TABLE_COLUMNS:
        return None
    width = len(columns)
    rows: List[List[str]] = []
    for raw_row in _as_list(item.get("rows")):
        if not isinstance(raw_row, list):
            continue
        cells = [_clean(cell) for cell in raw_row][:width]
        cells += [""] * (width - len(cells))
        if any(cells):
            rows.append(cells)
    if len(rows) < MIN_TABLE_ROWS:
        return None
    source_times = [_clean_time(time) for time in _as_list(item.get("source_times"))]
    return {
        "title": _clean(item.get("title")),
        "data": [columns] + rows,  # header row first
        "source_times": [time for time in source_times if time],
    }


def clean_summary(raw: object) -> Dict[str, object]:
    """Coerce a model reply into the Summary shape.

    Malformed entries are dropped rather than rejected: a partial summary on
    screen beats a 502 over one bad table row. Unknown fields are not kept.
    """
    source = raw if isinstance(raw, dict) else {}

    key_points: List[str] = []
    seen = set()
    for item in _as_list(source.get("key_points")):
        text = _clean(item)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            key_points.append(text)

    action_items: List[Dict[str, str]] = []
    for item in _as_list(source.get("action_items")):
        if not isinstance(item, dict):
            continue
        task = _clean(item.get("task"))
        if not task:
            continue
        action_items.append({"task": task, "source_time": _clean_time(item.get("source_time"))})

    tables: List[Dict[str, object]] = []
    for item in _as_list(source.get("tables")):
        table = _clean_table(item)
        if table is not None:
            tables.append(table)

    return {"key_points": key_points, "action_items": action_items, "tables": tables}


def _start_with_unit(unit: str, task: str) -> str:
    # An empty task stays empty, so clean_summary drops the item.
    if not task or unit.casefold() in task.casefold():
        return task
    first_word = task.split(" ", 1)[0]
    # "Rà soát..." becomes "Sở X rà soát...", but "GPMB..." keeps its capitals.
    if first_word[1:] == first_word[1:].lower():
        task = task[:1].lower() + task[1:]
    return "%s %s" % (unit, task)


def apply_units(raw: Dict[str, object]) -> Dict[str, object]:
    """Use the model's `unit` to drop ownerless action items and lead each task.

    `unit` exists only in the model's reply; the response carries the owner as
    the start of `task`. An empty or missing unit means nobody was named, so
    the item goes.
    """
    items: List[object] = []
    for item in _as_list(raw.get("action_items")):
        if not isinstance(item, dict):
            continue
        unit = _clean(item.get("unit"))
        if not unit:
            continue
        items.append(dict(item, task=_start_with_unit(unit, _clean(item.get("task")))))
    return dict(raw, action_items=items)


def _number_keys(text: str) -> List[Tuple[str, ...]]:
    # "3.200", "3,200" and "03.200" all become ("3", "200"): the separator
    # convention and a leading zero differ between speakers, the digits do not.
    keys = []
    for match in _NUMBER_PATTERN.finditer(text):
        groups = _NUMBER_SEPARATOR_PATTERN.split(match.group())
        keys.append((str(int(groups[0])),) + tuple(groups[1:]))
    return keys


def verify_sources(summary: Dict[str, object], lines: List[Line]) -> Dict[str, object]:
    """Drop what the transcript does not back.

    An action item must cite the start time of a real line. A table must cite
    at least one, and each of its rows may only contain numbers that occur
    somewhere in the transcript, which rules out invented and computed figures.
    """
    known_times: Set[str] = set()
    known_numbers: Set[Tuple[str, ...]] = set()
    for line in lines:
        known_times.add(line.time)
        known_numbers.update(_number_keys(line.text))

    action_items = [
        item for item in summary["action_items"] if item["source_time"] in known_times
    ]

    tables: List[Dict[str, object]] = []
    for table in summary["tables"]:
        times: List[str] = []
        for time in table["source_times"]:
            if time in known_times and time not in times:
                times.append(time)
        header, body = table["data"][0], table["data"][1:]
        rows = [
            row
            for row in body
            if all(key in known_numbers for cell in row for key in _number_keys(cell))
        ]
        if not times or len(rows) < MIN_TABLE_ROWS:
            continue
        tables.append(dict(table, data=[header] + rows, source_times=times))

    return {
        "key_points": summary["key_points"],
        "action_items": action_items,
        "tables": tables,
    }


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def hide_source_times(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "key_points": summary["key_points"],
        "action_items": [{"task": item["task"]} for item in summary["action_items"]],
        "tables": [
            {"title": table["title"], "data": table["data"]} for table in summary["tables"]
        ],
    }


def _render_action_item(item: Dict[str, str]) -> str:
    if "source_time" not in item:
        return "- " + item["task"]
    return "- %s _(%s)_" % (item["task"], item["source_time"])


def _render_table(table: Dict[str, object]) -> str:
    lines: List[str] = []
    if table["title"]:
        lines.append("**%s**" % table["title"])
        lines.append("")
    columns = table["data"][0]
    lines.append("| " + " | ".join(_escape_cell(column) for column in columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for row in table["data"][1:]:
        lines.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    if "source_times" in table:
        lines.append("")
        lines.append("_Nguồn: %s_" % ", ".join(table["source_times"]))
    return "\n".join(lines)


def render_markdown(summary: Dict[str, object]) -> str:
    """Render a verified summary. Sections with no entries are left out."""
    blocks: List[str] = []
    if summary["key_points"]:
        blocks.append(
            "\n".join([KEY_POINTS_HEADING] + ["- " + point for point in summary["key_points"]])
        )
    if summary["action_items"]:
        blocks.append(
            "\n".join(
                [ACTION_ITEMS_HEADING]
                + [_render_action_item(item) for item in summary["action_items"]]
            )
        )
    if summary["tables"]:
        blocks.append(
            "\n\n".join([TABLES_HEADING] + [_render_table(table) for table in summary["tables"]])
        )
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


# ----------------------------------------------------------------- prompt

SYSTEM_PROMPT = (
    "Bạn là thư ký tóm tắt trực tiếp cuộc họp. Mỗi lần bạn nhận một đoạn TRANSCRIPT khoảng "
    "2 phút vừa diễn ra.\n\n"
    "Mỗi dòng TRANSCRIPT có dạng \"[HH:MM:SS] Người nói: nội dung\". Người nói có ghi "
    "\"(chủ toạ)\" hoặc \"(chủ tọa)\" là người chủ trì cuộc họp; dưới đây viết chung là "
    "\"(chủ toạ)\". Dòng CHỦ TOẠ, nếu có, nêu người chủ trì, kể cả khi họ không nói trong "
    "đoạn này.\n\n"
    "Chỉ tóm tắt những gì có trong TRANSCRIPT này, gồm:\n"
    "- key_points: mọi ý chính của đoạn này, mỗi ý ngắn gọn, theo thứ tự xuất hiện. Đề xuất, kiến nghị, "
    "câu hỏi và báo cáo tình hình đều thuộc key_points.\n"
    "- action_items: CHỈ những việc đã được GIAO hoặc được NHẬN trong cuộc họp: người chủ trì "
    "giao hoặc yêu cầu một đơn vị/người thực hiện (\"giao\", \"yêu cầu\", \"đề nghị [đơn vị] "
    "khẩn trương...\"), hoặc một đơn vị tự nhận thực hiện (\"xin nhận\", \"sẽ hoàn thành...\"). "
    "Đề xuất hay kiến nghị gửi lên cấp trên (\"đề nghị tỉnh...\", \"kiến nghị...\", "
    "\"xin cho phép...\") KHÔNG phải action_items. Khi biết người chủ trì (dòng CHỦ TOẠ hoặc "
    "người nói ghi \"(chủ toạ)\"), CHỈ người chủ trì giao việc: câu \"đề nghị\" hay \"yêu cầu\" "
    "một đơn vị của BẤT KỲ người nói nào khác là kiến nghị, KHÔNG phải action_items, ghi vào "
    "key_points. Khi không biết người chủ trì, người nói nào cũng có thể giao việc. Nếu không chắc, KHÔNG ghi vào action_items "
    "mà ghi vào key_points. unit là tên đơn vị hoặc người thực hiện chính, đúng như người nói "
    "nêu; nếu người nói KHÔNG nêu đơn vị hay người thực hiện thì KHÔNG tạo action_items mà ghi "
    "vào key_points, KHÔNG tự suy ra. task là MỘT câu nêu hành động cụ thể, BẮT ĐẦU bằng unit "
    "(ví dụ \"Sở A chủ trì, Sở B phối hợp rà soát..., báo cáo trước ngày...\"), kèm thời hạn NẾU "
    "người nói có nêu; thời hạn ghi đúng như người nói, KHÔNG tự suy ra thời hạn khi không được "
    "nêu. source_time chép NGUYÊN mốc HH:MM:SS "
    "của dòng chứa câu giao hoặc nhận việc.\n"
    "- tables: CHỈ tạo khi có từ 2 số liệu hoặc nội dung cùng loại trở lên được nói rõ để so "
    "sánh (giữa các đơn vị, giữa các kỳ, kế hoạch và thực hiện, dự án và vướng mắc). Chép số "
    "đúng như transcript, giữ nguyên đơn vị. KHÔNG tính tổng, tỷ lệ hay chênh lệch. Mỗi dòng "
    "có đúng số ô bằng số cột. source_times là các mốc HH:MM:SS của những dòng chứa số liệu.\n\n"
    "Đoạn ngắn thường không có việc được giao hay số liệu để so sánh: khi đó trả về danh sách "
    "rỗng, KHÔNG tự thêm. Lời chào, lời mời phát biểu và thủ tục điều hành KHÔNG phải key_points.\n\n"
    "Viết bằng tiếng Việt. Chỉ trả về JSON theo schema."
)

_STRING = {"type": "string"}

SUMMARY_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "key_points": {"type": "array", "items": _STRING},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "unit": _STRING,
                    "task": _STRING,
                    "source_time": _STRING,
                },
                "required": ["unit", "task", "source_time"],
                "additionalProperties": False,
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": _STRING,
                    "columns": {"type": "array", "items": _STRING},
                    "rows": {"type": "array", "items": {"type": "array", "items": _STRING}},
                    "source_times": {"type": "array", "items": _STRING},
                },
                "required": ["title", "columns", "rows", "source_times"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["key_points", "action_items", "tables"],
    "additionalProperties": False,
}


def build_user_prompt(
    transcript_lines: List[str],
    *,
    agenda_title: str = "",
    chairs: Optional[List[str]] = None,
    truncated: bool = False,
) -> str:
    parts: List[str] = []
    if agenda_title:
        parts.append("MỤC NGHỊ SỰ: " + agenda_title)
    if chairs:
        parts.append("CHỦ TOẠ: " + "; ".join(chairs))
    body = ([TRUNCATION_MARKER] if truncated else []) + list(transcript_lines)
    parts.append("TRANSCRIPT:\n" + "\n".join(body))
    parts.append("Hãy tóm tắt đoạn này.")
    return "\n\n".join(parts)


# ----------------------------------------------------------- orchestrator

def chair_labels(speaker_roles: Optional[Dict[str, str]]) -> List[str]:
    """"name - role" of each speaker whose role marks the chair."""
    labels: List[str] = []
    for name, role in (speaker_roles or {}).items():
        role = _clean(role)
        if any(mark in unicodedata.normalize("NFC", role).casefold() for mark in CHAIR_MARKS):
            labels.append("%s - %s" % (_name_key(name), role))
    return labels


def _max_transcript_chars() -> int:
    raw = os.getenv(ENV_PREFIX + "_MAX_TRANSCRIPT_CHARS", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_TRANSCRIPT_CHARS
    return value if value > 0 else DEFAULT_MAX_TRANSCRIPT_CHARS


def generate_live_summary(
    srt_content: str,
    *,
    agenda_title: Optional[str] = None,
    speaker_roles: Optional[Dict[str, str]] = None,
    meeting_id: Optional[str] = None,
    client: Optional[InferenceClient] = None,
) -> Dict[str, object]:
    lines = build_lines(parse_transcript(srt_content), speaker_roles)
    if not lines:
        raise ValueError("Field 'srt_content' contains no spoken text.")
    transcript, truncated = truncate_lines(
        [line.render() for line in lines], _max_transcript_chars()
    )
    user_prompt = build_user_prompt(
        transcript,
        agenda_title=_clean(agenda_title),
        chairs=chair_labels(speaker_roles),
        truncated=truncated,
    )
    resolved_client = client if client is not None else client_from_env(ENV_PREFIX)
    try:
        raw = resolved_client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=SUMMARY_SCHEMA,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
    except ValueError as exc:
        raise LiveSummaryLLMError(str(exc)) from exc
    # complete_json repairs a reply cut off at LLM_MAX_TOKENS; missing keys mean it was cut.
    missing = [
        key for key in SUMMARY_SCHEMA["required"] if not isinstance(raw.get(key), list)
    ]
    if missing:
        raise LiveSummaryLLMError(
            "LLM reply is missing %s (cut off at the token limit?)" % ", ".join(missing)
        )

    returned_action_items = len(raw["action_items"])
    cleaned = clean_summary(apply_units(raw))
    summary = verify_sources(cleaned, lines)
    logger.info(
        "live summary meeting_id=%s lines=%d truncated=%s key_points=%d "
        "action_items=%d/%d tables=%d/%d",
        meeting_id or "-",
        len(lines),
        truncated,
        len(summary["key_points"]),
        len(summary["action_items"]),
        returned_action_items,
        len(summary["tables"]),
        len(cleaned["tables"]),
    )
    if not SHOW_SOURCE_TIMES:
        summary = hide_source_times(summary)
    return {
        "summary": summary,
        "content": render_markdown(summary)
    }
