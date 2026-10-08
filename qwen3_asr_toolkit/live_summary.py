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
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

from qwen3_asr_toolkit.action_items import (
    ACTION_ITEM_RULES,
    ACTION_ITEM_SCHEMA,
    TRANSCRIPT_FORMAT,
    chair_labels,
    clean_action_items,
    public_action_item,
    verify_action_items,
)
from qwen3_asr_toolkit.deadlines import parse_meeting_date
from qwen3_asr_toolkit.llm_inference import InferenceClient, client_from_env
from qwen3_asr_toolkit.srt_parser import (
    Line,
    _as_list,
    _clean,
    _clean_time,
    build_lines,
    parse_transcript,
)


logger = logging.getLogger(__name__)

ENV_PREFIX = "LIVE_SUMMARY"
# A 2-minute chunk is ~2k chars; this only guards against a caller that stops chunking.
DEFAULT_MAX_TRANSCRIPT_CHARS = 40000
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

_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")
_NUMBER_SEPARATOR_PATTERN = re.compile(r"[.,]")


class LiveSummaryLLMError(Exception):
    """The model produced no usable reply: server down, timeout or bad JSON.

    Kept apart from ValueError so the route can answer 502 (retry next round)
    instead of 400 (the request itself is wrong).
    """


# ------------------------------------------------------------- transcript

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


def clean_summary(raw: object, meeting_date: Optional[date] = None) -> Dict[str, object]:
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

    tables: List[Dict[str, object]] = []
    for item in _as_list(source.get("tables")):
        table = _clean_table(item)
        if table is not None:
            tables.append(table)

    return {
        "key_points": key_points,
        "action_items": clean_action_items(source.get("action_items"), meeting_date),
        "tables": tables,
    }


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
        "action_items": verify_action_items(summary["action_items"], lines),
        "tables": tables,
    }


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def hide_source_times(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "key_points": summary["key_points"],
        "action_items": [public_action_item(item) for item in summary["action_items"]],
        "tables": [
            {"title": table["title"], "data": table["data"]} for table in summary["tables"]
        ],
    }


def _render_action_item(item: Dict[str, object]) -> str:
    task = item["task"]
    text = "- %s: %s" % (", ".join(item["name"]), task[:1].upper() + task[1:])
    if item["deadline_raw"]:
        text += " (Thời hạn: %s)" % item["deadline_raw"]
    if "source_time" in item:
        text += " _(%s)_" % item["source_time"]
    return text


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
    + TRANSCRIPT_FORMAT + "\n\n"
    "Chỉ tóm tắt những gì có trong TRANSCRIPT này, gồm:\n"
    "- key_points: mọi ý chính của đoạn này, mỗi ý ngắn gọn, theo thứ tự xuất hiện. Đề xuất, kiến nghị, "
    "câu hỏi và báo cáo tình hình đều thuộc key_points.\n"
    "- action_items: " + ACTION_ITEM_RULES + " Kiến nghị và việc không chắc ghi vào "
    "key_points.\n"
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
        "action_items": {"type": "array", "items": ACTION_ITEM_SCHEMA},
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
    meeting_date: Optional[str] = None,
    client: Optional[InferenceClient] = None,
) -> Dict[str, object]:
    meeting_day = parse_meeting_date(meeting_date)
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
    cleaned = clean_summary(raw, meeting_day)
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
