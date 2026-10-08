"""The action-item contract shared by the live summary and the whole-meeting
task route: what counts as an action item, the shape the model writes it in,
and the code checks applied to every reply.

The model writes an item as {name, task, deadline, deadline_token,
source_time}: name lists who does the task, the leading unit first, then those
who support it. deadline is the deadline as spoken, and deadline_token the same
deadline as a token (deadlines.py). Code keeps the spoken deadline as
deadline_raw and puts the token's ISO 8601 datetime in deadline: those are the
response's names. source_time names the line the task was given or accepted
on; code checks it against the transcript. Responses leave out source_time and
deadline_token (public_action_item).

Kept apart from live_summary.py, which the tasks route ships without.
"""

import unicodedata
from datetime import date
from typing import Dict, List, Optional

from qwen3_asr_toolkit.deadlines import (
    DEADLINE_TOKEN_RULES,
    clean_deadline_token,
    resolve_deadline,
)
from qwen3_asr_toolkit.srt_parser import Line, _as_list, _clean, _clean_time, _name_key


# How speaker_roles marks the chair, after casefolding.
CHAIR_MARKS = ("chủ toạ", "chủ tọa")

_TRAILING_PUNCTUATION = " .;,"

_STRING = {"type": "string"}


# ----------------------------------------------------------------- prompt

TRANSCRIPT_FORMAT = (
    "Mỗi dòng TRANSCRIPT có dạng \"[HH:MM:SS] Người nói: nội dung\". Người nói có ghi "
    "\"(chủ toạ)\" hoặc \"(chủ tọa)\" là người chủ trì cuộc họp; dưới đây viết chung là "
    "\"(chủ toạ)\". Dòng CHỦ TOẠ, nếu có, nêu người chủ trì, kể cả khi họ không nói trong "
    "đoạn này."
)

ACTION_ITEM_RULES = (
    "CHỈ những việc đã được GIAO hoặc được NHẬN trong cuộc họp: người chủ trì "
    "giao hoặc yêu cầu một đơn vị/người thực hiện (\"giao\", \"yêu cầu\", \"đề nghị [đơn vị] "
    "khẩn trương...\"), hoặc một đơn vị tự nhận thực hiện (\"xin nhận\", \"sẽ hoàn thành...\"). "
    "Đề xuất hay kiến nghị gửi lên cấp trên (\"đề nghị tỉnh...\", \"kiến nghị...\", "
    "\"xin cho phép...\") KHÔNG phải action_items. Khi biết người chủ trì (dòng CHỦ TOẠ hoặc "
    "người nói ghi \"(chủ toạ)\"), CHỈ người chủ trì giao việc: câu \"đề nghị\" hay \"yêu cầu\" "
    "một đơn vị của BẤT KỲ người nói nào khác là kiến nghị, KHÔNG phải action_items. Khi không "
    "biết người chủ trì, người nói nào cũng có thể giao việc. Nếu không chắc, KHÔNG ghi vào "
    "action_items. Nếu người nói KHÔNG nêu đơn vị hay người thực hiện thì KHÔNG tạo "
    "action_items, KHÔNG tự suy ra. Các đơn vị được giao những việc khác nhau là các mục riêng. "
    "Mỗi mục gồm: name là danh sách đơn vị hoặc người thực hiện, đúng như người nói nêu: đơn vị "
    "chủ trì trước, rồi các đơn vị phối hợp nếu có. task là MỘT câu nêu hành động cụ thể, KHÔNG "
    "lặp lại name và thời hạn. deadline là thời hạn chép đúng như người nói; để trống nếu người "
    "nói không nêu, KHÔNG tự suy ra. " + DEADLINE_TOKEN_RULES + " source_time chép NGUYÊN mốc "
    "HH:MM:SS của dòng chứa câu giao hoặc nhận việc."
)

ACTION_ITEM_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "name": {"type": "array", "items": _STRING},
        "task": _STRING,
        "deadline": _STRING,
        "deadline_token": _STRING,
        "source_time": _STRING,
    },
    "required": ["name", "task", "deadline", "deadline_token", "source_time"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------ items

def _clean_names(value: object) -> List[str]:
    # A lone string is one name; repeats differing only in case go.
    names: List[str] = []
    seen = set()
    for raw in value if isinstance(value, list) else [value]:
        name = _clean(raw)
        if name and _name_key(name).casefold() not in seen:
            seen.add(_name_key(name).casefold())
            names.append(name)
    return names


def _clean_task(first_name: str, value: object) -> str:
    task, name = _name_key(value), _name_key(first_name)
    # The prompt asks for the task without its names; some replies repeat the first.
    if task.casefold().startswith(name.casefold()):
        task = task[len(name):].lstrip(" :,-")
    return task.rstrip(_TRAILING_PUNCTUATION)


def clean_action_items(
    raw: object, meeting_date: Optional[date] = None
) -> List[Dict[str, object]]:
    """Coerce the model's action items into {name, task, deadline_raw,
    deadline_token, deadline, source_time}.

    An item with no name or no task goes: nobody was named, or nothing to do.
    deadline is the end of the last day the token allows, as an ISO 8601
    datetime counted from meeting_date, or None.
    """
    items: List[Dict[str, object]] = []
    for item in _as_list(raw):
        if not isinstance(item, dict):
            continue
        names = _clean_names(item.get("name"))
        task = _clean_task(names[0], item.get("task")) if names else ""
        if not task:
            continue
        # The model's deadline is the spoken one; the response calls it deadline_raw.
        raw_deadline = _clean(item.get("deadline")).rstrip(_TRAILING_PUNCTUATION)
        token = clean_deadline_token(item.get("deadline_token"), raw_deadline)
        items.append(
            {
                "name": names,
                "task": task,
                "deadline_raw": raw_deadline,
                "deadline_token": token,
                "deadline": resolve_deadline(token, meeting_date),
                "source_time": _clean_time(item.get("source_time")),
            }
        )
    return items


def verify_action_items(
    items: List[Dict[str, object]], lines: List[Line]
) -> List[Dict[str, object]]:
    """The items citing the start time of one of lines."""
    known_times = {line.time for line in lines}
    return [item for item in items if item["source_time"] in known_times]


# Used inside the server, left out of responses.
_INTERNAL_FIELDS = ("source_time", "deadline_token")


def public_action_item(item: Dict[str, object]) -> Dict[str, object]:
    """An action item as a response carries it while source times are hidden."""
    return {key: value for key, value in item.items() if key not in _INTERNAL_FIELDS}


def chair_labels(speaker_roles: Optional[Dict[str, str]]) -> List[str]:
    """"name - role" of each speaker whose role marks the chair."""
    labels: List[str] = []
    for name, role in (speaker_roles or {}).items():
        role = _clean(role)
        if any(mark in unicodedata.normalize("NFC", role).casefold() for mark in CHAIR_MARKS):
            labels.append("%s - %s" % (_name_key(name), role))
    return labels
