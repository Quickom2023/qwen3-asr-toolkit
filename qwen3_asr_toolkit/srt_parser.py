"""The conference transcript format: "[start -> end] Speaker:" blocks.

Parses srt_content into timed segments and turns them into the one-line form
the models read, "[HH:MM:SS] Speaker: text", with each speaker's role attached
when the caller supplied one.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional


DEFAULT_SPEAKER = "Không rõ"

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


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float  # seconds since midnight of the day the transcript starts
    text: str


@dataclass(frozen=True)
class Line:
    time: str  # "HH:MM:SS", the key a model cites
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


def _clean_time(value: object) -> str:
    return _clean(value).strip("[] ")


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


def to_12_hour(clock: str) -> str:
    """A line time, "15:49:30", as the conference writes it: "3:49:30 PM"."""
    hours, minutes, seconds = clock.split(":")
    hour = int(hours)
    return "%d:%s:%s %s" % (hour % 12 or 12, minutes, seconds, "PM" if hour >= 12 else "AM")


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
