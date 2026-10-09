"""Deadline tokens and their ISO 8601 datetimes.

The model copies each deadline as spoken and also writes it as one token
(DEADLINE_TOKEN_RULES): a date, a period, an event, or NONE. Code checks the
token and turns dates and periods into the end of the last day they allow, or
the hour the speaker named, counted from the meeting date; the model does no
date arithmetic.
"""

import calendar
import re
import unicodedata
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional


NONE_TOKEN = "NONE"

# Vietnam has no daylight saving time.
MEETING_TIMEZONE = timezone(timedelta(hours=7))
# A deadline day lasts until its end, unless the speaker named an hour.
END_OF_DAY = time(23, 59, 59)

DEADLINE_TOKEN_RULES = (
    "deadline_token là deadline viết lại theo ĐÚNG MỘT trong các dạng sau, không giải thích, "
    "không thêm ký tự nào: DATE:DD/MM hoặc DATE:DD/MM/YYYY CHỈ khi người nói nêu rõ ngày và "
    "tháng (\"trước ngày 25 tháng 11\" -> DATE:25/11), thêm \" HH:MM\" khi người nói nêu giờ "
    "(\"trước 20 giờ ngày 20 tháng 10\" -> DATE:20/10 20:00); PERIOD:TODAY cho trong ngày, "
    "ngay hôm nay; PERIOD:TOMORROW cho ngày mai; hai dạng này cũng thêm \" HH:MM\" khi người "
    "nói nêu giờ (\"trước 17 giờ ngày mai\" -> PERIOD:TOMORROW 17:00); PERIOD:DAYS:N, "
    "PERIOD:WEEKS:N, "
    "PERIOD:MONTHS:N cho trong N ngày, tuần, tháng tới (\"trong 3 ngày\" -> PERIOD:DAYS:3, "
    "\"2 tuần nữa\" -> PERIOD:WEEKS:2); PERIOD:THIS_WEEK cho trong tuần này; PERIOD:NEXT_WEEK "
    "cho trong tuần sau; PERIOD:THIS_MONTH cho trong tháng này; PERIOD:MONTH:MM hoặc "
    "PERIOD:MONTH:MM/YYYY cho trong, giữa hoặc cuối tháng MM; PERIOD:Q1 đến PERIOD:Q4, hoặc "
    "PERIOD:Q1/YYYY, cho quý I đến quý IV; PERIOD:THIS_YEAR cho trong năm nay, cuối năm; "
    "PERIOD:YEAR:YYYY cho trong năm YYYY; EVENT:<thời hạn chép như người nói> khi thời hạn gắn "
    "với một sự kiện hoặc không khớp dạng nào ở trên (\"trước khi phê duyệt giá đất\" -> "
    "EVENT:trước khi phê duyệt giá đất); NONE khi deadline để trống. Số viết bằng chữ thì ghi "
    "bằng chữ số (\"ngày hai mươi tháng mười\" -> DATE:20/10). KHÔNG tự tính ngày từ \"mai\", "
    "\"tuần sau\", \"nữa\" hay ngày họp: dùng dạng PERIOD tương ứng. Chỉ ghi năm khi người nói "
    "nêu năm."
)

# Tolerant of spacing, so that it reads the model's token and the clean one.
_DATE_PATTERN = re.compile(
    r"DATE:\s*(\d{1,2})\s*/\s*(\d{1,2})(?:\s*/\s*(\d{4}))?(?:\s+(\d{1,2}):(\d{2}))?"
)
# Today and tomorrow may name an hour, like a date.
_DAY_PATTERN = re.compile(r"PERIOD:(TODAY|TOMORROW)(?:\s+(\d{1,2}):(\d{2}))?")
# The hour on a clean token, as _clean_hour writes it.
_HOUR_SUFFIX = re.compile(r" (\d{2}):(\d{2})$")
_MONTH_PATTERN = re.compile(r"PERIOD:MONTH:(\d{1,2})(?:/(\d{4}))?")
_QUARTER_PATTERN = re.compile(r"PERIOD:Q([1-4])(?:/(\d{4}))?")
_YEAR_PATTERN = re.compile(r"PERIOD:YEAR:(\d{4})")
_AHEAD_PATTERN = re.compile(r"PERIOD:(DAYS|WEEKS|MONTHS):(\d{1,3})")
_NAMED_PERIODS = (
    "PERIOD:TODAY",
    "PERIOD:TOMORROW",
    "PERIOD:THIS_WEEK",
    "PERIOD:NEXT_WEEK",
    "PERIOD:THIS_MONTH",
    "PERIOD:THIS_YEAR",
)
_EVENT_PREFIX = "EVENT:"
# A year that has 29 February, for checking a date given without a year.
_LEAP_YEAR = 2000

# Words for numbers, days and months as the transcript spells them. A DATE
# token whose spoken deadline has none of these and no digit was computed
# or invented by the model ("ngày mai" -> DATE:25/04).
_NUMBER_WORDS = frozenset(
    "một mốt hai ba bốn tư năm lăm nhăm sáu bảy tám chín mười mươi linh lẻ trăm nghìn ngàn "
    "mồng mùng rằm giêng chạp".split()
)


def _real_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _last_day(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _with_year(text: str, year: Optional[str]) -> str:
    return text + ("/" + year if year else "")


def _says_a_number(raw: str) -> bool:
    if re.search(r"\d", raw):
        return True
    words = re.findall(r"\w+", unicodedata.normalize("NFC", raw).casefold())
    return any(word in _NUMBER_WORDS for word in words)


def _clean_hour(hour: Optional[str], minute: Optional[str]) -> Optional[str]:
    """" HH:MM", "" when no hour was given, or None for an impossible one."""
    if hour is None:
        return ""
    if int(hour) > 23 or int(minute) > 59:
        return None
    return " %02d:%02d" % (int(hour), int(minute))


def _clean_date(match: "re.Match[str]", raw: str) -> str:
    day, month, year, hour, minute = match.groups()
    if _real_date(int(year or _LEAP_YEAR), int(month), int(day)) is None:
        return NONE_TOKEN
    at = _clean_hour(hour, minute)
    if at is None or not _says_a_number(raw):
        return NONE_TOKEN
    return _with_year("DATE:%02d/%02d" % (int(day), int(month)), year) + at


def _clean_day(match: "re.Match[str]", raw: str) -> str:
    day, hour, minute = match.groups()
    at = _clean_hour(hour, minute)
    if at is None:
        return NONE_TOKEN
    # An hour raw says no number for was invented; the day still holds.
    return "PERIOD:" + day + (at if _says_a_number(raw) else "")


def clean_deadline_token(value: object, raw: str) -> str:
    """The token in its canonical form, or NONE when it is not one of the
    forms, names an impossible date or hour, comes with no spoken deadline,
    raw, or is a date where raw says no number (the model inferred it).
    An hour on today or tomorrow where raw says no number is dropped."""
    if not isinstance(value, str) or not raw:
        return NONE_TOKEN
    text = value.strip()
    if text[:len(_EVENT_PREFIX)].upper() == _EVENT_PREFIX:
        event = text[len(_EVENT_PREFIX):].strip()
        return _EVENT_PREFIX + event if event else NONE_TOKEN

    squeezed = re.sub(r"\s+", " ", text).upper()
    match = _DATE_PATTERN.fullmatch(squeezed)
    if match:
        return _clean_date(match, raw)
    match = _DAY_PATTERN.fullmatch(squeezed)
    if match:
        return _clean_day(match, raw)

    token = squeezed.replace(" ", "")
    if token in _NAMED_PERIODS:
        return token
    match = _MONTH_PATTERN.fullmatch(token)
    if match:
        month, year = match.groups()
        if not 1 <= int(month) <= 12:
            return NONE_TOKEN
        return _with_year("PERIOD:MONTH:%02d" % int(month), year)
    match = _AHEAD_PATTERN.fullmatch(token)
    if match:
        unit, count = match.groups()
        return "PERIOD:%s:%d" % (unit, int(count)) if int(count) > 0 else NONE_TOKEN
    if _QUARTER_PATTERN.fullmatch(token) or _YEAR_PATTERN.fullmatch(token):
        return token
    return NONE_TOKEN


def resolve_deadline(token: str, meeting_date: Optional[date]) -> Optional[str]:
    """The end of the last day a clean token allows, or the hour it names, in
    Vietnam time, as YYYY-MM-DDTHH:MM:SS+07:00; or None for an event, NONE,
    or a token that needs the meeting date when there is none."""
    at = END_OF_DAY
    match = _HOUR_SUFFIX.search(token)
    if match:
        token = token[:match.start()]
        at = time(int(match.group(1)), int(match.group(2)))
    day = _last_day_allowed(token, meeting_date)
    if day is None:
        return None
    return datetime.combine(day, at, MEETING_TIMEZONE).isoformat()


def _months_ahead(start: date, months: int) -> date:
    years, month_index = divmod(start.month - 1 + months, 12)
    year, month = start.year + years, month_index + 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def _last_day_allowed(token: str, meeting_date: Optional[date]) -> Optional[date]:
    """A date or period given without a year is the next one that has not
    ended by the meeting date. A week ends on Sunday. N days, weeks or
    months run from the meeting date."""
    match = _DATE_PATTERN.fullmatch(token)
    if match:
        day, month, year = (int(group) if group else None for group in match.groups()[:3])
        if year is not None:
            return _real_date(year, month, day)
        if meeting_date is None:
            return None
        resolved = _real_date(meeting_date.year, month, day)
        if resolved is not None and resolved < meeting_date:
            resolved = _real_date(meeting_date.year + 1, month, day)
        return resolved

    match = _MONTH_PATTERN.fullmatch(token) or _QUARTER_PATTERN.fullmatch(token)
    if match:
        number, year = match.groups()
        month = int(number) * (3 if token.startswith("PERIOD:Q") else 1)
        if year:
            return _last_day(int(year), month)
        if meeting_date is None:
            return None
        resolved = _last_day(meeting_date.year, month)
        if resolved < meeting_date:
            resolved = _last_day(meeting_date.year + 1, month)
        return resolved

    match = _YEAR_PATTERN.fullmatch(token)
    if match:
        return date(int(match.group(1)), 12, 31)

    if meeting_date is None:
        return None
    match = _AHEAD_PATTERN.fullmatch(token)
    if match:
        unit, count = match.group(1), int(match.group(2))
        if unit == "MONTHS":
            return _months_ahead(meeting_date, count)
        return meeting_date + timedelta(days=count * (7 if unit == "WEEKS" else 1))

    if token == "PERIOD:TODAY":
        return meeting_date
    if token == "PERIOD:TOMORROW":
        return meeting_date + timedelta(days=1)
    if token == "PERIOD:THIS_WEEK":
        return meeting_date + timedelta(days=6 - meeting_date.weekday())
    if token == "PERIOD:NEXT_WEEK":
        return meeting_date + timedelta(days=13 - meeting_date.weekday())
    if token == "PERIOD:THIS_MONTH":
        return _last_day(meeting_date.year, meeting_date.month)
    if token == "PERIOD:THIS_YEAR":
        return date(meeting_date.year, 12, 31)
    return None


def parse_meeting_date(value: Optional[str]) -> Optional[date]:
    """The meeting's date in Vietnam from an ISO 8601 date or datetime, or
    None when none was given. A datetime with an offset is moved to Vietnam
    time first; one without is taken as Vietnam time."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        moment = datetime.fromisoformat(re.sub(r"[zZ]$", "+00:00", text))
    except ValueError:
        raise ValueError(
            "Field 'meeting_date' must be an ISO 8601 date such as 2026-10-08, got %r." % value
        ) from None
    if moment.tzinfo is not None:
        moment = moment.astimezone(MEETING_TIMEZONE)
    return moment.date()
