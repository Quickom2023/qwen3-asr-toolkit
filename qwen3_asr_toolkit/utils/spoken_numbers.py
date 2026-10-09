"""Undo a lisp the transcript writes down: "hai mươi năm tỷ" for "hai mươi lăm tỷ".

After "mươi" or "mười", a final five is said "lăm". A speaker who says l as n
makes it "năm", which also means "year", and the models then read 25 tỷ as
20 tỷ, or "ngày hai mươi lăm tháng mười" as 20 October. A "năm" is put back
to "lăm" only when a unit follows it, so "hai mươi năm qua" stays twenty years.
"""

import re
import unicodedata


# Words that cannot follow "năm" meaning "year". Nouns such as "người" or "hộ"
# are left out: "hai mươi năm người dân chờ đợi" is twenty years.
_UNITS = (
    "tỷ", "tỉ", "triệu", "nghìn", "ngàn", "đồng", "phần trăm",
    "tháng", "ngày", "tuần", "giờ", "phút", "giây",
    "km", "ki-lô-mét", "ha", "héc-ta", "mét", "tấn",
)
_LISPED_FIVE = re.compile(
    r"\b(mươi|mười)(\s+)(n)(ăm)(?=\s*%%|\s+(?:%s)\b)" % "|".join(map(re.escape, _UNITS)),
    re.IGNORECASE,
)


def _as_lam(match: re.Match) -> str:
    tens, space, n, rest = match.groups()
    return tens + space + ("L" if n == "N" else "l") + rest


def fix_lisped_five(text: str) -> str:
    """Turn "hai mươi năm tỷ" into "hai mươi lăm tỷ"; keep "hai mươi năm qua"."""
    return _LISPED_FIVE.sub(_as_lam, unicodedata.normalize("NFC", text))
