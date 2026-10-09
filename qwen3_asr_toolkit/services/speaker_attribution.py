"""Speaker attribution over ASR transcripts via LLM anchor extraction.

See docs/superpowers/specs/2026-08-19-speaker-attribution-design.md
"""

import argparse
import bisect
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import requests

try:
    from dotenv import find_dotenv, load_dotenv  # type: ignore
except Exception:
    find_dotenv = None
    load_dotenv = None

# Mirrors api_server.py:43. Without this the CLI resolves a different provider than
# the API server does, because only api_server.py was loading .env. override=False
# keeps real environment variables authoritative over the file.
if load_dotenv and find_dotenv:
    load_dotenv(find_dotenv(usecwd=True), override=False)

STREAM_SEPARATOR = "\n"

CUE_PATTERN = re.compile(
    r"(?:^[ \t]*\d+[ \t]*\r?\n)?"
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})"
    r"[ \t]*-->[ \t]*"
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})"
    r"[^\n]*\r?\n"
    r"(.*?)"
    r"(?=\r?\n[ \t]*\r?\n|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    cues: List[Cue]
    spans: List[Tuple[int, int]]
    _starts: List[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._starts = [span[0] for span in self.spans]

    def cue_at(self, offset: int) -> Cue:
        position = bisect.bisect_right(self._starts, offset) - 1
        return self.cues[min(max(position, 0), len(self.cues) - 1)]


def parse_cues(raw: str) -> List[Cue]:
    cues: List[Cue] = []
    for match in CUE_PATTERN.finditer(raw):
        text = " ".join(match.group(3).split())
        if not text:
            continue
        cues.append(
            Cue(
                index=len(cues),
                start=_parse_timestamp(match.group(1)),
                end=_parse_timestamp(match.group(2)),
                text=text,
            )
        )
    if not cues:
        raise ValueError("No cues found in transcript.")
    return cues


def build_transcript(cues: List[Cue]) -> Transcript:
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for cue in cues:
        spans.append((cursor, cursor + len(cue.text)))
        parts.append(cue.text)
        cursor += len(cue.text) + len(STREAM_SEPARATOR)
    return Transcript(text=STREAM_SEPARATOR.join(parts), cues=cues, spans=spans)


DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_NUM_CTX = 32768
DEFAULT_CHAT_MODEL = "qwen3.5:4b"
_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class LLMConfig:
    provider: str
    endpoint: str
    api_key: str
    model: str
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    num_ctx: int = DEFAULT_NUM_CTX


def _normalize_endpoint(base: str, provider: str) -> str:
    cleaned = base.strip().rstrip("/")
    if provider == "ollama":
        if cleaned.endswith("/api/chat"):
            return cleaned
        if cleaned.endswith("/v1"):
            cleaned = cleaned[: -len("/v1")]
        return cleaned + "/api/chat"
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if not cleaned.endswith("/v1"):
        cleaned = cleaned + "/v1"
    return cleaned + "/chat/completions"


def build_payload(
    config: LLMConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, object],
    max_tokens: int,
) -> Dict[str, object]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if config.provider == "ollama":
        return {
            "model": config.model,
            "stream": False,
            "think": False,
            "keep_alive": "-5m",
            "format": schema,
            "options": {
                "temperature": 0,
                "top_p": 0.8,
                "top_k": 20,
                "repeat_penalty": 1.0,
                "num_ctx": config.num_ctx,
                "num_predict": max_tokens,
            },
            "messages": messages,
        }
    return {
        "model": config.model,
        "stream": False,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }


def _extract_content(body: Dict[str, object], provider: str) -> str:
    if provider == "ollama":
        message = body.get("message")
    else:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response missing choices.")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message.")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ValueError("LLM response content is not text.")
    return _FENCE_PATTERN.sub("", content).strip()


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, object],
        max_tokens: int = 2048,
    ) -> Dict[str, object]:
        payload = build_payload(
            self.config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            max_tokens=max_tokens,
        )
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        try:
            response = requests.post(
                self.config.endpoint,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise ValueError("LLM API request failed: " + detail) from exc
        except requests.RequestException as exc:
            raise ValueError("LLM API request failed: " + str(exc)) from exc

        content = _extract_content(response.json(), self.config.provider)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned invalid JSON: " + content[:200]) from exc
        if not isinstance(parsed, dict):
            raise ValueError("LLM returned JSON that is not an object.")
        return parsed


def client_from_env(model: Optional[str] = None) -> LLMClient:
    base = (
        os.getenv("SPEAKER_ATTRIBUTION_API_URL", "").strip()
        or os.getenv("SPEAKER_ATTRIBUTION_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    )
    provider = os.getenv("SPEAKER_ATTRIBUTION_PROVIDER", "").strip().lower()
    if provider not in ("ollama", "openai"):
        provider = "ollama" if ("/api/chat" in base or ":11434" in base) else "openai"
    resolved_model = (
        (model or "").strip()
        or os.getenv("SPEAKER_ATTRIBUTION_MODEL", "").strip()
        or DEFAULT_CHAT_MODEL
    )
    api_key = (
        os.getenv("SPEAKER_ATTRIBUTION_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    return LLMClient(
        LLMConfig(
            provider=provider,
            endpoint=_normalize_endpoint(base, provider),
            api_key=api_key,
            model=resolved_model,
        )
    )


DEFAULT_CHUNK_SIZE = 3000
DEFAULT_OVERLAP = 300


@dataclass
class Chunk:
    start: int
    end: int
    text: str


def chunk_stream(
    text: str,
    *,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    if not text:
        return []
    step = size - overlap
    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(start=start, end=end, text=text[start:end]))
        if end == len(text):
            break
        start += step
    return chunks


EVENT_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["self_intro", "handoff"]},
                    "quote": {"type": "string"},
                    "name": {"type": "string"},
                },
                "required": ["type", "quote", "name"],
            },
        }
    },
    "required": ["events"],
}

EXTRACTION_PROMPT = """
Bạn là công cụ trích xuất sự kiện giới thiệu người nói trong biên bản họp tiếng Việt.

NHIỆM VỤ: Đọc đoạn văn bản và liệt kê MỌI vị trí có một trong hai loại sự kiện sau.

1. "self_intro" — người đang nói TỰ giới thiệu bản thân ở ngôi thứ nhất.
2. "handoff" — người điều hành MỜI một người khác lên phát biểu. Sự kiện này gọi tên người sẽ nói TIẾP THEO.

KHÔNG phải sự kiện (bỏ qua, kể cả khi nêu đầy đủ tên và chức danh):
- Lời chào / xưng hô: "Kính thưa đồng chí ...", "Thưa đồng chí ...".
- Giới thiệu, chào mừng đại biểu tới DỰ họp: "xin trân trọng giới thiệu và nhiệt liệt chào mừng đồng chí ...". Những người này chỉ ngồi dự, KHÔNG lên phát biểu.
- Nhắc lại tên người khác trong lúc trình bày, hoặc cảm ơn người vừa nói xong.

"handoff" BẮT BUỘC có động từ MỜI ("kính mời", "xin mời") ĐI KÈM việc người được mời sắp làm ("trình bày", "điều hành", "phát biểu", "báo cáo"). Chỉ "chào mừng" hay "giới thiệu ... tới dự" thì KHÔNG phải sự kiện.

QUY TẮC BẮT BUỘC:
- Trường "quote" phải được sao chép NGUYÊN VĂN từ đoạn văn bản: không sửa chữ, không rút gọn, không thêm hay bớt dấu câu.
- Trường "name" chỉ chứa họ tên riêng, không kèm chức danh hay đơn vị.
- Chỉ trả về JSON đúng schema. Không giải thích, không thêm văn bản nào khác.

VÍ DỤ 1
Đoạn văn bản: "Kính thưa các đồng chí, tôi là Nguyễn Văn Nam, đại biểu đoàn Hà Nội, tôi xin báo cáo về tình hình sản xuất."
Kết quả: {"events": [{"type": "self_intro", "quote": "tôi là Nguyễn Văn Nam", "name": "Nguyễn Văn Nam"}]}

VÍ DỤ 2
Đoạn văn bản: "Tiếp theo chương trình, xin mời đồng chí Trần Thị Bích, Chủ nhiệm Ủy ban, trình bày dự thảo nghị quyết."
Kết quả: {"events": [{"type": "handoff", "quote": "xin mời đồng chí Trần Thị Bích", "name": "Trần Thị Bích"}]}

VÍ DỤ 3
Đoạn văn bản: "Như đồng chí Nguyễn Văn Nam vừa trình bày, chỉ tiêu năm nay đã hoàn thành vượt mức kế hoạch đề ra."
Kết quả: {"events": []}

VÍ DỤ 3B
Đoạn văn bản: "Kính thưa đồng chí Tô Lâm, Tổng Bí thư Ban Chấp hành Trung ương Đảng! Về phía Quốc hội, tôi xin trân trọng giới thiệu và nhiệt liệt chào mừng đồng chí Trần Thanh Mẫn, Ủy viên Bộ Chính trị, Chủ tịch Quốc hội. Đồng chí Nguyễn Khắc Định, Phó Chủ tịch Quốc hội."
Kết quả: {"events": []}

VÍ DỤ 4
Đoạn văn bản: "Về nội dung thứ hai, chúng ta cần tập trung vào công tác giải ngân vốn đầu tư công trong quý tới."
Kết quả: {"events": []}
""".strip()


@dataclass
class RawEvent:
    type: str
    quote: str
    name: str


def extract_events(chunk: Chunk, client: "LLMClient") -> List[RawEvent]:
    payload = client.complete_json(
        system_prompt=EXTRACTION_PROMPT,
        user_prompt="Đoạn văn bản:\n\n" + chunk.text,
        schema=EVENT_SCHEMA,
    )
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return []
    events: List[RawEvent] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        event_type = item.get("type")
        quote = item.get("quote")
        name = item.get("name")
        if event_type not in ("self_intro", "handoff"):
            continue
        if not isinstance(quote, str) or not quote.strip():
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        # A self_intro's quote contains the speaker's own name by definition, so a name
        # sourced from outside it is fabricated: the model returned quote "Tôi đã trình
        # bày xong dự thảo nghị quyết" with name "Trần Thanh Mẫn", harvested from a
        # resolution signature line two cues away, producing a spurious turn.
        # Handoffs are deliberately NOT grounded -- a chair may invite by role after
        # naming the person earlier in the same sentence ("... Bộ trưởng Bộ Tư pháp
        # Hoàng Thanh Tùng ... Kính mời Bộ trưởng!"), and grounding dropped that one.
        # Comparing through normalize_name lets the existing confusable folding absorb
        # ASR spelling drift between the quote and the name the model reported.
        if event_type == "self_intro" and normalize_name(name) not in normalize_name(quote):
            continue
        events.append(RawEvent(type=event_type, quote=quote.strip(), name=name.strip()))
    return events


GREETING_MARKERS = (
    "kính thưa",
    "kính gửi",
    "thưa các đồng chí",
    "thưa quý vị",
    "thưa hội nghị",
    "xin chào",
)
_GREETING_PATTERN = re.compile("|".join(re.escape(marker) for marker in GREETING_MARKERS))
BACKTRACK_WINDOW = 200
# Zero, not a fuzzy window: an anchor seen through the chunk overlap resolves to the
# IDENTICAL global offset, because both `chunk.start + local` and _backtrack work in
# global coordinates. A non-zero window does not catch duplicates it would otherwise
# miss -- it only merges genuinely distinct speakers whose turns are that short.
DEDUPE_WINDOW = 0


@dataclass
class Anchor:
    offset: int
    name: str
    type: str
    quote: str


def _locate(haystack: str, quote: str) -> Optional[Tuple[int, int]]:
    position = haystack.find(quote)
    if position >= 0:
        return position, position + len(quote)
    # Models normalise capitalisation at the start of a returned quote ("xin mời" ->
    # "Xin mời"), which fails an exact match on an otherwise perfect copy. Fall back to
    # a case-insensitive, whitespace-flexible match: we only need the offset, so case
    # cannot change which span is located.
    tokens = [re.escape(token) for token in quote.split()]
    if not tokens:
        return None
    match = re.search(r"\s+".join(tokens), haystack, re.IGNORECASE)
    return (match.start(), match.end()) if match else None


def _backtrack(text: str, offset: int, window: int = BACKTRACK_WINDOW) -> int:
    # One non-overlapping alternation, then take the LAST match. Scanning per
    # marker and keeping the max start is wrong: in "Kính thưa các đồng chí",
    # the suffix marker "thưa các đồng chí" starts later than "kính thưa" and
    # would truncate the greeting. finditer consumes "kính thưa" first, so the
    # subsumed suffix never matches separately.
    lower_bound = max(0, offset - window)
    matches = list(_GREETING_PATTERN.finditer(text[lower_bound:offset].lower()))
    return lower_bound + matches[-1].start() if matches else offset


def _dedupe(anchors: List[Anchor], window: int = DEDUPE_WINDOW) -> List[Anchor]:
    kept: List[Anchor] = []
    for anchor in anchors:
        collision = None
        for index, existing in enumerate(kept):
            if abs(existing.offset - anchor.offset) <= window:
                collision = index
                break
        if collision is None:
            kept.append(anchor)
        elif kept[collision].type == "handoff" and anchor.type == "self_intro":
            kept[collision] = anchor
    return sorted(kept, key=lambda item: item.offset)


def resolve_anchors(
    events_by_chunk: List[Tuple[Chunk, List[RawEvent]]],
    transcript: Transcript,
) -> List[Anchor]:
    anchors: List[Anchor] = []
    for chunk, events in events_by_chunk:
        for event in events:
            location = _locate(chunk.text, event.quote)
            if location is None:
                continue
            local_start, local_end = location
            if event.type == "self_intro":
                offset = _backtrack(transcript.text, chunk.start + local_start)
            else:
                offset = chunk.start + local_end
            anchors.append(
                Anchor(offset=offset, name=event.name, type=event.type, quote=event.quote)
            )
    anchors.sort(key=lambda item: (item.offset, 0 if item.type == "self_intro" else 1))
    return _dedupe(anchors)


SIMILARITY_THRESHOLD = 0.85
_CONFUSABLE_RULES = (("tr", "ch"), ("x", "s"), ("gi", "d"), ("r", "d"), ("l", "n"))


def normalize_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFD", name.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = stripped.replace("đ", "d")
    for source, target in _CONFUSABLE_RULES:
        stripped = stripped.replace(source, target)
    return " ".join(stripped.split())


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def canonicalize(
    names: List[str],
    *,
    roster: Optional[List[str]] = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[str, str]:
    if not names:
        return {}

    if roster:
        normalized_roster = [(entry, normalize_name(entry)) for entry in roster]
        mapping: Dict[str, str] = {}
        for name in set(names):
            key = normalize_name(name)
            best_entry = None
            best_score = 0.0
            for entry, entry_key in normalized_roster:
                score = _similarity(key, entry_key)
                if score > best_score:
                    best_entry, best_score = entry, score
            mapping[name] = best_entry if best_score >= threshold else name
        return mapping

    counts = Counter(names)
    clusters: List[List[str]] = []
    for name, _count in counts.most_common():
        key = normalize_name(name)
        target = None
        for cluster in clusters:
            if _similarity(key, normalize_name(cluster[0])) >= threshold:
                target = cluster
                break
        if target is None:
            clusters.append([name])
        else:
            target.append(name)

    mapping = {}
    for cluster in clusters:
        canonical = cluster[0]
        for surface in cluster:
            mapping[surface] = canonical
    return mapping


UNKNOWN_SPEAKER = "UNKNOWN"
LONG_SPAN_WARNING_CHARS = 15000


@dataclass
class Turn:
    speaker: str
    start_offset: int
    end_offset: int
    text: str
    start_cue: int
    end_cue: int
    start_time: float
    end_time: float
    anchor_quote: Optional[str]


def _make_turn(
    transcript: Transcript,
    speaker: str,
    start: int,
    end: int,
    quote: Optional[str],
) -> Turn:
    start_cue = transcript.cue_at(start)
    end_cue = transcript.cue_at(max(start, end - 1))
    return Turn(
        speaker=speaker,
        start_offset=start,
        end_offset=end,
        text=transcript.text[start:end].strip(),
        start_cue=start_cue.index,
        end_cue=end_cue.index,
        start_time=start_cue.start,
        end_time=end_cue.end,
        anchor_quote=quote,
    )


def build_turns(
    anchors: List[Anchor],
    transcript: Transcript,
    *,
    name_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[Turn], List[str]]:
    warnings: List[str] = []
    if not anchors:
        warnings.append("No speaker anchors found; entire transcript is unattributed.")
        return (
            [_make_turn(transcript, UNKNOWN_SPEAKER, 0, len(transcript.text), None)],
            warnings,
        )

    mapping = name_map or {}
    turns: List[Turn] = []
    if anchors[0].offset > 0:
        turns.append(_make_turn(transcript, UNKNOWN_SPEAKER, 0, anchors[0].offset, None))

    for index, anchor in enumerate(anchors):
        if index + 1 < len(anchors):
            end = anchors[index + 1].offset
        else:
            end = len(transcript.text)
        speaker = mapping.get(anchor.name, anchor.name)
        turns.append(_make_turn(transcript, speaker, anchor.offset, end, anchor.quote))

    for turn in turns:
        span = turn.end_offset - turn.start_offset
        if span > LONG_SPAN_WARNING_CHARS:
            warnings.append(
                "Unanchored span of %d characters attributed to '%s' (cues %d-%d); "
                "speakers may have changed without introducing themselves."
                % (span, turn.speaker, turn.start_cue, turn.end_cue)
            )
    return turns, warnings


DEFAULT_MAX_WORKERS = 4


def attribute_speakers(
    raw_transcript: str,
    *,
    client: Optional[LLMClient] = None,
    roster: Optional[List[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Dict[str, object]:
    transcript = build_transcript(parse_cues(raw_transcript))
    chunks = chunk_stream(transcript.text, size=chunk_size, overlap=overlap)
    active_client = client if client is not None else client_from_env()

    results: List[List[RawEvent]] = [[] for _ in chunks]
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(extract_events, chunk, active_client): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except ValueError:
                failures += 1

    anchors = resolve_anchors(list(zip(chunks, results)), transcript)
    name_map = canonicalize([anchor.name for anchor in anchors], roster=roster)
    turns, warnings = build_turns(anchors, transcript, name_map=name_map)

    if failures:
        warnings.insert(
            0,
            "Anchor extraction failed for %d of %d chunks; attribution is incomplete."
            % (failures, len(chunks)),
        )

    return {
        "turns": [asdict(turn) for turn in turns],
        "speakers": sorted({turn.speaker for turn in turns if turn.speaker != UNKNOWN_SPEAKER}),
        "warnings": warnings,
    }


def format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return "%02d:%02d:%02d,%03d" % (hours, minutes, whole_seconds, milliseconds)


def render_srt(cues: List[Cue]) -> str:
    blocks = [
        "%s --> %s\n%s" % (format_timestamp(cue.start), format_timestamp(cue.end), cue.text)
        for cue in cues
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _cue_owner(cue_start: int, cue_end: int, turns: List[Dict[str, object]]) -> str:
    # Anchors sit at character offsets, so a turn boundary can fall in the middle of a
    # cue. A cue cannot be split without inventing timestamps, so it goes whole to the
    # turn covering most of its characters. Ties go to the earlier turn, and a cue no
    # turn overlaps stays unattributed.
    best_speaker = UNKNOWN_SPEAKER
    best_overlap = 0
    for turn in turns:
        overlap = min(cue_end, int(turn["end_offset"])) - max(cue_start, int(turn["start_offset"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(turn["speaker"])
    return best_speaker


def group_turns_as_srt(
    transcript: Transcript,
    turns: List[Dict[str, object]],
) -> List[Dict[str, str]]:
    """One entry per speaker, in order of first appearance, carrying every cue they own.

    A speaker who holds the floor more than once yields a single entry: their cues are
    concatenated in transcript order, so the returned srt_content is the merge of all
    their turns rather than one entry per turn.
    """
    ordered_names: List[str] = []
    grouped: Dict[str, List[Cue]] = {}
    for cue, (cue_start, cue_end) in zip(transcript.cues, transcript.spans):
        speaker = _cue_owner(cue_start, cue_end, turns)
        name = "" if speaker == UNKNOWN_SPEAKER else speaker
        if name not in grouped:
            grouped[name] = []
            ordered_names.append(name)
        grouped[name].append(cue)
    return [
        {"speaker_name": name, "srt_content": render_srt(grouped[name])}
        for name in ordered_names
    ]


def attribute_speakers_as_srt(
    raw_transcript: str,
    *,
    client: Optional[LLMClient] = None,
    roster: Optional[List[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> List[Dict[str, str]]:
    """Attribute speakers and return the transcript regrouped as one SRT per speaker."""
    result = attribute_speakers(
        raw_transcript,
        client=client,
        roster=roster,
        chunk_size=chunk_size,
        overlap=overlap,
        max_workers=max_workers,
    )
    transcript = build_transcript(parse_cues(raw_transcript))
    return group_turns_as_srt(transcript, result["turns"])  # type: ignore[arg-type]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attribute speakers in a cue-formatted ASR transcript."
    )
    parser.add_argument("input", help="Path to the transcript file")
    parser.add_argument("--roster", help="Path to a newline-separated attendee list")
    parser.add_argument("--model", help="Override the chat model name")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output", help="Write JSON here instead of stdout")
    args = parser.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as handle:
        raw_transcript = handle.read()

    roster = None
    if args.roster:
        with open(args.roster, "r", encoding="utf-8") as handle:
            roster = [line.strip() for line in handle if line.strip()]

    result = attribute_speakers(
        raw_transcript,
        client=client_from_env(args.model),
        roster=roster,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_workers=args.max_workers,
    )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
    else:
        print(rendered)

    for warning in result["warnings"]:
        print("WARNING: " + warning, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
