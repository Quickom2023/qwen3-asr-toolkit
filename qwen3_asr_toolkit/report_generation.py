"""Aggregate N per-group meeting transcripts into one Vietnamese report.

Five stages. Four ask the LLM for structured JSON; the fifth is pure Python
that computes every number in the document. The model never writes a count, a
quantifier or a footnote — it only writes sentences and says which source
opinions each sentence covers.

Stages run sequentially: the Ollama deployment behind this serves one request
at a time, so parallelism would only queue.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from qwen3_asr_toolkit.llm_inference import InferenceClient, client_from_env
from qwen3_asr_toolkit.qwen3asr import QwenASR


logger = logging.getLogger(__name__)

STANCES = ("nhất trí", "đề nghị", "băn khoăn", "không tán thành", "cho rằng")
DEFAULT_STANCE = "cho rằng"

PART_GENERAL = "I"
PART_SPECIFIC = "II"
PART_TITLES = {
    PART_GENERAL: "VỀ VẤN ĐỀ CHUNG",
    PART_SPECIFIC: "VỀ CÁC NỘI DUNG CỤ THỂ",
}
# Markdown markers for the rendered report: "# " for the top-level sections
# (the two parts and the outside-outline section) and "**" around
# each sub-heading.
SECTION_HEADING_PREFIX = "# "
SUBTITLE_MARKER = "**"

OUTSIDE_OUTLINE_TITLE = "NỘI DUNG NGOÀI ĐỀ CƯƠNG"
NO_OPINION_TEXT = "Không có ý kiến."

MAP_BATCH_SIZE = 20
MAJORITY_RATIO = 0.5
MANY_THRESHOLD = 10
SOME_THRESHOLD = 3

MAX_GENERAL_TITLES = 6
MAX_SPECIFIC_TITLES = 13


@dataclass(frozen=True)
class GroupTranscript:
    group_id: str
    transcript: str


@dataclass
class Point:
    id: str
    group_index: int
    group_id: str
    stance: str
    target: str
    content: str


@dataclass(frozen=True)
class OutlineItem:
    part: str
    index: int
    title: str

    @property
    def key(self) -> str:
        return "%s.%d" % (self.part, self.index)


@dataclass
class Bullet:
    sentence: str
    point_ids: List[str] = field(default_factory=list)


def normalize_stance(value: object) -> str:
    """Clamp a model-supplied stance to the five-value enum.

    Ollama's `format` does not enforce the enum, so out-of-enum values arrive
    routinely (`yêu cầu làm rõ`, `tán thành`, ...). Clamping keeps the point
    rather than dropping it, but it is lossy: a request recorded as a neutral
    assertion also stops stage 4 merging it with the requests it belongs with.
    Every clamp is logged so that drift is visible instead of silent — a rising
    rate here means the stage 1 prompt needs another synonym.
    """
    if isinstance(value, str):
        cleaned = " ".join(value.strip().lower().split())
        if cleaned in STANCES:
            return cleaned
    logger.warning(
        "Stance %r is outside STANCES; falling back to %r", value, DEFAULT_STANCE
    )
    return DEFAULT_STANCE


def stance_from_item(raw: Dict[str, object]) -> str:
    """Read the stance, recovering it from a misspelled key ("huang" for "huong").

    Only enum members are accepted, and the free-text fields are skipped.
    """
    value = raw.get("huong")
    if isinstance(value, str) and " ".join(value.strip().lower().split()) in STANCES:
        return normalize_stance(value)

    for key, candidate in raw.items():
        if key in ("huong", "doi_tuong", "noi_dung"):
            continue
        if (
            isinstance(candidate, str)
            and " ".join(candidate.strip().lower().split()) in STANCES
        ):
            logger.warning("Stance read from key %r instead of 'huong'", key)
            return normalize_stance(candidate)

    return normalize_stance(value)


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return QwenASR.remove_foreign_characters(value).strip()


# ---------------------------------------------------------------- stage 1

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "y_kien": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "huong": {"type": "string", "enum": list(STANCES)},
                    "doi_tuong": {"type": "string"},
                    "noi_dung": {"type": "string"},
                },
                "required": ["huong", "doi_tuong", "noi_dung"],
            },
        }
    },
    "required": ["y_kien"],
}

EXTRACT_SYSTEM_PROMPT = """Bạn bóc ý kiến từ biên bản thảo luận của một tổ.

Chỉ in ra DUY NHẤT một đối tượng JSON hợp lệ đúng theo mẫu {"y_kien":[...]}. Không viết lời dẫn, không giải thích, không dùng markdown, không bọc trong dấu ```. Kiểm tra trước khi trả lời: chuỗi phải bắt đầu bằng ký tự { và kết thúc bằng ký tự }. Sau dấu ] đóng danh sách y_kien phải còn đúng một dấu } đóng đối tượng.

Mỗi ý kiến độc lập là một phần tử, gồm ba trường:
- huong: chọn đúng một trong: nhất trí, đề nghị, băn khoăn, không tán thành, cho rằng
- doi_tuong: vấn đề được nói tới, 3-10 từ
- noi_dung: 1-2 câu, bắt đầu bằng động từ viết thường

Quy tắc:
- Chỉ bóc điều có trong biên bản. Không thêm, không suy diễn, không đổi số liệu.
- Bỏ lời chào hỏi, cảm ơn, thủ tục điều hành.
- Một câu chứa hai đề nghị khác nhau thì tách thành hai ý kiến.
- Nhiều câu cùng diễn giải một đề nghị thì gộp thành một ý kiến.
- "đồng ý nhưng đề nghị..." phải giữ cả hai vế thành hai ý kiến.
- huong phải là ĐÚNG một trong 5 giá trị, quy đổi từ đồng nghĩa:
  tán thành, đồng tình, thống nhất, nhất trí cao, ủng hộ -> "nhất trí";
  phản đối, không đồng tình, không nhất trí -> "không tán thành";
  yêu cầu, kiến nghị, mong muốn, cần làm rõ, cần bổ sung -> "đề nghị";
  lo ngại, e ngại, chưa yên tâm, chưa thuyết phục -> "băn khoăn".
  Không được tự đặt giá trị khác, kể cả khi biên bản dùng từ khác.
- Tổ không có ý kiến thì trả về danh sách rỗng, không bịa.

Ví dụ:
Biên bản: "Kính thưa Quốc hội. Tôi nhất trí với sự cần thiết đầu tư Dự án. Tuy nhiên đề nghị làm rõ nguồn vốn trung ương bố trí cho Dự án."
Kết quả: {"y_kien":[{"huong":"nhất trí","doi_tuong":"sự cần thiết đầu tư","noi_dung":"nhất trí với sự cần thiết đầu tư Dự án."},{"huong":"đề nghị","doi_tuong":"nguồn vốn","noi_dung":"đề nghị làm rõ nguồn vốn trung ương bố trí cho Dự án."}]}"""


def extract_points(
    groups: List[GroupTranscript],
    client: InferenceClient,
    *,
    max_tokens: int = 2000,
) -> List[Point]:
    points: List[Point] = []
    for group_index, group in enumerate(groups):
        if not group.transcript or not group.transcript.strip():
            raise ValueError(
                "Biên bản của '%s' rỗng, không bóc được ý kiến." % group.group_id
            )

        response = client.complete_json(
            system_prompt=EXTRACT_SYSTEM_PROMPT,
            user_prompt="Biên bản:\n" + group.transcript.strip(),
            schema=EXTRACT_SCHEMA,
            max_tokens=max_tokens,
            temperature=0.0,
            array_property="y_kien",
        )
        raw_items = response.get("y_kien")
        if not isinstance(raw_items, list):
            raw_items = []

        local_index = 0
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            content = _clean(raw.get("noi_dung"))
            if not content:
                continue
            local_index += 1
            points.append(
                Point(
                    id="T%02d-%03d" % (group_index + 1, local_index),
                    group_index=group_index,
                    group_id=group.group_id,
                    stance=stance_from_item(raw),
                    target=_clean(raw.get("doi_tuong")),
                    content=content,
                )
            )
    return points


# ---------------------------------------------------------------- stage 2

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "chung": {"type": "array", "items": {"type": "string"}},
        "cu_the": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["chung", "cu_the"],
}

OUTLINE_SYSTEM_PROMPT = """Bạn lập đề cương cho báo cáo tổng hợp ý kiến thảo luận tổ.

Chỉ in ra DUY NHẤT một đối tượng JSON hợp lệ đúng theo mẫu {"chung":[...],"cu_the":[...]}. Không viết lời dẫn, không giải thích, không dùng markdown, không bọc trong dấu ```. Kiểm tra trước khi trả lời: chuỗi phải bắt đầu bằng ký tự { và kết thúc bằng ký tự }.

Đọc danh sách ý kiến và đặt tiêu đề cho các nhóm vấn đề. Trả về hai danh sách:
- chung: các vấn đề bao trùm (sự cần thiết, thẩm quyền, trình tự thủ tục, hồ sơ, sự phù hợp với quy hoạch). Tối đa 6 tiêu đề.
- cu_the: các nội dung cụ thể (phạm vi, quy mô, hình thức đầu tư, tổng mức đầu tư, nguồn vốn, tiến độ, cơ chế đặc thù...). Tối đa 13 tiêu đề.

Quy tắc:
- Mỗi tiêu đề bắt đầu bằng "Về " và dài 3-12 từ.
- Tiêu đề đặt theo vấn đề, không theo hướng quan điểm. Không đặt "Về ý kiến đồng ý", "Về ý kiến phản đối".
- Không trùng nghĩa giữa các tiêu đề.
- Chỉ đặt tiêu đề cho vấn đề thực sự xuất hiện trong danh sách.

Ví dụ:
Danh sách: "- [nhất trí] sự cần thiết: nhất trí với sự cần thiết đầu tư. / - [đề nghị] nguồn vốn: đề nghị làm rõ vốn trung ương. / - [đề nghị] hồ sơ: đề nghị bổ sung đánh giá tác động môi trường."
Kết quả: {"chung":["Về sự cần thiết đầu tư","Về hồ sơ chủ trương đầu tư"],"cu_the":["Về nguồn vốn và khả năng cân đối vốn"]}"""


def format_points_for_outline(points: List[Point]) -> str:
    lines = []
    for point in points:
        target = point.target or "nội dung khác"
        lines.append("- [%s] %s: %s" % (point.stance, target, point.content))
    return "\n".join(lines)


def _dedupe_titles(raw_titles: object, limit: int) -> List[str]:
    titles: List[str] = []
    seen = set()
    if not isinstance(raw_titles, list):
        return titles
    for raw in raw_titles:
        title = _clean(raw)
        if not title:
            continue
        marker = title.lower()
        if marker in seen:
            continue
        seen.add(marker)
        titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def build_outline(
    points: List[Point],
    client: InferenceClient,
    *,
    max_tokens: int = 1200,
) -> List[OutlineItem]:
    if not points:
        raise ValueError("Không bóc được ý kiến nào từ các tổ.")

    response = client.complete_json(
        system_prompt=OUTLINE_SYSTEM_PROMPT,
        user_prompt="Danh sách ý kiến:\n" + format_points_for_outline(points),
        schema=OUTLINE_SCHEMA,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    general = _dedupe_titles(response.get("chung"), MAX_GENERAL_TITLES)
    specific = _dedupe_titles(response.get("cu_the"), MAX_SPECIFIC_TITLES)
    if not general and not specific:
        raise ValueError("Mô hình không tạo được đề cương cho báo cáo.")

    outline: List[OutlineItem] = []
    for index, title in enumerate(general, start=1):
        outline.append(OutlineItem(part=PART_GENERAL, index=index, title=title))
    for index, title in enumerate(specific, start=1):
        outline.append(OutlineItem(part=PART_SPECIFIC, index=index, title=title))
    return outline


# ---------------------------------------------------------------- stage 3

MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "anh_xa": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "muc": {"type": "string"},
                },
                "required": ["id", "muc"],
            },
        }
    },
    "required": ["anh_xa"],
}

MAP_SYSTEM_PROMPT = """Bạn xếp từng ý kiến vào đúng mục của đề cương.

Chỉ in ra DUY NHẤT một đối tượng JSON hợp lệ đúng theo mẫu {"anh_xa":[...]}. Không viết lời dẫn, không giải thích, không dùng markdown, không bọc trong dấu ```. Kiểm tra trước khi trả lời: chuỗi phải bắt đầu bằng ký tự { và kết thúc bằng ký tự }. Sau dấu ] đóng danh sách anh_xa phải còn đúng một dấu } đóng đối tượng.

Với mỗi ý kiến trong danh sách, trả về id của ý kiến và mã mục phù hợp nhất.

Quy tắc:
- Mã mục phải lấy nguyên văn từ đề cương, ví dụ I.1 hoặc II.3. Không tự đặt mã mới.
- Mỗi id xuất hiện đúng một lần.
- Xếp theo vấn đề được bàn, không theo hướng quan điểm. Ý nhất trí và ý không tán thành về cùng một vấn đề vào cùng một mục.
- Ý không thuộc mục nào thì ghi muc là "NGOAI".

Ví dụ:
Đề cương: "I.1: Về sự cần thiết đầu tư / II.1: Về nguồn vốn"
Danh sách: "T01-001 | [nhất trí] sự cần thiết: nhất trí với sự cần thiết đầu tư. / T02-004 | [đề nghị] vốn trung ương: đề nghị làm rõ vốn trung ương."
Kết quả: {"anh_xa":[{"id":"T01-001","muc":"I.1"},{"id":"T02-004","muc":"II.1"}]}"""


def format_outline_for_prompt(outline: List[OutlineItem]) -> str:
    return "\n".join("%s: %s" % (item.key, item.title) for item in outline)


def format_points_for_mapping(points: List[Point]) -> str:
    lines = []
    for point in points:
        target = point.target or "nội dung khác"
        lines.append(
            "%s | [%s] %s: %s" % (point.id, point.stance, target, point.content)
        )
    return "\n".join(lines)


def map_points(
    points: List[Point],
    outline: List[OutlineItem],
    client: InferenceClient,
    *,
    batch_size: int = MAP_BATCH_SIZE,
    max_tokens: int = 1500,
) -> Tuple[Dict[str, List[Point]], List[Point]]:
    valid_keys = set(item.key for item in outline)
    outline_block = format_outline_for_prompt(outline)

    # id -> outline key. Batched because a 4b model drops entries from a long
    # list; each batch repeats the outline so the prompt stays self-contained.
    assignments: Dict[str, str] = {}
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        batch_ids = set(point.id for point in batch)
        response = client.complete_json(
            system_prompt=MAP_SYSTEM_PROMPT,
            user_prompt=(
                "Đề cương:\n"
                + outline_block
                + "\n\nDanh sách ý kiến:\n"
                + format_points_for_mapping(batch)
            ),
            schema=MAP_SCHEMA,
            max_tokens=max_tokens,
            temperature=0.0,
            array_property="anh_xa",
        )
        raw_rows = response.get("anh_xa")
        if not isinstance(raw_rows, list):
            continue
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            point_id = _clean(raw.get("id"))
            key = _clean(raw.get("muc"))
            if point_id not in batch_ids or point_id in assignments:
                continue
            if key not in valid_keys:
                continue
            assignments[point_id] = key

    mapped: Dict[str, List[Point]] = dict((key, []) for key in valid_keys)
    unmapped: List[Point] = []
    for point in points:
        key = assignments.get(point.id)
        if key is None:
            unmapped.append(point)
        else:
            mapped[key].append(point)
    return mapped, unmapped


# ---------------------------------------------------------------- stage 4

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "y": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cau": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["cau", "ids"],
            },
        }
    },
    "required": ["y"],
}

WRITE_SYSTEM_PROMPT = """Bạn viết nội dung một mục của báo cáo tổng hợp ý kiến thảo luận tổ.

Chỉ in ra DUY NHẤT một đối tượng JSON hợp lệ đúng theo mẫu {"y":[...]}. Không viết lời dẫn, không giải thích, không dùng markdown, không bọc trong dấu ```. Kiểm tra trước khi trả lời: chuỗi phải bắt đầu bằng ký tự { và kết thúc bằng ký tự }. Sau dấu ] đóng danh sách y phải còn đúng một dấu } đóng đối tượng.

Gộp các ý kiến cùng hướng về cùng một vấn đề thành một câu khái quát. Với mỗi câu, liệt kê id của tất cả ý kiến đã gộp vào câu đó.

Quy tắc:
- Câu bắt đầu bằng động từ viết thường: nhất trí với..., đề nghị..., cho rằng..., bày tỏ băn khoăn về..., không tán thành...
- TUYỆT ĐỐI không viết số lượng ý kiến, không viết "Đa số", "Nhiều", "Một số", "Có ý kiến". Không viết tên đại biểu.
- Mỗi id xuất hiện đúng một lần trong toàn bộ kết quả. Không bỏ sót id nào.
- Ý khác hướng quan điểm thì tách thành câu riêng. "băn khoăn" khác "không tán thành".
- Giữ nguyên số liệu, mốc thời gian, điều kiện kèm theo. Khi gộp nhiều đề nghị, dùng (i), (ii), (iii) để giữ đủ chi tiết.
- Không thêm nhận định hay kết luận của cơ quan tổng hợp.
- BẮT BUỘC GỘP: nhiều ý kiến cùng hướng về cùng một vấn đề phải gộp thành DUY NHẤT một câu, kể cả khi các tổ diễn đạt khác nhau. Số câu trả về phải ít hơn hẳn số id. Không viết mỗi id thành một câu riêng. Câu gộp viết khái quát, bỏ chi tiết riêng của từng tổ.

Ví dụ:
Mục: "Về nguồn vốn"
Ý kiến: "T01-002 | đề nghị làm rõ vốn trung ương. / T03-001 | đề nghị làm rõ tỷ lệ đối ứng của địa phương. / T05-004 | không tán thành việc tăng phần vốn địa phương."
Kết quả: {"y":[{"cau":"đề nghị Chính phủ làm rõ phương án huy động vốn, gồm: (i) cơ cấu vốn trung ương; (ii) tỷ lệ đối ứng của từng địa phương.","ids":["T01-002","T03-001"]},{"cau":"không tán thành việc tăng phần vốn do địa phương bảo đảm.","ids":["T05-004"]}]}"""


def validate_bullets(
    raw_rows: object,
    allowed_ids: List[str],
) -> Tuple[List[Bullet], List[str]]:
    """Keep only bullets the data supports, and report which ids went unused.

    Drops ids that are not in allowed_ids and ids a previous bullet already
    claimed, then drops any bullet left with no sentence or no ids. The caller
    uses the missing list to recover points the model forgot.
    """
    allowed = set(allowed_ids)
    claimed = set()
    bullets: List[Bullet] = []

    if isinstance(raw_rows, list):
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            sentence = _clean(raw.get("cau"))
            if not sentence:
                continue
            raw_ids = raw.get("ids")
            if not isinstance(raw_ids, list):
                continue
            point_ids = []
            for raw_id in raw_ids:
                point_id = _clean(raw_id)
                if point_id in allowed and point_id not in claimed:
                    claimed.add(point_id)
                    point_ids.append(point_id)
            if not point_ids:
                continue
            bullets.append(Bullet(sentence=sentence, point_ids=point_ids))

    missing = [point_id for point_id in allowed_ids if point_id not in claimed]
    return bullets, missing


def _call_write(
    title: str,
    points: List[Point],
    client: InferenceClient,
    max_tokens: int,
) -> object:
    response = client.complete_json(
        system_prompt=WRITE_SYSTEM_PROMPT,
        user_prompt=(
            "Mục: " + title + "\n\nÝ kiến:\n" + format_points_for_mapping(points)
        ),
        schema=WRITE_SCHEMA,
        max_tokens=max_tokens,
        temperature=0.1,
        array_property="y",
    )
    return response.get("y")


def write_section(
    title: str,
    points: List[Point],
    client: InferenceClient,
    *,
    max_tokens: int = 2000,
) -> List[Bullet]:
    if not points:
        return []

    by_id = dict((point.id, point) for point in points)
    allowed_ids = [point.id for point in points]

    bullets, missing = validate_bullets(_call_write(title, points, client, max_tokens), allowed_ids)

    if missing:
        # One repair pass over only the forgotten points, then a literal
        # fallback. A point must never vanish: the footnote counts depend on
        # every mapped point landing in exactly one bullet.
        repair_points = [by_id[point_id] for point_id in missing]
        repaired, still_missing = validate_bullets(
            _call_write(title, repair_points, client, max_tokens), missing
        )
        bullets.extend(repaired)
        for point_id in still_missing:
            bullets.append(Bullet(sentence=by_id[point_id].content, point_ids=[point_id]))

    return bullets


# ---------------------------------------------------------------- stage 5

def quantifier(count: int, total: int) -> str:
    if total > 0 and float(count) / float(total) > MAJORITY_RATIO:
        return "Đa số ý kiến "
    if count >= MANY_THRESHOLD:
        return "Nhiều ý kiến "
    if count >= SOME_THRESHOLD:
        return "Một số ý kiến "
    return "Có ý kiến "


def format_count(count: int) -> str:
    return "%02d" % count


def footnote_breakdown(point_ids: List[str], points_by_id: Dict[str, Point]) -> str:
    # Keyed on group_index, not on the label: two uploaded sections may carry
    # the same heading text, and merging them would misstate the breakdown.
    counts: Dict[int, int] = {}
    labels: Dict[int, str] = {}
    for point_id in point_ids:
        point = points_by_id.get(point_id)
        if point is None:
            continue
        counts[point.group_index] = counts.get(point.group_index, 0) + 1
        labels[point.group_index] = point.group_id

    parts = []
    for group_index in sorted(counts):
        parts.append(
            "%s có %s ý kiến" % (labels[group_index], format_count(counts[group_index]))
        )
    return "Gồm: " + "; ".join(parts) + "."


def _render_bullet(bullet: Bullet, total: int, footnote_number: int) -> str:
    count = len(bullet.point_ids)
    sentence = bullet.sentence.strip().rstrip(".")
    return "- %s%s (%s ý kiến)[^%d]." % (
        quantifier(count, total),
        sentence,
        format_count(count),
        footnote_number,
    )


def render_report_document(
    outline: List[OutlineItem],
    bullets_by_key: Dict[str, List[Bullet]],
    outside_bullets: List[Bullet],
    points: List[Point],
) -> Tuple[str, List[Dict[str, str]]]:
    """Render the markdown plus its footnotes as {"[^1]": "Gồm: ..."} entries."""
    points_by_id = dict((point.id, point) for point in points)
    total = len(points)

    lines: List[str] = []
    footnotes: List[Dict[str, str]] = []

    def emit_bullets(bullets: List[Bullet]) -> None:
        # Descending count puts a majority view immediately above the dissent
        # it draws, which is how the reference report reads.
        for bullet in sorted(bullets, key=lambda item: len(item.point_ids), reverse=True):
            number = len(footnotes) + 1
            lines.append(_render_bullet(bullet, total, number))
            lines.append("")
            footnotes.append(
                {"[^%d]" % number: footnote_breakdown(bullet.point_ids, points_by_id)}
            )

    # A heading with no bullets means stage 2 minted a title that stage 3 then
    # filed under a different heading — a mapping error, not a topic nobody
    # raised. Printing it would advertise the error as if it were a finding, so
    # drop it, and drop a part left with nothing. Surviving headings are
    # renumbered per part so the reader never sees a gap where one was removed;
    # OutlineItem.index stays the mapping key and is not what gets printed.
    for part in (PART_GENERAL, PART_SPECIFIC):
        items = [
            item
            for item in outline
            if item.part == part and bullets_by_key.get(item.key)
        ]
        if not items:
            continue
        lines.append(SECTION_HEADING_PREFIX + "%s. %s" % (part, PART_TITLES[part]))
        lines.append("")
        for position, item in enumerate(items, start=1):
            lines.append(
                "%s%d. %s%s"
                % (SUBTITLE_MARKER, position, item.title, SUBTITLE_MARKER)
            )
            lines.append("")
            emit_bullets(bullets_by_key[item.key])

    if outside_bullets:
        lines.append(SECTION_HEADING_PREFIX + OUTSIDE_OUTLINE_TITLE)
        lines.append("")
        emit_bullets(outside_bullets)

    # The footnote text is returned separately, so the content carries only the
    # [^n] markers that index into it.
    if not lines:
        return NO_OPINION_TEXT + "\n", footnotes

    return "\n".join(lines).rstrip() + "\n", footnotes


def render_report(
    outline: List[OutlineItem],
    bullets_by_key: Dict[str, List[Bullet]],
    outside_bullets: List[Bullet],
    points: List[Point],
) -> str:
    """Markdown only, for callers that do not need the footnote list."""
    content, _ = render_report_document(outline, bullets_by_key, outside_bullets, points)
    return content


# ------------------------------------------------------------- orchestrator

def generate_report_from_groups(
    groups: List[GroupTranscript],
    *,
    model: Optional[str] = None,
    client: Optional[InferenceClient] = None,
) -> Dict[str, object]:
    if not groups:
        raise ValueError("Cần ít nhất một tổ để tổng hợp báo cáo.")

    resolved_client = client if client is not None else client_from_env("REPORT", model)

    points = extract_points(groups, resolved_client)
    outline = build_outline(points, resolved_client)
    mapped, unmapped = map_points(points, outline, resolved_client)

    bullets_by_key: Dict[str, List[Bullet]] = {}
    for item in outline:
        bullets_by_key[item.key] = write_section(
            item.title, mapped.get(item.key, []), resolved_client
        )

    outside_bullets = write_section(OUTSIDE_OUTLINE_TITLE, unmapped, resolved_client)

    content, footnotes = render_report_document(
        outline, bullets_by_key, outside_bullets, points
    )
    # total_count is every extracted point, so it equals the sum of the bullet
    # counts — including any that landed outside the đề cương.
    return {
        "content": content,
        "total_count": len(points),
        "footnotes": footnotes,
    }
