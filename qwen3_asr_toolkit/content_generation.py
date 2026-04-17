import os
from pathlib import Path
from typing import Dict, List, Optional

import requests


DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_TIMEOUT_SECONDS = 120


def _normalize_chat_endpoint(value: str) -> str:
    cleaned = value.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return cleaned + "/chat/completions"
    return cleaned + "/v1/chat/completions"


def _get_endpoint(prefix: str) -> str:
    api_url = os.getenv(f"{prefix}_API_URL", "").strip()
    if api_url:
        return _normalize_chat_endpoint(api_url)

    base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
    if base_url:
        return _normalize_chat_endpoint(base_url)

    fallback_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    return _normalize_chat_endpoint(fallback_base_url)


def _get_api_key(prefix: str) -> str:
    api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
    if api_key:
        return api_key

    fallback_key = os.getenv("OPENAI_API_KEY", "").strip()
    if fallback_key:
        return fallback_key

    raise ValueError(f"{prefix}_API_KEY or OPENAI_API_KEY is not configured.")


def _get_model(prefix: str, explicit_model: Optional[str]) -> str:
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()

    configured_model = os.getenv(f"{prefix}_MODEL", "").strip()
    if configured_model:
        return configured_model

    fallback_model = os.getenv("OPENAI_SUMMARY_MODEL", "").strip()
    if fallback_model:
        return fallback_model

    return DEFAULT_CHAT_MODEL


def _load_reference_text(filename: str) -> str:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / filename
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    return ""


def _call_chat_completion(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = 0.5,
    max_tokens: int,
) -> Dict[str, object]:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise ValueError(f"LLM API request failed: {detail}") from exc
    except requests.RequestException as exc:
        raise ValueError(f"LLM API request failed: {exc}") from exc

    return response.json()


def _extract_message_content(response_json: Dict[str, object]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("LLM response choice has invalid format.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message.")

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    chunks.append(text_value.strip())
        return "\n".join(chunks).strip()

    return str(content).strip()


def build_each_person_prompt(locale: Optional[str] = None) -> str:
    _ = locale
    style_template = _load_reference_text("data/minutes_template.txt")
    style_template_block = style_template if style_template else "[Không có style template]"
    EACH_PERSON_PROMPT = """
Bạn là trợ lý chuyên tóm tắt biên bản họp của cơ quan nhà nước bằng tiếng Việt.

Nhiệm vụ của bạn: Tóm tắt văn bản thành nội dung súc tích, chính xác, đúng văn phong hành chính.

YÊU CẦU ĐẦU RA:

1. Tiêu đề: BẮT BUỘC ghi đúng thứ tự họ tên và chức danh của người phát biểu theo định dạng `# [Họ Tên], [Chức danh + Cơ quan công tác]`
    - Hướng dẫn: Cho dù input đầu vào ghi chức danh trước tên (ví dụ: "Giám đốc A, ông B"), bạn PHẢI trích xuất tên riêng ra trước, sau đó mới đến chức vụ.
    - Ví dụ xử lý: 
     + Input: "Phó Bí thư tỉnh Yết Kiêu, Đào Thị Hồng Hạnh" 
     + Output: "Đào Thị Hồng Hạnh, Phó Bí thư tỉnh Yết Kiêu"

2. Cấu trúc nội dung (Dùng gạch đầu dòng):
   - Mỗi gạch đầu dòng (-) đại diện cho một nhóm vấn đề lớn.
   - QUY TẮC CỐT LÕI: Viết thẳng vào vấn đề. Loại bỏ mọi từ ngữ thừa thãi, lời dẫn giải, lời chào hỏi. 

3. Kiểm soát độ dài:
   - Tổng độ dài: Tối đa 300 từ (với phát biểu thường) và 500 từ (với phát biểu chỉ đạo).
   - Tuyệt đối không viết rườm rà để kéo dài văn bản. Mỗi ý chỉ cần đủ thông tin: [Hành động] + [Đối tượng] + [Số liệu/Thời hạn].

4. Văn phong:
   - Hành chính, quyết liệt, cô đọng.
   - Ví dụ thay vì viết: "Chúng ta cần phải nỗ lực để đẩy nhanh tiến độ hơn nữa" -> Viết: "Đẩy nhanh tiến độ thực hiện dự án."
   - Dùng các cụm từ quen thuộc như: "tập trung", "phấn đấu", "triển khai", "đề xuất", "tháo gỡ", "đẩy nhanh tiến độ", v.v.

5. Cấu trúc ưu tiên theo nội dung thực tế:
    - Nếu phát biểu có tính chỉ đạo, ưu tiên nêu rõ hành động và yêu cầu đối với các đơn vị liên quan.
    - Nếu phát biểu mang tính báo cáo, ưu tiên nêu rõ kết quả đạt được, khó khăn vướng mắc, đề xuất giải pháp.
    - Nếu phát biểu mang tính chất chất vấn, ưu tiên nêu rõ vấn đề được đặt ra và yêu cầu giải trình.

**Ví dụ về style tóm tắt:**
    """
    return EACH_PERSON_PROMPT + "\n" + style_template_block


def build_conclusions_prompt(locale: Optional[str] = None) -> str:
    _ = locale
    style_template = _load_reference_text("data/conclusions_template.txt")
    style_template_block = style_template if style_template else "[Không có style template]"
    CONCLUSIONS_PROMPT = """
Bạn là chuyên gia soạn thảo văn bản hành chính nhà nước Việt Nam, chuyên về 
thông báo kết luận hội nghị và văn bản chỉ đạo sau sự kiện.

## CẤU TRÚC VĂN BẢN

Luôn tạo văn bản theo các phần sau, theo đúng thứ tự:

**I. TÌNH HÌNH / ĐÁNH GIÁ KẾT QUẢ** — Đánh giá tổng quan tình hình, sự kiện 
hoặc công tác được tổng kết.

**II. MỘT SỐ NỘI DUNG CẦN RÚT KINH NGHIỆM** — Bài học kinh nghiệm và các điểm 
cần khắc phục, cải thiện trong thời gian tới.

**III. NHIỆM VỤ TRỌNG TÂM** — Các nhiệm vụ, chỉ đạo cụ thể cho từng cơ quan, 
đơn vị (được đánh số thứ tự theo từng lĩnh vực).

Điều chỉnh số lượng và tiêu đề các phần cho phù hợp với nội dung người dùng cung cấp.

---

## NGÔN NGỮ & VĂN PHONG

- Văn phong hành chính trang trọng, mang tính thể chế và thẩm quyền.
- Sử dụng văn phong chỉ đạo, giao nhiệm vụ: "Giao [Cơ quan] chủ trì...", 
  "Đề nghị...", "Yêu cầu...", "Chỉ đạo..."
- Ưu tiên các động từ hành động mạnh: chủ trì, phối hợp, tham mưu, triển khai, 
  rà soát, đẩy mạnh, bảo đảm, tăng cường, khẩn trương, hoàn thành, thực hiện.
- Phần đánh giá dùng ngôn ngữ tích cực, trân trọng: 
  "đã được triển khai toàn diện, đồng bộ, an toàn và hiệu quả."
- Ghi nhận đóng góp của các lực lượng, cá nhân trước khi nêu bài học kinh nghiệm.

---

## ĐỊNH DẠNG GIAO NHIỆM VỤ

Với mỗi nhiệm vụ trong Phần III, áp dụng cấu trúc sau:
- Mở đầu bằng: "Giao [Tên cơ quan/đơn vị] chủ trì, phối hợp với [...]:"
- Dùng dấu "+" để liệt kê các nhiệm vụ cụ thể của từng cơ quan.
- Ghi rõ thời hạn khi có yêu cầu: "Thời gian hoàn thành trong tháng [X]/[YYYY]."
- Mỗi chỉ đạo phải cụ thể, có thể thực hiện được và giao đúng cho cơ quan phụ trách.

---

## NGUYÊN TẮC NỘI DUNG

1. **Phần đánh giá (Phần I)**: Tổng kết kết quả theo hướng tích cực. Ghi nhận 
   đóng góp của các sở, ban, ngành, lực lượng và cá nhân. Dùng cấu trúc khen ngợi:
   "Biểu dương và ghi nhận sự nỗ lực, tinh thần trách nhiệm của..."

2. **Phần rút kinh nghiệm (Phần II)**: Diễn đạt mang tính xây dựng, định hướng. 
   Mở đầu mỗi ý bằng: "Chủ động", "Đẩy mạnh", "Rà soát", "Tăng cường". 
   Tuyệt đối không dùng ngôn ngữ quy trách nhiệm cá nhân.

3. **Phần nhiệm vụ trọng tâm (Phần III)**: Mỗi mục đánh số là một lĩnh vực 
   (bầu cử, kinh tế, an ninh, văn hóa...). Trong mỗi lĩnh vực, giao việc cụ thể 
   cho từng cơ quan bằng "Giao [Tên cơ quan]". Luôn nêu rõ: làm gì, ai làm, 
   làm khi nào.

---

## CÂU KẾT VĂN BẢN

Kết thúc mỗi văn bản bằng câu cố định:
"[Văn phòng / Cơ quan ban hành] thông báo ý kiến kết luận nêu trên đến các cơ quan, 
đơn vị, địa phương biết, thực hiện./."

---

## ĐẦU VÀO TỪ NGƯỜI DÙNG

Người dùng sẽ cung cấp:
- Chủ đề / sự kiện cần tổng kết (ví dụ: tổ chức Tết, hội nghị sơ kết, 
  kiểm tra chuyên đề...)
- Các cơ quan liên quan, kết quả đạt được và nhiệm vụ cần giao
- Các chỉ đạo cụ thể hoặc thời hạn hoàn thành (nếu có)

Dựa trên thông tin người dùng cung cấp, tạo ra văn bản kết luận hoàn chỉnh, 
đúng định dạng và chuẩn văn phong hành chính nhà nước Việt Nam.

**Ví dụ về style tóm tắt:**
    """
    return CONCLUSIONS_PROMPT + "\n" + style_template_block


def generate_each_person_from_transcript(
    transcript: str,
    *,
    model: Optional[str] = None,
    locale: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 3000,
    include_prompt: bool = False,
) -> Dict[str, object]:
    if not transcript or not transcript.strip():
        raise ValueError("Field 'transcript' must not be empty.")

    prompt = build_each_person_prompt(locale)
    endpoint = _get_endpoint("TRANSCRIPT_SUMMARY")
    api_key = _get_api_key("TRANSCRIPT_SUMMARY")
    model_name = _get_model("TRANSCRIPT_SUMMARY", model)

    user_prompt = (
        "Convert the following transcript excerpt into the target format.\n\n"
        "Transcript:\n"
        f"{transcript.strip()}"
    )

    response_json = _call_chat_completion(
        endpoint=endpoint,
        api_key=api_key,
        model=model_name,
        system_prompt=prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    generated_text = _extract_message_content(response_json)

    response: Dict[str, object] = {
        "content": generated_text,
    }
    if include_prompt:
        response["prompt"] = prompt
    return response


def generate_conclusions_from_summaries(
    transcript: str,
    *,
    model: Optional[str] = None,
    locale: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 10000,
    include_prompt: bool = False,
) -> Dict[str, object]:
    if not transcript or not transcript.strip():
        raise ValueError("Field 'transcript' must not be empty.")

    prompt = build_conclusions_prompt(locale)
    endpoint = _get_endpoint("MEETING_MINUTES")
    api_key = _get_api_key("MEETING_MINUTES")
    model_name = _get_model("MEETING_MINUTES", model)

    user_prompt = (
        "Generate the complete conclusions document from the following transcript.\n\n"
        "Transcript:\n"
        f"{transcript.strip()}"
    )

    response_json = _call_chat_completion(
        endpoint=endpoint,
        api_key=api_key,
        model=model_name,
        system_prompt=prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    generated_text = _extract_message_content(response_json)

    response: Dict[str, object] = {
        "content": generated_text,
    }
    if include_prompt:
        response["prompt"] = prompt
    return response
