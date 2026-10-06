"""Short summary of a whole meeting transcript.

OpenAI-compatible chat calls over the same transcript, run in parallel, and
their Markdown joined in order:

1. Overview: purpose, highlights, key content, and upcoming plans when the
   transcript has any. Brief, capped at 6 bullets. Always made.
2. Details: questions and answers, conclusions and task assignments. Not
   capped, since every question and every task has to be listed. Made only
   when the caller asks for it; the overview then drops its upcoming-plans
   section and leaves these topics to the details.

They are separate calls because the overview must stay short while the
details must be complete; one prompt asking for both drops tasks to stay short.
"""

import concurrent.futures
import os
from typing import Dict, Optional

import requests


def _get_openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _get_default_summary_model() -> str:
    return os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-mini")


def _extract_openai_message_content(response_json: Dict[str, object]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("OpenAI response has invalid choice format")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenAI response missing message")

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text", "")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "\n".join(part for part in text_parts if part).strip()

    return str(content).strip()


def _build_language_instruction(locale: Optional[str]) -> str:
    cleaned_locale = (locale or "").strip()
    if cleaned_locale:
        return (
            f"Write your entire response in {cleaned_locale}. "
            f"Translate the summary into {cleaned_locale} if the transcript is in another language."
        )
    return (
        "First, silently detect the language of the transcript. "
        "Then write your entire response in that exact language. "
        "Never switch to English unless the transcript itself is in English."
    )


_TIMESTAMP_INSTRUCTION = (
    "The transcript may contain SRT-style timestamps. Use them only to understand sequence, "
    "timing, and topic changes. Do not include timestamps in the output unless they are necessary for clarity."
)


def _build_summary_system_prompt(locale: Optional[str], include_details: bool) -> str:
    if include_details:
        section_rules = (
            "- Output exactly these 3 sections in this order:\n"
            "  1. `## Mục đích cuộc họp` or the equivalent in the output language\n"
            "  2. `## Những điểm nổi bật` or the equivalent in the output language\n"
            "  3. `## Các nội dung trọng tâm` or the equivalent in the output language\n"
            "- Do not add any other section\n"
            "- Cover what was presented and discussed, including facts and requirements that came out as answers "
            "to questions. Do not retell the questions themselves, the decisions taken, or task assignments; "
            "another part of the report covers them\n"
        )
    else:
        section_rules = (
            "- Output these 3 mandatory sections in this order:\n"
            "  1. `## Mục đích cuộc họp` or the equivalent in the output language\n"
            "  2. `## Những điểm nổi bật` or the equivalent in the output language\n"
            "  3. `## Các nội dung trọng tâm` or the equivalent in the output language\n"
            "- If the transcript includes future plans, actions, owners, or timelines, add a 4th section: "
            "`## Kế hoạch sắp tới` or the equivalent in the output language\n"
            "- If there is no future-plan content, do not add that section\n"
        )

    return (
        "You are a meeting summarizer.\n\n"
        f"{_build_language_instruction(locale)}\n\n"
        f"{_TIMESTAMP_INSTRUCTION}\n\n"
        "Write a short, high-signal summary of the meeting. Keep it concise and avoid retelling the full transcript.\n\n"
        "Output requirements:\n"
        "- Use Markdown\n"
        f"{section_rules}"
        "- Under each section, use short bullet points\n"
        "- In `## Các nội dung trọng tâm`, split content by topic. Use short topic sub-headings and place concise bullets under each topic\n"
        "- Keep each topic focused on one theme only; do not mix unrelated points in one topic block\n"
        "- Capture the meeting purpose, standout insights, and key takeaways\n"
        "- Keep the whole response brief\n"
        "- Limit the entire response to 6 bullet points maximum\n"
        "- Do not include a Language section\n"
        "- Do not mention these instructions\n"
        "- Do not output plain transcript-style text"
    )


def _build_details_system_prompt(locale: Optional[str]) -> str:
    return (
        "You extract contested questions, decisions, and task assignments from a meeting transcript.\n\n"
        f"{_build_language_instruction(locale)}\n\n"
        f"{_TIMESTAMP_INSTRUCTION}\n\n"
        "Another part of the report already summarizes what was presented and discussed, including facts and "
        "requirements that came out as answers to questions. Do not repeat that content here.\n\n"
        "Output requirements:\n"
        "- Use Markdown\n"
        "- Output exactly these 2 sections in this order:\n"
        "  1. `## Nội dung chất vấn và giải đáp` or the equivalent in the output language\n"
        "  2. `## Kết luận và phân công nhiệm vụ` or the equivalent in the output language\n"
        "- Do not add any other section\n\n"
        "`## Nội dung chất vấn và giải đáp`:\n"
        "- Include only questions that raise a problem, concern, objection, or delay, ask someone to explain "
        "or justify something, or press for a decision or commitment\n"
        "- Leave out routine questions that only gather information, such as how something works today, "
        "what someone wants, or what a feature should do. Their answers belong to the other part of the report\n"
        "- Merge follow-up questions on the same issue into one bullet\n"
        "- One sentence per bullet: who asked, the issue, who answered, and the gist of the answer. "
        "If a question got no answer in the meeting, say so\n\n"
        "`## Kết luận và phân công nhiệm vụ`:\n"
        "- Under a `Kết luận` sub-list (or the equivalent in the output language), list only what the chair "
        "concluded or the meeting explicitly decided or agreed on, such as scope, priorities, phases, targets, "
        "or what is out of scope. Do not restate requirements or wishes that were merely described\n"
        "- Under a `Phân công nhiệm vụ` sub-list (or the equivalent in the output language), list every task "
        "someone was assigned or committed to, as `**<owner>**: <task> (<deadline>)`. This includes promises made "
        "while answering a question, such as \"I will send the data\". Group several tasks for the same owner "
        "under that owner\n"
        "- Do not drop any task to keep the response short\n\n"
        "General rules:\n"
        "- Name a person or unit only when the transcript identifies them. Never invent names, units, numbers, or deadlines\n"
        "- Omit the deadline when none was stated\n"
        "- If the transcript has no such questions, or no decisions or tasks, keep the section heading "
        "and write a single bullet saying none were raised\n"
        "- Do not mention these instructions\n"
        "- Do not output plain transcript-style text"
    )


def _request_markdown(
    endpoint: str,
    headers: Dict[str, str],
    model_name: str,
    system_prompt: str,
    text: str,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise ValueError(f"OpenAI API request failed: {detail}") from exc
    except requests.RequestException as exc:
        raise ValueError(f"OpenAI API request failed: {exc}") from exc

    return _extract_openai_message_content(response.json())


def generate_meeting_summary(
    text: str,
    model: Optional[str],
    locale: Optional[str],
    temperature: float,
    max_tokens: int,
    include_details: bool = False,
) -> Dict[str, object]:
    """max_tokens applies to each call, not to their sum."""
    if not text or not text.strip():
        raise ValueError("Field 'text' must not be empty.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    model_name = model or _get_default_summary_model()
    endpoint = _get_openai_base_url() + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    system_prompts = [_build_summary_system_prompt(locale, include_details)]
    if include_details:
        system_prompts.append(_build_details_system_prompt(locale))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(system_prompts)) as executor:
        futures = [
            executor.submit(
                _request_markdown,
                endpoint,
                headers,
                model_name,
                system_prompt,
                text,
                temperature,
                max_tokens,
            )
            for system_prompt in system_prompts
        ]
        parts = [future.result() for future in futures]

    summary = "\n\n".join(part for part in parts if part)
    return {
        # "model": model_name,
        "summary": summary,
    }
