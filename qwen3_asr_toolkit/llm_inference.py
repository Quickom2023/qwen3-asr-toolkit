"""Provider-agnostic chat inference.

Isolates the two things that differ between Ollama and OpenAI-compatible
servers (vLLM, llama.cpp, OpenAI itself): the request payload shape and where
the assistant's text sits in the response. Callers see one interface, so
switching backends is an environment change rather than a code change.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

import requests

try:
    from dotenv import find_dotenv, load_dotenv  # type: ignore
except Exception:
    find_dotenv = None
    load_dotenv = None

# Mirrors api_server.py and speaker_attribution.py: load .env directly rather
# than relying on being imported after another module's load_dotenv() call.
# override=False keeps real environment variables authoritative over the file.
if load_dotenv and find_dotenv:
    load_dotenv(find_dotenv(usecwd=True), override=False)


DEFAULT_CHAT_MODEL = "qwen3.5:4b"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "32768"))
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.0
OLLAMA_KEEP_ALIVE = "-5m"
OLLAMA_THINK = False

# Anchored to the whole string, and deliberately tolerant of backtick runs of
# any length: qwen3.5:4b emits unfenced JSON with a single backtick tacked on
# the end, which a ```-only pattern leaves in place to kill json.loads.
_OPEN_FENCE_PATTERN = re.compile(r"\A`+[ \t]*(?:json|JSON)?[ \t]*\r?\n?")
_CLOSE_FENCE_PATTERN = re.compile(r"\r?\n?[ \t]*`+\Z")


@dataclass
class InferenceConfig:
    provider: str
    endpoint: str
    api_key: str = ""
    model: str = DEFAULT_CHAT_MODEL
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    num_ctx: int = DEFAULT_NUM_CTX


def normalize_endpoint(base: str, provider: str) -> str:
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
    config: InferenceConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: Optional[Dict[str, object]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, object]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if config.provider == "ollama":
        payload = {
            "model": config.model,
            "stream": False,
            "think": OLLAMA_THINK,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": temperature,
                "top_p": DEFAULT_TOP_P,
                "top_k": DEFAULT_TOP_K,
                "repeat_penalty": DEFAULT_REPEAT_PENALTY,
                "num_ctx": config.num_ctx,
                "num_predict": max_tokens,
            },
            "messages": messages,
        }
        if schema is not None:
            payload["format"] = schema
        return payload

    payload = {
        "model": config.model,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }
    return payload


def extract_content(body: Dict[str, object], provider: str) -> str:
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
    cleaned = content.strip()
    cleaned = _OPEN_FENCE_PATTERN.sub("", cleaned)
    cleaned = _CLOSE_FENCE_PATTERN.sub("", cleaned)
    return cleaned.strip()


def _close_unbalanced(text: str) -> Optional[str]:
    """Append the closing delimiters an unterminated JSON document is missing.

    Ollama's `format` does not constrain generation, so qwen3.5:4b sometimes
    stops (done_reason "stop", not truncation) having closed an inner array but
    not the root object. Only ever appends closers, and returns None whenever
    guessing could fabricate content: a string left open mid-value, a
    mismatched delimiter, or text already balanced (whose parse failed for some
    other reason, so repair would hide the real problem).
    """
    stack = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            opener = "{" if char == "}" else "["
            if not stack or stack[-1] != opener:
                return None
            stack.pop()
    if in_string or not stack:
        return None
    return text + "".join("}" if char == "{" else "]" for char in reversed(stack))


class InferenceClient:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def _post(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, object]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = build_payload(
            self.config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
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

        return extract_content(response.json(), self.config.provider)

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        return self._post(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, object],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        array_property: Optional[str] = None,
    ) -> Dict[str, object]:
        """Post and parse a JSON object.

        array_property names the schema's single array field. Ollama's `format`
        is advisory in practice (0.31.1 + qwen3.5:4b returns a bare array with
        or without it), so a caller that knows its envelope can accept the
        array and have it wrapped. Left unset, a non-object is still an error.
        """
        content = self._post(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            repaired = _close_unbalanced(content)
            if repaired is None:
                raise ValueError("LLM returned invalid JSON: " + content[:200]) from exc
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                raise ValueError("LLM returned invalid JSON: " + content[:200]) from exc
        if not isinstance(parsed, dict):
            if array_property and isinstance(parsed, list):
                return {array_property: parsed}
            raise ValueError("LLM returned JSON that is not an object: " + content[:200])
        return parsed


def client_from_env(prefix: str = "REPORT", model: Optional[str] = None) -> InferenceClient:
    base = (
        os.getenv(prefix + "_API_URL", "").strip()
        or os.getenv(prefix + "_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    )
    provider = os.getenv(prefix + "_PROVIDER", "").strip().lower()
    if provider not in ("ollama", "openai"):
        provider = "ollama" if ("/api/chat" in base or ":11434" in base) else "openai"
    resolved_model = (
        (model or "").strip()
        or os.getenv(prefix + "_MODEL", "").strip()
        or os.getenv("OPENAI_SUMMARY_MODEL", "").strip()
        or DEFAULT_CHAT_MODEL
    )
    api_key = (
        os.getenv(prefix + "_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    return InferenceClient(
        InferenceConfig(
            provider=provider,
            endpoint=normalize_endpoint(base, provider),
            api_key=api_key,
            model=resolved_model,
        )
    )
