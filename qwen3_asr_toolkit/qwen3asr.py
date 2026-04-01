import os
import time
import random
import re
import base64
import mimetypes
from typing import Optional
from urllib.parse import urlparse

from pydub import AudioSegment
import requests


MAX_API_RETRY = 10
API_RETRY_SLEEP = (1, 2)


language_code_mapping = {
    "ar": "Arabic",
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "vi": "Vietnamese",
}


class QwenASR:
    def __init__(
        self,
        api_url: str,
        model: Optional[str] = None,
        timeout_s: int = 300,
        temperature: Optional[float] = None,
        max_retries: int = MAX_API_RETRY,
    ):
        self.api_url = api_url
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_retries = max_retries

    def post_text_process(self, text, threshold=20):
        def fix_char_repeats(s, thresh):
            res = []
            i = 0
            n = len(s)
            while i < n:
                count = 1
                while i + count < n and s[i + count] == s[i]:
                    count += 1

                if count > thresh:
                    res.append(s[i])
                    i += count
                else:
                    res.append(s[i:i + count])
                    i += count
            return ''.join(res)

        def fix_pattern_repeats(s, thresh, max_len=20):
            n = len(s)
            min_repeat_chars = thresh * 2
            if n < min_repeat_chars:
                return s

            i = 0
            result = []
            while i <= n - min_repeat_chars:
                found = False
                for k in range(1, max_len + 1):
                    if i + k * thresh > n:
                        break

                    pattern = s[i:i + k]

                    valid = True
                    for rep in range(1, thresh):
                        start_idx = i + rep * k
                        if s[start_idx:start_idx + k] != pattern:
                            valid = False
                            break

                    if valid:
                        total_rep = thresh
                        end_index = i + thresh * k
                        while end_index + k <= n and s[end_index:end_index + k] == pattern:
                            total_rep += 1
                            end_index += k

                        result.append(pattern)
                        result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                        i = n
                        found = True
                        break

                if found:
                    break
                else:
                    result.append(s[i])
                    i += 1

            if not found:
                result.append(s[i:])
            return ''.join(result)

        text = fix_char_repeats(text, threshold)
        return fix_pattern_repeats(text, threshold)

    def _normalize_content(self, content):
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _redact_base64(self, text: str) -> str:
        if not text:
            return text
        return re.sub(r"[A-Za-z0-9+/=]{80,}", "<base64-redacted>", text)

    def _summarize_error(self, error: Exception) -> str:
        message = f"{error.__class__.__name__}: {error}"
        message = self._redact_base64(message)
        if len(message) > 300:
            message = message[:300] + "...(truncated)"
        return message

    def _display_audio_ref(self, audio_ref: str) -> str:
        if not audio_ref:
            return audio_ref
        if audio_ref.startswith("data:"):
            return "<data-url-audio>"
        return audio_ref

    def _parse_asr_output(self, content):
        try:
            from qwen_asr import parse_asr_output  # type: ignore
            return parse_asr_output(content)
        except Exception:
            pass

        content = self._normalize_content(content).strip()
        lang_match = re.search(r"(?i)language\s*[:=]\s*([A-Za-z-]+)", content)
        text_match = re.search(r"(?i)text\s*[:=]\s*(.+)", content, re.S)
        lang_code = lang_match.group(1).lower() if lang_match else None
        language = language_code_mapping.get(lang_code, "Unknown")
        text = text_match.group(1).strip() if text_match else content
        text = self._strip_inline_language_markers(text)
        return language, text

    def _to_data_url(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _is_audio_transcriptions_endpoint(self) -> bool:
        parsed = urlparse(self.api_url)
        return parsed.path.rstrip("/") == "/v1/audio/transcriptions"

    def _parse_audio_transcription_response(self, response_json):
        if isinstance(response_json, dict):
            text = response_json.get("text", "")
            language = response_json.get("language")

            # Some servers return metadata inline in `text`, e.g.
            # "language Vietnamese<asr_text>...". Reuse qwen_asr parser if available.
            try:
                from qwen_asr import parse_asr_output  # type: ignore
                parsed_language, parsed_text = parse_asr_output(text)
                if parsed_text:
                    text = parsed_text
                if not language and parsed_language:
                    language = parsed_language
            except Exception:
                pass

            text = self._strip_inline_language_markers(text)

            if language:
                lang_code = str(language).strip().lower()
                mapped_language = language_code_mapping.get(lang_code, str(language))
            else:
                mapped_language = "Unknown"
            return mapped_language, text
        return "Unknown", self._normalize_content(response_json)

    def _strip_inline_language_markers(self, text: str) -> str:
        if not text:
            return text
        cleaned = re.sub(
            r"(?is)\s*language\s+[^\s<:=>]+(?:\s+[^\s<:=>]+)*\s*<asr_text>\s*",
            " ",
            text,
        )
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def remove_chinese_characters(self, text: str) -> str:
        if not text:
            return text
        cleaned = re.sub(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def asr(self, wav_url: str, context: str = ""):
        local_file_path = None
        if not wav_url.startswith("http"):
            assert os.path.exists(wav_url), f"{wav_url} not exists!"
            local_file_path = wav_url
            file_size = os.path.getsize(local_file_path)

            # file size > 10M: convert to mp3
            if file_size > 10 * 1024 * 1024:
                mp3_path = os.path.splitext(local_file_path)[0] + ".mp3"
                audio = AudioSegment.from_file(local_file_path)
                audio.export(mp3_path, format="mp3")
                local_file_path = mp3_path

        if self._is_audio_transcriptions_endpoint():
            if not local_file_path:
                raise ValueError("Remote URL input is not supported for /v1/audio/transcriptions. Please provide a local file.")
            return self._asr_audio_transcriptions(local_file_path, context)

        if local_file_path:
            wav_url = self._to_data_url(local_file_path)

        response = None
        display_ref = self._display_audio_ref(wav_url)
        for _ in range(self.max_retries):
            try:
                messages = []
                if context:
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": context}],
                    })
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "audio_url",
                        "audio_url": {"url": wav_url},
                    }],
                })
                payload = {"messages": messages}
                if self.model:
                    payload["model"] = self.model
                if self.temperature is not None:
                    payload["temperature"] = self.temperature

                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                response_json = response.json()
                output = response_json["choices"][0]["message"]["content"]
                language, recog_text = self._parse_asr_output(output)
                recog_text = self.post_text_process(recog_text)
                recog_text = self.remove_chinese_characters(recog_text)
                return language, recog_text
            except Exception as e:
                try:
                    status_code = getattr(response, "status_code", "unknown")
                    reason = getattr(response, "reason", "")
                    error_summary = self._summarize_error(e)
                    print(f"Retry {_ + 1}...  {display_ref}\nHTTP {status_code} {reason}\n{error_summary}")
                except Exception:
                    error_summary = self._summarize_error(e)
                    print(f"Retry {_ + 1}...  {display_ref}\n{error_summary}")
            time.sleep(random.uniform(*API_RETRY_SLEEP))
        raise Exception(f"{display_ref} task failed!\n{response}")

    def _asr_audio_transcriptions(self, file_path: str, context: str = ""):
        response = None
        display_ref = file_path

        for _ in range(self.max_retries):
            try:
                data = {}
                if self.model:
                    data["model"] = self.model
                if context:
                    data["prompt"] = context
                if self.temperature is not None:
                    data["temperature"] = str(self.temperature)

                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    mime_type = "application/octet-stream"

                with open(file_path, "rb") as f:
                    files = {
                        "file": (os.path.basename(file_path), f, mime_type),
                    }
                    response = requests.post(
                        self.api_url,
                        data=data,
                        files=files,
                        timeout=self.timeout_s,
                    )
                response.raise_for_status()
                response_json = response.json()
                language, recog_text = self._parse_audio_transcription_response(response_json)
                recog_text = self.post_text_process(recog_text)
                recog_text = self.remove_chinese_characters(recog_text)
                return language, recog_text
            except Exception as e:
                try:
                    status_code = getattr(response, "status_code", "unknown")
                    reason = getattr(response, "reason", "")
                    error_summary = self._summarize_error(e)
                    print(f"Retry {_ + 1}...  {display_ref}\nHTTP {status_code} {reason}\n{error_summary}")
                except Exception:
                    error_summary = self._summarize_error(e)
                    print(f"Retry {_ + 1}...  {display_ref}\n{error_summary}")
            time.sleep(random.uniform(*API_RETRY_SLEEP))
        raise Exception(f"{display_ref} task failed!\n{response}")


if __name__ == "__main__":
    qwen_asr = QwenASR(api_url="http://localhost:8000/v1/audio/transcriptions", model=None)
    asr_text = qwen_asr.asr(wav_url="/path/to/your/wav_file.wav")
    print(asr_text)
