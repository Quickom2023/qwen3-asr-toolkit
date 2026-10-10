"""Enroll a user's voiceprint and find the enrolled users whose voice is closest to a sample."""

import math
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from huggingface_hub import hf_hub_download

from qwen3_asr_toolkit.services.voiceprint_store import AlreadyEnrolled, VoiceprintStore
from qwen3_asr_toolkit.utils.audio_tools import WAV_SAMPLE_RATE, load_audio
from qwen3_asr_toolkit.utils.speaker_embedding import MODEL_FILE, MODEL_REPO, speech_only


ENROLL_MIN_SEC = 15.0
QUERY_MIN_SEC = 5.0
EMBED_MAX_SEC = 20.0
MAX_UPLOAD_MB = 10
# Only the first MAX_AUDIO_SEC of a file are used: 20 s of net speech needs ~30 s of
# audio, and anything past this would cost decode and VAD time for no accuracy.
MAX_AUDIO_SEC = 60.0
DEFAULT_TOP_K = 5
MAX_TOP_K = 20


def db_path() -> str:
    default = Path(__file__).resolve().parents[1] / "db" / "voiceprints.db"
    return os.getenv("VP_DB_PATH") or str(default)


def model_path() -> str:
    return os.getenv("VP_MODEL_PATH") or hf_hub_download(MODEL_REPO, MODEL_FILE)


class VoiceprintError(Exception):
    """A request the voiceprint routes answer with {"error", "detail", "speech_seconds"}."""

    def __init__(self, status: int, code: str, detail: str, speech_seconds: Optional[float] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.speech_seconds = speech_seconds


def _reported_seconds(seconds: float) -> float:
    """Seconds rounded down to 0.1: a rejected 14.96 s must not read as the 15 s it failed."""
    return math.floor(round(seconds * 10, 6)) / 10


def _already_exists(exc: AlreadyEnrolled) -> VoiceprintError:
    return VoiceprintError(
        409,
        "already_exists",
        f"User '{exc.user_id}' already has voiceprint {exc.voiceprint_id}; "
        "send overwrite=true to replace it.",
    )


class VoiceprintService:
    def __init__(self, embedder, vad_model, store: VoiceprintStore) -> None:
        self._embedder = embedder
        self._vad_model = vad_model
        self._vad_lock = threading.Lock()
        self._store = store

    def _embed_speech(self, audio_path: str, min_sec: float):
        """Returns (embedding, net speech seconds) or raises VoiceprintError."""
        try:
            # A small upload can still hold hours of audio: decode only the first MAX_AUDIO_SEC.
            samples = load_audio(audio_path, max_seconds=MAX_AUDIO_SEC)
        except Exception:
            raise VoiceprintError(415, "unsupported_audio", "The audio file could not be decoded.")
        samples = samples[: int(MAX_AUDIO_SEC * WAV_SAMPLE_RATE)]

        speech = speech_only(samples, self._vad_model, self._vad_lock)
        speech_seconds = len(speech) / WAV_SAMPLE_RATE
        if speech_seconds < min_sec:
            reported = _reported_seconds(speech_seconds)
            raise VoiceprintError(
                422,
                "not_enough_speech",
                f"Found {reported:.1f} s of speech; at least {min_sec:g} s is needed.",
                speech_seconds=reported,
            )
        vector = self._embedder.embed(speech[: int(EMBED_MAX_SEC * WAV_SAMPLE_RATE)])
        return vector, _reported_seconds(speech_seconds)

    def enroll(self, user_id: str, audio_path: str, overwrite: bool = False) -> Dict[str, object]:
        # Checked before decoding so a refused duplicate costs no CPU; the store checks again on write.
        if not overwrite:
            existing_id = self._store.voiceprint_id(user_id)
            if existing_id is not None:
                raise _already_exists(AlreadyEnrolled(user_id, existing_id))
        vector, speech_seconds = self._embed_speech(audio_path, ENROLL_MIN_SEC)
        try:
            voiceprint_id, replaced = self._store.add(user_id, vector, speech_seconds, overwrite=overwrite)
        except AlreadyEnrolled as exc:
            raise _already_exists(exc)
        return {
            "voiceprint_id": voiceprint_id,
            "user_id": user_id,
            "speech_seconds": speech_seconds,
            "replaced": replaced,
        }

    def search(
        self,
        audio_path: str,
        top_k: int = DEFAULT_TOP_K,
        user_ids: Optional[Sequence[str]] = None,
        room_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """Ranks users selected by room or explicit ids (all users when neither is set)."""
        if room_id is not None:
            user_ids = self._store.room_user_ids(room_id)
        vector, speech_seconds = self._embed_speech(audio_path, QUERY_MIN_SEC)
        return {
            "speech_seconds": speech_seconds,
            "candidates": [
                {"user_id": c["user_id"], "score": round(c["score"], 3)}
                for c in self._store.search(vector, top_k, user_ids)
            ],
        }

    def add_room_member(self, room_id: str, user_id: str) -> Dict[str, object]:
        return self._store.add_room_member(room_id, user_id)

    def add_room_members(self, rooms: Sequence[Tuple[str, Sequence[str]]]) -> Dict[str, object]:
        return self._store.add_room_members(rooms)

    def list(self, limit: Optional[int] = None) -> Dict[str, object]:
        total, voiceprints = self._store.list(limit)
        return {"total": total, "voiceprints": voiceprints}

    def delete(self, voiceprint_id: int) -> Dict[str, object]:
        deleted = self._store.delete(voiceprint_id)
        if deleted is None:
            raise VoiceprintError(404, "not_found", f"No voiceprint with id {voiceprint_id}.")
        return {**deleted, "deleted": True}
