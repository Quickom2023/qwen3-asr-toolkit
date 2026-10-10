"""Voiceprints in SQLite, searched from an in-memory matrix by cosine score.

Writes go through one lock: commit to SQLite, then publish a new
(matrix, user_ids, index) tuple in a single assignment. Readers take a
reference to the current tuple and never lock; a published tuple is never
edited in place, so a reader holding it is unaffected by later writes.
"""

import os
import sqlite3
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EMBEDDING_DIM = 256

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voiceprint (
  id             INTEGER PRIMARY KEY,
  user_id        TEXT NOT NULL,
  model_version  TEXT NOT NULL,
  embedding      BLOB NOT NULL,
  speech_seconds REAL NOT NULL,
  created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (user_id, model_version)
);
CREATE TABLE IF NOT EXISTS voiceprint_room_member (
  room_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (room_id, user_id)
)
"""

_SELECT_ID = "SELECT id FROM voiceprint WHERE user_id = ? AND model_version = ?"

_INSERT = """
INSERT INTO voiceprint (user_id, model_version, embedding, speech_seconds)
VALUES (?, ?, ?, ?)
"""

_REPLACE = """
UPDATE voiceprint
SET embedding = ?, speech_seconds = ?, created_at = CURRENT_TIMESTAMP
WHERE id = ?
"""

Snapshot = Tuple[np.ndarray, List[str], Dict[str, int]]


class AlreadyEnrolled(Exception):
    """The user already has a voiceprint and the caller did not ask to overwrite it."""

    def __init__(self, user_id: str, voiceprint_id: int) -> None:
        super().__init__(f"User '{user_id}' already has voiceprint {voiceprint_id}.")
        self.user_id = user_id
        self.voiceprint_id = voiceprint_id


class VoiceprintStore:
    """One L2-normalized vector per user for a single model version."""

    def __init__(self, path: str, model_version: str, dim: int = EMBEDDING_DIM) -> None:
        self.model_version = model_version
        self.dim = dim
        folder = os.path.dirname(os.path.abspath(path))
        # Biometric data: a folder created here is readable by its owner only.
        os.makedirs(folder, mode=0o700, exist_ok=True)
        # Every use of the connection goes through this lock, reads included:
        # one sqlite3 connection must not be used by two threads at once.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()
        self.snapshot: Snapshot = self._load()

    def _migrate(self) -> None:
        columns = [row[1] for row in self._conn.execute("PRAGMA table_info(voiceprint)")]
        if "speech_sec" in columns:
            self._conn.execute("ALTER TABLE voiceprint RENAME COLUMN speech_sec TO speech_seconds")

    def _load(self) -> Snapshot:
        rows = self._conn.execute(
            "SELECT user_id, embedding FROM voiceprint WHERE model_version = ? ORDER BY id",
            (self.model_version,),
        ).fetchall()
        matrix = np.zeros((len(rows), self.dim), dtype=np.float32)
        user_ids: List[str] = []
        for row, (user_id, blob) in enumerate(rows):
            matrix[row] = np.frombuffer(blob, dtype="<f4")
            user_ids.append(user_id)
        return matrix, user_ids, {user_id: row for row, user_id in enumerate(user_ids)}

    def __len__(self) -> int:
        return len(self.snapshot[1])

    def voiceprint_id(self, user_id: str) -> Optional[int]:
        """The id of the user's voiceprint, or None if the user has none."""
        with self._lock:
            found = self._conn.execute(_SELECT_ID, (user_id, self.model_version)).fetchone()
        return None if found is None else int(found[0])

    def add_room_member(self, room_id: str, user_id: str) -> Dict[str, object]:
        """Adds a user to a room. Repeating the operation is safe."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO voiceprint_room_member (room_id, user_id) VALUES (?, ?)",
                (room_id, user_id),
            )
            (created_at,) = self._conn.execute(
                "SELECT created_at FROM voiceprint_room_member WHERE room_id = ? AND user_id = ?",
                (room_id, user_id),
            ).fetchone()
        return {"room_id": room_id, "user_id": user_id, "created_at": created_at}

    def room_user_ids(self, room_id: str) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id FROM voiceprint_room_member WHERE room_id = ? ORDER BY user_id",
                (room_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def add(
        self,
        user_id: str,
        vector: np.ndarray,
        speech_seconds: float,
        overwrite: bool = False,
    ) -> Tuple[int, bool]:
        """Saves the user's voiceprint; returns (row id, replaced).

        Raises AlreadyEnrolled when the user has one and `overwrite` is false.
        """
        vector = np.asarray(vector, dtype=np.float32).reshape(self.dim)
        blob = vector.astype("<f4").tobytes()
        with self._lock:
            with self._conn:
                # From the db, not the snapshot: another process may have written this user.
                existing = self._conn.execute(_SELECT_ID, (user_id, self.model_version)).fetchone()
                if existing is not None and not overwrite:
                    raise AlreadyEnrolled(user_id, int(existing[0]))
                if existing is not None:
                    voiceprint_id = int(existing[0])
                    self._conn.execute(_REPLACE, (blob, float(speech_seconds), voiceprint_id))
                else:
                    cursor = self._conn.execute(
                        _INSERT, (user_id, self.model_version, blob, float(speech_seconds))
                    )
                    voiceprint_id = int(cursor.lastrowid)

            matrix, user_ids, index = self.snapshot
            row = index.get(user_id)
            if row is not None:
                matrix = matrix.copy()
                matrix[row] = vector
            else:
                matrix = np.vstack([matrix, vector[None, :]])
                user_ids = user_ids + [user_id]
                index = {**index, user_id: len(user_ids) - 1}
            self.snapshot = (matrix, user_ids, index)
        return voiceprint_id, existing is not None

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        user_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        """The top_k users closest to `vector`, best first.

        With `user_ids`, only those users are ranked; ids without a voiceprint are skipped.
        """
        matrix, enrolled, index = self.snapshot
        if user_ids is None:
            rows = np.arange(len(enrolled))
        else:
            rows = np.array(sorted({index[u] for u in user_ids if u in index}), dtype=np.intp)
        if rows.size == 0:
            return []
        scores = matrix[rows] @ np.asarray(vector, dtype=np.float32).reshape(self.dim)
        order = np.argsort(-scores, kind="stable")
        return [
            {"user_id": enrolled[rows[i]], "score": float(scores[i])}
            for i in order[:top_k]
        ]

    def list(self, limit: Optional[int] = None) -> Tuple[int, List[Dict[str, object]]]:
        """(number of voiceprints, the newest `limit` of them, or all when None)."""
        query = (
            "SELECT id, user_id, speech_seconds, created_at FROM voiceprint "
            "WHERE model_version = ? ORDER BY created_at DESC, id DESC"
        )
        params: Tuple[object, ...] = (self.model_version,)
        if limit is not None:
            query += " LIMIT ?"
            params += (int(limit),)
        with self._lock:
            (total,) = self._conn.execute(
                "SELECT count(*) FROM voiceprint WHERE model_version = ?", (self.model_version,)
            ).fetchone()
            rows = self._conn.execute(query, params).fetchall()
        return int(total), [
            {
                "voiceprint_id": int(voiceprint_id),
                "user_id": user_id,
                "speech_seconds": float(speech_seconds),
                "created_at": created_at,
            }
            for voiceprint_id, user_id, speech_seconds, created_at in rows
        ]

    def delete(self, voiceprint_id: int) -> Optional[Dict[str, object]]:
        """Removes one voiceprint; returns {"voiceprint_id", "user_id"}, or None if there is none."""
        with self._lock:
            with self._conn:
                found = self._conn.execute(
                    "SELECT user_id FROM voiceprint WHERE id = ? AND model_version = ?",
                    (int(voiceprint_id), self.model_version),
                ).fetchone()
                if found is None:
                    return None
                self._conn.execute("DELETE FROM voiceprint WHERE id = ?", (int(voiceprint_id),))
            (user_id,) = found

            matrix, user_ids, index = self.snapshot
            row = index.get(user_id)
            if row is not None:
                matrix = np.delete(matrix, row, axis=0)
                user_ids = user_ids[:row] + user_ids[row + 1:]
                index = {uid: position for position, uid in enumerate(user_ids)}
                self.snapshot = (matrix, user_ids, index)
        return {"voiceprint_id": int(voiceprint_id), "user_id": user_id}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
