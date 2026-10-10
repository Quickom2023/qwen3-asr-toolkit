import tempfile
import unittest
from pathlib import Path

import numpy as np

from qwen3_asr_toolkit.services.voiceprint_store import VoiceprintStore


class VoiceprintRoomTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "voiceprints.db")
        self.store = VoiceprintStore(self.db_path, "test-model")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_room_membership_is_idempotent_and_persists(self):
        first = self.store.add_room_member("room-a", "user-a")
        repeated = self.store.add_room_member("room-a", "user-a")

        self.assertEqual(first, repeated)
        self.assertEqual(["user-a"], self.store.room_user_ids("room-a"))

        self.store.close()
        self.store = VoiceprintStore(self.db_path, "test-model")
        self.assertEqual(["user-a"], self.store.room_user_ids("room-a"))

    def test_add_room_members_deduplicates_and_is_idempotent(self):
        rooms = [
            ("room-a", ["user-a", "user-b", "user-a"]),
            ("room-b", ["user-b"]),
            ("room-a", ["user-c"]),
        ]
        first = self.store.add_room_members(rooms)
        repeated = self.store.add_room_members(
            [("room-a", ["user-a", "user-b", "user-c"]), ("room-b", ["user-b"])]
        )

        self.assertEqual(
            {
                "rooms": [
                    {"room_id": "room-a", "user_ids": ["user-a", "user-b", "user-c"]},
                    {"room_id": "room-b", "user_ids": ["user-b"]},
                ],
                "added_count": 4,
            },
            first,
        )
        self.assertEqual(0, repeated["added_count"])
        self.assertEqual(["user-a", "user-b", "user-c"], self.store.room_user_ids("room-a"))
        self.assertEqual(["user-b"], self.store.room_user_ids("room-b"))

    def test_search_can_filter_by_room_members(self):
        user_a = np.zeros(256, dtype=np.float32)
        user_a[0] = 1.0
        user_b = np.zeros(256, dtype=np.float32)
        user_b[1] = 1.0
        query = user_b.copy()

        self.store.add("user-a", user_a, 15.0)
        self.store.add("user-b", user_b, 15.0)
        self.store.add_room_member("room-a", "user-a")
        self.store.add_room_member("room-b", "user-b")

        candidates = self.store.search(query, top_k=5, user_ids=self.store.room_user_ids("room-a"))
        self.assertEqual(["user-a"], [candidate["user_id"] for candidate in candidates])

        candidates = self.store.search(query, top_k=5, user_ids=self.store.room_user_ids("room-b"))
        self.assertEqual(["user-b"], [candidate["user_id"] for candidate in candidates])


if __name__ == "__main__":
    unittest.main()
