"""
test_faces_unit.py — Unit tests for `FaceTracker` (tick aging) and `InMemoryFaceStore`.
"""
from __future__ import annotations

import unittest
import numpy as np

from faces.store import InMemoryFaceStore
from faces.tracker import FaceTracker


class TestFaceTracker(unittest.TestCase):
    def test_tracker_update_and_aging(self):
        tracker = FaceTracker(iou_thresh=0.5, history_len=3, max_age=2)

        # Frame 0: detection
        dets = [{"box": [10, 10, 50, 50], "uuid": "u1", "name": "Alice", "score": 0.9}]
        out = tracker.update(dets)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "Alice")
        track_id = out[0]["track_id"]

        # Frame 1: skip frame (tick)
        tracker.tick()
        active = tracker.get_active()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["track_id"], track_id)

        # Frame 2: second skip frame (tick) -> age=2 (<= max_age 2, still kept)
        tracker.tick()
        active = tracker.get_active()
        self.assertEqual(len(active), 1)

        # Frame 3: third skip frame (tick) -> age=3 (> max_age 2, dropped)
        tracker.tick()
        active = tracker.get_active()
        self.assertEqual(len(active), 0)

    def test_majority_vote_smoothing(self):
        tracker = FaceTracker(iou_thresh=0.5, history_len=5, max_age=5)
        # 3 frames of Alice, 1 frame of Bob on same box
        for _ in range(3):
            tracker.update([{"box": [10, 10, 50, 50], "uuid": "u1", "name": "Alice", "score": 0.9}])
        out = tracker.update([{"box": [12, 12, 52, 52], "uuid": "u2", "name": "Bob", "score": 0.8}])
        # Majority vote should keep Alice
        self.assertEqual(out[0]["name"], "Alice")


class TestInMemoryFaceStore(unittest.TestCase):
    def test_in_memory_crud_and_search(self):
        store = InMemoryFaceStore()
        store.connect()

        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        store.upsert_row(
            uid="id-1", person_id="pid-1", embedding=emb1.tolist(),
            name="Alice", role="Dev", department="Eng", notes="",
            site_id="site-1", camera_id="cam-1",
        )
        store.upsert_row(
            uid="id-2", person_id="pid-2", embedding=emb2.tolist(),
            name="Bob", role="PM", department="Eng", notes="",
            site_id="site-1", camera_id="cam-1",
        )

        rows = store.fetch_site_rows("site-1")
        self.assertEqual(len(rows), 2)

        # ANN search
        query = np.array([0.99, 0.1, 0.0], dtype=np.float32)
        query /= np.linalg.norm(query)
        hits = store.search(query, "site-1", limit=1, output_fields=["name"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["name"], "Alice")

        # Delete
        store.delete_identity("id-1")
        rows = store.fetch_site_rows("site-1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "Bob")

        store.close()


if __name__ == "__main__":
    unittest.main()
