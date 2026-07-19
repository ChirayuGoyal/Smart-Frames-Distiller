"""
faces/tracker.py — IoU-based face tracker with majority-vote smoothing.

Extracted from ``face_recognizer.py`` ``_TrackRegistry`` with the following
improvements:

* **Public API** — renamed ``_TrackRegistry`` → ``FaceTracker``.
* **No pre-seeded history** — new tracks start with only the first
  observation, eliminating majority-vote bias from inherited identity.
* **``tick()`` method** — ages and prunes *all* tracks by one step so
  that skip-frames (where detection is not run) properly advance track
  staleness.
* **``smooth``** — exposed as a module-level function (no underscore).
* **``compute_iou``** — exposed as a module-level function.
"""
from __future__ import annotations

from collections import Counter


# ── IoU computation ───────────────────────────────────────────────────────────

def compute_iou(box_a, box_b) -> float:
    """Compute Intersection-over-Union between two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (ua + ub - inter)


# ── Majority-vote smoothing ──────────────────────────────────────────────────

def smooth(hist: list) -> dict:
    """Majority-vote on UUID; tie-break by highest average score.

    Parameters
    ----------
    hist : list[tuple[str, str, float]]
        Each entry is ``(uuid, name, score)``.

    Returns
    -------
    dict
        ``{"uuid": str, "name": str, "score": float}``
    """
    if not hist:
        return {"uuid": "", "name": "", "score": 0.0}

    uuids = [h[0] for h in hist]
    counts = Counter(uuids)
    top = max(counts.values())
    tied = [u for u, c in counts.items() if c == top]

    if len(tied) == 1:
        winner = tied[0]
    else:
        avgs = {
            u: sum(h[2] for h in hist if h[0] == u) / counts[u]
            for u in tied
        }
        winner = max(avgs, key=avgs.get)

    matching = [(h[1], h[2]) for h in hist if h[0] == winner]
    name = matching[-1][0]
    score = sum(m[1] for m in matching) / len(matching)
    return {"uuid": winner, "name": name, "score": score}


# ── FaceTracker ───────────────────────────────────────────────────────────────

class FaceTracker:
    """Lightweight IoU-based tracker with majority-vote smoothing.

    History entries: ``(uuid, name, score)``

    Smoothed result: majority-vote UUID → latest name for that UUID,
    average score across matching entries.

    Parameters
    ----------
    iou_thresh : float
        Minimum IoU to match a detection to an existing track.
    history_len : int
        Maximum rolling-history length per track.
    max_age : int
        Drop a track after this many consecutive frames without a match.
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        history_len: int = 5,
        max_age: int = 8,
    ) -> None:
        self._iou_thresh = iou_thresh
        self._history_len = history_len
        self._max_age = max_age
        self._tracks: dict[int, dict] = {}
        self._next_id: int = 0

    # ── core update ───────────────────────────────────────────────────────

    def update(self, detections: list[dict]) -> list[dict]:
        """Match detections to existing tracks via greedy IoU assignment.

        Parameters
        ----------
        detections : list[dict]
            Each dict must contain ``{"box", "uuid", "name", "score"}``.

        Returns
        -------
        list[dict]
            Smoothed results for this cycle with keys
            ``{"track_id", "box", "uuid", "name", "score"}``.
        """
        # Compute all pairwise IoU matches above threshold.
        pairs: list[tuple[float, int, int]] = []
        for tid, tr in self._tracks.items():
            for di, det in enumerate(detections):
                iou = compute_iou(tr["box"], det["box"])
                if iou >= self._iou_thresh:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        # Greedy assignment (best IoU first, no double-matching).
        used_t: set[int] = set()
        used_d: set[int] = set()
        assignments: list[tuple[int, int]] = []
        for _, tid, di in pairs:
            if tid not in used_t and di not in used_d:
                assignments.append((tid, di))
                used_t.add(tid)
                used_d.add(di)

        # Update matched tracks.
        results: list[dict] = []
        for tid, di in assignments:
            det = detections[di]
            tr = self._tracks[tid]
            tr["box"] = det["box"]
            tr["age"] = 0
            tr["hist"].append((det["uuid"], det["name"], det["score"]))
            if len(tr["hist"]) > self._history_len:
                tr["hist"].pop(0)
            results.append(
                {"track_id": tid, "box": det["box"], **smooth(tr["hist"])}
            )

        # Age unmatched tracks.
        for tid in list(self._tracks):
            if tid not in used_t:
                self._tracks[tid]["age"] += 1

        # Prune stale tracks.
        self._tracks = {
            t: v for t, v in self._tracks.items() if v["age"] <= self._max_age
        }

        # Create new tracks for unmatched detections — no pre-seeded history.
        for di, det in enumerate(detections):
            if di not in used_d:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "box": det["box"],
                    "hist": [(det["uuid"], det["name"], det["score"])],
                    "age": 0,
                }
                results.append(
                    {"track_id": tid, "box": det["box"],
                     **smooth(self._tracks[tid]["hist"])}
                )

        return results

    # ── skip-frame aging ──────────────────────────────────────────────────

    def tick(self) -> None:
        """Age all tracks by one step and prune stale ones.

        Call this on *skip-frames* (frames where detection is not run) so
        that stale tracks are properly aged even when ``update()`` is not
        called.
        """
        for tr in self._tracks.values():
            tr["age"] += 1
        self._tracks = {
            t: v for t, v in self._tracks.items() if v["age"] <= self._max_age
        }

    # ── active-track query ────────────────────────────────────────────────

    def get_active(self) -> list[dict]:
        """Return smoothed results for all currently-alive tracks."""
        return [
            {"track_id": tid, "box": tr["box"], **smooth(tr["hist"])}
            for tid, tr in self._tracks.items()
        ]
