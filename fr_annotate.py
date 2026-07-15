"""
fr_annotate.py — Annotate a video with person detection overlays and stable random names.

Pipeline per frame:
  1. YOLOv8n person detection  → person bounding boxes (all ages)
  2. _AppearanceTracker        → two-stage tracking:
       stage-1  IoU matching   → links detection to active tracks each frame
       stage-2  HSV re-ID      → recovers same track ID when a person reappears
  3. _NameAssigner             → one fixed random name per track ID for the whole clip

Audio is copied directly from the input clip (no segment extraction, no re-encode drift).
No face detection, no Milvus, no ArcFace.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from fr_core import PersonDetector

_BASE = Path(__file__).parent
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from common.video_io import try_reencode_h264

log = logging.getLogger(__name__)

_GREEN  = (0, 200, 0)
_WHITE  = (255, 255, 255)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_FSCALE = 0.55
_THICK  = 1

_NAME_POOL = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nancy", "Oliver", "Patricia",
    "Quinn", "Rachel", "Samuel", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane", "Aaron", "Bella", "Carlos", "Donna", "Ethan", "Faith",
    "Grant", "Holly", "Isaac", "Jade", "Karl", "Lily", "Mason", "Nina",
    "Oscar", "Pam", "Ray", "Sara", "Troy", "Ursula", "Vince", "Wren",
    "Xena", "Yale",
]


# ── Name assigner ─────────────────────────────────────────────────────────────

class _NameAssigner:
    """
    Maps a stable track ID → one fixed random name for the whole clip.

    The completed name→track mapping is saved to a JSON sidecar file
    (same stem as the output video, suffix ``_names.json``) so that:
      • re-running detect on the same clip reuses the same names, and
      • a human can inspect / override the assignments.

    If a sidecar already exists when ``annotate_video`` is called it is
    loaded first, so any previously assigned names are preserved.
    """

    def __init__(self, pool: list[str], sidecar: Path | None = None):
        self._pool    = list(pool)
        self._map: dict[int, str] = {}
        self._counter = 0
        self._sidecar = sidecar
        if sidecar and sidecar.is_file():
            try:
                stored = json.loads(sidecar.read_text(encoding="utf-8"))
                # keys stored as strings in JSON — convert back to int
                self._map = {int(k): v for k, v in stored.items()}
                self._counter = len(self._map)
                log.info("detect  loaded %d name assignments from %s", self._counter, sidecar.name)
            except Exception as exc:
                log.warning("detect  could not load name sidecar %s: %s", sidecar, exc)

    def get(self, tid: int) -> str:
        if tid not in self._map:
            self._map[tid] = (
                self._pool[self._counter]
                if self._counter < len(self._pool)
                else f"Person_{tid}"
            )
            self._counter += 1
        return self._map[tid]

    def save(self) -> None:
        if self._sidecar and self._map:
            try:
                self._sidecar.write_text(
                    json.dumps({str(k): v for k, v in self._map.items()}, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                log.warning("detect  could not save name sidecar: %s", exc)


# ── Appearance tracker ────────────────────────────────────────────────────────

class _AppTrack:
    """Single tracked person with a rolling HSV histogram history."""

    _HIST_LEN = 7

    def __init__(self, tid: int, box, hist):
        self.tid  = tid
        self.box  = np.array(box, dtype=np.float32)
        self.age  = 0           # frames since last matched detection
        self._buf = [hist] * self._HIST_LEN
        self._ptr = 0

    def update(self, box, hist):
        self.box = np.array(box, dtype=np.float32)
        if hist is not None:
            self._buf[self._ptr % self._HIST_LEN] = hist
            self._ptr += 1
        self.age = 0

    @property
    def avg_hist(self) -> np.ndarray | None:
        valid = [h for h in self._buf if h is not None]
        if not valid:
            return None
        return np.mean(np.stack(valid, axis=0), axis=0).astype(np.float32)


class _AppearanceTracker:
    """
    Two-stage tracker — stable track IDs across the whole clip.

    Stage 1 — IoU matching
        Each frame: match detections to active tracks by overlap.
        Active tracks that go unmatched for > max_age frames move to the lost pool.

    Stage 2 — HSV histogram re-ID
        Detections still unmatched after stage 1 are compared against lost tracks
        by upper-torso colour histogram (cv2.HISTCMP_CORREL).
        If similarity >= hist_thresh the old track ID is restored → same name.
        Otherwise a new track ID (and new name) is assigned.

    Parameters
    ----------
    iou_thresh   Minimum IoU for stage-1 match (0.25 — lenient for fast movers / children)
    max_age      Frames before an unmatched active track moves to lost pool (90 ≈ 3 s at 30 fps)
    lost_age     Frames a lost track survives in the re-ID pool (300 ≈ 10 s at 30 fps)
    hist_thresh  Minimum CORREL similarity to recover a lost track (0.45 — lenient for re-ID)
    """

    def __init__(
        self,
        iou_thresh:  float = 0.25,
        max_age:     int   = 90,
        lost_age:    int   = 300,
        hist_thresh: float = 0.45,
    ):
        self.iou_thresh  = iou_thresh
        self.max_age     = max_age
        self.lost_age    = lost_age
        self.hist_thresh = hist_thresh
        self._active: list[_AppTrack] = []
        self._lost:   list[_AppTrack] = []
        self._next_id = 0

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_hist(frame: np.ndarray, box) -> np.ndarray | None:
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(frame.shape[1], int(box[2])); y2 = min(frame.shape[0], int(box[3]))
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] < 4 or roi.shape[1] < 4:
            return None
        # Upper torso only — more stable than full body (legs move a lot)
        mid   = roi.shape[0] // 2
        upper = roi[:mid] if mid >= 4 else roi
        hsv   = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        hist  = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    @staticmethod
    def _iou(a, b) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua    = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    @staticmethod
    def _sim(h1, h2) -> float:
        if h1 is None or h2 is None:
            return 0.0
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

    # ── main update ────────────────────────────────────────────────────────────

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        matched_act: set[int] = set()
        matched_det: set[int] = set()

        # Stage 1: IoU — greedy assignment ordered by highest IoU first
        pairs = sorted(
            (
                (self._iou(d["box"], t.box), di, ti)
                for di, d in enumerate(detections)
                for ti, t in enumerate(self._active)
            ),
            reverse=True,
        )
        for iou_val, di, ti in pairs:
            if iou_val < self.iou_thresh or di in matched_det or ti in matched_act:
                continue
            hist = self._extract_hist(frame, detections[di]["box"])
            self._active[ti].update(detections[di]["box"], hist)
            matched_act.add(ti)
            matched_det.add(di)

        # Age unmatched active tracks
        for ti, trk in enumerate(self._active):
            if ti not in matched_act:
                trk.age += 1

        # Stage 2: HSV re-ID for detections still unmatched
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            det_hist = self._extract_hist(frame, det["box"])
            best_sim, best_li = 0.0, -1
            for li, lost in enumerate(self._lost):
                s = self._sim(det_hist, lost.avg_hist)
                if s > best_sim:
                    best_sim, best_li = s, li
            if best_sim >= self.hist_thresh:
                # Recover lost track — same ID, same name
                recovered = self._lost.pop(best_li)
                recovered.update(det["box"], det_hist)
                self._active.append(recovered)
                log.debug("re-id  track=%d  sim=%.2f", recovered.tid, best_sim)
            else:
                # New person
                hist = self._extract_hist(frame, det["box"])
                self._active.append(_AppTrack(self._next_id, det["box"], hist))
                self._next_id += 1

        # Move dead active tracks to lost pool; expire old lost tracks
        new_active, to_lost = [], []
        for trk in self._active:
            (to_lost if trk.age > self.max_age else new_active).append(trk)
        self._active = new_active
        for trk in self._lost:
            trk.age += 1
        self._lost.extend(to_lost)
        self._lost = [t for t in self._lost if t.age <= self.lost_age]

        # Return only tracks that were matched this frame (age == 0)
        return [{"box": t.box, "tid": t.tid} for t in self._active if t.age == 0]


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw(frame: np.ndarray, box, name: str) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
    label = name or "Unknown"
    cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)
    (tw, th), _ = cv2.getTextSize(label, _FONT, _FSCALE, _THICK)
    lx, ly = x1, max(y1 - 1, th + 6)
    cv2.rectangle(frame, (lx, ly - th - 5), (lx + tw + 6, ly + 1), _GREEN, -1)
    cv2.putText(frame, label, (lx + 3, ly - 2), _FONT, _FSCALE, _WHITE, _THICK, cv2.LINE_AA)


# ── Audio helper ──────────────────────────────────────────────────────────────

def _copy_audio_stream(video_path: Path, audio_src: Path) -> bool:
    """
    Mux audio from audio_src into video_path without re-encoding either stream.
    This preserves the exact original audio timing — no drift, no segment extraction.
    """
    tmp = video_path.with_suffix(".with_audio.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_src),
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "copy",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(tmp),
            ],
            check=True, capture_output=True, timeout=300,
        )
        tmp.replace(video_path)
        return True
    except Exception as exc:
        log.debug("audio copy failed: %s", exc)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


# ── Main annotation function ──────────────────────────────────────────────────

def annotate_video(input_path: str, output_path: str, cfg: dict) -> dict:
    """
    Annotate video: YOLO person detection + appearance tracker + stable random names.
    Audio is copied directly from input_path — no re-encode, no timing drift.
    """
    person_conf = float(cfg.get("person_conf", 0.4))
    device      = cfg.get("device", "auto")

    person_det = PersonDetector(cfg["person_model"], device=device)
    tracker    = _AppearanceTracker(
        iou_thresh  = float(cfg.get("track_iou_thresh",  0.25)),
        max_age     = int(cfg.get("track_max_age",       90)),
        lost_age    = int(cfg.get("track_lost_age",      300)),
        hist_thresh = float(cfg.get("track_hist_thresh", 0.45)),
    )
    sidecar  = Path(output_path).with_name(Path(output_path).stem + "_names.json")
    assigner = _NameAssigner(_NAME_POOL, sidecar=sidecar)

    cap    = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )

    frame_idx  = 0
    recognised = 0

    log.info("detect  frames=%d  %s → %s",
             total, Path(input_path).name, Path(output_path).name)

    _bar = _tqdm(
        total=total or None,
        unit="fr", desc="  detect",
        dynamic_ncols=True, leave=True,
    ) if _TQDM else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        p_boxes       = person_det.detect(frame, conf=person_conf)
        dets          = [{"box": b, "name": "", "score": 1.0} for b in p_boxes]
        tracked_state = tracker.update(dets, frame)

        for t in tracked_state:
            name = assigner.get(t["tid"])
            _draw(frame, t["box"], name)
            recognised += 1

        writer.write(frame)
        frame_idx += 1
        if _bar:
            _bar.update(1)

    if _bar:
        _bar.close()
    cap.release()
    writer.release()

    out_p = Path(output_path)
    try_reencode_h264(out_p)
    _copy_audio_stream(out_p, Path(input_path))

    assigner.save()   # persist track-id → name map next to output file

    log.info("detect done  frames=%d  recognised=%d", frame_idx, recognised)
    return {
        "output":           output_path,
        "total_frames":     frame_idx,
        "recognised_draws": recognised,
        "unknown_draws":    0,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(
        description="Annotate video with person detection and stable random names"
    )
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("-c", "--config", default=str(_BASE / "config.json"))
    args = p.parse_args()

    cfg   = json.loads(Path(args.config).read_text(encoding="utf-8")).get("face_recognition", {})
    out   = args.output or str(Path(args.input).stem) + "_annotated.mp4"
    stats = annotate_video(args.input, out, cfg)

    print(f"\nOutput      : {stats['output']}")
    print(f"Frames      : {stats['total_frames']}")
    print(f"Recognised  : {stats['recognised_draws']}")


if __name__ == "__main__":
    main()
