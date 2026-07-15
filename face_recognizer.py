"""
face_recognizer.py — Face recognition overlay for the action-aware pipeline.

Runs on the compressed (filtered) video produced by action-aware selection.
Detects faces with SCRFD, embeds with ArcFace R50, queries Milvus, and writes
an annotated copy with colour-coded overlays:

  green  — tagged face  (name stored in Milvus)
  orange — untagged face (in Milvus DB but no name assigned yet)

Only faces with a Milvus match above similarity_thresh are drawn; purely
undetected or low-confidence faces are not overlaid.
"""

from __future__ import annotations

import logging
import subprocess
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from pymilvus import connections, Collection

log = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent


def resolve_model_path(path: str) -> str:
    """Resolve model path: absolute paths pass through; relative paths are resolved
    from the 07-action-aware directory where this file lives."""
    p = Path(path)
    return str(p) if p.is_absolute() else str(_BASE_DIR / p)


# ── Visual constants ─────────────────────────────────────────────────────────
TAGGED_COLOR   = (0, 200, 0)    # BGR green  — named identity
UNTAGGED_COLOR = (0, 140, 255)  # BGR orange — in DB but unnamed
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.55
FONT_THICKNESS = 2


# ── SCRFD face detector ───────────────────────────────────────────────────────

def _dist2box(centers: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return np.stack([
        centers[:, 0] - dist[:, 0],
        centers[:, 1] - dist[:, 1],
        centers[:, 0] + dist[:, 2],
        centers[:, 1] + dist[:, 3],
    ], axis=-1)


def _dist2kps(centers: np.ndarray, dist: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, dist.shape[1], 2):
        preds.append(centers[:, 0] + dist[:, i])
        preds.append(centers[:, 1] + dist[:, i + 1])
    return np.stack(preds, axis=-1).reshape(-1, 5, 2)


class SCRFD:
    _STRIDES    = (8, 16, 32)
    _INPUT_HW   = (640, 640)

    def __init__(self, model_path: str, provider: str, device_id: int) -> None:
        opts = [{"device_id": device_id} if "CUDA" in provider else {}, {}]
        self._sess = ort.InferenceSession(
            model_path,
            providers=[provider, "CPUExecutionProvider"],
            provider_options=opts,
        )
        self._in_name = self._sess.get_inputs()[0].name
        self._out_names = [o.name for o in self._sess.get_outputs()]
        log.info("SCRFD: %s", self._sess.get_providers()[0])

    def detect(self, bgr: np.ndarray, thresh: float = 0.5):
        """
        Returns (boxes, kps) in original frame pixel coords.
        boxes: float32 (N, 4)  [x1, y1, x2, y2]
        kps  : float32 (N, 5, 2)
        """
        h, w = bgr.shape[:2]
        ih, iw = self._INPUT_HW

        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        blob = cv2.resize(rgb, (iw, ih)).astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None]

        outs = self._sess.run(self._out_names, {self._in_name: blob})

        boxes_acc, kps_acc = [], []
        for i, stride in enumerate(self._STRIDES):
            scores = outs[i].reshape(-1)
            bbox   = outs[i + 3].reshape(-1, 4) * stride
            kps    = outs[i + 6].reshape(-1, 10) * stride

            fm_h, fm_w = ih // stride, iw // stride
            yy, xx = np.mgrid[:fm_h, :fm_w]
            centers = (np.stack([xx, yy], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
            centers = np.repeat(centers, 2, axis=0)

            inds = scores > thresh
            if not inds.any():
                continue
            boxes_acc.append(_dist2box(centers, bbox)[inds])
            kps_acc.append(_dist2kps(centers, kps)[inds])

        if not boxes_acc:
            return np.empty((0, 4), np.float32), np.empty((0, 5, 2), np.float32)

        boxes = np.concatenate(boxes_acc)
        kps   = np.concatenate(kps_acc)

        sx, sy = w / iw, h / ih
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        kps[:, :, 0] *= sx
        kps[:, :, 1] *= sy
        return boxes, kps


# ── ArcFace R50 embedding ─────────────────────────────────────────────────────

_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Procrustes alignment to 112×112 canonical face crop."""
    src = kps.astype(np.float32)
    dst = _TEMPLATE
    sm, dm = src.mean(0), dst.mean(0)
    sc, dc = src - sm, dst - dm

    U, S, Vt = np.linalg.svd(sc.T @ dc)
    R = (U @ Vt).T
    scale = np.trace(np.diag(S)) / np.sum(sc ** 2)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:, 2]   = dm - scale * R @ sm

    # M maps face→canonical; warpAffine needs the inverse (canonical→face)
    return cv2.warpAffine(bgr, cv2.invertAffineTransform(M), (112, 112), flags=cv2.INTER_LINEAR)


class ArcFace:
    def __init__(self, model_path: str, provider: str, device_id: int) -> None:
        opts = [{"device_id": device_id} if "CUDA" in provider else {}, {}]
        self._sess = ort.InferenceSession(
            model_path,
            providers=[provider, "CPUExecutionProvider"],
            provider_options=opts,
        )
        self._in_name = self._sess.get_inputs()[0].name
        log.info("ArcFace: %s", self._sess.get_providers()[0])

    def embed(self, bgr_112: np.ndarray) -> np.ndarray:
        """Return unit-norm 512-d embedding from a 112×112 BGR crop."""
        rgb  = cv2.cvtColor(bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None]
        emb  = self._sess.run(None, {self._in_name: blob})[0].flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


# ── Milvus lookup ─────────────────────────────────────────────────────────────

def _query_milvus(col: Collection, emb: np.ndarray, sim_thresh: float, site_id: str) -> dict | None:
    """Return best Milvus match within site_id, or None when score < sim_thresh."""
    from fr_milvus import search_same_site

    if not site_id:
        return None
    results = search_same_site(
        col, emb, site_id, limit=1, output_fields=["id", "name", "role", "department"],
    )
    if not results or not results[0]:
        return None
    hit   = results[0][0]
    score = float(hit.distance)
    if score < sim_thresh:
        return None
    return {
        "uuid": hit.entity.get("id") or "",
        "name": hit.entity.get("name") or "",
        "role": hit.entity.get("role") or "",
        "score": round(score, 4),
    }


# ── IoU tracking ─────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (ua + ub - inter)


class _TrackRegistry:
    """
    Lightweight IoU-based tracker with majority-vote smoothing.

    history entries: (uuid, name, score)
    Smoothed result: majority-vote UUID → latest name for that UUID, avg score.
    """

    def __init__(self, iou_thresh: float, history_len: int, max_age: int) -> None:
        self._iou_thresh  = iou_thresh
        self._history_len = history_len
        self._max_age     = max_age
        self._tracks: dict = {}
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: [{"box", "uuid", "name", "score"}]
        Returns smoothed track results for this cycle.
        """
        pairs = []
        for tid, tr in self._tracks.items():
            for di, det in enumerate(detections):
                iou = _iou(tr["box"], det["box"])
                if iou >= self._iou_thresh:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        used_t, used_d = set(), set()
        assignments = []
        for _, tid, di in pairs:
            if tid not in used_t and di not in used_d:
                assignments.append((tid, di))
                used_t.add(tid)
                used_d.add(di)

        results = []
        for tid, di in assignments:
            det = detections[di]
            tr  = self._tracks[tid]
            tr["box"] = det["box"]
            tr["age"] = 0
            tr["hist"].append((det["uuid"], det["name"], det["score"]))
            if len(tr["hist"]) > self._history_len:
                tr["hist"].pop(0)
            results.append({"track_id": tid, "box": det["box"], **_smooth(tr["hist"])})

        for tid in list(self._tracks):
            if tid not in used_t:
                self._tracks[tid]["age"] += 1
        self._tracks = {t: v for t, v in self._tracks.items() if v["age"] <= self._max_age}

        for di, det in enumerate(detections):
            if di not in used_d:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "box":  det["box"],
                    "hist": [(det["uuid"], det["name"], det["score"])],
                    "age":  0,
                }
                results.append({"track_id": tid, "box": det["box"], **_smooth(self._tracks[tid]["hist"])})

        return results

    def get_active(self) -> list[dict]:
        return [
            {"track_id": tid, "box": tr["box"], **_smooth(tr["hist"])}
            for tid, tr in self._tracks.items()
        ]


def _smooth(hist: list) -> dict:
    """Majority-vote on UUID; tie-break by highest average score."""
    uuids  = [h[0] for h in hist]
    counts = Counter(uuids)
    top    = max(counts.values())
    tied   = [u for u, c in counts.items() if c == top]
    if len(tied) == 1:
        winner = tied[0]
    else:
        avgs   = {u: sum(h[2] for h in hist if h[0] == u) / counts[u] for u in tied}
        winner = max(avgs, key=avgs.get)
    matching = [(h[1], h[2]) for h in hist if h[0] == winner]
    name     = matching[-1][0]
    score    = sum(m[1] for m in matching) / len(matching)
    return {"uuid": winner, "name": name, "score": score}


# ── Overlay drawing ───────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, box, name: str, uuid: str, score: float) -> None:
    """
    Draw a colour-coded face box + label.

    green  → tagged (name known): "Alice (84%)"
    orange → untagged (no name): "Unknown [3f7a1b2c]"
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    is_tagged = bool(name)
    color     = TAGGED_COLOR if is_tagged else UNTAGGED_COLOR
    label     = f"{name} ({round(score * 100)}%)" if is_tagged else f"Unknown [{uuid[:8]}]"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    tag_y1 = max(y1 - th - bl - 4, 0)
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


# ── FaceRecognizer ────────────────────────────────────────────────────────────

class FaceRecognizer:
    """
    SCRFD + ArcFace + Milvus in a single stateful processor.

    Expected cfg keys (from config.json → "face_recognition" sub-dict):
      detector_model       — path to det_10g.onnx
      embedding_model      — path to w600k_r50.onnx
      milvus.host/port/collection
      det_thresh_high      — primary detection threshold   (default 0.5)
      det_thresh_low       — fallback detection threshold  (default 0.35)
      similarity_thresh    — Milvus cosine match threshold (default 0.42)
      execution_provider   — CPUExecutionProvider | CUDAExecutionProvider
      gpu_device_id        — GPU index when using CUDA     (default 0)
      frame_skip           — detect every N-th frame       (default 2)
      iou_match_thresh     — IoU threshold for tracking    (default 0.5)
      track_history_len    — rolling history per track     (default 5)
      track_max_age        — drop track after N misses     (default 8)
    """

    def __init__(self, cfg: dict) -> None:
        provider  = cfg.get("execution_provider", "CPUExecutionProvider")
        device_id = int(cfg.get("gpu_device_id", 0))

        self._detector   = SCRFD(resolve_model_path(cfg["detector_model"]), provider, device_id)
        self._arcface    = ArcFace(resolve_model_path(cfg["embedding_model"]), provider, device_id)
        self._det_high   = float(cfg.get("det_thresh_high", 0.5))
        self._det_low    = float(cfg.get("det_thresh_low", 0.35))
        self._sim_thresh = float(cfg.get("similarity_thresh", 0.42))
        self.frame_skip  = int(cfg.get("frame_skip", 2))
        self._site_id = str(cfg.get("site_id") or cfg.get("milvus", {}).get("site_id") or "").strip()

        mc = cfg.get("milvus", {})
        connections.connect(alias="default", host=mc.get("host", "localhost"), port=int(mc.get("port", 19530)))
        self._col = Collection(mc.get("collection", "face_embeddings"))
        self._col.load()
        log.info(
            "FaceRecognizer ready — collection '%s' site_id='%s'",
            self._col.name, self._site_id or "(none)",
        )

        self._tracker = _TrackRegistry(
            iou_thresh  = float(cfg.get("iou_match_thresh", 0.5)),
            history_len = int(cfg.get("track_history_len", 5)),
            max_age     = int(cfg.get("track_max_age", 8)),
        )

    def process_frame(self, bgr: np.ndarray) -> list[dict]:
        """Full pipeline: detect → align → embed → search. Returns tracker results."""
        boxes, kps_arr = self._detector.detect(bgr, self._det_high)
        if len(boxes) == 0:
            boxes, kps_arr = self._detector.detect(bgr, self._det_low)

        detections = []
        for i in range(len(boxes)):
            crop  = align_face(bgr, kps_arr[i])
            emb   = self._arcface.embed(crop)
            match = _query_milvus(self._col, emb, self._sim_thresh, self._site_id)
            if match:
                detections.append({"box": boxes[i], **match})
            else:
                detections.append({"box": boxes[i], "uuid": "", "name": "", "score": 0.0})

        return self._tracker.update(detections)

    def get_active_tracks(self) -> list[dict]:
        return self._tracker.get_active()


# ── Top-level annotator ───────────────────────────────────────────────────────

def annotate_video_with_faces(
    video_path: Path,
    output_path: Path,
    cfg: dict,
    reencode_h264: bool = True,
) -> dict:
    """
    Read the compressed filtered video, overlay face detections, write output.

    Detection runs on every frame_skip-th frame; tracking fills gap frames.
    Returns a summary dict merged into the pipeline report.
    """
    recognizer = FaceRecognizer(cfg)
    frame_skip  = recognizer.frame_skip

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )
    log.info("Face annotation: %s → %s (frame_skip=%d)", video_path.name, output_path.name, frame_skip)

    frame_idx     = 0
    tagged_draws  = 0
    untagged_draws = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tracks = (
            recognizer.process_frame(frame)
            if frame_idx % frame_skip == 0
            else recognizer.get_active_tracks()
        )

        for tr in tracks:
            if tr["uuid"]:  # skip detections with no Milvus match
                _draw_overlay(frame, tr["box"], tr["name"], tr["uuid"], tr["score"])
                if tr["name"]:
                    tagged_draws += 1
                else:
                    untagged_draws += 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if reencode_h264 and shutil.which("ffmpeg"):
        tmp = output_path.with_suffix(".h264_tmp.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", str(output_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(tmp),
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0:
            output_path.unlink()
            tmp.rename(output_path)
            log.info("H.264 re-encode done: %s", output_path.name)
        else:
            tmp.unlink(missing_ok=True)
            log.warning("ffmpeg re-encode failed, keeping mp4v output")

    log.info(
        "Face annotation done: %d frames | %d tagged draws | %d untagged draws",
        frame_idx, tagged_draws, untagged_draws,
    )
    return {
        "output": str(output_path.resolve()),
        "total_frames": frame_idx,
        "tagged_face_draws": tagged_draws,
        "untagged_face_draws": untagged_draws,
        "frame_skip": frame_skip,
    }
