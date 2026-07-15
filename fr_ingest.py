"""
fr_ingest.py — Step 1: Ingest a video into the face recognition database.

Detects faces in every frame_skip-th frame, embeds them with ArcFace R50,
deduplicates within the same site_id (in-memory + Milvus), and inserts new
unique faces into Milvus with site_id and camera_id.

Usage:
    python fr_ingest.py video.mp4 --site-id site-001 --camera-id cam-001
    python fr_ingest.py video.mp4 -c config.json
    python fr_ingest.py video.mp4 --frame-skip 4 --cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from face_recognizer import SCRFD, ArcFace, align_face, resolve_model_path
from fr_milvus import (
    get_or_create_collection,
    insert_batch,
    require_site_id,
    resolve_site_camera,
    search_same_site,
)

_BASE = Path(__file__).parent
log = logging.getLogger(__name__)


def _load_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")).get("face_recognition", {})


class _HybridDedup:
    """Gate 1: in-video cache. Gate 2: Milvus search scoped to site_id."""

    def __init__(self, col, site_id: str, thresh: float) -> None:
        self._col = col
        self._site_id = site_id
        self._thresh = thresh
        self._cache = np.empty((0, 0), dtype=np.float32)
        self._ids: list[str] = []
        self.g1_hits = self.g2_hits = self.g2_calls = 0

    def is_duplicate(self, emb: np.ndarray) -> bool:
        if len(self._ids) > 0 and float(np.max(self._cache @ emb)) >= self._thresh:
            self.g1_hits += 1
            return True
        self.g2_calls += 1
        res = search_same_site(self._col, emb, self._site_id, limit=1, output_fields=["id"])
        if res and res[0]:
            score = float(res[0][0].distance)
            if score >= self._thresh:
                self.g2_hits += 1
                return True
        return False

    def register(self, emb: np.ndarray, uid: str) -> None:
        self._cache = emb[np.newaxis] if len(self._ids) == 0 else np.vstack([self._cache, emb[np.newaxis]])
        self._ids.append(uid)

    def summary(self) -> str:
        return (
            f"Gate1 skipped: {self.g1_hits} | "
            f"Gate2 called: {self.g2_calls} | Gate2 skipped: {self.g2_hits}"
        )


def _check_quality(frame, box, min_size: int, blur_thresh: float,
                   min_asp: float, max_asp: float) -> tuple[bool, str]:
    x1, y1, x2, y2 = (max(0, int(v)) for v in box)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False, "invalid_box"
    if w < min_size or h < min_size:
        return False, "too_small"
    asp = w / h
    if asp < min_asp or asp > max_asp:
        return False, "bad_aspect"
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "empty_crop"
    blur = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if blur < blur_thresh:
        return False, "too_blurry"
    return True, "ok"


def _calibrate(video: str, detector: SCRFD, det_high: float, det_low: float,
               stride: int, size_pct: float, blur_pct: float, min_samples: int):
    sizes, blurs = [], []
    cap = cv2.VideoCapture(video)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            boxes, _ = detector.detect(frame, det_high)
            if len(boxes) == 0:
                boxes, _ = detector.detect(frame, det_low)
            for box in boxes:
                x1, y1, x2, y2 = (max(0, int(v)) for v in box)
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                sizes.append(min(w, h))
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    blurs.append(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        idx += 1
    cap.release()

    if len(sizes) < min_samples:
        log.warning(
            "Calibration: only %d samples (need %d) — using fallbacks (size=20, blur=30).",
            len(sizes), min_samples,
        )
        return 20, 30.0

    min_size = int(np.percentile(sizes, size_pct))
    blur_floor = float(np.percentile(blurs, blur_pct))
    log.info("Calibration: %d samples | min_face_size=%d | blur_thresh=%.1f", len(sizes), min_size, blur_floor)
    return min_size, blur_floor


def ingest(video: str, cfg: dict) -> None:
    site_id, camera_id = resolve_site_camera(cfg)
    require_site_id(site_id, tool="fr_ingest")

    provider = cfg.get("execution_provider", "CPUExecutionProvider")
    device_id = int(cfg.get("gpu_device_id", 0))
    det_high = float(cfg.get("det_thresh_high", 0.5))
    det_low = float(cfg.get("det_thresh_low", 0.35))
    dedup_thr = float(cfg.get("dedup_thresh", 0.95))
    skip = int(cfg.get("frame_skip", 2))
    batch_sz = int(cfg.get("milvus", {}).get("batch_size", 100))

    q = cfg.get("quality", {})
    size_pct = float(q.get("face_size_percentile", 10))
    blur_pct = float(q.get("blur_percentile", 10))
    min_asp = float(q.get("min_aspect", 0.6))
    max_asp = float(q.get("max_aspect", 1.8))
    cal_stride = int(q.get("sample_frame_stride", 5))
    min_samp = int(q.get("min_samples_required", 30))

    detector = SCRFD(resolve_model_path(cfg["detector_model"]), provider, device_id)
    arcface = ArcFace(resolve_model_path(cfg["embedding_model"]), provider, device_id)

    min_size, blur_thresh = _calibrate(
        video, detector, det_high, det_low, cal_stride, size_pct, blur_pct, min_samp,
    )

    col = get_or_create_collection(cfg.get("milvus", {}))
    dedup = _HybridDedup(col, site_id, dedup_thr)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log.info(
        "Video: %s | site=%s | camera=%s | frames=%d | fps=%.1f | sampling 1/%d",
        video, site_id, camera_id or "(none)", total_f, fps, skip,
    )

    batch_ids, batch_embs = [], []
    counts = Counter()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        boxes, kps = detector.detect(frame, det_high)
        if len(boxes) == 0:
            boxes, kps = detector.detect(frame, det_low)

        for i in range(len(boxes)):
            counts["detected"] += 1
            ok_q, reason = _check_quality(frame, boxes[i], min_size, blur_thresh, min_asp, max_asp)
            if not ok_q:
                counts[f"rejected_{reason}"] += 1
                continue

            crop = align_face(frame, kps[i])
            emb = arcface.embed(crop)

            if dedup.is_duplicate(emb):
                counts["dedup_skipped"] += 1
                continue

            uid = str(uuid.uuid4())
            dedup.register(emb, uid)
            batch_ids.append(uid)
            batch_embs.append(emb.tolist())
            counts["inserted"] += 1

            if len(batch_ids) >= batch_sz:
                insert_batch(col, batch_ids, batch_embs, site_id, camera_id)
                batch_ids, batch_embs = [], []

        frame_idx += 1

    cap.release()

    if batch_ids:
        insert_batch(col, batch_ids, batch_embs, site_id, camera_id)

    log.info(
        "Ingest done | site=%s | frames=%d | detected=%d | inserted=%d | dedup_skipped=%d | %s",
        site_id,
        frame_idx // max(skip, 1),
        counts["detected"],
        counts["inserted"],
        counts["dedup_skipped"],
        dedup.summary(),
    )
    rejections = {k: v for k, v in counts.items() if k.startswith("rejected_")}
    if rejections:
        log.info("Rejections: %s", rejections)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Ingest a video into the face recognition database")
    parser.add_argument("video", help="Path to input video (.mp4 / .avi / .mkv)")
    parser.add_argument("-c", "--config", type=Path, default=_BASE / "config.json")
    parser.add_argument("--site-id", default=None, help="Site identifier (required)")
    parser.add_argument("--camera-id", default=None, help="Camera identifier")
    parser.add_argument("--frame-skip", type=int, default=None, help="Process every N-th frame")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    if args.frame_skip is not None:
        cfg["frame_skip"] = args.frame_skip
    if args.cpu:
        cfg["execution_provider"] = "CPUExecutionProvider"
    if args.site_id is not None:
        cfg["site_id"] = args.site_id
    if args.camera_id is not None:
        cfg["camera_id"] = args.camera_id

    ingest(args.video, cfg)


if __name__ == "__main__":
    main()
