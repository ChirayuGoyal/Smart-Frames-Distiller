"""
faces/annotate.py — Video annotation with face recognition overlays and optional person detection.

Evolved from ``face_recognizer.py`` (`annotate_video_with_faces`) with:
- Dependency injection for ``recognizer`` and ``person_detector``.
- Proper resource cleanup (try/finally around ``cv2.VideoCapture`` / ``cv2.VideoWriter``).
- Atomic output (write to temporary `.h264_tmp.mp4` / `.tmp.mp4` then replace).
- Progress callbacks (`progress_cb`).
- Tracker aging (`tracker.tick()`) on skip frames.
- Audio copying from source video (`_copy_audio_stream`).
- Compatibility with both Stack A and Stack B runner stats return contract.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from faces.engine import SCRFD, ArcFace, align_face, resolve_model_path
from faces.persons import PersonDetector
from faces.store import (
    FaceStore,
    FaceStoreError,
    MilvusFaceStore,
    resolve_site_camera,
)
from faces.tracker import FaceTracker

log = logging.getLogger(__name__)

# Colors and drawing constants
TAGGED_COLOR = (0, 200, 0)      # Green for known faces
UNTAGGED_COLOR = (0, 140, 255)  # Orange for unknown matched faces
PERSON_COLOR = (0, 255, 0)      # Green for person body box
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1


def _draw_overlay(frame: np.ndarray, box, name: str, uuid: str, score: float) -> None:
    """Draw a colour-coded face bounding box and text label."""
    x1, y1, x2, y2 = (int(v) for v in box)
    is_tagged = bool(name)
    color = TAGGED_COLOR if is_tagged else UNTAGGED_COLOR
    label = f"{name} ({round(score * 100)}%)" if is_tagged else f"Unknown [{uuid[:8]}]"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    tag_y1 = max(y1 - th - bl - 4, 0)
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - bl - 2),
        FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA
    )


def _draw_person_overlay(frame: np.ndarray, box) -> None:
    """Draw a person bounding box (dashed or thin rectangle)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_COLOR, 1)


def _copy_audio_stream(video_path: Path, audio_src: Path) -> bool:
    """Mux audio from audio_src into video_path without re-encoding either stream."""
    if not shutil.which("ffmpeg"):
        return False
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


def _store_from_cfg(cfg: dict) -> FaceStore:
    """Helper to instantiate MilvusFaceStore from face_recognition cfg dict."""
    milvus_cfg = cfg.get("milvus", {}) if isinstance(cfg.get("milvus"), dict) else {}
    host = str(milvus_cfg.get("host", "localhost"))
    port = int(milvus_cfg.get("port", 19530))
    collection = str(milvus_cfg.get("collection", "face_registry"))
    return MilvusFaceStore(host=host, port=port, collection=collection)


class FaceRecognizer:
    """SCRFD + ArcFace + FaceStore in a stateful processing pipeline."""

    def __init__(self, cfg: dict, *, store: FaceStore | None = None) -> None:
        provider = cfg.get("execution_provider") or cfg.get("device", "auto")
        device_id = int(cfg.get("gpu_device_id", 0))

        self._detector = SCRFD(resolve_model_path(cfg["detector_model"]), provider, device_id)
        self._arcface = ArcFace(resolve_model_path(cfg["embedding_model"]), provider, device_id)
        self._det_high = float(cfg.get("det_thresh_high", 0.5))
        self._det_low = float(cfg.get("det_thresh_low", 0.35))
        self._sim_thresh = float(cfg.get("similarity_thresh") or cfg.get("similarity_threshold") or 0.42)
        self.frame_skip = max(1, int(cfg.get("frame_skip", 2)))
        self._site_id, self._camera_id = resolve_site_camera(cfg)

        # Recognition requires a reachable store with a loaded collection.
        # If either is unavailable and face_recognition.required is false,
        # degrade gracefully: faces are still detected/tracked but stay
        # "Unknown" (no store lookups) instead of crashing the run.
        required = bool(cfg.get("required", False))
        self._store_ready = False
        if store is not None:
            self._store = store
            self._owns_store = False
        else:
            self._store = _store_from_cfg(cfg)
            self._owns_store = True
        try:
            if self._owns_store:
                self._store.connect()
            self._store.require_collection()
            self._store_ready = True
        except Exception as exc:
            if required:
                raise
            log.warning(
                "Face store unavailable (%s) — continuing with detection only, "
                "all faces will be Unknown. Set face_recognition.required=true "
                "to fail instead.", exc,
            )

        self._tracker = FaceTracker(
            iou_thresh=float(cfg.get("iou_match_thresh", 0.5)),
            history_len=int(cfg.get("track_history_len", 5)),
            max_age=int(cfg.get("track_max_age", 8)),
        )

    def process_frame(self, bgr: np.ndarray) -> list[dict]:
        """Detect, align, embed, query store, and update tracker."""
        boxes, kps_arr = self._detector.detect(bgr, self._det_high)
        if len(boxes) == 0:
            boxes, kps_arr = self._detector.detect(bgr, self._det_low)

        detections = []
        for i in range(len(boxes)):
            crop = align_face(bgr, kps_arr[i])
            emb = self._arcface.embed(crop)
            match = self._query_store(emb)
            if match:
                detections.append({"box": boxes[i], **match})
            else:
                detections.append({"box": boxes[i], "uuid": "", "name": "", "score": 0.0})

        return self._tracker.update(detections)

    def get_active_tracks(self, tick: bool = True) -> list[dict]:
        """Return currently active tracks, optionally aging all tracks by 1 step."""
        if tick:
            self._tracker.tick()
        return self._tracker.get_active()

    def _query_store(self, emb: np.ndarray) -> dict | None:
        if not self._site_id or not self._store_ready:
            return None
        try:
            results = self._store.search(
                emb, self._site_id, limit=1, output_fields=["id", "name", "role", "department"]
            )
        except Exception as exc:
            # One store failure disables lookups for the rest of the run
            # rather than raising per frame.
            log.warning("Face store search failed (%s) — disabling lookups for this run", exc)
            self._store_ready = False
            return None
        if not results or not results[0]:
            return None
        hit = results[0]
        score = float(hit.get("score", 0.0))
        if score < self._sim_thresh:
            return None
        return {
            "uuid": str(hit.get("id") or ""),
            "name": str(hit.get("name") or ""),
            "role": str(hit.get("role") or ""),
            "score": round(score, 4),
        }

    def close(self) -> None:
        if self._owns_store and self._store is not None:
            self._store.close()


def annotate_video(
    video_path: Path | str,
    output_path: Path | str,
    cfg: dict,
    *,
    recognizer: FaceRecognizer | None = None,
    person_detector: PersonDetector | None = None,
    progress_cb: Callable[[str, float, str], None] | None = None,
    reencode_h264: bool = True,
) -> dict:
    """Annotate video with face detections/identities and optional person boxes.

    Returns dict with keys needed by ``runner.py`` (`recognised_draws`, `unknown_draws`).
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    owns_recognizer = False
    if recognizer is None:
        recognizer = FaceRecognizer(cfg)
        owns_recognizer = True

    owns_person_det = False
    if person_detector is None and cfg.get("person_detector_enabled", False):
        person_model = cfg.get("person_model", "models/yolov8n.onnx")
        device = cfg.get("device", "auto")
        person_detector = PersonDetector(resolve_model_path(person_model), device=device)
        owns_person_det = True

    frame_skip = recognizer.frame_skip
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if owns_recognizer:
            recognizer.close()
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_out = output_path.with_suffix(".tmp_writing.mp4")
    writer = cv2.VideoWriter(
        str(tmp_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )

    frame_idx = 0
    tagged_draws = 0
    untagged_draws = 0

    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {tmp_out}")
        if progress_cb:
            progress_cb("detection", 0.0, f"Starting face detection on {video_path.name}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if person_detector is not None:
                person_conf = float(cfg.get("person_conf", 0.4))
                p_boxes = person_detector.detect(frame, conf=person_conf)
                for pb in p_boxes:
                    _draw_person_overlay(frame, pb)

            if frame_idx % frame_skip == 0:
                tracks = recognizer.process_frame(frame)
            else:
                tracks = recognizer.get_active_tracks(tick=True)

            for tr in tracks:
                if tr.get("uuid"):
                    _draw_overlay(frame, tr["box"], tr.get("name", ""), tr["uuid"], tr.get("score", 0.0))
                    if tr.get("name"):
                        tagged_draws += 1
                    else:
                        untagged_draws += 1

            writer.write(frame)
            frame_idx += 1

            if progress_cb and total_frames_header > 0 and frame_idx % 20 == 0:
                frac = min(1.0, frame_idx / total_frames_header)
                progress_cb("detection", frac, f"Annotating frame {frame_idx}/{total_frames_header}")

    finally:
        cap.release()
        writer.release()
        if owns_recognizer:
            recognizer.close()

    if progress_cb:
        progress_cb("detection", 1.0, "Processing output video")

    # Atomic rename or re-encode
    if reencode_h264 and shutil.which("ffmpeg"):
        h264_tmp = output_path.with_suffix(".h264_tmp.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", str(tmp_out),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(h264_tmp),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=600)
        except subprocess.TimeoutExpired:
            h264_tmp.unlink(missing_ok=True)
            tmp_out.replace(output_path)
            log.warning("ffmpeg re-encode timed out, kept mp4v output")
            res = None
        if res is not None and res.returncode == 0:
            tmp_out.unlink(missing_ok=True)
            h264_tmp.replace(output_path)
            log.info("H.264 re-encode done: %s", output_path.name)
        elif res is not None:
            h264_tmp.unlink(missing_ok=True)
            tmp_out.replace(output_path)
            log.warning("ffmpeg re-encode failed, kept mp4v output")
    else:
        tmp_out.replace(output_path)

    _copy_audio_stream(output_path, video_path)

    log.info(
        "Face annotation done: %d frames | %d tagged draws | %d untagged draws",
        frame_idx, tagged_draws, untagged_draws,
    )
    return {
        "output": str(output_path.resolve()),
        "total_frames": frame_idx,
        "recognised_draws": tagged_draws,
        "unknown_draws": untagged_draws,
        "tagged_face_draws": tagged_draws,
        "untagged_face_draws": untagged_draws,
        "frame_skip": frame_skip,
    }
