"""IR night-mode frame filter for the action-aware pipeline.

Replaces do_filter() when the camera feed is detected as night/IR.
Uses IRMotionDetector to flag frames with real motion, then applies
neighbour-padding and writes the kept frames — returning the same dict
shape as do_filter() so runner.py needs no special-casing after the call.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detectors.ir_motion_detector import IRMotionConfig, IRMotionDetector


def do_ir_filter(
    video: Path,
    opts: Any,      # RunOptions — not imported to avoid circular dep
    dst_path: Path,
) -> dict[str, Any]:
    """
    Process *video* with IRMotionDetector; write kept frames to *dst_path*.

    Returns a dict with the same keys as do_filter() so the caller is agnostic
    to which filter ran.
    """
    ir_cfg_d: dict[str, Any] = getattr(opts, "ir_mode_cfg", {})
    method      = str(ir_cfg_d.get("method",      "ensemble"))
    sensitivity = str(ir_cfg_d.get("sensitivity",  "medium"))

    ir_cfg = IRMotionConfig(method=method, sensitivity=sensitivity)
    ir_cfg.apply_preset()

    detector = IRMotionDetector(ir_cfg)
    pad      = int(getattr(opts, "neighbor_pad", 2))
    out_w    = int(getattr(opts, "output_width",  640))
    out_h    = int(getattr(opts, "output_height", 480))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_res = f"{w_src}x{h_src}"
    out_res = f"{out_w}x{out_h}"

    # Load ROI mask once (if configured)
    roi_mask = None
    roi_path = getattr(opts, "roi_path", None)
    if roi_path:
        from roi_loader import load_roi
        roi_mask = load_roi(roi_path, w_src, h_src).mask

    t0 = time.perf_counter()

    # ── Pass: collect frames and per-frame motion flag ─────────────────────────
    frames: list[np.ndarray] = []
    motion_flags: list[bool] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Apply ROI to the frame fed to the detector; full frame is kept for output
        det_frame = frame if roi_mask is None else cv2.bitwise_and(frame, frame, mask=roi_mask)
        result = detector.process_frame(det_frame)
        frames.append(frame)        # original — written to output unchanged
        motion_flags.append(result.is_motion)
    cap.release()

    actual_total = len(frames)

    # ── Expand motion indices with neighbour pad ───────────────────────────────
    motion_set: set[int] = {i for i, m in enumerate(motion_flags) if m}
    padded: set[int] = set()
    for i in motion_set:
        for j in range(max(0, i - pad), min(actual_total, i + pad + 1)):
            padded.add(j)
    kept_indices = sorted(padded)

    processing_ms = (time.perf_counter() - t0) * 1000
    kept_frames   = len(kept_indices)
    reduction_ratio = actual_total / kept_frames if kept_frames else 0.0
    duration_sec    = round(kept_frames / fps, 3) if fps > 0 else 0.0

    # ── Write kept frames ──────────────────────────────────────────────────────
    reencoded = False
    if kept_indices:
        _write_frames(
            [frames[i] for i in kept_indices],
            dst_path,
            fps=fps,
            out_w=out_w,
            out_h=out_h,
        )
        reencoded = True

    # ── Build segment list ─────────────────────────────────────────────────────
    segments: list[dict[str, Any]] = []
    if kept_indices:
        seg_start = seg_end = kept_indices[0]
        for idx in kept_indices[1:]:
            if idx == seg_end + 1:
                seg_end = idx
            else:
                segments.append({"start": seg_start, "end": seg_end})
                seg_start = seg_end = idx
        segments.append({"start": seg_start, "end": seg_end})

    return {
        "kept_indices":         kept_indices,
        "total_frames":         actual_total,
        "kept_frames":          kept_frames,
        "reduction_ratio":      round(reduction_ratio, 4),
        "model":                f"ir_motion/{method}/{sensitivity}",
        "device":               "cpu",
        "inference_resolution": src_res,
        "source_resolution":    src_res,
        "processing_ms":        round(processing_ms, 1),
        "segments":             segments,
        "correlation_timeline": [],
        "predictions":          [],
        "action_changes":       len(motion_set),
        "reencoded_h264":       reencoded,
        "fps":                  fps,
        "duration_sec":         duration_sec,
        "output_resolution":    out_res,
    }


def _write_frames(
    frames: list[np.ndarray],
    dst_path: Path,
    *,
    fps: float,
    out_w: int,
    out_h: int,
) -> None:
    """Write frames to dst_path as H.264 MP4 (ffmpeg pipe when available)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        tmp = dst_path.with_suffix(".ir_tmp.mp4")
        proc = subprocess.Popen(
            [
                ffmpeg, "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{out_w}x{out_h}",
                "-r", str(fps),
                "-i", "-",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                str(tmp),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for frame in frames:
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))
            proc.stdin.write(frame.astype(np.uint8).tobytes())
        proc.stdin.close()
        proc.wait(timeout=300)
        tmp.replace(dst_path)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (out_w, out_h))
        for frame in frames:
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))
            writer.write(frame)
        writer.release()
