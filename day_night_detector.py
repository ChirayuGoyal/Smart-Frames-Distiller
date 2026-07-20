"""Classify a video source as day (colour) or night (IR/near-grayscale).

IR cameras in night mode produce nearly monochromatic output regardless of
brightness because infrared radiation carries no colour information.
Mean HSV saturation across a sample of frames is a robust, zero-model
discriminator:
  - Daytime colour cameras: mean saturation ≈ 40–120
  - IR / night cameras:     mean saturation ≈ 3–20
Threshold of 30 comfortably separates the two populations.

Two entry points are provided:
  detect_from_frames(frames)     — when you already have the frames (preferred
                                    for RTSP so the stream is only opened once)
  detect_view_mode(source_path)  — opens + reads the source itself (file mode)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

log = logging.getLogger(__name__)

ViewMode = Literal["day", "night"]

_DEFAULT_SAT_THRESH    = 30   # HSV S in [0, 255]
_DEFAULT_SAMPLE_FRAMES = 30


def _mean_saturation(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def detect_from_frames(
    frames: list[np.ndarray],
    *,
    saturation_threshold: float = _DEFAULT_SAT_THRESH,
) -> ViewMode:
    """Classify pre-read frames as 'day' or 'night'.

    Uses the median of per-frame mean-saturation values — more robust than the
    plain mean because one bright/colourful frame in an otherwise-IR stream
    won't skew the result.
    """
    if not frames:
        log.warning("day_night_detector: no frames to analyse — defaulting to day")
        return "day"

    saturations = [_mean_saturation(f) for f in frames]
    median_sat  = float(np.median(saturations))
    detected: ViewMode = "night" if median_sat < saturation_threshold else "day"

    log.info(
        "day/night detection: median_saturation=%.1f  threshold=%.0f  frames=%d  → %s",
        median_sat, saturation_threshold, len(frames), detected,
    )
    return detected


def detect_view_mode(
    source: str | Path,
    *,
    saturation_threshold: float = _DEFAULT_SAT_THRESH,
    sample_frames: int = _DEFAULT_SAMPLE_FRAMES,
    forced_mode: str = "auto",
) -> ViewMode:
    """Return 'night' when source appears IR/near-grayscale, 'day' otherwise.

    Used in file mode (runner.py).  For RTSP streams use detect_from_frames()
    on frames already read from the open cap — that avoids opening a second
    connection.
    """
    if forced_mode in ("day", "night"):
        log.info("day/night detection: forced_mode=%s", forced_mode)
        return forced_mode  # type: ignore[return-value]

    src     = str(source)
    is_rtsp = src.lower().startswith(("rtsp://", "rtsps://"))

    # Use the FFMPEG backend for RTSP so OpenCV doesn't fall back to a broken
    # built-in demuxer.
    cap = (
        cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if is_rtsp
        else cv2.VideoCapture(src)
    )
    if not cap.isOpened():
        log.warning(
            "day_night_detector: cannot open '%s' — defaulting to day", src
        )
        return "day"

    # RTSP streams sometimes deliver a handful of black/corrupt frames while
    # the connection is being established.  Skip them before sampling.
    if is_rtsp:
        for _ in range(5):
            cap.read()

    frames: list[np.ndarray] = []
    for _ in range(sample_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    return detect_from_frames(frames, saturation_threshold=saturation_threshold)
