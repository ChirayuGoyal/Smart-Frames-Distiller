#!/usr/bin/env python3
"""
IR Night Vision Motion Detection System
========================================
Purpose-built for infrared / night-vision CCTV cameras.

WHY a separate system?
-----------------------
Standard motion detectors (MOG2 defaults, simple frame-diff) fail on IR video because:
  1. Sensor noise  — IR CMOS sensors generate far more pixel-level noise than visible-light
     sensors; simple absdiff fires on every frame even in an empty scene.
  2. H.264/H.265 compression artifacts — block-DCT artefacts cause ~4-8 pixel blobs per
     macro-block boundary that look identical to motion to a naive detector.
  3. Low contrast  — foreground objects often have similar IR reflectance as the background.
  4. Uneven illumination — IR LED arrays create radial hotspots; intensity gradients shift
     as LEDs warm up, confusing single-frame background models.
  5. No colour cues — all discrimination must happen in luminance only.

Research-backed pipeline (per: bwsw noise-tolerant detector, PyImageSearch, ScienceDirect
near-IR background subtraction benchmark, CLAHE clip studies):
  Pre-process  → Gaussian blur (kills compression + sensor noise) + CLAHE (restores contrast)
  Background   → Weighted running average (alpha 0.05) — slower to adapt, more stable than MOG2
                 MOG2 with high varThreshold (50-80) as second opinion
  Differencing → absdiff → threshold (25-35 for IR) → morphological close→open→dilate
  Temporal     → Ring-buffer confirmation: pixel must be "hot" in K of last N masks
  Output       → Bounding boxes, heat overlay, HUD, sparkline

Methods
-------
  running_avg   Weighted exponential average background. Best for slow-changing IR scenes.
  mog2          Adaptive Gaussian mixture. Better for scenes with periodic background motion.
  frame_diff    Rolling lag-frame differencing. Zero warmup, zero memory — fallback only.
  ensemble      All three combined with OR vote (highest sensitivity, lowest misses).

Usage
-----
  # Annotated output video
  python ir_motion_detector.py --input clip.mp4 --output ir_motion_out.mp4

  # RTSP night-vision stream, live window
  python ir_motion_detector.py --input rtsp://10.x.x.x:8554/cam --show

  # Explicit sensitivity + method
  python ir_motion_detector.py --input clip.mp4 --sensitivity high --method ensemble --output out.mp4

  # Auto-calibrate noise floor from first 60 frames, then detect
  python ir_motion_detector.py --input clip.mp4 --auto-calibrate --output out.mp4

  # Diagnose: show split view (preprocessed | annotated)
  python ir_motion_detector.py --input clip.mp4 --debug-view --output out.mp4
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ir_motion_detector")


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity presets  (tuned for IR noise characteristics)
# ─────────────────────────────────────────────────────────────────────────────

_PRESETS = {
    # Use when scene is very cluttered or camera is low-quality (lots of noise/artifacts)
    "low": dict(
        pixel_threshold=35,        # only count large pixel jumps
        min_motion_ratio=0.008,    # 0.8 % of frame
        min_region_area=1500,      # ignore tiny blobs
        confirm_frames=4,          # need 4 consistent frames
        temporal_k=3,              # hot in ≥3 of last N masks
        temporal_n=5,
        blur_ksize=21,             # heavy blur kills compression artifacts
        clahe_clip=2.0,
    ),
    # Good general-purpose starting point for most IR cameras
    "medium": dict(
        pixel_threshold=25,
        min_motion_ratio=0.002,    # 0.2 %
        min_region_area=600,
        confirm_frames=2,
        temporal_k=2,
        temporal_n=4,
        blur_ksize=11,
        clahe_clip=2.5,
    ),
    # Use when motion is subtle (slow walking, slight hand movement)
    "high": dict(
        pixel_threshold=15,
        min_motion_ratio=0.0005,   # 0.05 %
        min_region_area=150,
        confirm_frames=1,
        temporal_k=1,
        temporal_n=3,
        blur_ksize=7,
        clahe_clip=3.0,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IRMotionConfig:
    # Method
    method: str = "ensemble"           # running_avg | mog2 | frame_diff | ensemble
    sensitivity: str = "medium"

    # Pre-processing
    scale_factor: float = 0.5          # process at half resolution (noise + speed)
    blur_ksize: int = 11               # Gaussian blur kernel (must be odd) — kills H.264 artifacts
    clahe_clip: float = 2.5            # CLAHE clip limit (>4.0 amplifies IR noise — avoid)
    clahe_tile: int = 8                # CLAHE tile grid size

    # Threshold
    pixel_threshold: int = 25          # gray-level diff to count as changed pixel
    min_motion_ratio: float = 0.002    # fraction of (scaled) frame area
    min_region_area: int = 600         # minimum contour area in SCALED pixels

    # Morphology (applied after threshold)
    morph_close_k: int = 5            # close fills holes caused by low-contrast IR
    morph_open_k: int = 3             # open removes remaining noise blobs
    morph_dilate_k: int = 3           # dilate expands + merges nearby regions

    # Temporal ring-buffer confirmation
    # A pixel is "confirmed motion" only if it was hot in ≥ temporal_k of last temporal_n masks
    temporal_k: int = 2
    temporal_n: int = 4

    # Temporal state machine
    confirm_frames: int = 2            # consecutive confirmed frames before motion=True
    clear_frames: int = 10             # still frames before resetting motion=True→False

    # Background model settings
    warmup_frames: int = 30            # frames to warm up background model
    running_avg_alpha: float = 0.05    # lower = slower to adapt = more stable for IR
    mog2_history: int = 200
    mog2_var_threshold: int = 60       # higher than default (16) — IR needs this
    frame_diff_lag: int = 3            # lag for frame_diff fallback

    # Auto-calibration
    auto_calibrate: bool = False
    calibration_frames: int = 60       # frames to sample for noise-floor estimation

    # Visualization
    show_overlay: bool = True
    show_bboxes: bool = True
    show_hud: bool = True
    show_intensity_bar: bool = True
    show_sparkline: bool = True
    debug_view: bool = False           # side-by-side: preprocessed | annotated

    def apply_preset(self) -> None:
        p = _PRESETS.get(self.sensitivity, _PRESETS["medium"])
        for k, v in p.items():
            setattr(self, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionRegion:
    x: int; y: int; w: int; h: int
    area: int
    cx: int; cy: int     # centroid


@dataclass
class IRMotionResult:
    is_motion: bool
    motion_ratio: float          # fraction of FULL frame (rescaled back)
    changed_pixels: int          # in SCALED frame
    regions: List[MotionRegion]  # bboxes in FULL frame coords
    mask_full: Optional[np.ndarray]   # uint8 binary mask, full frame size
    preprocessed: Optional[np.ndarray]  # gray, full frame — for debug view
    warmup: bool = False         # still in warmup phase


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processor
# ─────────────────────────────────────────────────────────────────────────────

class IRPreprocessor:
    """
    Convert BGR frame → noise-suppressed, contrast-enhanced grayscale.

    Pipeline per research:
      1. BGR → GRAY   (IR cameras are near-monochrome; colour adds nothing)
      2. Gaussian blur  kills sensor shot-noise and H.264 block artifacts
         Kernel 11×11 handles typical RTSP stream compression (720p/1080p)
      3. CLAHE         restores contrast that IR + blur softened; clip≤3.0 keeps noise down
    """

    def __init__(self, cfg: IRMotionConfig) -> None:
        self.cfg = cfg
        k = cfg.blur_ksize | 1   # ensure odd
        self._blur_ksize = (k, k)
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_tile, cfg.clahe_tile),
        )
        self._scale = cfg.scale_factor

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (scaled_processed_gray, full_gray_for_display)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_gray = gray.copy()

        # Scale down — single biggest noise reducer for high-res IR streams
        if self._scale != 1.0:
            h, w = gray.shape[:2]
            nh = max(1, int(h * self._scale))
            nw = max(1, int(w * self._scale))
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

        # Gaussian blur: kills H.264 8×8 DCT block artifacts and sensor shot noise
        gray = cv2.GaussianBlur(gray, self._blur_ksize, 0)

        # CLAHE: enhance local contrast (clip<=3.0 is safe for IR noise levels)
        gray = self._clahe.apply(gray)

        return gray, full_gray


# ─────────────────────────────────────────────────────────────────────────────
# Background models
# ─────────────────────────────────────────────────────────────────────────────

class RunningAvgBackground:
    """
    cv2.accumulateWeighted — weighted exponential moving average.
    Best for IR: slow alpha (0.05) means background adapts to gradual
    IR LED warm-up and ambient changes, not to fast-moving objects.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._bg: Optional[np.ndarray] = None

    def apply(self, gray: np.ndarray) -> np.ndarray:
        """Returns foreground mask uint8."""
        if self._bg is None:
            self._bg = gray.astype(np.float32)
            return np.zeros_like(gray)
        cv2.accumulateWeighted(gray, self._bg, self.alpha)
        bg8 = cv2.convertScaleAbs(self._bg)
        diff = cv2.absdiff(gray, bg8)
        return diff

    def reset(self) -> None:
        self._bg = None


class MOG2Background:
    """MOG2 with IR-tuned parameters (high varThreshold)."""

    def __init__(self, history: int = 200, var_threshold: int = 60) -> None:
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )

    def apply(self, gray: np.ndarray) -> np.ndarray:
        fg = self._sub.apply(gray)
        # MOG2 returns 255=fg, 127=shadow; binarize strictly
        _, binary = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        return binary

    def reset(self) -> None:
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=self._sub.getHistory(),
            varThreshold=self._sub.getVarThreshold(),
            detectShadows=False,
        )


class FrameDiffBackground:
    """
    Lag-frame differencing on PREPROCESSED frames.
    Using preprocessed (blurred+CLAHE) instead of raw frames is critical —
    raw frame diff picks up every compression artifact.
    """

    def __init__(self, lag: int = 3) -> None:
        self._buf: deque = deque(maxlen=max(2, lag + 1))
        self._lag = lag

    def apply(self, gray: np.ndarray) -> np.ndarray:
        self._buf.append(gray.copy())
        if len(self._buf) <= self._lag:
            return np.zeros_like(gray)
        ref = self._buf[0]
        diff = cv2.absdiff(ref, gray)
        return diff

    def reset(self) -> None:
        self._buf.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Morphological post-processor
# ─────────────────────────────────────────────────────────────────────────────

class MorphProcessor:
    """
    Three-stage morphology optimised for IR noise patterns:
      close  → fills the low-contrast "holes" that appear inside moving IR objects
      open   → removes residual small blobs from sensor noise that survived blur+threshold
      dilate → merges nearby blobs into a single region; improves bounding box quality
    """

    def __init__(self, cfg: IRMotionConfig) -> None:
        def _k(size: int) -> np.ndarray:
            s = max(3, size | 1)
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))

        self._close_k = _k(cfg.morph_close_k)
        self._open_k  = _k(cfg.morph_open_k)
        self._dilate_k = _k(cfg.morph_dilate_k)

    def process(self, binary: np.ndarray) -> np.ndarray:
        out = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._close_k)
        out = cv2.morphologyEx(out,    cv2.MORPH_OPEN,  self._open_k)
        out = cv2.dilate(out, self._dilate_k, iterations=1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Temporal ring-buffer confirmation
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConfirmation:
    """
    Eliminates single-frame noise bursts.

    Maintains a ring buffer of N binary masks.  A pixel is "truly moving" only
    if it was flagged in at least K of the last N masks.  This kills the common
    IR failure mode where one noisy frame creates a spurious detection.
    """

    def __init__(self, k: int, n: int) -> None:
        self._k = k
        self._n = n
        self._buf: deque = deque(maxlen=n)

    def confirm(self, binary: np.ndarray) -> np.ndarray:
        self._buf.append(binary.astype(np.uint8) // 255)  # 0/1
        if len(self._buf) < self._k:
            return np.zeros_like(binary)
        stack = np.stack(list(self._buf), axis=0)   # (n, H, W) uint8
        votes = stack.sum(axis=0)                    # (H, W) how many frames had this pixel
        confirmed = (votes >= self._k).astype(np.uint8) * 255
        return confirmed

    def reset(self) -> None:
        self._buf.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-calibrator  (noise floor estimation)
# ─────────────────────────────────────────────────────────────────────────────

class NoiseCalibrator:
    """
    Samples the first N preprocessed frames from a static scene to estimate
    the 99th-percentile pixel noise level, then sets pixel_threshold accordingly.
    """

    def __init__(self, n_frames: int = 60) -> None:
        self._n = n_frames
        self._diffs: List[float] = []
        self._prev: Optional[np.ndarray] = None

    @property
    def done(self) -> bool:
        return len(self._diffs) >= self._n

    def feed(self, gray: np.ndarray) -> None:
        if self._prev is not None:
            diff = cv2.absdiff(self._prev, gray)
            self._diffs.append(float(np.percentile(diff, 99)))
        self._prev = gray.copy()

    def recommended_threshold(self) -> int:
        if not self._diffs:
            return 25
        noise_p99 = float(np.percentile(self._diffs, 90))
        # Set threshold at 1.5× the 90th-percentile noise level
        thresh = max(10, int(noise_p99 * 1.5))
        logger.info(
            "Auto-calibration: noise P99=%.1f → pixel_threshold=%d", noise_p99, thresh
        )
        return thresh


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

class IRMotionDetector:
    """
    Full IR motion detection pipeline.

    Call process_frame(bgr_frame) for each frame → IRMotionResult.
    """

    def __init__(self, cfg: IRMotionConfig) -> None:
        self.cfg = cfg
        self._preprocessor = IRPreprocessor(cfg)
        self._morph = MorphProcessor(cfg)
        self._temporal = TemporalConfirmation(cfg.temporal_k, cfg.temporal_n)

        # Background models
        self._run_avg = RunningAvgBackground(alpha=cfg.running_avg_alpha)
        self._mog2    = MOG2Background(cfg.mog2_history, cfg.mog2_var_threshold)
        self._fdiff   = FrameDiffBackground(cfg.frame_diff_lag)

        # Auto-calibration
        self._calibrator = NoiseCalibrator(cfg.calibration_frames) if cfg.auto_calibrate else None
        self._calibrated = not cfg.auto_calibrate

        # State machine
        self._warmup_left = cfg.warmup_frames
        self._confirm_count = 0
        self._clear_count = 0
        self._motion_active = False
        self._frame_size: Tuple[int, int] = (0, 0)   # (W, H) full frame

    def reset(self) -> None:
        self._run_avg.reset()
        self._mog2.reset()
        self._fdiff.reset()
        self._temporal.reset()
        self._warmup_left = self.cfg.warmup_frames
        self._confirm_count = 0
        self._clear_count = 0
        self._motion_active = False
        if self.cfg.auto_calibrate:
            self._calibrator = NoiseCalibrator(self.cfg.calibration_frames)
            self._calibrated = False

    def _threshold_to_binary(self, diff: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(
            diff, self.cfg.pixel_threshold, 255, cv2.THRESH_BINARY
        )
        return binary

    def _get_raw_mask(self, gray: np.ndarray) -> np.ndarray:
        """Run all selected background models → combined binary mask."""
        method = self.cfg.method

        if method == "running_avg":
            diff = self._run_avg.apply(gray)
            return self._threshold_to_binary(diff)

        elif method == "mog2":
            return self._mog2.apply(gray)

        elif method == "frame_diff":
            diff = self._fdiff.apply(gray)
            return self._threshold_to_binary(diff)

        elif method == "ensemble":
            # Warm ALL three, combine with OR
            diff_ra  = self._run_avg.apply(gray)
            mask_ra  = self._threshold_to_binary(diff_ra)
            mask_mog = self._mog2.apply(gray)
            diff_fd  = self._fdiff.apply(gray)
            mask_fd  = self._threshold_to_binary(diff_fd)
            # OR vote: if ANY model sees motion at a pixel, count it
            combined = cv2.bitwise_or(mask_ra, mask_mog)
            combined = cv2.bitwise_or(combined, mask_fd)
            return combined

        return np.zeros_like(gray)

    def _find_regions(
        self, mask_scaled: np.ndarray, scale: float, full_w: int, full_h: int
    ) -> List[MotionRegion]:
        """Find contours in scaled mask, convert bboxes to full-frame coords."""
        contours, _ = cv2.findContours(
            mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        inv = 1.0 / scale if scale > 0 else 1.0
        regions: List[MotionRegion] = []
        for cnt in contours:
            area_scaled = cv2.contourArea(cnt)
            if area_scaled < self.cfg.min_region_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # scale back to full frame
            fx = min(int(x * inv), full_w - 1)
            fy = min(int(y * inv), full_h - 1)
            fw = min(int(w * inv), full_w - fx)
            fh = min(int(h * inv), full_h - fy)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int((M["m10"] / M["m00"]) * inv)
                cy = int((M["m01"] / M["m00"]) * inv)
            else:
                cx, cy = fx + fw // 2, fy + fh // 2
            regions.append(MotionRegion(
                x=fx, y=fy, w=fw, h=fh,
                area=int(area_scaled / (scale * scale)),  # full-frame area estimate
                cx=cx, cy=cy,
            ))
        regions.sort(key=lambda r: r.area, reverse=True)
        return regions

    def process_frame(self, frame: np.ndarray) -> IRMotionResult:
        h, w = frame.shape[:2]
        self._frame_size = (w, h)

        gray_scaled, gray_full = self._preprocessor.process(frame)
        sh, sw = gray_scaled.shape[:2]
        scale = self.cfg.scale_factor

        # ── Auto-calibration phase ──────────────────────────────────────────
        if not self._calibrated and self._calibrator is not None:
            self._calibrator.feed(gray_scaled)
            # Still warm the background model during calibration
            self._run_avg.apply(gray_scaled)
            self._mog2.apply(gray_scaled)
            self._fdiff.apply(gray_scaled)
            if self._calibrator.done:
                self.cfg.pixel_threshold = self._calibrator.recommended_threshold()
                self._calibrated = True
                logger.info("Calibration done. pixel_threshold → %d", self.cfg.pixel_threshold)
            return IRMotionResult(
                is_motion=False, motion_ratio=0.0, changed_pixels=0,
                regions=[], mask_full=None, preprocessed=gray_full, warmup=True,
            )

        # ── Warmup phase ────────────────────────────────────────────────────
        if self._warmup_left > 0:
            self._run_avg.apply(gray_scaled)
            self._mog2.apply(gray_scaled)
            self._fdiff.apply(gray_scaled)
            self._warmup_left -= 1
            return IRMotionResult(
                is_motion=False, motion_ratio=0.0, changed_pixels=0,
                regions=[], mask_full=None, preprocessed=gray_full, warmup=True,
            )

        # ── Detection ───────────────────────────────────────────────────────
        raw_mask = self._get_raw_mask(gray_scaled)
        clean_mask = self._morph.process(raw_mask)
        confirmed_mask = self._temporal.confirm(clean_mask)

        # Compute ratio against full-frame area for consistency
        changed = int(np.count_nonzero(confirmed_mask))
        total_scaled = sh * sw
        raw_ratio = changed / total_scaled if total_scaled else 0.0

        regions = self._find_regions(confirmed_mask, scale, w, h)

        # Scale mask back to full frame for overlay drawing
        if np.any(confirmed_mask):
            mask_full = cv2.resize(confirmed_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_full = np.zeros((h, w), dtype=np.uint8)

        # ── State machine ────────────────────────────────────────────────────
        over = raw_ratio >= self.cfg.min_motion_ratio and len(regions) > 0
        if over:
            self._confirm_count += 1
            self._clear_count = 0
        else:
            self._clear_count += 1
            self._confirm_count = 0

        if not self._motion_active and self._confirm_count >= self.cfg.confirm_frames:
            self._motion_active = True
        elif self._motion_active and self._clear_count >= self.cfg.clear_frames:
            self._motion_active = False

        return IRMotionResult(
            is_motion=self._motion_active,
            motion_ratio=raw_ratio,
            changed_pixels=changed,
            regions=regions,
            mask_full=mask_full,
            preprocessed=gray_full,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_H  = 38
_SLINE_H = 44


class IRVisualizer:
    """
    Overlay rich motion annotations on BGR frames.

    Design choices for IR output:
      - Bright cyan/lime bounding boxes (visible on both bright and dark IR)
      - HOT colormap for motion heat overlay (contrasts against IR greyscale)
      - Intensity bar on right edge with threshold marker
      - Bottom sparkline showing motion ratio history
      - Flashing red "MOTION DETECTED" banner
      - Debug split-view: left = preprocessed grayscale (what detector sees),
                          right = annotated colour frame
    """

    def __init__(self, cfg: IRMotionConfig) -> None:
        self.cfg = cfg
        self._history: deque = deque(maxlen=200)

    def draw(
        self,
        frame: np.ndarray,
        result: IRMotionResult,
        frame_idx: int,
        fps: float,
    ) -> np.ndarray:
        self._history.append(result.motion_ratio)

        if self.cfg.debug_view:
            return self._debug_split(frame, result, frame_idx, fps)

        out = frame.copy()

        if result.warmup:
            self._draw_warmup_banner(out, frame_idx)
            return out

        if self.cfg.show_overlay and result.mask_full is not None:
            self._draw_heat_overlay(out, result.mask_full)
        if self.cfg.show_bboxes:
            self._draw_bboxes(out, result.regions, frame.shape)
        if self.cfg.show_intensity_bar:
            self._draw_intensity_bar(out, result.motion_ratio)
        if self.cfg.show_hud:
            self._draw_hud(out, result, frame_idx, fps)
        if self.cfg.show_sparkline:
            self._draw_sparkline(out)
        if result.is_motion:
            self._draw_motion_banner(out)

        return out

    # ── heat overlay ────────────────────────────────────────────────────────

    def _draw_heat_overlay(self, frame: np.ndarray, mask: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        hot = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        motion_pixels = mask > 0
        alpha = self.cfg.show_overlay and 0.45 or 0.0
        blended = cv2.addWeighted(frame, 1 - alpha, hot, alpha, 0)
        frame[motion_pixels] = blended[motion_pixels]

    # ── bounding boxes ───────────────────────────────────────────────────────

    def _draw_bboxes(
        self, frame: np.ndarray, regions: List[MotionRegion], shape: tuple
    ) -> None:
        fh, fw = shape[:2]
        large_thresh = int(fw * fh * 0.04)  # >4% of frame = large

        for r in regions:
            color = (0, 255, 180) if r.area < large_thresh else (0, 160, 255)
            thick = 2 if r.area < large_thresh else 3
            cv2.rectangle(frame, (r.x, r.y), (r.x + r.w, r.y + r.h), color, thick)
            cv2.circle(frame, (r.cx, r.cy), 5, color, -1)
            label = f"{r.area // 100}×100px"
            lx = max(0, r.x + 4)
            ly = max(14, r.y + r.h - 6)
            cv2.putText(frame, label, (lx, ly), _FONT, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (lx, ly), _FONT, 0.42, color, 1, cv2.LINE_AA)

    # ── intensity bar ────────────────────────────────────────────────────────

    def _draw_intensity_bar(self, frame: np.ndarray, ratio: float) -> None:
        h, w = frame.shape[:2]
        bw, bx = 16, w - 22
        by0, by1 = _HUD_H + 4, h - _SLINE_H - 4
        bh = by1 - by0

        cv2.rectangle(frame, (bx, by0), (bx + bw, by1), (20, 20, 20), -1)

        scale_max = max(self.cfg.min_motion_ratio * 12, 0.03)
        frac = min(ratio / scale_max, 1.0)
        fill = int(bh * frac)
        if fill > 0:
            r_col = int(min(255, frac * 2 * 255))
            g_col = int(min(255, (1 - frac) * 2 * 255))
            cv2.rectangle(
                frame, (bx, by1 - fill), (bx + bw, by1), (0, g_col, r_col), -1
            )

        # Threshold marker
        t_frac = self.cfg.min_motion_ratio / scale_max
        ty = int(by0 + bh * (1.0 - t_frac))
        cv2.line(frame, (bx - 3, ty), (bx + bw + 3, ty), (255, 255, 0), 1)
        cv2.rectangle(frame, (bx, by0), (bx + bw, by1), (60, 60, 60), 1)

    # ── HUD strip ────────────────────────────────────────────────────────────

    def _draw_hud(
        self, frame: np.ndarray, result: IRMotionResult, frame_idx: int, fps: float
    ) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, _HUD_H), (10, 10, 10), -1)

        status_col = (0, 255, 120) if result.is_motion else (100, 100, 100)
        info = (
            f"IR-MOTION  |  Method:{self.cfg.method.upper()}"
            f"  Frame:{frame_idx}"
            f"  FPS:{fps:.1f}"
            f"  Motion:{result.motion_ratio * 100:.3f}%"
            f"  Thresh:{self.cfg.min_motion_ratio * 100:.3f}%"
            f"  Regions:{len(result.regions)}"
            f"  Sens:{self.cfg.sensitivity}"
        )
        cv2.putText(frame, info, (11, 26), _FONT, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info, (10, 26), _FONT, 0.45, status_col, 1, cv2.LINE_AA)

    # ── sparkline ────────────────────────────────────────────────────────────

    def _draw_sparkline(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        sy = h - _SLINE_H
        cv2.rectangle(frame, (0, sy), (w, h), (8, 8, 8), -1)

        hist = list(self._history)
        if len(hist) < 2:
            return

        max_v = max(max(hist), self.cfg.min_motion_ratio * 2, 1e-6)
        inner = _SLINE_H - 10
        n = len(hist)
        xs = [int(i * w / n) for i in range(n)]
        ys = [sy + _SLINE_H - 5 - int((v / max_v) * inner) for v in hist]

        # Threshold line
        ty = sy + _SLINE_H - 5 - int((self.cfg.min_motion_ratio / max_v) * inner)
        cv2.line(frame, (0, ty), (w, ty), (40, 40, 160), 1)

        for i in range(len(hist) - 1):
            over = hist[i] >= self.cfg.min_motion_ratio
            col = (0, 200, 80) if over else (60, 60, 160)
            cv2.line(frame, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), col, 2)

        cv2.putText(
            frame, "Motion History",
            (4, sy + 14), _FONT, 0.38, (90, 90, 90), 1, cv2.LINE_AA,
        )

    # ── motion banner ────────────────────────────────────────────────────────

    def _draw_motion_banner(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        txt = "!! MOTION DETECTED !!"
        fs = 1.2
        th = 2
        (tw, tht), _ = cv2.getTextSize(txt, _FONT, fs, th)
        x = (w - tw) // 2
        y = h // 2 + _HUD_H

        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 18, y - tht - 14), (x + tw + 18, y + 14), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        pulse = int(abs(np.sin(time.monotonic() * 5)) * 180 + 75)
        cv2.putText(frame, txt, (x + 2, y + 2), _FONT, fs, (0, 0, int(pulse * 0.4)), th + 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (x, y), _FONT, fs, (0, pulse // 2, pulse), th, cv2.LINE_AA)

        border_col = (0, 0, pulse)
        cv2.rectangle(frame, (3, _HUD_H + 3), (w - 22, h - _SLINE_H - 3), border_col, 3)

    # ── warmup banner ────────────────────────────────────────────────────────

    def _draw_warmup_banner(self, frame: np.ndarray, frame_idx: int) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, _HUD_H), (10, 10, 40), -1)
        cv2.putText(
            frame,
            f"IR-MOTION  |  WARMING UP / CALIBRATING...  Frame:{frame_idx}",
            (10, 26), _FONT, 0.48, (100, 160, 255), 1, cv2.LINE_AA,
        )

    # ── debug split view ─────────────────────────────────────────────────────

    def _debug_split(
        self,
        frame: np.ndarray,
        result: IRMotionResult,
        frame_idx: int,
        fps: float,
    ) -> np.ndarray:
        """Left: preprocessed grayscale (what the detector sees). Right: annotated."""
        h, w = frame.shape[:2]

        # Left: preprocessed gray → BGR
        if result.preprocessed is not None:
            pre = cv2.cvtColor(result.preprocessed, cv2.COLOR_GRAY2BGR)
            if pre.shape[:2] != (h, w):
                pre = cv2.resize(pre, (w, h))
        else:
            pre = np.zeros_like(frame)

        cv2.putText(pre, "PREPROCESSED (detector input)", (8, 28),
                    _FONT, 0.52, (80, 200, 255), 1, cv2.LINE_AA)

        # Right: annotated
        right = frame.copy()
        if not result.warmup:
            if self.cfg.show_overlay and result.mask_full is not None:
                self._draw_heat_overlay(right, result.mask_full)
            self._draw_bboxes(right, result.regions, right.shape)
        self._draw_hud(right, result, frame_idx, fps)
        self._draw_sparkline(right)
        if result.warmup:
            self._draw_warmup_banner(right, frame_idx)
        elif result.is_motion:
            self._draw_motion_banner(right)

        # Stack horizontally; add divider line
        combined = np.hstack([pre, right])
        mid = w
        cv2.line(combined, (mid, 0), (mid, h), (0, 255, 0), 2)
        return combined


# ─────────────────────────────────────────────────────────────────────────────
# H.264 writer (web-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_ffmpeg() -> Optional[str]:
    """Find ffmpeg — system binary or imageio-ffmpeg bundle."""
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        return exe if exe else None
    except Exception:
        return None


class _H264Writer:
    """
    Encode output as web-compatible H.264 MP4 via ffmpeg rawvideo pipe.
    Falls back to OpenCV mp4v if ffmpeg is unavailable.
    """

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._w = width
        self._h = height
        self._fps = max(fps, 1.0)
        self._proc: Optional[subprocess.Popen] = None
        self._cv_writer: Optional[cv2.VideoWriter] = None
        self.output_path = self._path

        ffmpeg = _resolve_ffmpeg()
        if ffmpeg:
            self._proc = subprocess.Popen(
                [
                    ffmpeg, "-y",
                    "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}",
                    "-r", str(self._fps),
                    "-i", "-",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ensure even dims
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",   # required for browser compatibility
                    "-movflags", "+faststart",  # moov atom at front for streaming
                    "-an",
                    str(self._path),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("H.264 writer: ffmpeg pipe → %s", self._path)
        else:
            logger.warning("ffmpeg not found — falling back to mp4v (not web-compatible). Install: pip install imageio-ffmpeg")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._cv_writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self._w or frame.shape[0] != self._h:
            frame = cv2.resize(frame, (self._w, self._h))
        if self._proc and self._proc.stdin:
            self._proc.stdin.write(frame.astype(np.uint8).tobytes())
        elif self._cv_writer:
            self._cv_writer.write(frame)

    def release(self) -> None:
        if self._proc:
            if self._proc.stdin:
                self._proc.stdin.close()
            err = self._proc.stderr.read().decode("utf-8", errors="replace") if self._proc.stderr else ""
            rc = self._proc.wait(timeout=300)
            if rc != 0:
                raise RuntimeError(f"ffmpeg H.264 encoding failed (exit {rc}): {err[-600:]}")
        if self._cv_writer:
            self._cv_writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run(
    source: str,
    cfg: IRMotionConfig,
    *,
    output_path: Optional[str] = None,
    show: bool = False,
    max_frames: Optional[int] = None,
    log_interval: int = 60,
) -> dict:

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if (
        source.startswith("rtsp") or source.startswith("rtsps")
    ) else cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    full_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output size: if debug_view, double-wide
    out_w = full_w * 2 if cfg.debug_view else full_w
    out_h = full_h

    logger.info(
        "Source: %s | %dx%d @ %.1f FPS | Total frames: %s",
        source, full_w, full_h, fps_src, total if total > 0 else "unknown",
    )
    logger.info(
        "Config: method=%s sensitivity=%s scale=%.1f blur=%d clahe_clip=%.1f "
        "pixel_thresh=%d min_ratio=%.4f confirm=%d clear=%d warmup=%d",
        cfg.method, cfg.sensitivity, cfg.scale_factor,
        cfg.blur_ksize, cfg.clahe_clip, cfg.pixel_threshold,
        cfg.min_motion_ratio, cfg.confirm_frames, cfg.clear_frames, cfg.warmup_frames,
    )

    writer: Optional[_H264Writer] = None
    if output_path:
        writer = _H264Writer(output_path, fps_src, out_w, out_h)
        logger.info("Writing output → %s", output_path)

    detector   = IRMotionDetector(cfg)
    visualizer = IRVisualizer(cfg)

    fidx        = 0
    motion_frs  = 0
    total_regs  = 0
    fps_t       = time.monotonic()
    fps_cnt     = 0
    display_fps = fps_src

    try:
        while True:
            if max_frames and fidx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                logger.info("End of stream at frame %d", fidx)
                break

            result = detector.process_frame(frame)

            fps_cnt += 1
            if fps_cnt >= 30:
                display_fps = fps_cnt / max(time.monotonic() - fps_t, 1e-6)
                fps_t = time.monotonic()
                fps_cnt = 0

            annotated = visualizer.draw(frame, result, fidx, display_fps)

            if result.is_motion:
                motion_frs += 1
                total_regs += len(result.regions)

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("IR Motion Detector", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    logger.info("User quit")
                    break

            if fidx % log_interval == 0 and not result.warmup:
                logger.info(
                    "Frame %5d | motion=%-5s | ratio=%.4f | regions=%d",
                    fidx, result.is_motion, result.motion_ratio, len(result.regions),
                )

            fidx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted at frame %d", fidx)
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    summary = {
        "total_frames":   fidx,
        "motion_frames":  motion_frs,
        "motion_pct":     round(100.0 * motion_frs / max(fidx, 1), 2),
        "avg_regions":    round(total_regs / max(motion_frs, 1), 2),
        "method":         cfg.method,
        "sensitivity":    cfg.sensitivity,
        "pixel_threshold": cfg.pixel_threshold,
        "output":         output_path,
    }
    logger.info("Done. %s", summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IR Night Vision Motion Detection — purpose-built for infrared CCTV cameras.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True, help="Video file or RTSP URL")
    p.add_argument("--output", "-o", default=None,  help="Output annotated .mp4 file")
    p.add_argument("--show",   action="store_true",  help="Live display window")
    p.add_argument(
        "--method",
        choices=["running_avg", "mog2", "frame_diff", "ensemble"],
        default="ensemble",
        help="Background model. ensemble = all three OR-voted (default)",
    )
    p.add_argument(
        "--sensitivity", choices=["low", "medium", "high"], default="medium",
        help="Preset sensitivity (default: medium). Start here, tune from results.",
    )
    p.add_argument("--scale",    type=float, default=0.5,  help="Process at this fraction of resolution (default 0.5)")
    p.add_argument("--blur",     type=int,   default=None, help="Gaussian blur kernel size (odd number)")
    p.add_argument("--clahe-clip", type=float, default=None, help="CLAHE clip limit (2.0-3.0 recommended for IR)")
    p.add_argument("--pixel-thresh", type=int, default=None, help="Pixel intensity diff threshold")
    p.add_argument("--min-ratio",    type=float, default=None, help="Min motion fraction (e.g. 0.002)")
    p.add_argument("--warmup",       type=int,   default=30,   help="Warmup frames (default 30)")
    p.add_argument("--auto-calibrate", action="store_true",
                   help="Auto-estimate pixel threshold from first 60 frames (recommended for new cameras)")
    p.add_argument("--debug-view", action="store_true",
                   help="Side-by-side: preprocessed vs annotated (great for tuning)")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    p.add_argument("--log-interval", type=int, default=60, help="Log every N frames")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    cfg = IRMotionConfig(
        method=args.method,
        sensitivity=args.sensitivity,
        scale_factor=args.scale,
        warmup_frames=args.warmup,
        auto_calibrate=args.auto_calibrate,
        debug_view=args.debug_view,
    )
    cfg.apply_preset()

    # CLI overrides (after preset so they win)
    if args.blur is not None:
        cfg.blur_ksize = args.blur
    if args.clahe_clip is not None:
        cfg.clahe_clip = args.clahe_clip
    if args.pixel_thresh is not None:
        cfg.pixel_threshold = args.pixel_thresh
    if args.min_ratio is not None:
        cfg.min_motion_ratio = args.min_ratio

    if not args.output and not args.show:
        logger.warning("No --output and no --show — results will not be saved or displayed.")

    summary = run(
        source=args.input,
        cfg=cfg,
        output_path=args.output,
        show=args.show,
        max_frames=args.max_frames,
        log_interval=args.log_interval,
    )

    print("\n=== IR Motion Detection Summary ===")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()
