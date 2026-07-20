"""
Action-aware frame selection per RESEARCH.md:

  logits_t = action_model(clip around t)
  keep(t) if argmax(logits_t) != argmax(logits_{t-k})
           OR |max_conf_t - max_conf_{t-k}| > delta

Also enforces:
  - First and last frame always kept
  - Max gap between kept frames (anchor frames)
  - Neighbor padding around detected action changes
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from action_model import ActionModelBackend, ActionPrediction, create_action_model, resolve_device
from correlation_plot import build_correlation_timeline
from preprocess import resize_for_inference

from common.types import DetectedEvent, FrameSelectionResult, SelectionStats
from common.video_io import iter_frames, read_video_meta

# Number of clips to batch into one GPU forward pass.
# Larger = more GPU utilisation; smaller = lower latency per decision.
_INFER_BATCH = 8


class ActionAwareSelector:
    def __init__(
        self,
        clip_len: int = 16,
        sample_stride: int = 4,
        compare_stride: int | None = None,
        conf_delta: float = 0.15,
        max_gap: int = 30,
        neighbor_pad: int = 2,
        prefer_torch: bool = True,
        device: str | None = "auto",
        inference_scale: float = 1.0,
        inference_max_side: int | None = None,
        ensemble: bool = False,
        model: ActionModelBackend | None = None,
        audio_spikes: bool = False,
        audio_rms_z: float = 2.5,
        audio_delta_z: float = 2.0,
        model_path: str | None = None,
        model_cache_dir: str | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
        roi_mask: np.ndarray | None = None,   # uint8 mask (h,w); 255=inside ROI
    ):
        self.clip_len = clip_len
        self.sample_stride = sample_stride
        self.compare_stride = compare_stride or sample_stride
        self.conf_delta = conf_delta
        self.max_gap = max_gap
        self.neighbor_pad = neighbor_pad
        self.device = resolve_device(device)
        self.inference_scale = inference_scale
        self.inference_max_side = inference_max_side
        self.audio_spikes  = audio_spikes
        self.audio_rms_z   = audio_rms_z
        self.audio_delta_z = audio_delta_z
        self.progress_cb = progress_cb
        self.cancel_check = cancel_check
        self.roi_mask      = roi_mask
        self.model = model or create_action_model(
            clip_len=clip_len, prefer_torch=prefer_torch, device=self.device,
            ensemble=ensemble, model_path=model_path, cache_dir=model_cache_dir,
        )

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def _build_clip(self, ring: deque[np.ndarray]) -> list[np.ndarray]:
        """Pad ring buffer to clip_len by repeating the first (oldest) frame."""
        clip = list(ring)
        while len(clip) < self.clip_len:
            clip.insert(0, clip[0])
        return clip[: self.clip_len]

    # ------------------------------------------------------------------
    # Main selection
    # ------------------------------------------------------------------

    def select(self, video_path: str | Path) -> FrameSelectionResult:
        t0 = time.perf_counter()
        meta = read_video_meta(video_path)

        log.info(
            "filter  model=%s  device=%s  stride=%d  clip_len=%d  batch=%d  video=%s",
            type(self.model).__name__, self.device,
            self.sample_stride, self.clip_len, _INFER_BATCH,
            Path(video_path).name,
        )

        # Sliding window of the last clip_len resized frames.
        # Memory: O(clip_len * H * W * 3) instead of O(N * H * W * 3).
        ring: deque[np.ndarray] = deque(maxlen=self.clip_len)
        inference_size: tuple[int, int] | None = None
        predictions: list[ActionPrediction] = []
        pending: list[tuple[int, list[np.ndarray]]] = []
        n = 0  # total frames seen

        def _flush() -> None:
            if pending:
                if self.cancel_check:
                    self.cancel_check()
                predictions.extend(self.model.predict_batch(pending))
                pending.clear()

        _frame_iter = iter_frames(video_path)
        if _TQDM:
            _frame_iter = _tqdm(
                _frame_iter,
                total=meta.frame_count or None,
                unit="fr", desc="  filter",
                dynamic_ncols=True, leave=True,
            )
        for frame_idx, frame in _frame_iter:
            if self.cancel_check:
                self.cancel_check()
            inf_frame = frame
            if self.roi_mask is not None:
                from roi_loader import apply_roi_mask
                inf_frame = apply_roi_mask(frame, self.roi_mask)
            resized = resize_for_inference(inf_frame, self.inference_scale, self.inference_max_side)
            if inference_size is None:
                h, w = resized.shape[:2]
                inference_size = (w, h)
            ring.append(resized)

            if frame_idx % self.sample_stride == 0:
                pending.append((frame_idx, self._build_clip(ring)))
                if len(pending) >= _INFER_BATCH:
                    _flush()

            n = frame_idx + 1
            if self.progress_cb and frame_idx % 20 == 0:
                self.progress_cb(n, meta.frame_count)

        # Ensure the very last frame is always predicted
        last_idx = n - 1
        if n > 0 and (last_idx % self.sample_stride != 0):
            pending.append((last_idx, self._build_clip(ring)))

        _flush()
        if self.progress_cb:
            self.progress_cb(n, meta.frame_count)

        if n == 0:
            raise ValueError(f"No frames in {video_path}")

        # Predictions arrive in frame order (streaming is sequential)
        keep: set[int] = {0, n - 1}
        events: list[DetectedEvent] = []

        for i in range(1, len(predictions)):
            prev = predictions[i - 1]
            curr = predictions[i]
            label_changed = curr.class_id != prev.class_id
            conf_jump = abs(curr.confidence - prev.confidence) > self.conf_delta

            if label_changed or conf_jump:
                reason = []
                if label_changed:
                    reason.append(f"label {prev.top_label}->{curr.top_label}")
                if conf_jump:
                    reason.append(f"conf Δ={abs(curr.confidence - prev.confidence):.3f}")
                for frame_idx in range(
                    max(0, curr.frame_index - self.neighbor_pad),
                    min(n, curr.frame_index + self.neighbor_pad + 1),
                ):
                    keep.add(frame_idx)
                events.append(
                    DetectedEvent(
                        frame_index=curr.frame_index,
                        timestamp_sec=curr.frame_index / meta.fps,
                        event_type="action_change",
                        score=max(curr.confidence, abs(curr.confidence - prev.confidence)),
                        detail="; ".join(reason),
                    )
                )

        # Audio energy spikes — union with visual keeps before gap enforcement
        if self.audio_spikes:
            from audio_filter import AudioSpikeFinder
            audio_keep = AudioSpikeFinder(
                rms_z_thresh=self.audio_rms_z,
                delta_z_thresh=self.audio_delta_z,
                neighbor_pad=self.neighbor_pad,
            ).find_spike_frames(video_path, meta.fps, n)
            new_from_audio = len(audio_keep - keep)
            keep |= audio_keep
            if new_from_audio:
                log.info("audio spikes added %d new frames to keep set", new_from_audio)

        # Anchor frames: bound the maximum skip gap
        sorted_keep = sorted(keep)
        for i in range(len(sorted_keep) - 1):
            gap = sorted_keep[i + 1] - sorted_keep[i]
            if gap > self.max_gap:
                mid = sorted_keep[i] + gap // 2
                keep.add(mid)

        kept_sorted = sorted(keep)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ratio = n / len(kept_sorted) if kept_sorted else 0.0
        log.info(
            "filter done  kept=%d/%d  ratio=%.1f×  %.1fs  resolution=%s",
            len(kept_sorted), n, ratio, elapsed_ms / 1000,
            f"{inference_size[0]}x{inference_size[1]}" if inference_size else "?",
        )
        correlation_timeline = build_correlation_timeline(
            predictions, meta.fps, conf_delta=self.conf_delta
        )

        return FrameSelectionResult(
            video_path=str(video_path),
            kept_indices=kept_sorted,
            events=events,
            stats=SelectionStats(
                total_frames=n,
                kept_frames=len(kept_sorted),
                reduction_ratio=n / len(kept_sorted) if kept_sorted else 0.0,
                processing_ms=elapsed_ms,
            ),
            metadata={
                "method": "action-aware",
                "model": type(self.model).__name__,
                "device": self.device,
                "source_resolution": f"{meta.width}x{meta.height}",
                "inference_resolution": f"{inference_size[0]}x{inference_size[1]}" if inference_size else None,
                "inference_scale": self.inference_scale,
                "inference_max_side": self.inference_max_side,
                "sample_stride": self.sample_stride,
                "conf_delta": self.conf_delta,
                "predictions": [
                    {
                        "frame": p.frame_index,
                        "class_id": p.class_id,
                        "label": p.top_label,
                        "confidence": round(p.confidence, 4),
                    }
                    for p in predictions
                ],
                "correlation_timeline": correlation_timeline,
            },
        )

    def label_timeline(self, video_path: str | Path, stride: int = 4) -> list[tuple[int, int]]:
        """Return (frame, class_id) for action_label_stability metric."""
        ring: deque[np.ndarray] = deque(maxlen=self.clip_len)
        pending: list[tuple[int, list[np.ndarray]]] = []

        for frame_idx, frame in iter_frames(video_path):
            resized = resize_for_inference(frame, self.inference_scale, self.inference_max_side)
            ring.append(resized)
            if frame_idx % stride == 0:
                pending.append((frame_idx, self._build_clip(ring)))

        out: list[tuple[int, int]] = []
        for i in range(0, len(pending), _INFER_BATCH):
            for pred in self.model.predict_batch(pending[i : i + _INFER_BATCH]):
                out.append((pred.frame_index, pred.class_id))
        return sorted(out)
