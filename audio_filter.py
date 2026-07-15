"""
Audio energy spike detector for action-aware frame selection.

Extracts mono audio from the video via ffmpeg, computes per-frame RMS energy
aligned to video frame boundaries, then returns frame indices to keep based on
two complementary signals:

  1. RMS z-score   — frames where energy is unusually HIGH  (shouts, impacts,
                     alarms, glass breaking)
  2. RMS delta     — frames where energy CHANGES suddenly   (onset / offset of
                     activity — someone starts talking, a door slams)

No extra Python dependencies — uses only numpy + the ffmpeg binary already
required by the pipeline.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_TARGET_SR = 22050   # resample all audio to this rate for consistent hop sizes


# ── ffmpeg audio extraction ────────────────────────────────────────────────────

def _load_audio(video_path: str | Path, sr: int = _TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Extract mono audio from video as a float32 numpy array via ffmpeg pipe.
    Returns (samples, sample_rate).  On any failure returns (empty array, sr).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1",          # force mono
        "-ar", str(sr),      # resample to target_sr
        "-f", "f32le",       # raw float32 little-endian PCM
        "-vn",               # strip video stream
        "-",                 # pipe to stdout
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0 or not result.stdout:
            stderr = result.stderr.decode(errors="replace")
            if "no audio" in stderr.lower() or "Audio" not in stderr:
                log.debug("no audio stream in %s", Path(video_path).name)
            else:
                log.debug("ffmpeg audio extract failed: %s", stderr[-200:])
            return np.empty(0, dtype=np.float32), sr
        arr = np.frombuffer(result.stdout, dtype=np.float32).copy()
        return arr, sr
    except Exception as exc:
        log.debug("audio load failed for %s: %s", Path(video_path).name, exc)
        return np.empty(0, dtype=np.float32), sr


# ── RMS per video frame ────────────────────────────────────────────────────────

def _rms_per_frame(
    audio: np.ndarray,
    sr: int,
    fps: float,
    n_frames: int,
) -> np.ndarray:
    """Compute RMS energy per video frame, one bucket per frame."""
    hop = max(1, int(sr / fps))
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        e = min(s + hop, len(audio))
        if s >= len(audio):
            rms[i] = 0.0
        else:
            w = audio[s:e]
            rms[i] = float(np.sqrt(np.mean(w * w))) if len(w) > 0 else 0.0
    return rms


# ── Spike finder ───────────────────────────────────────────────────────────────

class AudioSpikeFinder:
    """
    Return frame indices to keep based on audio energy spikes.

    Parameters
    ----------
    rms_z_thresh   : Keep frame if RMS > mean + z·std  (loud events).
    delta_z_thresh : Keep frame if |ΔRMS| > mean + z·std  (sudden changes).
    neighbor_pad   : Extra frames kept around each spike centre.
    min_gap_sec    : Minimum seconds between reported spike events (suppresses
                     consecutive detections from a single long sound).
    """

    def __init__(
        self,
        rms_z_thresh:   float = 2.5,
        delta_z_thresh: float = 2.0,
        neighbor_pad:   int   = 2,
        min_gap_sec:    float = 0.5,
    ):
        self.rms_z_thresh   = rms_z_thresh
        self.delta_z_thresh = delta_z_thresh
        self.neighbor_pad   = neighbor_pad
        self.min_gap_sec    = min_gap_sec

    def find_spike_frames(
        self,
        video_path:   str | Path,
        video_fps:    float,
        total_frames: int,
    ) -> set[int]:
        """Return set of video frame indices to keep because of audio spikes."""
        if total_frames == 0 or video_fps <= 0:
            return set()

        audio, sr = _load_audio(video_path)
        if len(audio) == 0:
            log.info("audio  no audio track — skipping audio spike detection")
            return set()

        rms = _rms_per_frame(audio, sr, video_fps, total_frames)

        # ── Signal 1: loud events (high absolute RMS) ─────────────────────────
        mean_r = float(np.mean(rms))
        std_r  = float(np.std(rms))
        if std_r > 1e-9:
            loud = rms > mean_r + self.rms_z_thresh * std_r
        else:
            loud = np.zeros(len(rms), dtype=bool)

        # ── Signal 2: sudden energy changes (onset / offset) ──────────────────
        delta  = np.abs(np.diff(rms.astype(np.float64), prepend=float(rms[0])))
        mean_d = float(np.mean(delta))
        std_d  = float(np.std(delta))
        if std_d > 1e-9:
            sudden = delta > mean_d + self.delta_z_thresh * std_d
        else:
            sudden = np.zeros(len(delta), dtype=bool)

        spike_mask = loud | sudden
        raw = [int(i) for i in np.where(spike_mask)[0]]

        # ── Suppress consecutive spikes closer than min_gap_sec ───────────────
        min_gap_frames = max(1, int(self.min_gap_sec * video_fps))
        filtered: list[int] = []
        last = -min_gap_frames
        for idx in raw:
            if idx - last >= min_gap_frames:
                filtered.append(idx)
                last = idx

        # ── Expand with neighbor_pad ───────────────────────────────────────────
        keep: set[int] = set()
        for idx in filtered:
            for f in range(
                max(0, idx - self.neighbor_pad),
                min(total_frames, idx + self.neighbor_pad + 1),
            ):
                keep.add(f)

        log.info(
            "audio  spikes=%d (loud=%d delta=%d) → after gap-suppress=%d → frames=%d",
            int(spike_mask.sum()), int(loud.sum()), int(sudden.sum()),
            len(filtered), len(keep),
        )
        return keep
