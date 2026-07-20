"""Synthetic test video generation for validation and tests.

Regimes are designed around MotionEnergyActionModel's decision thresholds
(motion energy > 8.0, edge density > 12.0) so class changes occur exactly
at the regime boundaries.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_FPS = 25.0
_W, _H = 640, 480
_REGIME_LEN = 50  # frames per motion regime


def _checkerboard(cell: int = 16) -> np.ndarray:
    """High edge-density background."""
    tile = np.kron(
        (np.indices((_H // cell + 1, _W // cell + 1)).sum(axis=0) % 2) * 255,
        np.ones((cell, cell)),
    )[:_H, :_W].astype(np.uint8)
    return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)


def make_action_aware_test_video(base_dir: Path) -> dict:
    """Write a deterministic 200-frame clip with 4 motion regimes.

    Regime 0 (frames   0- 49): static plain gray        → static_low
    Regime 1 (frames  50- 99): fast high-contrast box   → motion_high
    Regime 2 (frames 100-149): static checkerboard      → static_high
    Regime 3 (frames 150-199): moving blurred blob      → motion_low

    Returns {"video": str, "change_points": [50, 100, 150]}.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    video_path = base_dir / "synthetic_action_aware.mp4"

    plain = np.full((_H, _W, 3), 96, dtype=np.uint8)
    board = _checkerboard()

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_W, _H)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {video_path}")

    try:
        for i in range(4 * _REGIME_LEN):
            regime = i // _REGIME_LEN
            t = i % _REGIME_LEN

            if regime == 0:
                frame = plain.copy()
            elif regime == 1:
                # Fast-moving high-contrast rectangle: high energy + high edges
                frame = plain.copy()
                x = (t * 24) % (_W - 120)
                cv2.rectangle(frame, (x, 140), (x + 120, 340), (255, 255, 255), -1)
                cv2.rectangle(frame, (x, 140), (x + 120, 340), (0, 0, 0), 4)
            elif regime == 2:
                frame = board.copy()
            else:
                # Slow smooth blob on plain bg: moderate energy, low edges
                frame = plain.copy()
                x = 60 + (t * 20) % (_W - 200)
                cv2.circle(frame, (x + 70, 240), 90, (150, 150, 150), -1)
                frame = cv2.GaussianBlur(frame, (31, 31), 0)

            writer.write(frame)
    finally:
        writer.release()

    return {
        "video": str(video_path),
        "change_points": [_REGIME_LEN, 2 * _REGIME_LEN, 3 * _REGIME_LEN],
    }
