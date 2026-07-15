"""Frame preprocessing for faster CPU inference."""

from __future__ import annotations

import cv2
import numpy as np


def resize_for_inference(
    frame: np.ndarray,
    scale: float = 1.0,
    max_side: int | None = None,
) -> np.ndarray:
    """Downscale frame before model inference (INTER_AREA for decimation)."""
    h, w = frame.shape[:2]
    if scale != 1.0:
        w = max(1, int(w * scale))
        h = max(1, int(h * scale))
    if max_side is not None and max(w, h) > max_side:
        ratio = max_side / max(w, h)
        w = max(1, int(w * ratio))
        h = max(1, int(h * ratio))
    if w == frame.shape[1] and h == frame.shape[0]:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
