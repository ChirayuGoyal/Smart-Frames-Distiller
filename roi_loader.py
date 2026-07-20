"""Load a LabelMe-format ROI JSON and produce a binary mask for frame masking.

The mask is applied to frames *before* inference so the detector only "sees"
pixels inside the ROI.  Output videos are always written with the original
full frames — the ROI only affects which frames are selected, not what is saved.

Supported LabelMe shape types:
  polygon   — arbitrary convex/concave regions
  rectangle — two-corner bounding box
  circle    — centre + edge point

Multiple shapes are OR-combined (union).  Unknown shape types are skipped.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


class ROIConfig(NamedTuple):
    mask:   np.ndarray          # uint8, shape (h, w), 255 = inside ROI
    bbox:   tuple[int, int, int, int]   # (x, y, w, h) bounding rect of all shapes
    source: str                 # path to JSON, for logging


def load_roi(
    json_path: str | Path,
    frame_width: int,
    frame_height: int,
) -> ROIConfig:
    """Parse a LabelMe JSON and return a binary mask at (frame_height, frame_width).

    The annotation may have been made on a different resolution image; the mask
    is scaled to match the actual frame dimensions using nearest-neighbour
    interpolation so the polygon edges don't soften.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    ann_w = int(data.get("imageWidth",  frame_width))
    ann_h = int(data.get("imageHeight", frame_height))

    mask = np.zeros((ann_h, ann_w), dtype=np.uint8)

    n_drawn = 0
    for shape in data.get("shapes", []):
        pts   = np.array(shape["points"], dtype=np.float32)
        stype = shape.get("shape_type", "polygon")

        if stype == "polygon":
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
                n_drawn += 1

        elif stype == "rectangle":
            x1 = int(min(pts[:, 0]));  y1 = int(min(pts[:, 1]))
            x2 = int(max(pts[:, 0]));  y2 = int(max(pts[:, 1]))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            n_drawn += 1

        elif stype == "circle":
            # LabelMe stores [[cx, cy], [rx, ry]] where (rx,ry) is a point on the rim
            if len(pts) >= 2:
                cx, cy = int(pts[0][0]), int(pts[0][1])
                radius = int(np.linalg.norm(pts[1].astype(float) - pts[0].astype(float)))
                cv2.circle(mask, (cx, cy), radius, 255, -1)
                n_drawn += 1

        else:
            log.debug("roi_loader: skipping unsupported shape type '%s'", stype)

    if n_drawn == 0:
        log.warning(
            "roi_loader: no drawable shapes found in %s — using full frame", json_path
        )
        mask[:] = 255

    # Scale to actual frame dimensions if annotation resolution differs
    if (ann_w, ann_h) != (frame_width, frame_height):
        mask = cv2.resize(
            mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST
        )
        log.debug(
            "roi_loader: rescaled mask %dx%d → %dx%d",
            ann_w, ann_h, frame_width, frame_height,
        )

    # Bounding rect of the union of all shapes (used for logging)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        bbox = tuple(cv2.boundingRect(coords))   # (x, y, w, h)
    else:
        bbox = (0, 0, frame_width, frame_height)

    log.info(
        "ROI loaded: %s | shapes=%d | bbox=(%d,%d,%d,%d) | mask=%dx%d",
        Path(json_path).name, n_drawn, *bbox, frame_width, frame_height,
    )
    return ROIConfig(mask=mask, bbox=bbox, source=str(json_path))


def apply_roi_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out all pixels outside the ROI mask.  Returns a new array."""
    h, w = frame.shape[:2]
    m = mask
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(frame, frame, mask=m)
