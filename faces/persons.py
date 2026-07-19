"""
Person detector based on YOLOv8n (COCO class-0 = person).

Catches all ages via body shape — complementary to face detection which may
miss small children or people facing away from the camera.

Model: yolov8n.onnx from Ultralytics (exported with ``opset=12, imgsz=640``).
Input : 640×640 RGB float32 [0, 1].
Output: ``(1, 84, 8400)`` — 8400 proposals × (4 box + 80 class scores).
Runs on CUDA when ``onnxruntime-gpu`` is installed and a GPU is available.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from faces.engine import (
    _ensure_onnx,
    resolve_execution_provider,
    resolve_model_path,
)

log = logging.getLogger(__name__)


def _provider_label(providers: list) -> str:
    """Human-readable label for the first execution provider."""
    first = providers[0]
    name = first[0] if isinstance(first, tuple) else first
    return name.replace("ExecutionProvider", "")


class PersonDetector:
    """YOLOv8n person detector — catches all ages via body shape.

    Parameters
    ----------
    model_path:
        Path to the ONNX model file (absolute or relative to the project root).
    device:
        ``"auto"`` (default) to prefer CUDA when available, ``"cuda"`` to
        require it, or ``"cpu"`` for CPU-only inference.
    """

    _INPUT: int = 640
    _PERSON_CLS: int = 0

    def __init__(self, model_path: str, device: str = "auto") -> None:
        resolved = Path(resolve_model_path(model_path))
        _ensure_onnx(resolved)
        providers, _ = resolve_execution_provider(device)
        self.sess = ort.InferenceSession(str(resolved), providers=providers)
        self.iname: str = self.sess.get_inputs()[0].name
        log.info(
            "PersonDetector (%s): %s", _provider_label(providers), resolved.name,
        )

    def detect(self, bgr: np.ndarray, conf: float = 0.4) -> np.ndarray:
        """Run person detection on a BGR frame.

        Parameters
        ----------
        bgr:
            Input image in OpenCV BGR format, shape ``(H, W, 3)``.
        conf:
            Minimum confidence threshold for person detections.

        Returns
        -------
        np.ndarray
            Detected bounding boxes as ``[N, 4]`` float32 array with columns
            ``(x1, y1, x2, y2)`` in the original frame coordinates.
        """
        oh, ow = bgr.shape[:2]
        scale = self._INPUT / max(oh, ow)
        nw, nh = int(ow * scale), int(oh * scale)
        pad_x = (self._INPUT - nw) // 2
        pad_y = (self._INPUT - nh) // 2

        # Letterbox to 640×640 with grey padding (114)
        canvas = np.full((self._INPUT, self._INPUT, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = cv2.resize(bgr, (nw, nh))
        inp = (
            canvas[:, :, ::-1].astype(np.float32) / 255.0
        ).transpose(2, 0, 1)[np.newaxis]

        outs = self.sess.run(None, {self.iname: inp})
        pred = outs[0].squeeze(0).T  # (8400, 84)

        scores = pred[:, 4 + self._PERSON_CLS]
        keep = scores >= conf
        if not keep.any():
            return np.empty((0, 4), np.float32)

        pred = pred[keep]
        scores = scores[keep]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

        x1 = np.clip((cx - w / 2 - pad_x) / scale, 0, ow)
        y1 = np.clip((cy - h / 2 - pad_y) / scale, 0, oh)
        x2 = np.clip((cx + w / 2 - pad_x) / scale, 0, ow)
        y2 = np.clip((cy + h / 2 - pad_y) / scale, 0, oh)
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        keep_nms = self._nms(boxes, scores, iou_thresh=0.45)
        return boxes[keep_nms]

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float = 0.45,
    ) -> list[int]:
        """Greedy non-maximum suppression.

        Parameters
        ----------
        boxes:
            ``[N, 4]`` array of ``(x1, y1, x2, y2)`` boxes.
        scores:
            ``[N]`` array of confidence scores.
        iou_thresh:
            IoU threshold above which overlapping boxes are suppressed.

        Returns
        -------
        list[int]
            Indices of the boxes to keep.
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]
        return keep
