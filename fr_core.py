"""
fr_core.py — Face detection, embedding, Milvus registry, and IoU tracker.

ONNX inference uses CUDA when onnxruntime-gpu is installed and a CUDA device is
available; falls back to CPU automatically otherwise.
YuNet (cv2.FaceDetectorYN) is always CPU — OpenCV's standard build has no CUDA
support for DNN, so the face-detection step runs on CPU regardless.
Milvus collection is site-scoped: every read/write filters by site_id.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)
_BASE = Path(__file__).parent
_DIM  = 512


# ── Model path resolver ────────────────────────────────────────────────────────

def resolve_model(path: str) -> str:
    p = Path(path)
    return str(p) if p.is_absolute() else str(_BASE / p)


# ── Auto-build / auto-download helpers ────────────────────────────────────────

# Direct download URLs for lightweight models
_ONNX_URLS: dict[str, str] = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
}

# InsightFace buffalo_l pack names (det_10g, w600k_r50)
_INSIGHTFACE_PACK = "buffalo_l"


def _ensure_onnx(resolved: Path) -> None:
    """
    Verify that ``resolved`` exists.  If not, attempt to build or download it.

    Strategy by filename:
      yolo*.onnx              → export from ultralytics
      yunet*.onnx             → download from OpenCV zoo (GitHub raw)
      det_10g / w600k / r50   → look in insightface local cache, else helpful error
      anything else           → raise FileNotFoundError with clear instructions
    """
    if resolved.is_file():
        return

    resolved.parent.mkdir(parents=True, exist_ok=True)
    name = resolved.name.lower()

    if "yolo" in name:
        _build_yolov8_onnx(resolved)

    elif resolved.name in _ONNX_URLS:
        _download_onnx(resolved, _ONNX_URLS[resolved.name])

    elif any(k in name for k in ("det_10g", "w600k", "r50")):
        _locate_insightface_model(resolved)

    else:
        raise FileNotFoundError(
            f"Model not found: {resolved}\n"
            "Place the ONNX file at the path above, "
            "or update the model path in config.json."
        )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Auto-setup for {resolved.name} completed but file still missing: {resolved}"
        )


def _build_yolov8_onnx(dst: Path) -> None:
    """Export YOLOv8n (or the matching variant) to ONNX via ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            f"YOLO ONNX model not found: {dst}\n"
            "Install ultralytics to auto-export it:\n"
            "  pip install ultralytics\n"
            f"Or copy the ONNX file manually to: {dst}"
        ) from None

    import shutil

    # Infer .pt model name from the requested filename
    # e.g. "yolov8n.onnx" → "yolov8n.pt", "yolo.onnx" → "yolov8n.pt"
    stem = dst.stem                           # e.g. "yolov8n" or "yolo"
    pt_name = stem if stem.startswith("yolov") else "yolov8n"

    log.info("building %s from %s.pt (one-time setup, may take a minute) ...",
             dst.name, pt_name)
    model       = YOLO(f"{pt_name}.pt")       # downloads .pt from ultralytics if absent
    export_path = Path(str(model.export(
        format="onnx", opset=12, simplify=True, imgsz=640,
    )))

    if not export_path.is_file():
        raise FileNotFoundError(
            f"ultralytics export returned {export_path!r} but file not found.\n"
            f"Copy {pt_name}.onnx manually to {dst}"
        )

    if export_path.resolve() != dst.resolve():
        shutil.copy2(str(export_path), str(dst))

    log.info("%s ready → %s", dst.name, dst)


def _download_onnx(dst: Path, url: str) -> None:
    """Download an ONNX model from a URL."""
    import urllib.request

    log.info("downloading %s ...", dst.name)
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as exc:
        dst.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {dst.name} from {url}: {exc}\n"
            f"Download it manually and copy to: {dst}"
        ) from exc
    log.info("%s downloaded → %s", dst.name, dst)


def _locate_insightface_model(dst: Path) -> None:
    """
    Copy an InsightFace model from the local insightface cache if present.
    Common cache locations: ~/.insightface/models/buffalo_l/
    """
    home = Path.home()
    search_dirs = [
        home / ".insightface" / "models" / _INSIGHTFACE_PACK,
        home / ".insightface" / "models" / "buffalo_s",
        home / ".insightface" / "models",
    ]

    for d in search_dirs:
        candidate = d / dst.name
        if candidate.is_file():
            import shutil
            shutil.copy2(str(candidate), str(dst))
            log.info("copied %s from insightface cache → %s", dst.name, dst)
            return

    raise FileNotFoundError(
        f"InsightFace model not found: {dst}\n"
        "Download it by running once:\n"
        "  pip install insightface\n"
        "  python -c \"import insightface; "
        f"insightface.app.FaceAnalysis(name='{_INSIGHTFACE_PACK}').prepare(ctx_id=0)\"\n"
        f"Then copy {dst.name} to: {dst.parent}"
    )


# ── ORT provider selection ────────────────────────────────────────────────────

def _ort_providers(device: str = "auto") -> list:
    """
    Return an ORT execution-provider list for the requested device.

    device="auto"  → CUDA if onnxruntime-gpu is installed, else CPU
    device="cuda"  → CUDA (raises at session creation if not available)
    device="cpu"   → always CPU
    """
    if device == "cpu":
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider",
        ]
    if device == "cuda":
        raise RuntimeError(
            "CUDAExecutionProvider not found in onnxruntime. "
            "Install onnxruntime-gpu:  pip install onnxruntime-gpu>=1.17"
        )
    return ["CPUExecutionProvider"]


def _provider_label(providers: list) -> str:
    first = providers[0]
    name  = first[0] if isinstance(first, tuple) else first
    return name.replace("ExecutionProvider", "")


# ── SCRFD face detector (CPU) ─────────────────────────────────────────────────

_REF_KPS = np.float32([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
])
_STRIDES  = [8, 16, 32]
_NUM_ANCH = 2


class FaceDetector:
    """SCRFD face detector. Runs on CUDA when onnxruntime-gpu is available."""

    def __init__(self, model_path: str, device: str = "auto"):
        resolved   = Path(resolve_model(model_path))
        _ensure_onnx(resolved)
        providers  = _ort_providers(device)
        self.sess  = ort.InferenceSession(str(resolved), providers=providers)
        self.iname = self.sess.get_inputs()[0].name
        log.info("FaceDetector (%s): %s", _provider_label(providers), resolved.name)

    def detect(self, bgr: np.ndarray, thresh: float = 0.5):
        """Returns (boxes [N,4], kps [N,5,2]) in original frame coords."""
        h, w   = bgr.shape[:2]
        img, scale = self._preprocess(bgr)
        outs   = self.sess.run(None, {self.iname: img})
        boxes, kps = self._decode(outs, scale, h, w, thresh)
        return boxes, kps

    def _preprocess(self, bgr):
        oh, ow = bgr.shape[:2]
        scale  = 640 / max(oh, ow)
        nh, nw = int(oh * scale), int(ow * scale)
        resized = cv2.resize(bgr, (nw, nh))
        canvas  = np.zeros((640, 640, 3), dtype=np.float32)
        canvas[:nh, :nw] = resized.astype(np.float32)
        canvas /= 255.0
        return canvas.transpose(2, 0, 1)[np.newaxis], scale

    def _decode(self, outs, scale, orig_h, orig_w, thresh):
        # det_10g.onnx output layout:
        #   outs[0..n-1]       → scores  for each stride  (N,)
        #   outs[n..2n-1]      → bboxes  for each stride  (N, 4)
        #   outs[2n..3n-1]     → kps     for each stride  (N, 10)
        # where N = (640/stride)^2 * num_anchors
        n      = len(_STRIDES)
        all_boxes, all_kps = [], []

        for i, stride in enumerate(_STRIDES):
            gh, gw = 640 // stride, 640 // stride

            scores = outs[i].reshape(-1)
            bboxes = outs[i + n].reshape(-1, 4)
            lmks   = outs[i + 2 * n].reshape(-1, 10)   # (N, 5*xy)

            keep = scores >= thresh
            if not keep.any():
                continue

            bboxes = bboxes[keep]
            lmks   = lmks[keep]

            # Anchor centres for the kept detections
            anch   = np.where(keep)[0]
            cy     = (anch // (_NUM_ANCH * gw)).astype(np.float32)
            cx     = ((anch // _NUM_ANCH) % gw).astype(np.float32)
            cx_f   = (cx + 0.5) * stride
            cy_f   = (cy + 0.5) * stride

            x1 = (cx_f - bboxes[:, 0] * stride) / scale
            y1 = (cy_f - bboxes[:, 1] * stride) / scale
            x2 = (cx_f + bboxes[:, 2] * stride) / scale
            y2 = (cy_f + bboxes[:, 3] * stride) / scale
            boxes = np.stack([x1, y1, x2, y2], axis=1)

            # lmks columns: x0,y0,x1,y1,...  → shape (N,5,2)
            lm_x = (cx_f[:, None] + lmks[:, 0::2] * stride) / scale  # (N,5)
            lm_y = (cy_f[:, None] + lmks[:, 1::2] * stride) / scale  # (N,5)
            pts  = np.stack([lm_x, lm_y], axis=2)                     # (N,5,2)

            all_boxes.append(boxes)
            all_kps.append(pts)

        if not all_boxes:
            return np.empty((0, 4), np.float32), np.empty((0, 5, 2), np.float32)

        boxes = np.vstack(all_boxes)
        kps   = np.vstack(all_kps)
        keep  = self._nms(boxes, iou_thresh=0.4)
        return boxes[keep], kps[keep]

    @staticmethod
    def _nms(boxes, iou_thresh=0.4):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]
        keep  = []
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]
        return keep


# ── YuNet face detector (CPU) ─────────────────────────────────────────────────

class YuNetDetector:
    """YuNet face detector — robust across all ages (toddlers to elderly). CPU-only.

    Uses cv2.FaceDetectorYN (OpenCV >= 4.5.4).
    Model: face_detection_yunet_2023mar.onnx (~373 KB) from opencv/opencv_zoo.

    Output keypoint order (same as SCRFD _REF_KPS):
      0: right-eye-from-camera (= person's left eye)
      1: left-eye-from-camera  (= person's right eye)
      2: nose tip
      3: right mouth corner (camera) = person's left
      4: left mouth corner  (camera) = person's right
    """

    _W, _H = 640, 640   # letterbox target (width, height)

    def __init__(self, model_path: str):
        resolved = Path(resolve_model(model_path))
        _ensure_onnx(resolved)
        self._dn = cv2.FaceDetectorYN.create(
            str(resolved), "",
            (self._W, self._H),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )
        log.info("YuNetDetector (CPU): %s", resolved.name)

    def detect(self, bgr: np.ndarray, thresh: float = 0.5):
        """Returns (boxes [N,4], kps [N,5,2]) in original frame coords."""
        oh, ow  = bgr.shape[:2]
        scale   = min(self._W / ow, self._H / oh)
        nw, nh  = int(ow * scale), int(oh * scale)
        pad_x   = (self._W - nw) // 2
        pad_y   = (self._H - nh) // 2

        canvas  = np.zeros((self._H, self._W, 3), dtype=np.uint8)
        canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = cv2.resize(bgr, (nw, nh))

        self._dn.setScoreThreshold(float(thresh))
        _, faces = self._dn.detect(canvas)

        if faces is None or len(faces) == 0:
            return np.empty((0, 4), np.float32), np.empty((0, 5, 2), np.float32)

        # faces: (N, 15) — [x, y, w, h, kp0_x, kp0_y, ..., kp4_x, kp4_y, score]
        x1 = (faces[:, 0]                - pad_x) / scale
        y1 = (faces[:, 1]                - pad_y) / scale
        x2 = (faces[:, 0] + faces[:, 2] - pad_x) / scale
        y2 = (faces[:, 1] + faces[:, 3] - pad_y) / scale
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        kp_raw = faces[:, 4:14].reshape(-1, 5, 2)           # (N, 5, 2)
        kp_x   = (kp_raw[:, :, 0] - pad_x) / scale
        kp_y   = (kp_raw[:, :, 1] - pad_y) / scale
        kps    = np.stack([kp_x, kp_y], axis=2).astype(np.float32)

        return boxes, kps


# ── YOLOv8 person detector (CPU) ─────────────────────────────────────────────

class PersonDetector:
    """YOLOv8n person detector — catches all ages via body shape.

    Model: yolov8n.onnx from ultralytics (COCO, class-0 = person).
    Input: 640×640 RGB float32 0-1.  Output: (1, 84, 8400).
    Runs on CUDA when onnxruntime-gpu is available.
    """

    _INPUT       = 640
    _PERSON_CLS  = 0

    def __init__(self, model_path: str, device: str = "auto"):
        resolved   = Path(resolve_model(model_path))
        _ensure_onnx(resolved)
        providers  = _ort_providers(device)
        self.sess  = ort.InferenceSession(str(resolved), providers=providers)
        self.iname = self.sess.get_inputs()[0].name
        log.info("PersonDetector (%s): %s", _provider_label(providers), resolved.name)

    def detect(self, bgr: np.ndarray, conf: float = 0.4) -> np.ndarray:
        """Returns boxes [N,4] (x1,y1,x2,y2) in original frame coords."""
        oh, ow  = bgr.shape[:2]
        scale   = self._INPUT / max(oh, ow)
        nw, nh  = int(ow * scale), int(oh * scale)
        pad_x   = (self._INPUT - nw) // 2
        pad_y   = (self._INPUT - nh) // 2

        canvas  = np.full((self._INPUT, self._INPUT, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = cv2.resize(bgr, (nw, nh))
        inp     = (canvas[:, :, ::-1].astype(np.float32) / 255.0
                   ).transpose(2, 0, 1)[np.newaxis]

        outs    = self.sess.run(None, {self.iname: inp})
        pred    = outs[0].squeeze(0).T                    # (8400, 84)

        scores  = pred[:, 4 + self._PERSON_CLS]
        keep    = scores >= conf
        if not keep.any():
            return np.empty((0, 4), np.float32)

        pred    = pred[keep]
        scores  = scores[keep]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

        x1 = np.clip((cx - w / 2 - pad_x) / scale, 0, ow)
        y1 = np.clip((cy - h / 2 - pad_y) / scale, 0, oh)
        x2 = np.clip((cx + w / 2 - pad_x) / scale, 0, ow)
        y2 = np.clip((cy + h / 2 - pad_y) / scale, 0, oh)
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        keep_nms = self._nms(boxes, scores, iou_thresh=0.45)
        return boxes[keep_nms]

    @staticmethod
    def _nms(boxes, scores, iou_thresh=0.45):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep  = []
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]
        return keep


# ── Face preprocessor ─────────────────────────────────────────────────────────

def preprocess_face(bgr_112: np.ndarray) -> np.ndarray:
    """Suppress background in aligned 112×112 face crop using a soft ellipse mask.

    Fills background with neutral gray so ArcFace ignores it during embedding.
    """
    h, w  = bgr_112.shape[:2]
    mask  = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.47), int(h * 0.47)),
                0, 0, 360, 1.0, -1)
    mask  = cv2.GaussianBlur(mask, (15, 15), 7)
    face  = bgr_112.astype(np.float32)
    out   = face * mask[:, :, None] + 127.5 * (1.0 - mask[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Detector factory ──────────────────────────────────────────────────────────

def create_detector(cfg: dict):
    """Return YuNetDetector or FaceDetector based on cfg['detector_type'].

    config.json keys:
        detector_type  : "yunet"  (recommended) or "scrfd" (default)
        yunet_model    : path to face_detection_yunet_2023mar.onnx
        detector_model : path to det_10g.onnx  (used when type == scrfd)
        device         : "auto" | "cpu" | "cuda"  (only affects SCRFD, not YuNet)
    """
    device = cfg.get("device", "auto")
    dtype  = cfg.get("detector_type", "scrfd").lower()
    if dtype == "yunet":
        model = cfg.get("yunet_model") or cfg.get("detector_model", "")
        return YuNetDetector(model)        # always CPU — cv2.FaceDetectorYN has no CUDA
    return FaceDetector(cfg["detector_model"], device=device)


# ── Face alignment ────────────────────────────────────────────────────────────

def align_face(bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Procrustes-align face keypoints to 112×112 canonical crop."""
    src = kps.astype(np.float32)
    dst = _REF_KPS

    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(cov)
    R = (Vt.T @ U.T)

    scale_s = np.sum(dst_c * (src_c @ R)) / (np.sum(src_c ** 2) + 1e-6)
    t       = dst_mean - scale_s * (src_mean @ R)
    M       = np.hstack([scale_s * R, t[:, np.newaxis]])

    return cv2.warpAffine(bgr, cv2.invertAffineTransform(M), (112, 112))


# ── ArcFace embedder (CPU) ────────────────────────────────────────────────────

class FaceEmbedder:
    """ArcFace R50 — returns 512-d unit-norm vector.

    Runs on CUDA when onnxruntime-gpu is available.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        resolved   = Path(resolve_model(model_path))
        _ensure_onnx(resolved)
        providers  = _ort_providers(device)
        self.sess  = ort.InferenceSession(str(resolved), providers=providers)
        self.iname = self.sess.get_inputs()[0].name
        log.info("FaceEmbedder (%s): %s", _provider_label(providers), resolved.name)

    def embed(self, bgr_112: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        inp = rgb.transpose(2, 0, 1)[np.newaxis]
        out = self.sess.run(None, {self.iname: inp})[0][0]
        n   = np.linalg.norm(out)
        return out / n if n > 0 else out


# ── Milvus face registry ──────────────────────────────────────────────────────

class FaceDB:
    """Milvus-backed face registry, site-id scoped."""

    def __init__(self, host: str, port: int, collection: str = "face_registry"):
        from pymilvus import (  # lazy import — only needed when FaceDB is actually instantiated
            Collection, CollectionSchema, FieldSchema, DataType, connections, utility,
        )
        fields = [
            FieldSchema("id",          DataType.VARCHAR,      max_length=64,   is_primary=True, auto_id=False),
            FieldSchema("site_id",     DataType.VARCHAR,      max_length=128),
            FieldSchema("person_name", DataType.VARCHAR,      max_length=256),
            FieldSchema("embedding",   DataType.FLOAT_VECTOR, dim=_DIM),
            FieldSchema("notes",       DataType.VARCHAR,      max_length=512),
        ]
        connections.connect(alias="default", host=host, port=int(port))
        if not utility.has_collection(collection):
            schema = CollectionSchema(fields, description="Face registry by site")
            col = Collection(collection, schema)
            col.create_index("embedding", {
                "metric_type": "COSINE",
                "index_type":  "FLAT",
                "params":      {},
            })
            log.info("Created Milvus collection '%s'", collection)
        self.col = Collection(collection)
        self.col.load()
        log.info("FaceDB ready: %s:%s / %s", host, port, collection)

    # ── write ──────────────────────────────────────────────────────────────────

    def add(self, site_id: str, person_name: str, embedding: np.ndarray,
            notes: str = "", uid: str | None = None) -> str:
        """Store one embedding. Returns the UUID used (pass uid to keep a known ID)."""
        uid = uid or str(uuid.uuid4())
        self.col.insert([[uid], [site_id], [person_name], [embedding.tolist()], [notes]])
        self.col.flush()
        return uid

    def delete_person(self, site_id: str, person_name: str) -> int:
        """Remove all embeddings for a person at a site. Returns count deleted."""
        rows = self._query_all(f'site_id == "{site_id}" and person_name == "{person_name}"',
                               ["id"])
        if not rows:
            return 0
        ids_expr = "[" + ", ".join(f'"{r["id"]}"' for r in rows) + "]"
        self.col.delete(f"id in {ids_expr}")
        self.col.flush()
        return len(rows)

    # ── read ───────────────────────────────────────────────────────────────────

    def search(self, site_id: str, embedding: np.ndarray, threshold: float = 0.45
               ) -> Optional[dict]:
        """Best match within site. Returns {name, score, id} or None."""
        try:
            res = self.col.search(
                data=[embedding.tolist()],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=1,
                output_fields=["id", "person_name"],
                expr=f'site_id == "{site_id}"',
            )
        except Exception as e:
            log.debug("Milvus search error: %s", e)
            return None
        if not res or not res[0]:
            return None
        hit  = res[0][0]
        sc   = float(hit.score)
        if sc < threshold:
            return None
        return {
            "name":  hit.entity.get("person_name", ""),
            "score": sc,
            "id":    hit.entity.get("id", ""),
        }

    def list_people(self, site_id: str) -> list[dict]:
        """List all people at a site with embedding count."""
        rows = self._query_all(f'site_id == "{site_id}"', ["id", "person_name", "notes"])
        by_name: dict[str, dict] = {}
        for r in rows:
            n = r["person_name"]
            if n not in by_name:
                by_name[n] = {"name": n, "embeddings": 0, "notes": r.get("notes", "")}
            by_name[n]["embeddings"] += 1
        return sorted(by_name.values(), key=lambda x: x["name"])

    def count(self, site_id: str) -> int:
        return len(self._query_all(f'site_id == "{site_id}"', ["id"]))

    def _query_all(self, expr: str, fields: list) -> list:
        rows, offset = [], 0
        while True:
            page = self.col.query(expr=expr, output_fields=fields,
                                  offset=offset, limit=16_000)
            if not page:
                break
            rows.extend(page)
            if len(page) < 16_000:
                break
            offset += 16_000
        return rows


# ── IoU tracker ───────────────────────────────────────────────────────────────

class _Track:
    def __init__(self, tid: int, box, name: str, score: float, history_len: int):
        self.tid    = tid
        self.box    = np.array(box, dtype=np.float32)
        self.score  = score
        self.age    = 0
        self.hist   = [name] * history_len
        self.h_ptr  = 0
        self.h_len  = history_len

    def update(self, box, name: str, score: float):
        self.box             = np.array(box, dtype=np.float32)
        self.hist[self.h_ptr % self.h_len] = name
        self.h_ptr          += 1
        self.score           = score
        self.age             = 0

    @property
    def name(self) -> str:
        named = [n for n in self.hist if n]
        return max(set(named), key=named.count) if named else ""


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua    = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class IoUTracker:
    """Simple IoU-based multi-face tracker with majority-vote name smoothing."""

    def __init__(self, iou_thresh: float = 0.4, max_age: int = 30, history_len: int = 7):
        self.iou_thresh  = iou_thresh
        self.max_age     = max_age
        self.history_len = history_len
        self._tracks: list[_Track] = []
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Args:
            detections: [{"box": [x1,y1,x2,y2], "name": str, "score": float}]
        Returns:
            [{"box", "name", "score", "tid"}] — one entry per currently-visible track
        """
        # Age all tracks; they get reset if matched
        for t in self._tracks:
            t.age += 1

        if detections:
            matched_trk: set[int] = set()
            matched_det: set[int] = set()

            for di, det in enumerate(detections):
                best_iou, best_ti = 0.0, -1
                for ti, trk in enumerate(self._tracks):
                    if ti in matched_trk:
                        continue
                    v = _iou(det["box"], trk.box)
                    if v > best_iou:
                        best_iou, best_ti = v, ti
                if best_iou >= self.iou_thresh:
                    self._tracks[best_ti].update(det["box"], det["name"], det["score"])
                    matched_trk.add(best_ti)
                    matched_det.add(di)

            for di, det in enumerate(detections):
                if di not in matched_det:
                    self._tracks.append(
                        _Track(self._next_id, det["box"], det["name"], det["score"], self.history_len)
                    )
                    self._next_id += 1

        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

        return [
            {"box": t.box, "name": t.name, "score": t.score, "tid": t.tid}
            for t in self._tracks if t.age == 0
        ]
