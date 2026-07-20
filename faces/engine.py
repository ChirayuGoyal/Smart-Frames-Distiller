"""
faces/engine.py — Unified face detection and embedding engine.

Consolidates SCRFD face detection and ArcFace embedding from Stack B
(face_recognizer.py) with execution-provider resolution and auto-download
helpers from fr_core.py.

ONNX sessions are cached (keyed on path + provider + device_id) so that
multiple consumers sharing the same model file reuse a single session.
"""
from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

# Repo root — engine.py lives in  <repo>/faces/engine.py
_BASE_DIR = Path(__file__).resolve().parent.parent


# ── Model-path resolution ─────────────────────────────────────────────────────

def resolve_model_path(path: str) -> str:
    """Resolve *path*: absolute paths pass through, relative paths resolve
    from the repository root."""
    p = Path(path)
    return str(p) if p.is_absolute() else str(_BASE_DIR / p)


# ── Execution-provider resolution ─────────────────────────────────────────────

def resolve_execution_provider(
    device: str = "auto",
    device_id: int = 0,
) -> tuple[list, int]:
    """Map ``auto | cuda | cpu`` to an ORT execution-providers list and
    device_id.

    Returns
    -------
    (providers_list, device_id)
        *providers_list* is ready for ``ort.InferenceSession(..., providers=)``.
        CUDA entries are ``('CUDAExecutionProvider', {'device_id': N})``.
    """
    device = device.lower().strip()

    if device == "cpu":
        return (["CPUExecutionProvider"], device_id)

    available = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in available

    if device == "cuda" and not has_cuda:
        raise RuntimeError(
            "CUDAExecutionProvider not found in onnxruntime. "
            "Install onnxruntime-gpu:  pip install onnxruntime-gpu>=1.17"
        )

    if has_cuda:
        return (
            [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ],
            device_id,
        )

    # auto fallback → CPU
    return (["CPUExecutionProvider"], device_id)


# ── Auto-build / auto-download helpers ────────────────────────────────────────

_ONNX_URLS: dict[str, str] = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
}

_INSIGHTFACE_PACK = "buffalo_l"


def _ensure_onnx(resolved: Path) -> None:
    """Verify that *resolved* exists.  If not, attempt to build or download it.

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

    stem = dst.stem
    pt_name = stem if stem.startswith("yolov") else "yolov8n"

    log.info(
        "building %s from %s.pt (one-time setup, may take a minute) ...",
        dst.name, pt_name,
    )
    model = YOLO(f"{pt_name}.pt")
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
        with urllib.request.urlopen(url, timeout=60) as resp, open(dst, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:
        dst.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {dst.name} from {url}: {exc}\n"
            f"Download it manually and copy to: {dst}"
        ) from exc
    log.info("%s downloaded → %s", dst.name, dst)


def _locate_insightface_model(dst: Path) -> None:
    """Copy an InsightFace model from the local insightface cache if present.

    Common cache locations: ``~/.insightface/models/buffalo_l/``
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


# ── ONNX session cache ────────────────────────────────────────────────────────

_session_cache: dict[tuple, ort.InferenceSession] = {}
_session_lock = threading.Lock()


def _get_session(
    model_path: str,
    providers: list,
    device_id: int,
    *,
    intra_threads: int = 0,
) -> ort.InferenceSession:
    """Return a cached ``InferenceSession``, creating one if absent.

    Parameters
    ----------
    model_path : str
        Absolute or resolved path to the ONNX model.
    providers : list
        Provider list (may contain tuples for CUDA options).
    device_id : int
        GPU device index (used as part of the cache key).
    intra_threads : int
        ``SessionOptions.intra_op_num_threads``.  0 = ORT default.
    """
    # Build a hashable key from the provider list.
    def _hashable(prov):
        if isinstance(prov, tuple):
            name, opts = prov
            return (name, tuple(sorted(opts.items())))
        return prov

    key = (model_path, tuple(_hashable(p) for p in providers), device_id)

    with _session_lock:
        sess = _session_cache.get(key)
        if sess is not None:
            return sess

        opts = ort.SessionOptions()
        if intra_threads > 0:
            opts.intra_op_num_threads = intra_threads

        sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
        )
        _session_cache[key] = sess
        return sess


# ── SCRFD face detector ───────────────────────────────────────────────────────

def _dist2box(centers: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return np.stack([
        centers[:, 0] - dist[:, 0],
        centers[:, 1] - dist[:, 1],
        centers[:, 0] + dist[:, 2],
        centers[:, 1] + dist[:, 3],
    ], axis=-1)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.4) -> list[int]:
    """Greedy non-maximum suppression; returns indices of boxes to keep."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def _dist2kps(centers: np.ndarray, dist: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, dist.shape[1], 2):
        preds.append(centers[:, 0] + dist[:, i])
        preds.append(centers[:, 1] + dist[:, i + 1])
    return np.stack(preds, axis=-1).reshape(-1, 5, 2)


class SCRFD:
    """SCRFD face detector using an ONNX model.

    Input is resized to 640×640; strides (8, 16, 32) with 2 anchors per
    location.  Normalisation: ``(blob - 127.5) / 128.0``.
    """

    _STRIDES = (8, 16, 32)
    _INPUT_HW = (640, 640)

    def __init__(self, model_path: str, provider: str, device_id: int) -> None:
        resolved = Path(resolve_model_path(model_path))
        _ensure_onnx(resolved)

        providers, device_id = _build_providers(provider, device_id)
        self._sess = _get_session(str(resolved), providers, device_id)
        self._in_name = self._sess.get_inputs()[0].name
        self._out_names = [o.name for o in self._sess.get_outputs()]
        log.info("SCRFD: %s", self._sess.get_providers()[0])

    def detect(self, bgr: np.ndarray, thresh: float = 0.5):
        """Return ``(boxes, kps)`` in original-frame pixel coordinates.

        boxes : float32 (N, 4)  ``[x1, y1, x2, y2]``
        kps   : float32 (N, 5, 2)
        """
        h, w = bgr.shape[:2]
        ih, iw = self._INPUT_HW

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        blob = cv2.resize(rgb, (iw, ih)).astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None]

        outs = self._sess.run(self._out_names, {self._in_name: blob})

        boxes_acc, kps_acc, scores_acc = [], [], []
        for i, stride in enumerate(self._STRIDES):
            scores = outs[i].reshape(-1)
            bbox = outs[i + 3].reshape(-1, 4) * stride
            kps = outs[i + 6].reshape(-1, 10) * stride

            fm_h, fm_w = ih // stride, iw // stride
            yy, xx = np.mgrid[:fm_h, :fm_w]
            centers = (
                np.stack([xx, yy], axis=-1).astype(np.float32) * stride
            ).reshape(-1, 2)
            centers = np.repeat(centers, 2, axis=0)  # 2 anchors per location

            inds = scores > thresh
            if not inds.any():
                continue
            boxes_acc.append(_dist2box(centers, bbox)[inds])
            kps_acc.append(_dist2kps(centers, kps)[inds])
            scores_acc.append(scores[inds])

        if not boxes_acc:
            return np.empty((0, 4), np.float32), np.empty((0, 5, 2), np.float32)

        boxes = np.concatenate(boxes_acc)
        kps = np.concatenate(kps_acc)
        scores = np.concatenate(scores_acc)

        # Suppress duplicate detections of the same face across strides/anchors
        # — without this, every face spawns several overlapping tracks.
        keep = _nms(boxes, scores, iou_thresh=0.4)
        boxes, kps = boxes[keep], kps[keep]

        sx, sy = w / iw, h / ih
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        kps[:, :, 0] *= sx
        kps[:, :, 1] *= sy
        return boxes, kps


# ── ArcFace R50 embedding ─────────────────────────────────────────────────────

_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Procrustes alignment to 112×112 canonical face crop."""
    src = kps.astype(np.float32)
    dst = _TEMPLATE
    sm, dm = src.mean(0), dst.mean(0)
    sc, dc = src - sm, dst - dm

    U, S, Vt = np.linalg.svd(sc.T @ dc)
    R = (U @ Vt).T
    scale = np.trace(np.diag(S)) / np.sum(sc ** 2)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:, 2] = dm - scale * R @ sm

    # M maps face→canonical, which is exactly what warpAffine expects
    # (it inverts internally when sampling).
    return cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_LINEAR)


class ArcFace:
    """ArcFace R50 face embedder.

    Input: 112×112 BGR crop.  Output: unit-norm 512-d embedding.
    Normalisation: ``(rgb - 127.5) / 128.0``.
    """

    def __init__(self, model_path: str, provider: str, device_id: int) -> None:
        resolved = Path(resolve_model_path(model_path))
        _ensure_onnx(resolved)

        providers, device_id = _build_providers(provider, device_id)
        self._sess = _get_session(str(resolved), providers, device_id)
        self._in_name = self._sess.get_inputs()[0].name
        log.info("ArcFace: %s", self._sess.get_providers()[0])

    def embed(self, bgr_112: np.ndarray) -> np.ndarray:
        """Return unit-norm 512-d embedding from a 112×112 BGR crop."""
        rgb = cv2.cvtColor(bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None]
        emb = self._sess.run(None, {self._in_name: blob})[0].flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_providers(provider: str, device_id: int) -> tuple[list, int]:
    """Build a providers list from a single provider string + device_id.

    This is used by ``SCRFD.__init__`` and ``ArcFace.__init__`` to keep the
    same ``(model_path, provider, device_id)`` constructor signature as the
    original ``face_recognizer.py`` while integrating with the session cache.
    """
    normalized = provider.lower().strip()
    if normalized in {"auto", "cpu", "cuda"}:
        return resolve_execution_provider(normalized, device_id)
    if "CUDA" in provider:
        return (
            [
                (provider, {"device_id": device_id}),
                "CPUExecutionProvider",
            ],
            device_id,
        )
    return ([provider, "CPUExecutionProvider"], device_id)
