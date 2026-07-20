"""
server/model_cache.py — Singletons and thread-safe caches for models and infrastructure clients.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from action_model import ActionModelBackend, create_action_model
from faces.annotate import FaceRecognizer
from faces.store import FaceStore, InMemoryFaceStore, MilvusFaceStore
from kafka_producer import KafkaPublisher
from server.settings import Settings

log = logging.getLogger(__name__)

_gpu_semaphore: threading.BoundedSemaphore | None = None
_gpu_lock = threading.Lock()

# Caches are keyed by the parameters that change model behaviour — a second
# job with a different clip_len/device/site must not silently reuse the first
# job's model or recognizer.
_action_models: dict[tuple, ActionModelBackend] = {}
_action_lock = threading.Lock()

_face_store: FaceStore | None = None
_memory_store: FaceStore | None = None
_store_lock = threading.Lock()

_face_recognizers: dict[tuple, FaceRecognizer] = {}
_face_lock = threading.Lock()

_kafka_publisher: KafkaPublisher | None = None
_kafka_lock = threading.Lock()


def resolve_face_config(
    settings: Settings,
    overrides: dict[str, Any] | None = None,
    *,
    site_id: str | None = None,
    camera_id: str | None = None,
) -> dict[str, Any]:
    """Build a complete face-service config from settings and request overrides."""
    supplied = dict(overrides or {})
    supplied_milvus = supplied.pop("milvus", {})
    if not isinstance(supplied_milvus, dict):
        supplied_milvus = {}

    resolved: dict[str, Any] = {
        "device": settings.models.device,
        "person_model": settings.models.person_model,
        "detector_model": settings.models.detector_model,
        "embedding_model": settings.models.embedding_model,
        "milvus": {
            "host": settings.milvus.host,
            "port": settings.milvus.port,
            "collection": settings.milvus.collection,
        },
    }
    resolved.update(supplied)
    resolved["milvus"] = {**resolved["milvus"], **supplied_milvus}
    if site_id:
        resolved["site_id"] = site_id
    if camera_id is not None:
        resolved["camera_id"] = camera_id
    return resolved


def get_gpu_semaphore(settings: Settings) -> threading.BoundedSemaphore:
    global _gpu_semaphore
    if _gpu_semaphore is None:
        with _gpu_lock:
            if _gpu_semaphore is None:
                slots = max(1, settings.gpu_slots)
                _gpu_semaphore = threading.BoundedSemaphore(slots)
    return _gpu_semaphore


def get_action_model(settings: Settings, cfg: dict[str, Any] | None = None) -> ActionModelBackend:
    cfg = cfg or {}
    clip_len = int(cfg.get("clip_len", 16))
    prefer_torch = bool(cfg.get("prefer_torch", True))
    device = str(cfg.get("device", settings.models.device))
    ensemble = bool(cfg.get("ensemble", False))
    mpath = cfg.get("model_path") or settings.models.r3d_model_path or None
    cdir = cfg.get("model_cache_dir") or settings.models.r3d_cache_dir or None

    key = (clip_len, prefer_torch, device, ensemble, mpath, cdir)
    with _action_lock:
        model = _action_models.get(key)
        if model is None:
            model = create_action_model(
                clip_len=clip_len,
                prefer_torch=prefer_torch,
                device=device,
                ensemble=ensemble,
                model_path=mpath,
                cache_dir=cdir,
            )
            _action_models[key] = model
    return model


def get_face_store(settings: Settings, *, use_memory: bool = False) -> FaceStore:
    global _face_store, _memory_store
    with _store_lock:
        if use_memory:
            # Kept separate from the Milvus store so tests requesting memory
            # always get one, regardless of initialization order.
            if _memory_store is None:
                _memory_store = InMemoryFaceStore()
                _memory_store.connect()
            return _memory_store
        if _face_store is None:
            _face_store = MilvusFaceStore(
                host=settings.milvus.host,
                port=settings.milvus.port,
                collection=settings.milvus.collection,
            )
            try:
                _face_store.connect()
            except Exception as exc:
                log.warning("Initial connection to FaceStore failed: %s", exc)
        return _face_store


def get_face_recognizer(settings: Settings, cfg: dict[str, Any] | None = None, *, store: FaceStore | None = None) -> FaceRecognizer:
    resolved = resolve_face_config(settings, cfg)
    key = (
        resolved.get("site_id", ""),
        resolved.get("camera_id", ""),
        str(resolved.get("device", "")),
        str(resolved.get("execution_provider", "")),
        str(resolved.get("detector_model", "")),
        str(resolved.get("embedding_model", "")),
        str(resolved.get("similarity_thresh") or resolved.get("similarity_threshold") or ""),
        id(store) if store is not None else None,
    )
    with _face_lock:
        rec = _face_recognizers.get(key)
        if rec is None:
            st = store or get_face_store(settings)
            rec = FaceRecognizer(resolved, store=st)
            _face_recognizers[key] = rec
    return rec


def get_kafka_publisher(settings: Settings) -> KafkaPublisher:
    global _kafka_publisher
    if _kafka_publisher is None:
        with _kafka_lock:
            if _kafka_publisher is None:
                overrides = settings.kafka_overrides()
                _kafka_publisher = KafkaPublisher(overrides=overrides, spool_path=settings.kafka_spool_path)
    return _kafka_publisher


def close_all() -> None:
    global _face_store, _memory_store, _kafka_publisher
    with _store_lock:
        for st in (_face_store, _memory_store):
            if st is not None:
                try:
                    st.close()
                except Exception as exc:
                    log.debug("Error closing face store: %s", exc)
        _face_store = None
        _memory_store = None

    with _kafka_lock:
        if _kafka_publisher is not None:
            try:
                _kafka_publisher.close()
            except Exception as exc:
                log.debug("Error closing kafka publisher: %s", exc)
            _kafka_publisher = None

    with _action_lock:
        _action_models.clear()

    with _face_lock:
        _face_recognizers.clear()
