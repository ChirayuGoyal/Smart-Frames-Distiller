"""
server/routers/health.py — Liveness and readiness endpoints probing dependencies.
"""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Request
from server.schemas import HealthResponse

router = APIRouter(prefix="/v1/health", tags=["health"])


@router.get("", response_model=dict)
async def health_liveness() -> dict[str, str]:
    """Basic liveness probe."""
    return {"status": "ok"}


@router.get("/ready", response_model=HealthResponse)
async def health_readiness(request: Request) -> HealthResponse:
    """Readiness probe checking Milvus, Kafka, model files, and GPU availability."""
    settings = request.app.state.settings
    deps: dict[str, str] = {}
    status = "ok"

    # Check model files (resolve relative paths against the repo root, the
    # same way the engine does — a cwd-relative check gives false negatives)
    from faces.engine import resolve_model_path
    models_ok = True
    for m in (settings.models.detector_model, settings.models.embedding_model, settings.models.person_model):
        if not Path(resolve_model_path(m)).exists():
            models_ok = False
            break
    deps["models"] = "ok" if models_ok else "missing"
    if not models_ok:
        status = "degraded"

    # Check GPU
    try:
        import torch
        deps["gpu"] = "ok" if torch.cuda.is_available() else "unavailable"
    except ImportError:
        deps["gpu"] = "unavailable"

    # Check Milvus (via face store connection check if initialized or quick probe)
    try:
        from server.model_cache import get_face_store
        store = get_face_store(settings)
        store.connect()
        deps["milvus"] = "ok"
    except Exception as exc:
        deps["milvus"] = f"error: {exc}"
        status = "degraded"

    # Check Kafka
    if settings.kafka.enabled:
        try:
            from server.model_cache import get_kafka_publisher
            pub = get_kafka_publisher(settings)
            ok = pub.connect(wait=False)
            deps["kafka"] = "ok" if ok else (pub.last_error or "unavailable")
            if not ok and settings.kafka.required:
                status = "error"
            elif not ok:
                if status == "ok":
                    status = "degraded"
        except Exception as exc:
            deps["kafka"] = f"error: {exc}"
            if settings.kafka.required:
                status = "error"
            elif status == "ok":
                status = "degraded"
    else:
        deps["kafka"] = "disabled"

    return HealthResponse(status=status, dependencies=deps)
