"""
server/app.py — FastAPI application factory and lifespan manager.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from server.errors import register_error_handlers
from server.jobs import JobManager
from server.model_cache import close_all, get_face_store, get_kafka_publisher
from server.routers import faces, health, jobs, streams
from server.settings import Settings, get_settings
from server.stream_manager import StreamManager

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    job_mgr = JobManager(settings)
    stream_mgr = StreamManager(settings)
    app.state.job_manager = job_mgr
    app.state.stream_manager = stream_mgr

    log.info("Starting JobManager and recovering interrupted jobs...")
    job_mgr.start()

    if settings.warm_models:
        log.info("Warming action and face models as requested...")
        try:
            from server.model_cache import get_action_model, get_face_recognizer
            get_action_model(settings)
            get_face_recognizer(settings)
        except Exception as exc:
            log.warning("Error warming models: %s", exc)

    # Proactively initialize connections without blocking or crashing if unreachable
    try:
        get_face_store(settings)
    except Exception as exc:
        log.warning("Initial connection to Milvus failed (endpoints will retry/503): %s", exc)

    try:
        if settings.kafka.enabled:
            get_kafka_publisher(settings).connect(wait=False)
    except Exception as exc:
        log.warning("Initial probe to Kafka broker failed: %s", exc)

    yield

    log.info("Shutting down StreamManager and JobManager...")
    stream_mgr.shutdown()
    job_mgr.shutdown()
    close_all()
    log.info("Server shutdown complete.")


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Smart Frames Distiller API",
        version="1.0.0",
        description="Action-aware semantic video chunking and face recognition server.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(streams.router)
    app.include_router(faces.router)

    return app
