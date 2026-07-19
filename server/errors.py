"""
server/errors.py — Application exception hierarchy and FastAPI error handlers.

All custom errors inherit from `AppError` and carry a machine-readable `code` and
HTTP status code. Exception handlers convert these into `{"error": {"code": ..., "detail": ...}}`.
"""
from __future__ import annotations

from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for all application errors."""

    def __init__(self, message: str, code: str = "APP_ERROR", status_code: int = 500, detail: Any = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.detail = detail or message


class JobNotFoundError(AppError):
    def __init__(self, job_id: str):
        super().__init__(f"Job not found: {job_id}", code="JOB_NOT_FOUND", status_code=404)


class JobNotCancellableError(AppError):
    def __init__(self, job_id: str, current_state: str):
        super().__init__(
            f"Job {job_id} cannot be cancelled from state '{current_state}'",
            code="JOB_NOT_CANCELLABLE",
            status_code=409,
        )


class JobCancelledError(Exception):
    """Raised internally within pipeline threads when cooperative cancellation is triggered."""
    pass


class StreamNotFoundError(AppError):
    def __init__(self, stream_id: str):
        super().__init__(f"Stream not found: {stream_id}", code="STREAM_NOT_FOUND", status_code=404)


class StreamLimitReachedError(AppError):
    def __init__(self, max_streams: int):
        super().__init__(
            f"Maximum concurrent RTSP streams reached ({max_streams})",
            code="STREAM_LIMIT_REACHED",
            status_code=429,
        )


class PathNotAllowedError(AppError):
    def __init__(self, path: str):
        super().__init__(
            f"Path not allowed by SFD_ALLOWED_INPUT_DIRS: {path}",
            code="PATH_NOT_ALLOWED",
            status_code=403,
        )


class DependencyUnavailableError(AppError):
    def __init__(self, dependency: str, message: str = ""):
        msg = f"Dependency unavailable: {dependency}" + (f" ({message})" if message else "")
        super().__init__(msg, code="DEPENDENCY_UNAVAILABLE", status_code=503)


class FaceStoreUnavailableError(DependencyUnavailableError):
    def __init__(self, message: str = ""):
        super().__init__("milvus", message)


class InvalidJobOptionsError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="INVALID_JOB_OPTIONS", status_code=422)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "detail": exc.message}},
        )

    @app.exception_handler(JobCancelledError)
    async def job_cancelled_handler(request: Request, exc: JobCancelledError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": {"code": "JOB_CANCELLED", "detail": "Job execution was cancelled"}},
        )
