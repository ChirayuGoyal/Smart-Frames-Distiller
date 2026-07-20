"""
server/routers/jobs.py — Pipeline job management endpoints.
"""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, status
# request.form() yields starlette UploadFile instances; fastapi.UploadFile is a
# SUBCLASS, so isinstance against it silently rejects every real upload.
from starlette.datastructures import UploadFile
from fastapi.responses import FileResponse, JSONResponse
from server.errors import AppError, JobNotFoundError, PathNotAllowedError
from server.schemas import CreateJobRequest, JobOptions, JobRecordResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


async def _resolve_and_check_path(request: Request, server_path: str) -> Path:
    settings = request.app.state.settings
    # Default-deny: server-side paths are only usable when an allowlist is
    # configured (SFD_ALLOWED_INPUT_DIRS). An empty allowlist must not mean
    # "the whole filesystem".
    if not settings.allowed_input_dirs:
        raise PathNotAllowedError(server_path)

    path = Path(server_path).resolve()
    allowed = [d.resolve() for d in settings.allowed_input_dirs]
    if not any(path.is_relative_to(a) for a in allowed):
        raise PathNotAllowedError(server_path)
    if not path.exists() or not path.is_file():
        raise AppError(f"Server file not found: {server_path}", code="FILE_NOT_FOUND", status_code=404)
    return path


def _sanitize_filename(filename: str) -> str:
    """Strip any path components — client filenames must never navigate."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = name.replace("..", "_").strip(". ")
    return name or "upload.bin"


@router.post("", response_model=JobRecordResponse, status_code=status.HTTP_201_CREATED)
async def create_job(request: Request) -> JobRecordResponse:
    """Submit a new pipeline job via multipart file upload or server-side path."""
    settings = request.app.state.settings
    job_mgr = request.app.state.job_manager

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("file")
        server_path_field = form.get("server_path")
        options_raw = form.get("options")

        if options_raw and isinstance(options_raw, str):
            try:
                options = JobOptions.model_validate_json(options_raw)
            except Exception as exc:
                raise AppError(f"Invalid options JSON: {exc}", code="INVALID_OPTIONS", status_code=422)
        else:
            options = JobOptions()

        if file and isinstance(file, UploadFile) and file.filename:
            settings.uploads_tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = settings.uploads_tmp_dir / f"{uuid.uuid4()}_{_sanitize_filename(file.filename)}"
            size_written = 0
            with tmp_path.open("wb") as out_f:
                while chunk := await file.read(65536):
                    size_written += len(chunk)
                    if size_written > settings.max_upload_bytes:
                        out_f.close()
                        tmp_path.unlink(missing_ok=True)
                        raise AppError("File upload exceeded size limit", code="UPLOAD_TOO_LARGE", status_code=413)
                    out_f.write(chunk)
            video_path = tmp_path
        elif server_path_field and isinstance(server_path_field, str):
            video_path = await _resolve_and_check_path(request, server_path_field)
        else:
            raise AppError("Must provide either a file upload or server_path", code="MISSING_INPUT", status_code=400)
    else:
        try:
            data = await request.json()
            req = CreateJobRequest.model_validate(data)
        except Exception as exc:
            raise AppError(f"Invalid request JSON: {exc}", code="INVALID_REQUEST", status_code=422)

        if not req.server_path:
            raise AppError("server_path is required when sending JSON body", code="MISSING_INPUT", status_code=400)
        video_path = await _resolve_and_check_path(request, req.server_path)
        options = req.options

    return job_mgr.submit_job(type="pipeline", options=options, video_path=video_path)


@router.get("", response_model=list[JobRecordResponse])
async def list_jobs(request: Request, state: str | None = None) -> list[JobRecordResponse]:
    """List all submitted pipeline jobs, optionally filtered by state."""
    return request.app.state.job_manager.list_jobs(state=state)


@router.get("/{job_id}", response_model=JobRecordResponse)
async def get_job(request: Request, job_id: str) -> JobRecordResponse:
    """Get status and details of a specific job."""
    return request.app.state.job_manager.get_job(job_id)


@router.post("/{job_id}/cancel", response_model=JobRecordResponse)
async def cancel_job(request: Request, job_id: str) -> JobRecordResponse:
    """Request cooperative cancellation of a queued or running job."""
    return request.app.state.job_manager.cancel_job(job_id)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(request: Request, job_id: str) -> None:
    """Cancel and delete all records and artifacts for a job."""
    if not request.app.state.job_manager.delete_job(job_id):
        raise JobNotFoundError(job_id)


@router.get("/{job_id}/report", response_model=dict[str, Any])
async def get_job_report(request: Request, job_id: str) -> dict[str, Any]:
    """Get the final structured summary report for a completed job."""
    record = request.app.state.job_manager.get_job(job_id)
    if record.report:
        return record.report

    # Try loading from disk if not cached in memory
    settings = request.app.state.settings
    out_dir = settings.jobs_dir / job_id / "output"
    for report_file in out_dir.glob("*_report.json"):
        if report_file.exists():
            return json.loads(report_file.read_text(encoding="utf-8"))

    raise AppError("Report not available (job may not be complete)", code="REPORT_NOT_FOUND", status_code=404)


@router.get("/{job_id}/artifacts", response_model=list[dict[str, Any]])
async def list_job_artifacts(request: Request, job_id: str) -> list[dict[str, Any]]:
    """List all output files produced by the job."""
    settings = request.app.state.settings
    out_dir = settings.jobs_dir / job_id / "output"
    if not out_dir.exists():
        return []

    artifacts = []
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            artifacts.append({
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "modified_at": f.stat().st_mtime,
            })
    return artifacts


@router.get("/{job_id}/artifacts/{name}")
async def get_job_artifact(request: Request, job_id: str, name: str) -> FileResponse:
    """Download a specific output artifact file."""
    settings = request.app.state.settings
    out_dir = (settings.jobs_dir / job_id / "output").resolve()
    file_path = (out_dir / name).resolve()

    if not file_path.is_relative_to(out_dir):
        raise AppError("Access denied (path traversal attempted)", code="PATH_NOT_ALLOWED", status_code=403)
    if not file_path.exists() or not file_path.is_file():
        raise AppError(f"Artifact not found: {name}", code="ARTIFACT_NOT_FOUND", status_code=404)

    return FileResponse(path=file_path, filename=name)
