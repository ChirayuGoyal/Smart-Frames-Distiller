"""
server/routers/faces.py — Face recognition identity management, search, and job endpoints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import uuid as _uuid

from fastapi import APIRouter, Request, status
from starlette.datastructures import UploadFile
from faces.service import (
    delete_identity,
    list_identities,
    onboard_identity,
    search_face,
    tag_identity,
)
from server.errors import AppError, FaceStoreUnavailableError
from server.model_cache import get_face_store, resolve_face_config
from server.routers.jobs import _resolve_and_check_path, _sanitize_filename
from server.schemas import (
    FaceIdentityCreateRequest,
    FaceMergeRequest,
    FaceOnboardRequest,
    FaceSearchRequest,
    FaceJobOptions,
    JobRecordResponse,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/faces", tags=["faces"])


async def _save_upload(request: Request, file: UploadFile) -> Path:
    """Stream a multipart upload into work_dir/tmp with the size cap applied."""
    settings = request.app.state.settings
    settings.uploads_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = settings.uploads_tmp_dir / f"{_uuid.uuid4()}_{_sanitize_filename(file.filename or 'upload')}"
    size_written = 0
    with tmp_path.open("wb") as out_f:
        while chunk := await file.read(65536):
            size_written += len(chunk)
            if size_written > settings.max_upload_bytes:
                out_f.close()
                tmp_path.unlink(missing_ok=True)
                raise AppError("File upload exceeded size limit", code="UPLOAD_TOO_LARGE", status_code=413)
            out_f.write(chunk)
    return tmp_path


async def _input_from_form(request: Request, form) -> tuple[Path, bool]:
    """Resolve (path, is_temp_upload) from a multipart form with file OR server_path."""
    file = form.get("file")
    server_path = form.get("server_path")
    if file is not None and isinstance(file, UploadFile) and file.filename:
        return await _save_upload(request, file), True
    if server_path and isinstance(server_path, str):
        return await _resolve_and_check_path(request, server_path), False
    raise AppError("Must provide either a file upload or server_path", code="MISSING_INPUT", status_code=400)


def _require_store(request: Request):
    settings = request.app.state.settings
    store = get_face_store(settings)
    try:
        store.connect()
        return store
    except Exception as exc:
        raise FaceStoreUnavailableError(str(exc))


@router.post("/identities", response_model=dict[str, Any])
async def create_or_update_identity(request: Request, body: FaceIdentityCreateRequest) -> dict[str, Any]:
    """Tag or update metadata for a specific face UUID."""
    store = _require_store(request)
    if not body.uid:
        raise AppError("uid is required for tagging", code="MISSING_INPUT", status_code=400)
    cfg = {"site_id": body.site_id, "camera_id": body.camera_id}
    try:
        return tag_identity(
            uid=body.uid,
            name=body.name,
            role=body.role,
            department=body.department,
            notes=body.notes,
            store=store,
            cfg=cfg,
        )
    except (SystemExit, KeyError) as exc:
        raise AppError(str(exc), code="FACE_NOT_FOUND", status_code=404)
    except Exception as exc:
        raise AppError(str(exc), code="FACE_ERROR", status_code=500)


@router.get("/identities", response_model=list[dict[str, Any]])
async def get_identities(
    request: Request, site_id: str, include_untagged: bool = False
) -> list[dict[str, Any]]:
    """List registered face entries for a site."""
    store = _require_store(request)
    try:
        return list_identities(site_id, store=store, include_untagged=include_untagged)
    except Exception as exc:
        raise AppError(str(exc), code="FACE_ERROR", status_code=500)


@router.patch("/identities/{uid}", response_model=dict[str, Any])
async def patch_identity(request: Request, uid: str, body: FaceIdentityCreateRequest) -> dict[str, Any]:
    """Update metadata fields for a face UUID."""
    store = _require_store(request)
    cfg = {"site_id": body.site_id, "camera_id": body.camera_id}
    try:
        return tag_identity(
            uid=uid,
            name=body.name,
            role=body.role,
            department=body.department,
            notes=body.notes,
            store=store,
            cfg=cfg,
        )
    except (SystemExit, KeyError) as exc:
        raise AppError(str(exc), code="FACE_NOT_FOUND", status_code=404)
    except Exception as exc:
        raise AppError(str(exc), code="FACE_ERROR", status_code=500)


@router.delete("/identities/{uid}", response_model=dict[str, bool])
async def remove_identity(request: Request, uid: str) -> dict[str, bool]:
    """Delete a face entry and its embeddings by UUID."""
    store = _require_store(request)
    try:
        ok = delete_identity(uid, store=store)
        return {"deleted": ok}
    except Exception as exc:
        raise AppError(str(exc), code="FACE_ERROR", status_code=500)


@router.post("/onboard", response_model=dict[str, Any])
async def onboard_person(request: Request) -> dict[str, Any]:
    """Enroll a person from a video/image — multipart upload OR JSON server_path."""
    store = _require_store(request)
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        path, is_temp = await _input_from_form(request, form)
        body = FaceOnboardRequest(
            name=str(form.get("name") or ""),
            site_id=str(form.get("site_id") or ""),
            max_frames=int(form.get("max_frames") or 8),
            single_embedding=str(form.get("single_embedding") or "true").lower() != "false",
            replace=str(form.get("replace") or "false").lower() == "true",
            notes=str(form.get("notes") or ""),
        )
    else:
        body = FaceOnboardRequest.model_validate(await request.json())
        if not body.server_path:
            raise AppError("server_path is required", code="MISSING_INPUT", status_code=400)
        path = await _resolve_and_check_path(request, body.server_path)
        is_temp = False

    cfg = resolve_face_config(request.app.state.settings, site_id=body.site_id)
    try:
        return onboard_identity(
            name=body.name,
            video_or_image=path,
            cfg=cfg,
            store=store,
            max_frames=body.max_frames,
            single_embedding=body.single_embedding,
            replace=body.replace,
            notes=body.notes,
        )
    except Exception as exc:
        raise AppError(str(exc), code="FACE_ERROR", status_code=500)
    finally:
        if is_temp:
            path.unlink(missing_ok=True)


@router.post("/search", response_model=list[dict[str, Any]])
async def search_identity(request: Request) -> list[dict[str, Any]]:
    """Search identities by face image — multipart upload OR JSON server_path."""
    store = _require_store(request)
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        path, is_temp = await _input_from_form(request, form)
        site_id = str(form.get("site_id") or "")
        limit = int(form.get("limit") or 5)
    else:
        body = FaceSearchRequest.model_validate(await request.json())
        if not body.server_path:
            raise AppError("server_path is required", code="MISSING_INPUT", status_code=400)
        path = await _resolve_and_check_path(request, body.server_path)
        is_temp = False
        site_id, limit = body.site_id, body.limit

    try:
        cfg = resolve_face_config(request.app.state.settings, site_id=site_id)
        return search_face(path, site_id=site_id, store=store, cfg=cfg, limit=limit)
    except Exception as exc:
        raise AppError(str(exc), code="FACE_ERROR", status_code=500)
    finally:
        if is_temp:
            path.unlink(missing_ok=True)


@router.post("/ingest", response_model=JobRecordResponse, status_code=status.HTTP_201_CREATED)
async def start_ingest_job(request: Request, server_path: str, site_id: str, camera_id: str = "") -> JobRecordResponse:
    """Enqueue a background job to ingest faces from a video into Milvus."""
    path = await _resolve_and_check_path(request, server_path)
    options = FaceJobOptions(site_id=site_id, camera_id=camera_id)
    return request.app.state.job_manager.submit_job(type="ingest", options=options, video_path=path)


@router.post("/merge", response_model=JobRecordResponse, status_code=status.HTTP_201_CREATED)
async def start_merge_job(request: Request, body: FaceMergeRequest) -> JobRecordResponse:
    """Enqueue a background job to cluster and merge duplicate faces for a site."""
    options = FaceJobOptions(site_id=body.site_id, dry_run=body.dry_run)
    # dummy path since merge operates on existing DB records
    dummy_path = request.app.state.settings.work_dir / "dummy_merge.tmp"
    dummy_path.touch(exist_ok=True)
    return request.app.state.job_manager.submit_job(type="merge", options=options, video_path=dummy_path)
