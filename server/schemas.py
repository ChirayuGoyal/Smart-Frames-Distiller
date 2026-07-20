"""
server/schemas.py — Pydantic request/response models for the FastAPI server.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, model_validator


class JobOptions(BaseModel):
    """Pipeline run options matching `runner.RunOptions` plus server controls."""

    # Stage flags
    filter_enabled: bool = False
    detection_enabled: bool = False
    chunk_enabled: bool = False
    kafka_enabled: bool = False

    # Parallelism / filter stage
    n_workers: int = 1
    clip_len: int = 16
    sample_stride: int = 4
    conf_delta: float = 0.15
    max_gap: int = 30
    neighbor_pad: int = 2
    prefer_torch: bool = True
    ensemble: bool = False
    audio_spikes: bool = False
    audio_rms_z: float = 2.5
    audio_delta_z: float = 2.0
    device: str = "auto"
    inference_scale: float = 1.0
    inference_max_side: int | None = None
    reencode_h264: bool = True
    output_width: int = 640
    output_height: int = 480
    output_fps: float | None = None

    # Server / stage options
    overwrite: bool = False

    # Day/night + ROI (see day_night_detector.py, roi_loader.py)
    view_mode: str = "auto"                 # auto | day | night
    roi_path: str | None = None             # LabelMe JSON restricting detection
    ir_mode_cfg: dict[str, Any] = Field(default_factory=dict)
    day_night_sat_threshold: float = 30.0
    day_night_sample_frames: int = 30
    app_id: int = 0

    # Stage configs
    face_recognition_cfg: dict[str, Any] = Field(default_factory=dict)
    kafka_cfg: dict[str, Any] = Field(default_factory=dict)
    visualization: dict[str, Any] = Field(default_factory=dict)

    # Chunk / run IDs
    run_id: str | None = None
    camera_id: str = ""
    site_id: str = ""
    chunk_duration_sec: float = 5.0
    chunk_width: int | None = None
    chunk_height: int | None = None
    base_timestamp_ms: int | None = None

    # Overrides
    kafka_sp_enabled: str | None = None
    kafka_critic_enabled: str | None = None
    kafka_sp: str | None = None
    kafka_critic: str | None = None
    model_path: str | None = None
    model_cache_dir: str | None = None

    @model_validator(mode="after")
    def validate_stage_rules(self) -> JobOptions:
        if self.view_mode not in ("auto", "day", "night"):
            raise ValueError("view_mode must be one of: auto, day, night")
        if not (self.filter_enabled or self.detection_enabled or self.chunk_enabled):
            raise ValueError("No stage selected — set at least one of: filter_enabled, detection_enabled, chunk_enabled")
        if self.kafka_enabled and not self.chunk_enabled:
            raise ValueError("kafka_enabled requires chunk_enabled")
        site = self.site_id or self.face_recognition_cfg.get("site_id")
        if self.detection_enabled and not site:
            raise ValueError("site_id is required when detection_enabled is true")
        if self.kafka_enabled and not self.camera_id:
            raise ValueError("camera_id is required when kafka_enabled is true")
        return self


class FaceJobOptions(BaseModel):
    """Options for maintenance jobs that do not execute pipeline stages."""

    site_id: str
    camera_id: str = ""
    dry_run: bool = False


class CreateJobRequest(BaseModel):
    """Payload for POST /v1/jobs when using server-side path instead of multipart upload."""
    server_path: str | None = None
    options: JobOptions = Field(default_factory=JobOptions)


class JobProgress(BaseModel):
    stage: str = ""
    frac: float = 0.0
    message: str = ""


class JobRecordResponse(BaseModel):
    job_id: str
    type: str = "pipeline"
    state: str  # queued | running | succeeded | failed | cancelled | interrupted
    progress: JobProgress = Field(default_factory=JobProgress)
    error: str | None = None
    report: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class StreamCreateRequest(BaseModel):
    rtsp_url: str
    site_id: str
    camera_id: str
    options: dict[str, Any] = Field(default_factory=dict)


class StreamStats(BaseModel):
    frames_read: int = 0
    chunks_sent: int = 0
    fps_actual: float = 0.0


class StreamResponse(BaseModel):
    stream_id: str
    rtsp_url: str
    site_id: str
    camera_id: str
    state: str  # connecting | running | reconnecting | stopping | stopped | failed
    stats: StreamStats = Field(default_factory=StreamStats)


class FaceIdentityCreateRequest(BaseModel):
    """Tag/update payload. None = keep existing value, "" = clear the field."""
    uid: str | None = None
    name: str | None = None
    role: str | None = None
    department: str | None = None
    notes: str | None = None
    site_id: str = ""
    camera_id: str = ""


class FaceOnboardRequest(BaseModel):
    name: str
    site_id: str
    server_path: str | None = None
    max_frames: int = 8
    single_embedding: bool = True
    replace: bool = False
    notes: str = ""


class FaceSearchRequest(BaseModel):
    site_id: str
    server_path: str | None = None
    limit: int = 5


class FaceMergeRequest(BaseModel):
    site_id: str
    dry_run: bool = False


class HealthResponse(BaseModel):
    status: str  # ok | degraded | error
    dependencies: dict[str, str]  # kafka, milvus, models, gpu -> ok / unavailable / error details
