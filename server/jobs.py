"""
server/jobs.py — In-process background job execution queue and state persistence.
"""
from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runner import JobCancelled, RunOptions, run_action_aware
from server.errors import (
    JobCancelledError,
    JobNotCancellableError,
    JobNotFoundError,
    PathNotAllowedError,
)
from server.model_cache import (
    get_action_model,
    get_face_store,
    get_gpu_semaphore,
    resolve_face_config,
)
from server.schemas import FaceJobOptions, JobOptions, JobProgress, JobRecordResponse
from server.settings import Settings

log = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    """Manages async pipeline jobs using a thread pool and file-based state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs_dir = settings.jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, JobRecordResponse] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._queue: queue.Queue[tuple[str, JobOptions | FaceJobOptions, Path]] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Scan disk for existing jobs and launch worker threads."""
        with self._lock:
            for job_dir in self.jobs_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                job_file = job_dir / "job.json"
                if not job_file.exists():
                    continue
                try:
                    data = json.loads(job_file.read_text(encoding="utf-8"))
                    record = JobRecordResponse.model_validate(data)
                    if record.state in ("queued", "running"):
                        record.state = "interrupted"
                        record.error = "Server restarted or crashed during job execution"
                        record.updated_at = _utc_now()
                        self._persist_record(record)
                    self._jobs[record.job_id] = record
                except Exception as exc:
                    log.warning("Failed to load job record from %s: %s", job_file, exc)

            num_workers = max(1, self.settings.max_concurrent_jobs)
            for i in range(num_workers):
                t = threading.Thread(target=self._worker_loop, name=f"JobWorker-{i}", daemon=True)
                t.start()
                self._threads.append(t)
            log.info("JobManager started with %d worker threads", num_workers)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Signal worker threads to stop and wait for completion."""
        self._shutdown_event.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    def _persist_record(self, record: JobRecordResponse) -> None:
        job_dir = self.jobs_dir / record.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job_file = job_dir / "job.json"
        data = record.model_dump(mode="json")
        tmp_file = job_dir / "job.json.tmp"
        tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_file.replace(job_file)

    def submit_job(
        self, type: str, options: JobOptions | FaceJobOptions, video_path: Path
    ) -> JobRecordResponse:
        job_id = str(uuid.uuid4())
        now = _utc_now()
        if isinstance(options, JobOptions):
            options.run_id = options.run_id or job_id

        out_dir = self.jobs_dir / job_id / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        record = JobRecordResponse(
            job_id=job_id,
            type=type,
            state="queued",
            progress=JobProgress(stage="queued", frac=0.0, message="Waiting in queue"),
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._cancel_events[job_id] = threading.Event()
            self._persist_record(record)

        self._queue.put((job_id, options, video_path))
        return record

    def get_job(self, job_id: str) -> JobRecordResponse:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]
        job_file = self.jobs_dir / job_id / "job.json"
        if job_file.exists():
            try:
                data = json.loads(job_file.read_text(encoding="utf-8"))
                record = JobRecordResponse.model_validate(data)
                with self._lock:
                    self._jobs[job_id] = record
                return record
            except Exception:
                pass
        raise JobNotFoundError(job_id)

    def list_jobs(self, state: str | None = None) -> list[JobRecordResponse]:
        with self._lock:
            records = list(self._jobs.values())
        if state:
            records = [r for r in records if r.state == state]
        records.sort(key=lambda r: r.created_at, reverse=True)
        # Reports can embed full per-frame chunk metadata — far too heavy for a
        # listing. Fetch a single job to get its report.
        return [r.model_copy(update={"report": None}) for r in records]

    def cancel_job(self, job_id: str) -> JobRecordResponse:
        record = self.get_job(job_id)
        with self._lock:
            if record.state == "queued":
                record.state = "cancelled"
                record.progress.message = "Cancelled while queued"
                record.updated_at = _utc_now()
                self._persist_record(record)
                return record
            elif record.state == "running":
                ev = self._cancel_events.get(job_id)
                if ev:
                    ev.set()
                record.progress.message = "Cancellation requested"
                record.updated_at = _utc_now()
                self._persist_record(record)
                return record
            else:
                raise JobNotCancellableError(job_id, record.state)

    def delete_job(self, job_id: str) -> bool:
        try:
            record = self.get_job(job_id)
            if record.state == "running":
                ev = self._cancel_events.get(job_id)
                if ev:
                    ev.set()
        except JobNotFoundError:
            pass

        with self._lock:
            self._jobs.pop(job_id, None)
            self._cancel_events.pop(job_id, None)

        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
            return True
        return False

    def _check_roi_path(self, roi_path: str | None) -> str | None:
        """ROI JSON is a client-supplied server-side path — hold it to the same
        allowlist as video inputs so it cannot be used to probe the filesystem."""
        if not roi_path:
            return None
        if not self.settings.allowed_input_dirs:
            raise PathNotAllowedError(roi_path)
        p = Path(roi_path).resolve()
        allowed = [d.resolve() for d in self.settings.allowed_input_dirs]
        if not any(p.is_relative_to(a) for a in allowed):
            raise PathNotAllowedError(roi_path)
        if not p.is_file():
            raise JobNotFoundError(f"ROI file not found: {roi_path}")
        return str(p)

    def _update_progress(self, job_id: str, stage: str, frac: float, message: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.progress = JobProgress(stage=stage, frac=frac, message=message)
            record.updated_at = _utc_now()
            self._persist_record(record)

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            job_id, options, video_path = item
            record = self._jobs.get(job_id)
            cancel_event = self._cancel_events.get(job_id, threading.Event())

            if not record or record.state == "cancelled" or cancel_event.is_set():
                if record:
                    record.state = "cancelled"
                    record.updated_at = _utc_now()
                    self._persist_record(record)
                self._queue.task_done()
                continue

            with self._lock:
                record.state = "running"
                record.progress = JobProgress(stage="starting", frac=0.0, message="Initializing pipeline")
                record.updated_at = _utc_now()
                self._persist_record(record)

            try:
                if record.type == "ingest":
                    from faces.service import ingest_video
                    store = get_face_store(self.settings)
                    cfg = resolve_face_config(
                        self.settings,
                        site_id=options.site_id,
                        camera_id=options.camera_id,
                    )
                    report = ingest_video(video_path, cfg, store=store)
                elif record.type == "merge":
                    from faces.service import merge_identities
                    store = get_face_store(self.settings)
                    cfg = resolve_face_config(self.settings, site_id=options.site_id)
                    dry_run = options.dry_run
                    report = merge_identities(cfg, store=store, dry_run=dry_run)
                else:
                    if not isinstance(options, JobOptions):
                        raise TypeError("Pipeline jobs require JobOptions")
                    n_workers = min(
                        max(1, options.n_workers), self.settings.max_filter_workers
                    )
                    face_cfg = resolve_face_config(
                        self.settings,
                        options.face_recognition_cfg,
                        site_id=options.site_id or None,
                        camera_id=options.camera_id,
                    )
                    kafka_cfg = {
                        **self.settings.kafka_overrides(),
                        **options.kafka_cfg,
                    }
                    run_opts = RunOptions(
                        video=video_path,
                        filter_enabled=options.filter_enabled,
                        detection_enabled=options.detection_enabled,
                        chunk_enabled=options.chunk_enabled,
                        kafka_enabled=options.kafka_enabled,
                        output_dir=self.jobs_dir / job_id / "output",
                        n_workers=n_workers,
                        clip_len=options.clip_len,
                        sample_stride=options.sample_stride,
                        conf_delta=options.conf_delta,
                        max_gap=options.max_gap,
                        neighbor_pad=options.neighbor_pad,
                        prefer_torch=options.prefer_torch,
                        ensemble=options.ensemble,
                        audio_spikes=options.audio_spikes,
                        audio_rms_z=options.audio_rms_z,
                        audio_delta_z=options.audio_delta_z,
                        device=options.device,
                        inference_scale=options.inference_scale,
                        inference_max_side=options.inference_max_side,
                        reencode_h264=options.reencode_h264,
                        output_width=options.output_width,
                        output_height=options.output_height,
                        output_fps=options.output_fps,
                        face_recognition_cfg=face_cfg,
                        run_id=options.run_id or job_id,
                        camera_id=options.camera_id,
                        site_id=options.site_id,
                        chunk_duration_sec=options.chunk_duration_sec,
                        chunk_width=options.chunk_width,
                        chunk_height=options.chunk_height,
                        base_timestamp_ms=options.base_timestamp_ms,
                        kafka_cfg=kafka_cfg,
                        kafka_sp_enabled=options.kafka_sp_enabled,
                        kafka_critic_enabled=options.kafka_critic_enabled,
                        kafka_sp=options.kafka_sp,
                        kafka_critic=options.kafka_critic,
                        visualization=options.visualization,
                        model_path=options.model_path,
                        model_cache_dir=options.model_cache_dir,
                        view_mode=options.view_mode,
                        roi_path=self._check_roi_path(options.roi_path),
                        ir_mode_cfg=options.ir_mode_cfg,
                        day_night_sat_threshold=options.day_night_sat_threshold,
                        day_night_sample_frames=options.day_night_sample_frames,
                        app_id=options.app_id,
                        overwrite=options.overwrite,
                        cancel_event=cancel_event,
                        progress_cb=lambda st, fr, msg: self._update_progress(job_id, st, fr, msg),
                    )

                    # A cached action model can be shared only by sequential
                    # jobs; parallel filtering runs in separate processes.
                    if n_workers == 1:
                        run_opts.action_model = get_action_model(
                            self.settings,
                            {
                                "clip_len": options.clip_len,
                                "prefer_torch": options.prefer_torch,
                                "device": options.device,
                                "ensemble": options.ensemble,
                                "model_path": options.model_path,
                                "model_cache_dir": options.model_cache_dir,
                            },
                        )

                    # Acquire GPU semaphore around inference if using torch
                    use_gpu = (run_opts.prefer_torch and run_opts.device != "cpu")
                    sem = get_gpu_semaphore(self.settings) if use_gpu else None

                    if sem:
                        sem.acquire()
                    try:
                        report = run_action_aware(run_opts)
                    finally:
                        if sem:
                            sem.release()

                with self._lock:
                    record.state = "succeeded"
                    record.progress = JobProgress(stage="done", frac=1.0, message="Completed successfully")
                    record.report = report
                    record.updated_at = _utc_now()
                    self._persist_record(record)

            except (JobCancelled, JobCancelledError):
                with self._lock:
                    record.state = "cancelled"
                    record.progress.message = "Cancelled during execution"
                    record.updated_at = _utc_now()
                    self._persist_record(record)
            except Exception as exc:
                log.exception("Job %s failed during execution", job_id)
                with self._lock:
                    record.state = "failed"
                    record.error = str(exc)
                    record.progress.message = f"Error: {exc}"
                    record.updated_at = _utc_now()
                    self._persist_record(record)
            finally:
                # Uploaded inputs live in work_dir/tmp — reclaim them once the
                # job reaches a terminal state (server paths are never touched).
                try:
                    vp = Path(video_path).resolve()
                    if vp.is_relative_to(self.settings.uploads_tmp_dir.resolve()):
                        vp.unlink(missing_ok=True)
                except OSError:
                    pass
                self._queue.task_done()
