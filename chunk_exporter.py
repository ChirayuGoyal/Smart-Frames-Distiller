"""Split filtered video into fixed-duration chunks, save them, build frame-level
metadata, and publish to Kafka.

Chunking is independent of Kafka:
  * --chunks-dir writes N-second UUID-named chunks to a local folder (works even
    with --skip-kafka).
  * When Kafka is enabled, each chunk is also written to the jvadata assets path
    and one message per chunk is published.

Frame-level metadata:
  * For every filtered frame we record frame_id, source_frame_number,
    filtered_index, position_in_chunk, source_time_sec, epoch timestamp_ms, and
    the chunk_id / chunk_index it belongs to.
  * Written as a run-level JSON and a per-chunk sidecar (<chunk>.frames.json),
    A compact summary (counts, time range, first/last frame, file refs) is
    embedded in each Kafka message under the "event_metadata" key.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import cv2

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.video_io import read_video_meta, mux_audio_to_video, try_reencode_h264

from kafka_producer import (
    build_chunk_message,
    chunk_asset_path,
    connect_kafka,
    kafka_settings,
    publish_chunk,
    log,
)


def _write_chunk_video(
    frames: list,
    dst_path: Path,
    fps: float,
    width: int,
    height: int,
    *,
    reencode_h264: bool,
    out_width: int | None = None,
    out_height: int | None = None,
) -> None:
    final_w = out_width  or width
    final_h = out_height or height
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # OpenCV can only write mp4v; write to a temp then re-encode to H.264+faststart
    # so the final file is always a web-compatible MP4, unconditionally.
    cv_path = dst_path.with_suffix(".tmp.mp4")

    writer = cv2.VideoWriter(
        str(cv_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (final_w, final_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create chunk video writer: {cv_path}")
    for frame in frames:
        if final_w != width or final_h != height:
            frame = cv2.resize(frame, (final_w, final_h))
        writer.write(frame)
    writer.release()

    ok = try_reencode_h264(cv_path, fps=fps)   # converts cv_path in-place, resamples to fps
    cv_path.replace(dst_path)                 # move result to final path
    if not ok:
        log.warning("chunk re-encode failed — output may not be web-compatible: %s", dst_path)


def _save_full_clip(filtered_video: Path, dest: str) -> str | None:
    """Copy the WHOLE filtered clip to the exact destination path from the CLI."""
    dst = Path(dest)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filtered_video, dst)
        log.info("full filtered clip saved to: %s", dst)
        return str(dst)
    except Exception as exc:
        log.error("failed to save full filtered clip to %s: %s", dst, exc)
        return None


def _frame_record(
    *,
    source_frame_number: int,
    filtered_index: int,
    position_in_chunk: int,
    source_fps: float | None,
    base_ts: int,
    chunk_id: str,
    chunk_index: int,
) -> dict[str, Any]:
    """One frame-level metadata record (both indices + both times)."""
    t_sec = (source_frame_number / source_fps) if source_fps else 0.0
    return {
        "frame_id": str(uuid.uuid4()),
        "source_frame_number": int(source_frame_number),
        "filtered_index": int(filtered_index),
        "position_in_chunk": int(position_in_chunk),
        "source_time_sec": round(t_sec, 4),
        "timestamp_ms": int(base_ts + round(t_sec * 1000)),
        "chunk_id": chunk_id,
        "chunk_index": int(chunk_index),
    }


def _brief_frame(fm: dict[str, Any] | None) -> dict[str, Any] | None:
    """Minimal frame ref for the compact in-message summary."""
    if not fm:
        return None
    return {k: fm[k] for k in ("frame_id", "source_frame_number", "filtered_index", "timestamp_ms")}


def split_and_publish_chunks(
    filtered_video: str | Path,
    *,
    site_id: str,
    camera_id: str,
    run_id: str,
    app_id: int = 0,
    chunk_duration_sec: float = 5.0,
    base_timestamp_ms: int | None = None,
    reencode_h264: bool = True,
    full_clip_dest: str | None = None,
    chunks_dir: str | None = None,
    kept_indices: list[int] | None = None,
    source_fps: float | None = None,
    frames_meta_path: str | None = None,
    kafka_overrides: dict[str, Any] | None = None,
    chunk_width: int | None = None,
    chunk_height: int | None = None,
    output_fps: float | None = None,
) -> dict[str, Any]:
    """
    Split filtered MP4 into chunk_duration_sec segments, build frame-level
    metadata, save chunks, and (when Kafka is enabled) publish one message per
    chunk with the frame metadata under "event_metadata".
    """
    filtered_video = Path(filtered_video)
    if not filtered_video.is_file():
        raise FileNotFoundError(f"Filtered video not found: {filtered_video}")

    cfg = kafka_settings(kafka_overrides)
    kafka_on = cfg["enabled"]
    embed_frames = cfg.get("embed_frame_metadata", True)
    # Run-level frames-metadata reference sent in the event must be resolvable
    # by downstream consumers: use a jvadata path when publishing, else local.
    assets_base = cfg["assets_base"]
    chunk_ms = int(chunk_duration_sec * 1000)
    base_ts = base_timestamp_ms if base_timestamp_ms is not None else int(time.time() * 1000)
    chunks_dir_path = Path(chunks_dir) if chunks_dir else None
    assets_frames_meta = f"{assets_base}/{site_id}/{camera_id}/full/{run_id}_frames_metadata.json"
    event_frames_meta_ref = assets_frames_meta if kafka_on else frames_meta_path

    meta = read_video_meta(filtered_video)
    # source_fps drives original frame numbers/timestamps; fall back to clip fps.
    src_fps = source_fps if source_fps else meta.fps
    frames_per_chunk = max(1, int(round(meta.fps * chunk_duration_sec)))
    log.info(
        "chunk export start: video=%s fps=%.3f src_fps=%.3f chunk_dur=%ss frames_per_chunk=%d "
        "kafka_on=%s chunks_dir=%s kept_indices=%s site=%s camera=%s run_id=%s",
        filtered_video.name, meta.fps, src_fps, chunk_duration_sec, frames_per_chunk,
        kafka_on, chunks_dir_path or "(none)",
        len(kept_indices) if kept_indices else "(none)", site_id, camera_id, run_id,
    )

    full_clip_path: str | None = None
    if full_clip_dest:
        full_clip_path = _save_full_clip(filtered_video, full_clip_dest)

    if kafka_on:
        if not connect_kafka(wait=True, overrides=kafka_overrides):
            if cfg["required"]:
                log.error("Kafka required but broker unreachable — aborting chunk export")
                raise RuntimeError("Kafka connection required but broker unreachable")
            log.warning("proceeding without Kafka — chunks saved locally, messages spooled")
    else:
        log.info("Kafka disabled — chunks-only mode (no publishing)")

    cap = cv2.VideoCapture(str(filtered_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open filtered video: {filtered_video}")

    buffer: list = []
    chunk_index = 0
    records: list[dict[str, Any]] = []
    all_frames_meta: list[dict[str, Any]] = []
    frame_cursor = 0  # running 0-based index across the whole filtered stream

    def _sidecar_path(chunk_file: Path) -> Path:
        return chunk_file.parent / f"{chunk_file.stem}_frames.json"

    def _flush_chunk(frames: list, index: int) -> None:
        nonlocal frame_cursor
        if not frames:
            return

        chunk_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())
        asset_path = chunk_asset_path(site_id, camera_id, chunk_id, assets_base)

        targets: list[Path] = []
        if chunks_dir_path is not None:
            targets.append(chunks_dir_path / f"{chunk_id}.mp4")   # user dir — exclusive
        elif kafka_on:
            targets.append(Path(asset_path))                        # jvadata default
        if not targets:
            targets.append(Path(asset_path))                        # last-resort fallback

        # Write primary target; copy to any additional targets.
        written_to: Path | None = None
        copy_start: int = 1
        for t_idx, t in enumerate(targets):
            try:
                _write_chunk_video(
                    frames, t,
                    output_fps if output_fps is not None else meta.fps,
                    meta.width, meta.height,
                    reencode_h264=reencode_h264,
                    out_width=chunk_width,
                    out_height=chunk_height,
                )
                written_to = t
                copy_start = t_idx + 1
                break
            except Exception as exc:
                log.warning("chunk %d: write failed at %s — %s", index, t, exc)

        if written_to is None:
            log.error("chunk %d: chunk_id=%s could not be saved to any target", index, chunk_id)
            return

        # Mux audio from filtered clip for this chunk's time window
        chunk_start_sec = index * chunk_duration_sec
        chunk_end_sec   = chunk_start_sec + (len(frames) / meta.fps if meta.fps else 0)
        mux_audio_to_video(
            written_to, filtered_video,
            [(chunk_start_sec, chunk_end_sec)],
        )

        for extra in targets[copy_start:]:
            try:
                extra.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(written_to, extra)
            except Exception as exc:
                log.warning("chunk %d: copy to %s failed — %s", index, extra, exc)

        # --- frame-level metadata for this chunk ---
        g0 = frame_cursor
        frames_meta: list[dict[str, Any]] = []
        for i in range(len(frames)):
            filt_idx = g0 + i
            if kept_indices is not None and filt_idx < len(kept_indices):
                src_idx = int(kept_indices[filt_idx])
            else:
                src_idx = filt_idx
            frames_meta.append(_frame_record(
                source_frame_number=src_idx,
                filtered_index=filt_idx,
                position_in_chunk=i,
                source_fps=src_fps,
                base_ts=base_ts,
                chunk_id=chunk_id,
                chunk_index=index,
            ))
        frame_cursor += len(frames)
        all_frames_meta.extend(frames_meta)

        duration_ms = int(round(len(frames) / meta.fps * 1000)) if meta.fps > 0 else chunk_ms
        start_ts = base_ts + index * chunk_ms
        end_ts = start_ts + duration_ms

        # Full frame-level metadata -> sidecar files + run-level JSON.
        chunk_meta_full = {
            "chunk_id": chunk_id,
            "chunk_index": index,
            "run_id": run_id,
            "site_id": site_id,
            "camera_id": camera_id,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "source_fps": round(src_fps, 6) if src_fps else None,
            "frame_count": len(frames_meta),
            "frames": frames_meta,
        }

        # Per-chunk sidecar JSON next to every saved copy.
        for t in targets:
            try:
                _sidecar_path(t).write_text(json.dumps(chunk_meta_full, indent=2), encoding="utf-8")
            except Exception as exc:
                log.error("failed to write frame sidecar for %s: %s", t, exc)

        # Metadata sent in the Kafka message. embed_frames=True -> full per-frame
        # records; False -> compact summary (full frames stay in sidecar/run JSON).
        sidecar_ref = _sidecar_path(Path(asset_path)) if kafka_on else _sidecar_path(targets[0])
        event_metadata_msg = {
            "chunk_id": chunk_id,
            "chunk_index": index,
            "asset_id": run_id,
            "site_id": site_id,
            "camera_id": camera_id,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "source_fps": round(src_fps, 6) if src_fps else None,
            "frame_count": len(frames_meta),
            "frames_sidecar": str(sidecar_ref),
            "frames_metadata_file": event_frames_meta_ref,
        }
        if embed_frames:
            event_metadata_msg["frames"] = frames_meta
        else:
            event_metadata_msg["first_frame"] = _brief_frame(frames_meta[0] if frames_meta else None)
            event_metadata_msg["last_frame"] = _brief_frame(frames_meta[-1] if frames_meta else None)

        message = build_chunk_message(
            event_id=event_id,
            run_id=run_id,
            chunk_id=chunk_id,
            camera_id=camera_id,
            site_id=site_id,
            app_id=app_id,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            chunk_path=asset_path,
            sp_enabled=cfg["sp_enabled"],
            critic_enabled=cfg["critic_enabled"],
            sp=cfg.get("sp"),
            critic=cfg.get("critic"),
        )
        message["event_metadata"] = event_metadata_msg  # full frames or compact summary (see embed_frame_metadata)

        published = False
        if kafka_on:
            log.info("chunk %d: %d frames (%d frame-meta) -> publishing chunk_id=%s",
                     index, len(frames), len(frames_meta), chunk_id)
            published = publish_chunk(message, cfg=cfg)
        else:
            log.info("chunk %d: %d frames (%d frame-meta) -> saved (chunks-only) chunk_id=%s",
                     index, len(frames), len(frames_meta), chunk_id)

        records.append({
            "chunk_index": index,
            "chunk_id": chunk_id,
            "event_id": event_id,
            "path": asset_path,
            "saved_paths": [str(t) for t in targets],
            "frames_sidecars": [str(_sidecar_path(t)) for t in targets],
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "frames": len(frames),
            "frame_meta_count": len(frames_meta),
            "duration_sec": round(len(frames) / meta.fps, 3) if meta.fps else 0,
            "kafka_published": published,
            "message": message,
        })

    _bar = _tqdm(
        total=meta.frame_count or None,
        unit="fr", desc="  chunk ",
        dynamic_ncols=True, leave=True,
    ) if _TQDM else None

    while True:
        ok, frame = cap.read()
        if not ok:
            _flush_chunk(buffer, chunk_index)
            break
        buffer.append(frame)
        if _bar:
            _bar.update(1)
        if len(buffer) >= frames_per_chunk:
            _flush_chunk(buffer, chunk_index)
            buffer = []
            chunk_index += 1

    if _bar:
        _bar.close()
    cap.release()

    total = len(records)
    published = sum(1 for r in records if r["kafka_published"])
    failed = (total - published) if kafka_on else 0

    # Run-level frame metadata file (all kept frames + which chunk they belong to).
    run_meta_doc = json.dumps({
        "run_id": run_id,
        "site_id": site_id,
        "camera_id": camera_id,
        "source_video": str(filtered_video.resolve()),
        "source_fps": round(src_fps, 6) if src_fps else None,
        "base_timestamp_ms": base_ts,
        "total_chunks": total,
        "total_frames": len(all_frames_meta),
        "frames": all_frames_meta,
    }, indent=2)

    frames_meta_written: str | None = None  # local copy
    if frames_meta_path:
        try:
            lp = Path(frames_meta_path)
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text(run_meta_doc, encoding="utf-8")
            frames_meta_written = str(lp)
            log.info("run-level frame metadata written (local): %s (%d frames)", lp, len(all_frames_meta))
        except Exception as exc:
            log.error("failed to write run-level frame metadata to %s: %s", frames_meta_path, exc)

    # jvadata copy at the path referenced in the event (so consumers can resolve it).
    if kafka_on:
        try:
            ap = Path(assets_frames_meta)
            ap.parent.mkdir(parents=True, exist_ok=True)
            ap.write_text(run_meta_doc, encoding="utf-8")
            log.info("run-level frame metadata written (assets): %s", ap)
        except Exception as exc:
            log.error("failed to write run-level frame metadata to %s: %s", assets_frames_meta, exc)

    log.info("chunk export done: total=%d published=%d failed=%d frames=%d chunks_dir=%s full_clip=%s",
             total, published, failed, len(all_frames_meta), chunks_dir_path or "(none)",
             full_clip_path or "(not saved)")
    if total == 0:
        log.warning("NO chunks produced — source video had no readable frames (check %s)",
                    filtered_video)

    return {
        "run_id": run_id,
        "site_id": site_id,
        "camera_id": camera_id,
        "source_video": str(filtered_video.resolve()),
        "chunk_duration_sec": chunk_duration_sec,
        "base_timestamp_ms": base_ts,
        "source_fps": round(src_fps, 6) if src_fps else None,
        "kafka_enabled": kafka_on,
        "chunks_dir": str(chunks_dir_path) if chunks_dir_path else None,
        "full_clip_path": full_clip_path,
        "frames_metadata_path": frames_meta_written,
        "frames_metadata_event_ref": event_frames_meta_ref,
        "total_chunks": total,
        "total_frames": len(all_frames_meta),
        "published_chunks": published,
        "failed_chunks": failed,
        "chunks": records,
    }
