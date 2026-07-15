"""
Action-aware pipeline orchestration — 3 explicit stages.

Stage 1  --filter     Original clip → action-aware selection → filtered clip
Stage 2  --detection  Filtered clip → person detection + face recognition → annotated clip
Stage 3  --chunk      Annotated clip → N-second chunks → Kafka (if --kafka flag set)

Each stage is opt-in. Only stages whose flag is True are executed.
The output of each stage becomes the input of the next.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sys

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.benchmark import BenchmarkSession, merge_benchmark_into_report
from common.selection import build_frame_reasons, removed_indices
from common.video_io import read_video_meta, write_frame_index
from common.visualize import AnnotateStyle, write_annotated_video

from parallel_filter import do_filter
from correlation_plot import plot_correlation_timeline
from chunk_exporter import split_and_publish_chunks


def _path_or_none(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def output_resolution_from_config(cfg: dict[str, Any]) -> tuple[int, int]:
    nested = cfg.get("output_resolution") or {}
    width  = cfg.get("output_width",  nested.get("width",  640))
    height = cfg.get("output_height", nested.get("height", 480))
    return int(width), int(height)


def resolve_output_paths(video: Path) -> dict[str, Path]:
    out_dir = video.parent / "action_aware_output"
    stem    = video.stem
    return {
        "output_dir":            out_dir,
        "annotated_video":       out_dir / f"{stem}_annotated.mp4",
        "filtered_clip":         out_dir / f"{stem}_filtered.mp4",
        "filter_metadata":       out_dir / f"{stem}_filter_metadata.json",
        "detection_clip":        out_dir / f"{stem}_detection.mp4",
        "benchmark_path":        out_dir / f"{stem}_benchmark.json",
        "report_path":           out_dir / f"{stem}_report.json",
        "kept_indices_path":     out_dir / f"{stem}_kept_indices.json",
        "correlation_plot_path": out_dir / f"{stem}_correlation.png",
        "frames_metadata_path":  out_dir / f"{stem}_frames_metadata.json",
    }


@dataclass
class RunOptions:
    video: Path

    # ── Stage flags ────────────────────────────────────────────────────────────
    filter_enabled:    bool = False   # Stage 1: action-aware filtering
    detection_enabled: bool = False   # Stage 2: person detection + face recognition
    chunk_enabled:     bool = False   # Stage 3: chunking
    kafka_enabled:     bool = False   # Sub-flag of chunk: publish to Kafka

    # ── Output path overrides ─────────────────────────────────────────────────
    output_dir:       Path | None = None
    filtered_clip:    Path | None = None   # override for stage-1 output
    detection_clip:   Path | None = None   # override for stage-2 output
    output_clip:      Path | None = None   # copy final stage output here

    # ── Parallelism ───────────────────────────────────────────────────────────
    n_workers:         int   = 1    # 1 = sequential; >1 = parallel segment filter

    # ── Filter stage config ───────────────────────────────────────────────────
    clip_len:          int   = 16
    sample_stride:     int   = 4
    conf_delta:        float = 0.15
    max_gap:           int   = 30
    neighbor_pad:      int   = 2
    prefer_torch:      bool  = True
    ensemble:          bool  = False   # combine R3D-18 + MotionEnergy (OR-logic triggers)
    audio_spikes:      bool  = False   # keep frames near audio energy spikes
    audio_rms_z:       float = 2.5    # z-score threshold for loud events
    audio_delta_z:     float = 2.0    # z-score threshold for sudden energy changes
    device:            str   = "auto"
    inference_scale:   float = 1.0
    inference_max_side: int | None = None
    reencode_h264:     bool  = True
    output_width:      int   = 640
    output_height:     int   = 480
    output_fps:        float = 10.0   # default 10 fps; override with --fps
    plot_correlation:  bool  = False
    correlation_plot_path: Path | None = None
    correlation_show:  bool  = False
    visualization:     dict[str, Any] = field(default_factory=dict)

    # ── Detection stage config ────────────────────────────────────────────────
    face_recognition_cfg: dict[str, Any] = field(default_factory=dict)

    # ── Chunk stage config ────────────────────────────────────────────────────
    run_id:             str   = "test-run-12345"
    camera_id:          str   = ""
    site_id:            str   = ""
    chunk_duration_sec: float = 5.0
    chunk_width:        int | None = None
    chunk_height:       int | None = None
    base_timestamp_ms:  int | None = None
    chunks_dir:         str | None = None
    full_clip_dest:     str | None = None
    kafka_cfg:          dict[str, Any] = field(default_factory=dict)

    # ── Kafka message field overrides (from CLI) ──────────────────────────────
    kafka_sp_enabled:     str | None = None   # "true"/"false" → metadata.sp_enabled
    kafka_critic_enabled: str | None = None   # "true"/"false" → metadata.critic_enabled
    kafka_sp:             str | None = None   # alert_level.sp  (None = inherit from sp_enabled)
    kafka_critic:         str | None = None   # alert_level.critic (None = inherit)

    # ── Model weights ─────────────────────────────────────────────────────────
    model_path:        str | None = None   # local .pt file → no network download
    model_cache_dir:   str | None = None   # redirect torch hub download location

    # ── Benchmark ─────────────────────────────────────────────────────────────
    benchmark_enabled: bool = False
    benchmark_cfg:     dict[str, Any] = field(default_factory=dict)
    save_benchmark:    bool = False        # write <stem>_benchmark.json
    benchmark_path:    Path | None = None  # override save path


def style_from_config(viz: dict[str, Any]) -> AnnotateStyle:
    return AnnotateStyle(
        remove_border_color=tuple(viz.get("remove_border_color", [0, 0, 255])),
        remove_border_thickness=viz.get("remove_border_thickness", 8),
        remove_label=viz.get("remove_label", "REMOVE"),
        remove_dim_factor=viz.get("remove_dim_factor", 0.45),
        show_frame_number=viz.get("show_frame_number", True),
        show_stats_banner=viz.get("show_stats_banner", True),
    )


def options_from_config(cfg: dict[str, Any], video: Path | None = None) -> RunOptions:
    resolved_video = video or _path_or_none(cfg.get("input_video"))
    if resolved_video is None:
        raise ValueError("input_video is required in config or as CLI argument")
    out_w, out_h = output_resolution_from_config(cfg)
    return RunOptions(
        video=resolved_video,
        clip_len=cfg.get("clip_len", 16),
        sample_stride=cfg.get("sample_stride", 4),
        conf_delta=cfg.get("conf_delta", 0.15),
        max_gap=cfg.get("max_gap", 30),
        neighbor_pad=cfg.get("neighbor_pad", 2),
        prefer_torch=cfg.get("prefer_torch", True),
        ensemble=cfg.get("ensemble", False),
        audio_spikes=cfg.get("audio_spikes", False),
        audio_rms_z=float(cfg.get("audio_rms_z", 2.5)),
        audio_delta_z=float(cfg.get("audio_delta_z", 2.0)),
        device=cfg.get("device", "auto"),
        inference_scale=cfg.get("inference_scale", 1.0),
        inference_max_side=cfg.get("inference_max_side"),
        benchmark_cfg=cfg.get("benchmark", {}),
        model_path=cfg.get("model_path") or None,
        model_cache_dir=cfg.get("model_cache_dir") or None,
        reencode_h264=cfg.get("reencode_h264", True),
        output_width=out_w,
        output_height=out_h,
        output_fps=float(cfg.get("output_fps") or 10.0),
        plot_correlation=cfg.get("plot_correlation", False),
        chunks_dir=(cfg.get("chunks_dir") or None),
        run_id=cfg.get("run_id", "test-run-12345"),
        camera_id=str(cfg.get("camera_id", "")),
        site_id=str(cfg.get("site_id", "")),
        n_workers=int(cfg.get("n_workers", 1)),
        chunk_duration_sec=float(cfg.get("chunk_duration_sec", 5.0)),
        base_timestamp_ms=cfg.get("base_timestamp_ms"),
        kafka_cfg=cfg.get("kafka", {}),
        visualization=cfg.get("visualization", {}),
        face_recognition_cfg=cfg.get("face_recognition", {}),
    )


# ── Stage-level trace helpers ──────────────────────────────────────────────────

def _hdr(msg: str) -> None:
    """Emit a visually distinct stage header to the log."""
    log.info("── %s %s", msg, "─" * max(0, 60 - len(msg)))


def _stage_done(name: str, elapsed: float, **kv) -> None:
    parts = "  ".join(f"{k}={v}" for k, v in kv.items())
    log.info("%s done  %.1fs  %s", name, elapsed, parts)


def _print_benchmark_summary(b: dict[str, Any]) -> None:
    timing   = b.get("timing_ms", {})
    tp       = b.get("throughput", {}).get("pipeline", {})
    mem      = b.get("memory", {})
    deploy   = b.get("deployment_recommendation", {})
    bottleneck = b.get("bottleneck_phase", "?")

    lines = [
        "── Benchmark summary ────────────────────────────────────────────",
        f"  Total wall time      : {timing.get('total', 0)/1000:.2f} s",
    ]
    for phase in ("filter", "detection", "chunk"):
        ms = timing.get(phase)
        if ms is not None:
            lines.append(f"  {phase:<20} : {ms/1000:.2f} s")
    lines += [
        f"  Bottleneck stage     : {bottleneck}",
        f"  Processing fps       : {tp.get('processing_fps', 0):.1f}",
        f"  ms / frame           : {tp.get('ms_per_frame', 0):.1f}",
        f"  Real-time factor     : {tp.get('realtime_factor', 0):.2f}×  "
        f"({tp.get('interpretation', '')})",
        f"  Peak RAM             : {mem.get('peak_rss_mb', 0):.0f} MB",
        f"  Peak GPU VRAM        : {mem.get('peak_gpu_allocated_mb', 0):.0f} MB",
        f"  CPU avg              : {mem.get('cpu_percent_avg') or '—'} %",
        f"  Deployment tier      : {deploy.get('label', '?')}",
        "─────────────────────────────────────────────────────────────────",
    ]
    for line in lines:
        log.info(line)


def run_action_aware(opts: RunOptions) -> dict[str, Any]:
    """
    Execute enabled pipeline stages in order.

    Stage 1 (filter_enabled):    select action frames → filtered clip
    Stage 2 (detection_enabled): person detect + face recog → annotated clip
    Stage 3 (chunk_enabled):     split → local chunks (+ Kafka if kafka_enabled)
    """
    pipeline_t0 = time.perf_counter()

    if not opts.video.is_file():
        raise FileNotFoundError(f"Video not found: {opts.video}")

    if not any([opts.filter_enabled, opts.detection_enabled, opts.chunk_enabled]):
        raise ValueError(
            "No stages enabled. Pass at least one of: --filter  --detect  --chunk"
        )

    paths = resolve_output_paths(opts.video)
    if opts.output_dir is not None:
        stem  = opts.run_id          # use run_id as filename stem inside --out-dir
        paths = {
            "output_dir":            opts.output_dir,
            "annotated_video":       opts.output_dir / f"{stem}_annotated.mp4",
            "filtered_clip":         opts.output_dir / f"{stem}_filtered.mp4",
            "filter_metadata":       opts.output_dir / f"{stem}_filter_metadata.json",
            "detection_clip":        opts.output_dir / f"{stem}_detection.mp4",
            "benchmark_path":        opts.output_dir / f"{stem}_benchmark.json",
            "report_path":           opts.output_dir / f"{stem}_report.json",
            "kept_indices_path":     opts.output_dir / f"{stem}_kept_indices.json",
            "correlation_plot_path": opts.output_dir / f"{stem}_correlation.png",
            "frames_metadata_path":  opts.output_dir / f"{stem}_frames_metadata.json",
        }
    out_dir = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    bench        = BenchmarkSession()
    meta         = read_video_meta(opts.video)
    current_clip = opts.video     # advances through stages

    # ── Pipeline header ───────────────────────────────────────────────────────
    active_stages = [
        s for s, on in [
            ("filter", opts.filter_enabled),
            ("detect", opts.detection_enabled),
            ("chunk",  opts.chunk_enabled),
            ("kafka",  opts.kafka_enabled),
        ] if on
    ]
    _hdr(f"pipeline: {opts.video.name}")
    log.info(
        "video    %s  %.0ffr  %.1ffps  %.1fs  %dx%d",
        opts.video.name, meta.frame_count or 0,
        meta.fps, meta.duration_sec or 0,
        meta.width, meta.height,
    )
    log.info("stages   %s", " → ".join(active_stages))

    report: dict[str, Any] = {
        "video":  str(opts.video.resolve()),
        "method": "action-aware",
        "stages": {
            "filter":    opts.filter_enabled,
            "detection": opts.detection_enabled,
            "chunk":     opts.chunk_enabled,
            "kafka":     opts.kafka_enabled,
        },
    }

    n_active   = len([s for s in [opts.filter_enabled, opts.detection_enabled, opts.chunk_enabled] if s])
    stage_num  = 0

    # ── Stage 1: Filtering ────────────────────────────────────────────────────
    if opts.filter_enabled:
        stage_num += 1
        _hdr(f"[{stage_num}/{n_active}] filter")
        log.info("config   device=%s  workers=%d  stride=%d  clip_len=%d",
                 opts.device, opts.n_workers, opts.sample_stride, opts.clip_len)

        if opts.filtered_clip:
            filtered_out = opts.filtered_clip
        elif opts.output_dir:
            filtered_out = paths["filtered_clip"]   # already under out_dir
        elif opts.run_id and opts.site_id and opts.camera_id:
            filtered_out = Path(
                f"/jvadata/vst/assets/{opts.site_id}/{opts.camera_id}/{opts.run_id}.mp4"
            )
        else:
            filtered_out = paths["filtered_clip"]
        filtered_out.parent.mkdir(parents=True, exist_ok=True)

        par_tmp = out_dir / "_filter_segs"
        bench.reset_gpu_peak()
        bench.start("filter")
        stage_t0 = time.perf_counter()
        fi = do_filter(opts.video, opts, opts.n_workers, filtered_out, tmp_dir=par_tmp)
        bench.end("filter")
        filter_elapsed = time.perf_counter() - stage_t0

        write_frame_index(
            paths["kept_indices_path"], fi["kept_indices"],
            extra={k: fi[k] for k in ("model", "device", "correlation_timeline", "predictions")},
        )

        # ── Filter metadata JSON ──────────────────────────────────────────────
        total_f  = fi["total_frames"]
        kept_idx = fi["kept_indices"]
        drop_idx = sorted(set(range(total_f)) - set(kept_idx))

        filter_meta = {
            "run_id":               opts.run_id,
            "site_id":              opts.site_id,
            "camera_id":            opts.camera_id,
            "source_video":         str(opts.video.resolve()),
            "filtered_clip":        str(filtered_out.resolve()),
            "model":                fi["model"],
            "device":               fi["device"],
            "inference_resolution": fi["inference_resolution"],
            "source_resolution":    fi["source_resolution"],
            "total_frames":         total_f,
            "kept_frames":          fi["kept_frames"],
            "dropped_frames":       len(drop_idx),
            "reduction_ratio":      fi["reduction_ratio"],
            "processing_ms":        fi["processing_ms"],
            "segments":             fi["segments"],
            "kept_indices":         kept_idx,
            "dropped_indices":      drop_idx,
            "correlation_timeline": fi["correlation_timeline"],
            "predictions":          fi["predictions"],
            "action_changes":       fi["action_changes"],
        }

        if opts.output_dir:
            filter_meta_path = paths["filter_metadata"]   # already under out_dir
        elif opts.run_id and opts.site_id and opts.camera_id:
            filter_meta_path = Path(
                f"/jvadata/vst/assets/{opts.site_id}/{opts.camera_id}/{opts.run_id}_metadata.json"
            )
        else:
            filter_meta_path = paths["filter_metadata"]

        filter_meta_path.parent.mkdir(parents=True, exist_ok=True)
        filter_meta_path.write_text(json.dumps(filter_meta, indent=2), encoding="utf-8")

        if opts.plot_correlation:
            plot_path = opts.correlation_plot_path or paths["correlation_plot_path"]
            plot_correlation_timeline(
                fi["correlation_timeline"], plot_path,
                title=f"Action correlation — {opts.video.stem}",
                show=opts.correlation_show,
            )

        report["filter"] = {
            "output":           str(filtered_out.resolve()),
            "metadata_path":    str(filter_meta_path.resolve()),
            "total_frames":     total_f,
            "kept_frames":      fi["kept_frames"],
            "removed_frames":   total_f - fi["kept_frames"],
            "reduction_ratio":  fi["reduction_ratio"],
            "processing_ms":    fi["processing_ms"],
            "model":            fi["model"],
            "device":           fi["device"],
            "segments":         fi["segments"],
            "reencoded_h264":   fi["reencoded_h264"],
            "fps":              fi.get("fps", 0.0),
            "duration_sec":     fi.get("duration_sec", 0.0),
            "output_resolution":fi.get("output_resolution", ""),
            **({"wall_sec": fi["wall_sec"]} if "wall_sec" in fi else {}),
        }
        current_clip = filtered_out
        ratio = total_f / fi["kept_frames"] if fi["kept_frames"] else 0.0
        _stage_done(
            "filter", filter_elapsed,
            kept=f"{fi['kept_frames']}/{total_f}",
            ratio=f"{ratio:.1f}×",
            model=fi["model"],
            device=fi["device"],
        )
        log.info("filter   output → %s", filtered_out)

    # ── Stage 2: Detection ────────────────────────────────────────────────────
    if opts.detection_enabled:
        stage_num += 1
        _hdr(f"[{stage_num}/{n_active}] detect")
        fr_cfg = opts.face_recognition_cfg
        log.info("config   site=%s  frame_skip=%d  sim_thresh=%.2f",
                 fr_cfg.get("site_id", ""), fr_cfg.get("frame_skip", "?"),
                 fr_cfg.get("similarity_threshold", 0.45))

        if not current_clip.is_file():
            raise FileNotFoundError(
                f"Detection stage: input clip not found: {current_clip}\n"
                "Run --filter first or point to an existing clip."
            )
        from fr_annotate import annotate_video
        detection_out = opts.detection_clip or paths["detection_clip"]
        bench.start("detection")
        stage_t0 = time.perf_counter()
        det_info = annotate_video(
            str(current_clip),
            str(detection_out),
            opts.face_recognition_cfg,
        )
        bench.end("detection")
        det_elapsed = time.perf_counter() - stage_t0
        report["detection"] = {
            "output": str(detection_out.resolve()),
            **det_info,
        }
        current_clip = detection_out
        _stage_done(
            "detect", det_elapsed,
            frames=det_info.get("total_frames", "?"),
            recognised=det_info.get("recognised_draws", "?"),
            unknown=det_info.get("unknown_draws", "?"),
        )

    # ── Stage 3: Chunking ─────────────────────────────────────────────────────
    if opts.chunk_enabled:
        stage_num += 1
        _hdr(f"[{stage_num}/{n_active}] chunk")
        log.info("config   duration=%.0fs  kafka=%s  site=%s  camera=%s  run=%s",
                 opts.chunk_duration_sec, opts.kafka_enabled,
                 opts.site_id, opts.camera_id, opts.run_id)

        if not current_clip.is_file():
            raise FileNotFoundError(
                f"Chunk stage: input clip not found: {current_clip}\n"
                "Run --filter or --detection first."
            )
        if opts.kafka_enabled and (not opts.camera_id or not opts.site_id):
            raise ValueError(
                "camera_id and site_id are required when --kafka is enabled."
            )

        # When out_dir is specified, use it as the primary chunk destination.
        if opts.output_dir and not opts.chunks_dir:
            opts.chunks_dir = str(opts.output_dir)

        # Merge kafka_cfg with CLI overrides for sp/critic
        kafka_overrides: dict[str, Any] = {**opts.kafka_cfg, "enabled": bool(opts.kafka_enabled)}
        if opts.kafka_sp_enabled is not None:
            kafka_overrides["sp_enabled"] = opts.kafka_sp_enabled
        if opts.kafka_critic_enabled is not None:
            kafka_overrides["critic_enabled"] = opts.kafka_critic_enabled
        if opts.kafka_sp is not None:
            kafka_overrides["sp"] = opts.kafka_sp
        if opts.kafka_critic is not None:
            kafka_overrides["critic"] = opts.kafka_critic

        frames_meta_path = paths["frames_metadata_path"]

        bench.start("chunk")
        stage_t0 = time.perf_counter()
        chunk_report = split_and_publish_chunks(
            current_clip,
            site_id=opts.site_id,
            camera_id=opts.camera_id,
            run_id=opts.run_id,
            chunk_duration_sec=opts.chunk_duration_sec,
            chunk_width=opts.chunk_width,
            chunk_height=opts.chunk_height,
            output_fps=opts.output_fps,
            base_timestamp_ms=opts.base_timestamp_ms,
            reencode_h264=opts.reencode_h264,
            full_clip_dest=opts.full_clip_dest,
            chunks_dir=opts.chunks_dir,
            kept_indices=None,
            source_fps=meta.fps,
            frames_meta_path=str(frames_meta_path),
            kafka_overrides=kafka_overrides,
        )
        bench.end("chunk")
        chunk_elapsed = time.perf_counter() - stage_t0
        report["chunk"] = chunk_report
        _stage_done(
            "chunk", chunk_elapsed,
            chunks=chunk_report.get("total_chunks", "?"),
            published=chunk_report.get("published_chunks", "?"),
            failed=chunk_report.get("failed_chunks", "?"),
        )

    # ── Optional output clip copy ─────────────────────────────────────────────
    if opts.output_clip and current_clip.is_file():
        import shutil as _shutil
        dst = opts.output_clip
        dst.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(current_clip, dst)
        from common.video_io import try_reencode_h264
        try_reencode_h264(dst)
        report["output_clip"] = str(dst.resolve())
        log.info("output clip saved → %s", dst)

    # ── Benchmark & report ────────────────────────────────────────────────────
    if opts.benchmark_enabled:
        # save_benchmark (or an explicit --benchmark-out path) → write standalone file
        bm_save_path = (
            opts.benchmark_path or paths["benchmark_path"]
            if (opts.save_benchmark or opts.benchmark_path)
            else None
        )
        merge_benchmark_into_report(
            report, bench,
            video_meta=meta,
            method="action-aware",
            selector_model=report.get("filter", {}).get("model"),
            benchmark_cfg=opts.benchmark_cfg,
            benchmark_path=bm_save_path,
        )
        _print_benchmark_summary(report["benchmark"])

    report_path = paths["report_path"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"]  = str(report_path.resolve())
    report["output_paths"] = {k: str(v.resolve()) for k, v in paths.items()
                               if isinstance(v, Path)}

    pipeline_elapsed = time.perf_counter() - pipeline_t0
    _hdr(f"pipeline done  {pipeline_elapsed:.1f}s total")

    return report


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
