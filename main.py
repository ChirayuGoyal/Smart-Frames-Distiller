#!/usr/bin/env python3
"""
Action-aware pipeline — 3 stages, each enabled with true/false.

  Original Clip
    --filter true   →  Filtered Clip  (action-aware selection)
    --detect true   →  Detection Clip (person boxes + face names)
    --chunk  true   →  N-second chunks
    --kafka  true   →  Publish chunks to Kafka (requires --chunk true)

── Quick examples ────────────────────────────────────────────
Filter only:
  python3 main.py N2.mp4 --filter true --site site-001 --camera cam-001 --run abc-123

Filter + detect:
  python3 main.py N2.mp4 --filter true --detect true --site site-001 --camera cam-001 --run abc-123

Full pipeline:
  python3 main.py N2.mp4 --filter true --detect true --chunk true --kafka true \\
    --site site-001 --camera cam-001 --run abc-123

Chunk only (no filter, no detection, no Kafka):
  python3 main.py N2.mp4 --chunk true --site site-001 --camera cam-001 --run abc-123 --chunks-dir ./out
──────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from runner import load_config, options_from_config, run_action_aware

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

log = logging.getLogger(__name__)

_TRUE_VALS  = {"true",  "1", "yes", "on",  "y"}
_FALSE_VALS = {"false", "0", "no",  "off", "n"}


def _bool(v: str) -> bool:
    low = str(v).lower()
    if low in _TRUE_VALS:
        return True
    if low in _FALSE_VALS:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got '{v}'")


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with a clean timestamp format on stderr."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-5s  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    # Quiet noisy third-party loggers that flood at INFO
    for noisy in ("PIL", "matplotlib", "urllib3", "confluent_kafka"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Action-aware video pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input / config ────────────────────────────────────────────────────────
    p.add_argument("video", nargs="?",
                   help="Input video file or RTSP stream URL  (rtsp://...)")
    p.add_argument("-c", "--config", type=Path,
                   default=Path(__file__).parent / "config.json",
                   help="Config JSON (default: config.json)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output folder (default: <video_dir>/action_aware_output/)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show DEBUG-level logs")

    # ── Stage flags ───────────────────────────────────────────────────────────
    stages = p.add_argument_group("Stages  (pass true/false for each)")
    stages.add_argument("--filter", type=_bool, default=False, metavar="true|false",
                        help="Stage 1 — action-aware filtering → filtered clip")
    stages.add_argument("--detect", type=_bool, default=False, metavar="true|false",
                        help="Stage 2 — person detection + face recognition → annotated clip")
    stages.add_argument("--chunk",  type=_bool, default=False, metavar="true|false",
                        help="Stage 3 — split into N-second chunks")
    stages.add_argument("--kafka",  type=_bool, default=False, metavar="true|false",
                        help="Publish chunks to Kafka (requires --chunk true)")

    # ── Identity (shared across stages) ──────────────────────────────────────
    ident = p.add_argument_group("Identity  (required for most stages)")
    ident.add_argument("--site",   default=None, metavar="SITE_ID",
                       help="Site identifier  e.g. site-001")
    ident.add_argument("--camera", default=None, metavar="CAMERA_ID",
                       help="Camera identifier  e.g. cam-001")
    ident.add_argument("--run",    default=None, metavar="RUN_ID",
                       help="Run UUID for this clip  e.g. abc-123")

    # ── Filter options ────────────────────────────────────────────────────────
    flt = p.add_argument_group("Filter options")
    flt.add_argument("--width",          type=int,   default=None,
                     help="Filtered clip width px  (default 640)")
    flt.add_argument("--height",         type=int,   default=None,
                     help="Filtered clip height px  (default 480)")
    flt.add_argument("--fps",            type=float, default=None,
                     help="Output FPS (default: same as input)")
    flt.add_argument("--clip-len",       type=int,   default=None,
                     help="Frames per clip window  (default 16)")
    flt.add_argument("--stride",         type=int,   default=None,
                     help="Frame sampling stride  (default 4)")
    flt.add_argument("--conf-delta",     type=float, default=None,
                     help="Action confidence threshold delta  (default 0.15)")
    flt.add_argument("--max-gap",        type=int,   default=None,
                     help="Max gap between kept segments in frames  (default 30)")
    flt.add_argument("--device",         choices=["auto", "cpu", "cuda"], default=None,
                     help="Inference device  (default auto)")
    flt.add_argument("--workers",        type=int,   default=None, metavar="N",
                     help="Parallel filter workers — split video into N segments and filter "
                          "each in a separate process (default: config n_workers or 1). "
                          "Try 4 for long clips on a multi-core server.")
    flt.add_argument("--no-torch",       action="store_true",
                     help="Use motion-energy fallback instead of R3D-18")
    flt.add_argument("--ensemble",        action="store_true",
                     help="Run R3D-18 AND MotionEnergy together — keep frames "
                          "flagged by either model (OR-logic triggers)")
    flt.add_argument("--audio-spikes",   type=_bool, default=None, metavar="true|false",
                     help="Keep frames near audio energy spikes "
                          "(loud events + sudden energy transitions)")
    flt.add_argument("--audio-rms-z",    type=float, default=None, metavar="FLOAT",
                     help="Z-score for loud events; keep if RMS > mean + z·std  (default 2.5)")
    flt.add_argument("--audio-delta-z",  type=float, default=None, metavar="FLOAT",
                     help="Z-score for sudden changes; keep if |ΔRMS| > mean + z·std  (default 2.0)")
    flt.add_argument("--max-side",       type=int,   default=None,
                     help="Max px on longest inference side  (default 480)")
    flt.add_argument("--scale",          type=float, default=None,
                     help="Inference scale factor  e.g. 0.5")

    # ── Chunk options ─────────────────────────────────────────────────────────
    chk = p.add_argument_group("Chunk options")
    chk.add_argument("--duration",       type=float, default=None,
                     help="Chunk length in seconds  (default 5)")
    chk.add_argument("--chunk-width",    type=int,   default=None,
                     help="Chunk output width px  (default: same as input)")
    chk.add_argument("--chunk-height",   type=int,   default=None,
                     help="Chunk output height px  (default: same as input)")
    chk.add_argument("--chunks-dir",     type=Path,  default=None,
                     help="Save chunks as UUID .mp4 files in this folder")
    chk.add_argument("--save-clip",      type=Path,  default=None,
                     help="Copy entire filtered clip to this exact path")
    chk.add_argument("--base-ts",        type=int,   default=None,
                     help="Epoch ms for first chunk start  (default: now)")

    # ── Kafka message overrides ───────────────────────────────────────────────
    kfk = p.add_argument_group("Kafka message overrides  (require --kafka true)")
    kfk.add_argument("--sp-enabled",     type=_bool, default=None, metavar="true|false",
                     help="Smart-processor enabled flag in Kafka message  (default: from config)")
    kfk.add_argument("--critic-enabled", type=_bool, default=None, metavar="true|false",
                     help="Critic enabled flag in Kafka message  (default: from config)")
    kfk.add_argument("--sp",             default=None, metavar="VALUE",
                     help="alert_level.sp value in Kafka message  (default: inherits --sp-enabled)")
    kfk.add_argument("--critic",         default=None, metavar="VALUE",
                     help="alert_level.critic value in Kafka message  (default: inherits --critic-enabled)")

    # ── Output overrides ─────────────────────────────────────────────────────
    ovr = p.add_argument_group("Output path overrides")
    ovr.add_argument("--filtered-clip",  type=Path, default=None,
                     help="Override stage-1 output path")
    ovr.add_argument("--detection-clip", type=Path, default=None,
                     help="Override stage-2 output path")
    ovr.add_argument("--output-clip",    type=Path, default=None,
                     help="Copy final stage output to this path (web-compatible H.264)")
    ovr.add_argument("--benchmark",       type=_bool, default=False, metavar="true|false",
                     help="Collect, print and save benchmark metrics to "
                          "<out_dir>/<stem>_benchmark.json")
    ovr.add_argument("--benchmark-out",  type=Path, default=None,
                     help="Custom path for benchmark JSON (implies --save-benchmark)")
    ovr.add_argument("--plot-correlation", action="store_true")

    args = p.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    setup_logging(verbose=args.verbose)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg       = load_config(args.config) if args.config.is_file() else {}
    video_raw = str(args.video or cfg.get("input_video", "") or "").strip()
    is_rtsp   = video_raw.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))

    if not video_raw:
        p.error(
            "No video input. Pass a file path, rtsp:// URL, or http(s):// stream URL, "
            "or set input_video in config.json"
        )
    elif not is_rtsp and not Path(video_raw).is_file():
        p.error(
            f"Video not found: '{video_raw}'. "
            "Pass a file path, rtsp:// URL, or http(s):// stream URL, "
            "or set input_video in config.json"
        )

    video    = None if is_rtsp else Path(video_raw)
    rtsp_url = video_raw if is_rtsp else None

    if args.out_dir is not None and args.out_dir.suffix.lower() in _VIDEO_SUFFIXES:
        p.error(f"--out-dir must be a folder, not a video file: {args.out_dir}")

    # ── Validate stages ───────────────────────────────────────────────────────
    if is_rtsp:
        if args.detect:
            log.warning("--detect is not supported for RTSP streams — it will be skipped")
        if args.kafka and not (args.camera or cfg.get("camera_id")):
            p.error("--camera CAMERA_ID is required when using --kafka with RTSP")
        if args.kafka and not (args.run or cfg.get("run_id")):
            p.error("--run RUN_ID is required when using --kafka with RTSP")
    else:
        if not any([args.filter, args.detect, args.chunk]):
            p.error("No stage selected — pass at least one of:  --filter  --detect  --chunk")
        if args.kafka and not args.chunk:
            p.error("--kafka true requires --chunk true")
        if args.detect and not (args.site or cfg.get("face_recognition", {}).get("site_id")):
            p.error("--site SITE_ID is required when using --detect")
        if args.filter and not (args.run or cfg.get("run_id")):
            p.error("--run RUN_ID is required when using --filter")
        if args.kafka and not (args.camera or cfg.get("camera_id")):
            p.error("--camera CAMERA_ID is required when using --kafka")

    # ── Build options ─────────────────────────────────────────────────────────
    opts = options_from_config(cfg, video=video or Path("."))

    # Overrides
    opts.output_dir        = args.out_dir
    opts.filtered_clip     = args.filtered_clip
    opts.detection_clip    = args.detection_clip
    opts.output_clip       = args.output_clip
    # --benchmark-out implies benchmarking (its help says so)
    opts.benchmark_enabled = args.benchmark or args.benchmark_out is not None
    opts.save_benchmark    = args.benchmark or args.benchmark_out is not None
    opts.benchmark_path    = args.benchmark_out

    # Stage flags
    opts.filter_enabled    = args.filter
    opts.detection_enabled = args.detect
    opts.chunk_enabled     = args.chunk
    opts.kafka_enabled     = args.kafka

    # Identity
    if args.site:
        opts.site_id = args.site
        opts.face_recognition_cfg = {**opts.face_recognition_cfg, "site_id": args.site}
    if args.camera: opts.camera_id          = args.camera
    if args.run:    opts.run_id             = args.run

    # Filter
    # Only override config when the flag was actually passed
    if args.workers is not None:
        opts.n_workers = max(1, args.workers)
    if args.width:       opts.output_width        = args.width
    if args.height:      opts.output_height       = args.height
    if args.fps:         opts.output_fps          = args.fps
    if args.clip_len:    opts.clip_len            = args.clip_len
    if args.stride:      opts.sample_stride       = args.stride
    if args.conf_delta:  opts.conf_delta          = args.conf_delta
    if args.max_gap:     opts.max_gap             = args.max_gap
    if args.device:      opts.device              = args.device
    if args.no_torch:    opts.prefer_torch        = False
    if args.ensemble:       opts.ensemble            = True
    if args.audio_spikes is not None:
        opts.audio_spikes = args.audio_spikes
    if args.audio_rms_z   is not None: opts.audio_rms_z   = args.audio_rms_z
    if args.audio_delta_z is not None: opts.audio_delta_z = args.audio_delta_z
    if args.max_side:    opts.inference_max_side  = args.max_side
    if args.scale:       opts.inference_scale     = args.scale
    opts.plot_correlation = args.plot_correlation or opts.plot_correlation

    # Chunk
    if args.duration:      opts.chunk_duration_sec = args.duration
    if args.chunk_width:   opts.chunk_width        = args.chunk_width
    if args.chunk_height:  opts.chunk_height       = args.chunk_height
    if args.chunks_dir:    opts.chunks_dir         = str(args.chunks_dir)
    if args.save_clip:     opts.full_clip_dest     = str(args.save_clip)
    if args.base_ts:       opts.base_timestamp_ms  = args.base_ts

    # Kafka sp / critic overrides
    if args.sp_enabled is not None:
        opts.kafka_sp_enabled = "true" if args.sp_enabled else "false"
    if args.critic_enabled is not None:
        opts.kafka_critic_enabled = "true" if args.critic_enabled else "false"
    if args.sp is not None:
        opts.kafka_sp = args.sp
    if args.critic is not None:
        opts.kafka_critic = args.critic

    # ── Run ───────────────────────────────────────────────────────────────────
    if is_rtsp:
        from rtsp_stream import RTSPStreamProcessor
        kafka_overrides: dict[str, Any] = {**opts.kafka_cfg, "enabled": bool(args.kafka)}
        if opts.kafka_sp_enabled is not None:
            kafka_overrides["sp_enabled"] = opts.kafka_sp_enabled
        if opts.kafka_critic_enabled is not None:
            kafka_overrides["critic_enabled"] = opts.kafka_critic_enabled
        if opts.kafka_sp is not None:
            kafka_overrides["sp"] = opts.kafka_sp
        if opts.kafka_critic is not None:
            kafka_overrides["critic"] = opts.kafka_critic
        opts.kafka_enabled = args.kafka
        report = RTSPStreamProcessor(
            rtsp_url, opts, kafka_overrides=kafka_overrides
        ).run()
    else:
        report = run_action_aware(opts)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
