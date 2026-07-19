"""
Parallel action-aware filter.

Split the source video into N equal-duration segments, run the action-aware
filter on each segment in a separate process, then concatenate the filtered
segment clips into a single web-compatible H.264 output.

Why processes (not threads)?
  - PyTorch / ONNX runtime both release the GIL during inference, but loading
    models and doing heavy preprocessing is still CPU-bound.
  - Each process gets its own GPU context (safe for CUDA, no shared-state issues).
  - On Linux (server), 'spawn' start-method avoids CUDA-after-fork bugs.

Usage:
  from parallel_filter import do_filter
  fi = do_filter(video, opts, n_workers=4, output_path=..., tmp_dir=...)
"""

from __future__ import annotations

import logging
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

_MIN_SEG_SEC = 15.0   # don't create segments shorter than this


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def _ffmpeg_split(video: Path, start_sec: float, duration_sec: float, dst: Path) -> None:
    """Extract one time segment, frame-accurately.

    Re-encodes (libx264 ultrafast) instead of stream-copying: -c copy cuts on
    keyframes, so segment frame counts don't sum to the source total and every
    downstream kept-index offset drifts. The encode cost buys exact indices.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.6f}",
            "-i", str(video),
            "-t", f"{duration_sec:.6f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-c:a", "copy",
            str(dst),
        ],
        capture_output=True, timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg split failed (rc={r.returncode}):\n{r.stderr.decode(errors='replace')}"
        )


def _ffmpeg_concat(clips: list[Path], output: Path) -> None:
    """Concatenate clips into a web-compatible H.264 MP4 (single encode pass)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lst = output.with_suffix(".concat_list.txt")
    try:
        # concat-demuxer quoting: single quotes in the path are escaped as '\''
        # and paths use forward slashes so Windows backslashes survive.
        lst.write_text(
            "\n".join(
                "file '{}'".format(p.resolve().as_posix().replace("'", "'\\''"))
                for p in clips
            ),
            encoding="utf-8",
        )
        r = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", str(lst),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(output),
            ],
            capture_output=True, timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed (rc={r.returncode}):\n{r.stderr.decode(errors='replace')}"
            )
    finally:
        lst.unlink(missing_ok=True)


# ── Segment worker (module-level so ProcessPoolExecutor can pickle it) ─────────

def _filter_segment_worker(args: dict) -> dict:
    """
    Filter one video segment in a subprocess.
    Returns a result dict consumed by the parent process.
    """
    seg_path = Path(args["seg_path"])
    out_path = Path(args["out_path"])
    idx      = args["seg_idx"]
    o        = args["opts"]

    # Restore sys.path so local modules are importable in the spawned process.
    for p in args.get("sys_paths", []):
        if p not in sys.path:
            sys.path.insert(0, p)

    import logging as _lg
    _lg.basicConfig(
        level=_lg.INFO,
        format=f"[worker-{idx}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from selector import ActionAwareSelector
    from common.video_io import write_kept_video

    _lg.getLogger(__name__).info("Starting segment %d: %s", idx, seg_path.name)
    t0 = time.perf_counter()

    sel = ActionAwareSelector(
        clip_len=o["clip_len"],
        sample_stride=o["sample_stride"],
        conf_delta=o["conf_delta"],
        max_gap=o["max_gap"],
        neighbor_pad=o["neighbor_pad"],
        prefer_torch=o["prefer_torch"],
        device=o["device"],
        inference_scale=o["inference_scale"],
        inference_max_side=o.get("inference_max_side"),
        ensemble=o.get("ensemble", False),
        audio_spikes=o.get("audio_spikes", False),
        audio_rms_z=o.get("audio_rms_z", 2.5),
        audio_delta_z=o.get("audio_delta_z", 2.0),
        model_path=o.get("model_path"),
        model_cache_dir=o.get("model_cache_dir"),
    )
    result = sel.select(seg_path)
    elapsed = time.perf_counter() - t0

    total_f = result.stats.total_frames if result.stats else 0
    kept_c  = result.stats.kept_frames  if result.stats else 0
    kept_idx = list(result.kept_indices)

    filtered_path: str | None = None
    if kept_c > 0:
        # Write raw mp4v — the parent will re-encode once during concat.
        write_kept_video(
            seg_path, out_path, kept_idx,
            fps=o.get("output_fps"),
            reencode_h264=False,
            output_width=o.get("output_width"),
            output_height=o.get("output_height"),
        )
        filtered_path = str(out_path)

    _lg.getLogger(__name__).info(
        "Segment %d done: %d/%d frames kept (%.1fs)", idx, kept_c, total_f, elapsed
    )
    return {
        "seg_idx":        idx,
        "filtered_path":  filtered_path,
        "kept_indices":   kept_idx,
        "total_frames":   total_f,
        "kept_frames":    kept_c,
        "processing_sec": round(elapsed, 2),
        "metadata":       result.metadata,
        "events":         list(getattr(result, "events", [])),
    }


# ── Sequential fallback ───────────────────────────────────────────────────────

def _sequential(video: Path, opts, output_path: Path) -> dict[str, Any]:
    from selector import ActionAwareSelector
    from common.video_io import write_kept_video

    sel = ActionAwareSelector(
        clip_len=opts.clip_len,
        sample_stride=opts.sample_stride,
        conf_delta=opts.conf_delta,
        max_gap=opts.max_gap,
        neighbor_pad=opts.neighbor_pad,
        prefer_torch=opts.prefer_torch,
        device=opts.device,
        inference_scale=opts.inference_scale,
        inference_max_side=opts.inference_max_side,
        ensemble=getattr(opts, "ensemble", False),
        audio_spikes=getattr(opts, "audio_spikes", False),
        audio_rms_z=getattr(opts, "audio_rms_z", 2.5),
        audio_delta_z=getattr(opts, "audio_delta_z", 2.0),
        model_path=getattr(opts, "model_path", None),
        model_cache_dir=getattr(opts, "model_cache_dir", None),
        model=getattr(opts, "action_model", None),
        progress_cb=getattr(opts, "filter_progress_cb", None),
        cancel_check=getattr(opts, "filter_cancel_check", None),
    )
    result = sel.select(video)
    compress = write_kept_video(
        video, output_path, result.kept_indices,
        fps=getattr(opts, "output_fps", None),
        reencode_h264=opts.reencode_h264,
        output_width=opts.output_width,
        output_height=opts.output_height,
    )
    s = result.stats
    m = result.metadata
    return {
        "kept_indices":         list(result.kept_indices),
        "total_frames":         s.total_frames if s else 0,
        "kept_frames":          s.kept_frames  if s else 0,
        "reduction_ratio":      round(s.reduction_ratio, 2) if s else 0.0,
        "processing_ms":        round(s.processing_ms, 1)  if s else 0.0,
        "model":                m.get("model"),
        "device":               m.get("device"),
        "inference_resolution": m.get("inference_resolution"),
        "source_resolution":    m.get("source_resolution"),
        "correlation_timeline": m.get("correlation_timeline", []),
        "predictions":          m.get("predictions", []),
        "action_changes":       len(getattr(result, "events", [])),
        "segments":             1,
        "reencoded_h264":       compress.get("reencoded_h264", False),
        "fps":                  compress.get("fps", 0.0),
        "duration_sec":         compress.get("duration_sec", 0.0),
        "output_resolution":    compress.get("output_resolution", ""),
    }


# ── Parallel implementation ───────────────────────────────────────────────────

def _parallel(
    video: Path,
    opts,
    n_workers: int,
    output_path: Path,
    tmp_dir: Path,
) -> dict[str, Any]:
    from common.video_io import read_video_meta

    meta = read_video_meta(video)
    dur  = meta.duration_sec

    # Clamp workers so no segment is shorter than _MIN_SEG_SEC
    max_w    = max(1, int(dur // _MIN_SEG_SEC))
    n_workers = min(n_workers, max_w)

    if n_workers == 1:
        log.info("Video too short for parallel split (%.0fs) — running sequentially.", dur)
        return _sequential(video, opts, output_path)

    log.info(
        "Parallel filter: %d workers × ~%.0fs segments  (total %.0fs)",
        n_workers, dur / n_workers, dur,
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Split ─────────────────────────────────────────────────────────────────
    seg_dur   = dur / n_workers
    seg_paths: list[Path] = []
    for i in range(n_workers):
        sp    = tmp_dir / f"seg_{i:03d}_in.mp4"
        start = i * seg_dur
        d     = seg_dur if i < n_workers - 1 else (dur - start)
        log.info("  split seg %d  %.1fs – %.1fs  → %s", i, start, start + d, sp.name)
        _ffmpeg_split(video, start, d, sp)
        seg_paths.append(sp)

    # ── Build worker args ─────────────────────────────────────────────────────
    here     = str(Path(__file__).resolve().parent)
    # Preserve existing sys.path entries + prepend repo root (Windows spawn needs it)
    sys_paths = list(dict.fromkeys([here] + sys.path))

    opts_dict: dict[str, Any] = {
        "clip_len":          opts.clip_len,
        "sample_stride":     opts.sample_stride,
        "conf_delta":        opts.conf_delta,
        "max_gap":           opts.max_gap,
        "neighbor_pad":      opts.neighbor_pad,
        "prefer_torch":      opts.prefer_torch,
        "ensemble":          getattr(opts, "ensemble", False),
        "audio_spikes":      getattr(opts, "audio_spikes", False),
        "audio_rms_z":       getattr(opts, "audio_rms_z", 2.5),
        "audio_delta_z":     getattr(opts, "audio_delta_z", 2.0),
        "device":            opts.device,
        "inference_scale":   opts.inference_scale,
        "inference_max_side":opts.inference_max_side,
        "output_width":      opts.output_width,
        "output_height":     opts.output_height,
        "output_fps":        getattr(opts, "output_fps", None),
        "model_path":        getattr(opts, "model_path", None),
        "model_cache_dir":   getattr(opts, "model_cache_dir", None),
    }
    worker_args = [
        {
            "seg_idx":   i,
            "seg_path":  str(seg_paths[i]),
            "out_path":  str(tmp_dir / f"seg_{i:03d}_filtered.mp4"),
            "opts":      opts_dict,
            "sys_paths": sys_paths,
        }
        for i in range(n_workers)
    ]

    # ── Run in parallel ───────────────────────────────────────────────────────
    # Use 'spawn' to avoid CUDA-after-fork bugs on Linux.
    mp_ctx = multiprocessing.get_context("spawn")
    wall_t0 = time.perf_counter()
    seg_results: list[dict | None] = [None] * n_workers

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        futs = {pool.submit(_filter_segment_worker, wa): wa["seg_idx"] for wa in worker_args}
        completed_iter = as_completed(futs)
        if _TQDM:
            completed_iter = _tqdm(
                completed_iter,
                total=n_workers,
                unit="seg", desc="  filter",
                dynamic_ncols=True, leave=True,
            )
        for fut in completed_iter:
            i = futs[fut]
            try:
                seg_results[i] = fut.result()
                r = seg_results[i]
                log.info(
                    "  seg %d done: %d/%d frames kept (%.1fs)",
                    i, r["kept_frames"], r["total_frames"], r["processing_sec"],
                )
            except Exception as exc:
                log.error("  seg %d FAILED: %s", i, exc, exc_info=True)
                raise

    wall_elapsed = time.perf_counter() - wall_t0
    log.info("All segments done in %.1f s wall time.", wall_elapsed)

    # ── Collect filtered clips (skip empty segments) ──────────────────────────
    filtered_clips = [
        Path(r["filtered_path"])
        for r in seg_results
        if r and r["filtered_path"] and Path(r["filtered_path"]).is_file()
    ]
    if not filtered_clips:
        raise RuntimeError("All segments produced 0 kept frames — nothing to output.")

    # ── Global kept_indices ───────────────────────────────────────────────────
    # Exact: _ffmpeg_split is frame-accurate (re-encode), so per-segment decoded
    # totals sum to the source total and local→global offsets are precise.
    global_kept: list[int] = []
    frame_offset = 0
    for r in seg_results:
        if r:
            for li in r["kept_indices"]:
                global_kept.append(frame_offset + li)
            frame_offset += r["total_frames"]

    # ── Concat + single-pass web-compatible encode ────────────────────────────
    log.info("Concatenating %d clip(s) → %s", len(filtered_clips), output_path.name)
    _ffmpeg_concat(filtered_clips, output_path)

    # ── Cleanup temp files ────────────────────────────────────────────────────
    for sp in seg_paths:
        sp.unlink(missing_ok=True)
    for fc in filtered_clips:
        fc.unlink(missing_ok=True)
    try:
        if tmp_dir.is_dir() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass

    # ── Aggregate stats ───────────────────────────────────────────────────────
    total_f   = sum(r["total_frames"] for r in seg_results if r)
    total_kept = sum(r["kept_frames"] for r in seg_results if r)
    first_m   = next((r["metadata"] for r in seg_results if r), {})
    all_events: list = []
    for r in seg_results:
        if r:
            all_events.extend(r.get("events", []))

    try:
        from common.video_io import read_video_meta as _rvm
        out_m  = _rvm(output_path)
        out_fps, out_dur, out_res = out_m.fps, out_m.duration_sec, f"{out_m.width}x{out_m.height}"
    except Exception:
        out_fps, out_dur, out_res = meta.fps, 0.0, ""

    return {
        "kept_indices":         global_kept,
        "total_frames":         total_f,
        "kept_frames":          total_kept,
        "reduction_ratio":      round(total_f / total_kept, 2) if total_kept else 0.0,
        "processing_ms":        round(wall_elapsed * 1000, 1),
        "model":                first_m.get("model"),
        "device":               first_m.get("device"),
        "inference_resolution": first_m.get("inference_resolution"),
        "source_resolution":    first_m.get("source_resolution"),
        "correlation_timeline": first_m.get("correlation_timeline", []),
        "predictions":          first_m.get("predictions", []),
        "action_changes":       len(all_events),
        "segments":             n_workers,
        "reencoded_h264":       True,
        "fps":                  out_fps,
        "duration_sec":         out_dur,
        "output_resolution":    out_res,
        "wall_sec":             round(wall_elapsed, 2),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def do_filter(
    video: Path,
    opts,            # RunOptions — accessed read-only
    n_workers: int,
    output_path: Path,
    tmp_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run the action-aware filter stage.

    n_workers = 1  → sequential (original behaviour, lower overhead)
    n_workers > 1  → split → parallel filter → concat

    Returns a FilterInfo dict with all fields runner.py needs.
    """
    if n_workers > 1:
        if getattr(opts, "cancel_event", None) is not None:
            log.info("Server cancellation requires sequential filtering; using one worker")
            n_workers = 1

    if n_workers > 1:
        # Each worker spawns its own CUDA context and loads R3D-18 into VRAM
        # independently. On a shared GPU this exhausts VRAM very quickly.
        # Parallelism only makes sense when workers use CPU inference.
        device = getattr(opts, "device", "auto")
        if device in ("auto", "cuda"):
            log.warning(
                "GPU mode: %d parallel workers would each load R3D-18 into VRAM "
                "independently — capping to 1 worker to avoid OOM. "
                "Pass --device cpu to use multi-worker CPU parallelism.",
                n_workers,
            )
            n_workers = 1

    if n_workers > 1:
        td = tmp_dir or output_path.parent / "_filter_segs"
        return _parallel(video, opts, n_workers, output_path, td)
    return _sequential(video, opts, output_path)
