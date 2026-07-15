#!/usr/bin/env python3
"""
Action-aware benchmark runner — compute cost & edge requirements per run.

Profiles selection only (no video encode) or full pipeline depending on flags.

Usage:
  python benchmark.py video.mp4
  python benchmark.py -c config.json
  python benchmark.py video.mp4 --compare-torch --compare-cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runner import load_config, options_from_config, resolve_output_paths, run_action_aware


def _print_summary(label: str, report: dict) -> None:
    bench = report.get("benchmark", {})
    tp = bench.get("throughput", {}).get("selection_only", {})
    edge = bench.get("edge_compatibility", [])
    feasible = [e for e in edge if e.get("realtime_feasible")]

    summary = {
        "label": label,
        "model": report.get("model"),
        "video": report.get("video"),
        "total_frames": report.get("total_frames"),
        "reduction_ratio": report.get("reduction_ratio"),
        "ms_per_frame_selection": tp.get("ms_per_frame"),
        "processing_fps": tp.get("processing_fps"),
        "realtime_factor": bench.get("throughput", {}).get("pipeline", {}).get("realtime_factor"),
        "peak_rss_mb": bench.get("memory", {}).get("peak_rss_mb"),
        "peak_gpu_mb": bench.get("memory", {}).get("peak_gpu_allocated_mb"),
        "deployment": bench.get("deployment_recommendation"),
        "edge_devices_ok": [e["device_label"] for e in feasible],
        "compute_cost": bench.get("compute_cost"),
        "system_requirements": bench.get("system_requirements"),
        "benchmark_path": report.get("benchmark_path"),
    }
    print(json.dumps(summary, indent=2))


def _run_profile(
    video: Path,
    cfg: dict,
    *,
    label: str,
    prefer_torch: bool,
    out_path: Path | None,
) -> dict:
    opts = options_from_config(cfg, video=video)
    opts.benchmark_path = out_path
    opts.prefer_torch = prefer_torch
    report = run_action_aware(opts)
    _print_summary(label, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark action-aware frame selection (compute cost + edge requirements)"
    )
    parser.add_argument("video", type=Path, nargs="?", help="Input video")
    parser.add_argument("-c", "--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("-o", "--benchmark-out", type=Path, help="Benchmark JSON output path")
    parser.add_argument("--sample-stride", type=int)
    parser.add_argument("--no-torch", action="store_true", help="Force motion-energy CPU fallback")
    parser.add_argument(
        "--compare-torch",
        action="store_true",
        help="Run both R3D-18 (GPU/CPU torch) and motion fallback; print comparison",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config.is_file() else {}
    video = args.video or Path(cfg.get("input_video", ""))
    if not video or not Path(video).is_file():
        parser.error("video path required (argument or config.input_video)")

    video = Path(video)
    paths = resolve_output_paths(video)
    out_dir = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_stride:
        cfg["sample_stride"] = args.sample_stride

    if args.compare_torch:
        reports = {}
        reports["r3d18"] = _run_profile(
            video,
            cfg,
            label="R3D-18 (torch)",
            prefer_torch=True,
            out_path=out_dir / f"{video.stem}_benchmark_r3d18.json",
        )
        reports["motion_cpu"] = _run_profile(
            video,
            cfg,
            label="Motion-energy (CPU)",
            prefer_torch=False,
            out_path=out_dir / f"{video.stem}_benchmark_motion.json",
        )
        comparison = {
            "video": str(video),
            "r3d18_ms_per_frame": reports["r3d18"]
            .get("benchmark", {})
            .get("throughput", {})
            .get("selection_only", {})
            .get("ms_per_frame"),
            "motion_ms_per_frame": reports["motion_cpu"]
            .get("benchmark", {})
            .get("throughput", {})
            .get("selection_only", {})
            .get("ms_per_frame"),
            "recommendation": (
                "Use motion fallback on edge if R3D-18 exceeds device budget"
            ),
        }
        comp_path = out_dir / f"{video.stem}_benchmark_comparison.json"
        comp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print("\n=== Comparison ===")
        print(json.dumps(comparison, indent=2))
        return 0

    bench_out = args.benchmark_out or paths["benchmark_path"]
    _run_profile(
        video,
        cfg,
        label="R3D-18" if not args.no_torch else "Motion-energy",
        prefer_torch=not args.no_torch,
        out_path=bench_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
