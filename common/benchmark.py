"""Benchmark session + report builder for pipeline runs.

Phase timing via start()/end() pairs, RSS/CPU sampling via psutil, and
CUDA peak-memory via torch when available.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import psutil

    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _cuda_peak_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
    except ImportError:
        pass
    return 0.0


class BenchmarkSession:
    """Accumulates per-phase wall time plus memory/CPU samples."""

    def __init__(self) -> None:
        self._created = time.perf_counter()
        self.phases_ms: dict[str, float] = {}
        self._open: dict[str, float] = {}
        self._peak_rss_mb = 0.0
        self._cpu_samples: list[float] = []
        self._proc = psutil.Process() if _PSUTIL else None
        self._sample()

    def _sample(self) -> None:
        if self._proc is None:
            return
        try:
            rss = self._proc.memory_info().rss / (1024**2)
            self._peak_rss_mb = max(self._peak_rss_mb, rss)
            cpu = self._proc.cpu_percent(interval=None)
            if cpu > 0:
                self._cpu_samples.append(cpu)
        except Exception:  # process gone / access denied — sampling only
            pass

    def start(self, name: str) -> None:
        self._open[name] = time.perf_counter()
        self._sample()

    def end(self, name: str) -> None:
        t0 = self._open.pop(name, None)
        if t0 is None:
            log.warning("BenchmarkSession.end(%r) without start", name)
            return
        self.phases_ms[name] = self.phases_ms.get(name, 0.0) + (
            time.perf_counter() - t0
        ) * 1000
        self._sample()

    def reset_gpu_peak(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    @property
    def total_ms(self) -> float:
        return sum(self.phases_ms.values())

    @property
    def peak_rss_mb(self) -> float:
        self._sample()
        return self._peak_rss_mb

    @property
    def cpu_percent_avg(self) -> float | None:
        if not self._cpu_samples:
            return None
        return sum(self._cpu_samples) / len(self._cpu_samples)


# Reference edge devices: rough relative throughput vs a modern x86 CPU core
# running this pipeline. Used only for indicative feasibility flags.
_EDGE_DEVICES = [
    {"device_label": "Jetson Orin Nano", "relative_speed": 1.6},
    {"device_label": "Jetson Xavier NX", "relative_speed": 1.0},
    {"device_label": "Raspberry Pi 5", "relative_speed": 0.35},
    {"device_label": "Intel N100 mini-PC", "relative_speed": 0.7},
]


def build_benchmark_report(
    bench: BenchmarkSession,
    *,
    video_meta: Any,
    method: str,
    selector_model: str | None,
    benchmark_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = benchmark_cfg or {}
    total_ms = bench.total_ms
    frames = int(getattr(video_meta, "frame_count", 0) or 0)
    duration = float(getattr(video_meta, "duration_sec", 0.0) or 0.0)

    processing_fps = frames / (total_ms / 1000) if total_ms > 0 else 0.0
    ms_per_frame = total_ms / frames if frames else 0.0
    realtime_factor = duration / (total_ms / 1000) if total_ms > 0 else 0.0
    interpretation = (
        f"{realtime_factor:.1f}x faster than real-time"
        if realtime_factor >= 1
        else f"{1 / realtime_factor:.1f}x slower than real-time"
        if realtime_factor > 0
        else "n/a"
    )

    # Selection-only throughput: the pure inference phase when present
    sel_ms = bench.phases_ms.get("filter", bench.phases_ms.get("inference", 0.0))
    sel_fps = frames / (sel_ms / 1000) if sel_ms > 0 else 0.0
    sel_mspf = sel_ms / frames if frames else 0.0

    bottleneck = (
        max(bench.phases_ms, key=bench.phases_ms.get) if bench.phases_ms else "?"
    )

    peak_gpu_mb = _cuda_peak_mb()
    used_gpu = peak_gpu_mb > 0

    if realtime_factor >= 4:
        label = "edge-capable (CPU headroom)"
    elif realtime_factor >= 1:
        label = "edge-capable (near real-time)" if not used_gpu else "gpu-workstation"
    else:
        label = "gpu-server recommended" if used_gpu else "batch/offline only"

    edge = [
        {
            "device_label": d["device_label"],
            "relative_speed": d["relative_speed"],
            "estimated_realtime_factor": round(
                realtime_factor * d["relative_speed"], 2
            ),
            "realtime_feasible": realtime_factor * d["relative_speed"] >= 1.0,
        }
        for d in _EDGE_DEVICES
    ]

    hourly = float(cfg.get("cloud_gpu_hourly_usd", 0.45))
    # Cost to process one hour of footage at the measured speed
    cost_per_video_hour = hourly / realtime_factor if realtime_factor > 0 else None

    return {
        "method": method,
        "selector_model": selector_model,
        "timing_ms": {"total": round(total_ms, 1)}
        | {k: round(v, 1) for k, v in bench.phases_ms.items()},
        "throughput": {
            "pipeline": {
                "processing_fps": round(processing_fps, 2),
                "ms_per_frame": round(ms_per_frame, 2),
                "realtime_factor": round(realtime_factor, 3),
                "interpretation": interpretation,
            },
            "selection_only": {
                "processing_fps": round(sel_fps, 2),
                "ms_per_frame": round(sel_mspf, 2),
            },
        },
        "memory": {
            "peak_rss_mb": round(bench.peak_rss_mb, 1),
            "peak_gpu_allocated_mb": round(peak_gpu_mb, 1),
            "cpu_percent_avg": (
                round(bench.cpu_percent_avg, 1)
                if bench.cpu_percent_avg is not None
                else None
            ),
        },
        "bottleneck_phase": bottleneck,
        "deployment_recommendation": {
            "label": label,
            "used_gpu": used_gpu,
        },
        "edge_compatibility": edge,
        "compute_cost": {
            "cloud_gpu_hourly_usd": hourly,
            "usd_per_video_hour": (
                round(cost_per_video_hour, 4) if cost_per_video_hour else None
            ),
        },
        "system_requirements": {
            "min_ram_mb": round(bench.peak_rss_mb * 1.5, 0),
            "gpu_required": used_gpu,
        },
        "video": {
            "frames": frames,
            "duration_sec": round(duration, 2),
            "fps": float(getattr(video_meta, "fps", 0.0) or 0.0),
        },
        "notes": cfg.get("notes", ""),
    }


def save_benchmark_report(report: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")


def merge_benchmark_into_report(
    report: dict[str, Any],
    bench: BenchmarkSession,
    *,
    video_meta: Any,
    method: str,
    selector_model: str | None,
    benchmark_cfg: dict[str, Any] | None = None,
    benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    bm = build_benchmark_report(
        bench,
        video_meta=video_meta,
        method=method,
        selector_model=selector_model,
        benchmark_cfg=benchmark_cfg,
    )
    report["benchmark"] = bm
    if benchmark_path:
        save_benchmark_report(bm, benchmark_path)
        report["benchmark_path"] = str(Path(benchmark_path).resolve())
        log.info("benchmark → %s", benchmark_path)
    return report
