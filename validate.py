#!/usr/bin/env python3
"""
Cross-validation for action-aware frame selection.

Tests:
  1. Synthetic video change-point detection (precision/recall)
  2. Action label stability on kept vs original timeline
  3. Determinism (two runs produce identical keep set)
  4. Compression ratio within expected bounds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common.metrics import action_label_stability, change_point_precision_recall, event_retention_rate
from common.synthetic import make_action_aware_test_video

from selector import ActionAwareSelector


def run_validation(use_torch: bool = False) -> dict:
    base = Path(__file__).parent / "test_data"
    gt = make_action_aware_test_video(base)
    video = Path(gt["video"])
    change_points = gt["change_points"]

    selector = ActionAwareSelector(
        sample_stride=2,
        conf_delta=0.08,
        max_gap=25,
        prefer_torch=use_torch,
    )

    # Determinism
    r1 = selector.select(video)
    r2 = selector.select(video)
    deterministic = r1.kept_indices == r2.kept_indices

    # Change-point detection from action_change events
    predicted_changes = [e.frame_index for e in r1.events]
    cp_metrics = change_point_precision_recall(predicted_changes, change_points, tolerance=4)

    # Event retention: each GT change must have a kept frame nearby
    retention = event_retention_rate(change_points, r1.kept_indices, tolerance=5)

    # Action label stability
    labels_orig = selector.label_timeline(video, stride=2)
    labels_kept = [(f, c) for f, c in labels_orig if f in set(r1.kept_indices)]
    stability = action_label_stability(labels_orig, labels_kept, tolerance=6)

    stats = r1.stats
    results = {
        "video": str(video),
        "model": r1.metadata.get("model"),
        "deterministic": deterministic,
        "total_frames": stats.total_frames if stats else 0,
        "kept_frames": stats.kept_frames if stats else 0,
        "reduction_ratio": round(stats.reduction_ratio, 2) if stats else 0,
        "change_point_metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in cp_metrics.items()},
        "event_retention": round(retention, 3),
        "action_label_stability": round(stability, 3),
        "predicted_changes": predicted_changes,
        "ground_truth_changes": change_points,
        "processing_ms": round(stats.processing_ms, 1) if stats else 0,
    }

    # Pass criteria aligned with RESEARCH.md / benchmark framework
    results["passed"] = (
        deterministic
        and retention >= 0.95
        and cp_metrics["recall"] >= 0.5
        and (stats.kept_frames if stats else 0) < (stats.total_frames if stats else 1)
    )
    return results


def main() -> int:
    print("=== Action-Aware Validation (motion fallback) ===")
    motion_results = run_validation(use_torch=False)
    print(json.dumps(motion_results, indent=2))

    torch_results = None
    try:
        import torch  # noqa: F401

        print("\n=== Action-Aware Validation (torchvision R3D-18) ===")
        torch_results = run_validation(use_torch=True)
        print(json.dumps(torch_results, indent=2))
    except ImportError:
        print("\n(torch not installed — skipping R3D-18 validation)")

    out_path = Path(__file__).parent / "test_data" / "validation_report.json"
    out_path.write_text(
        json.dumps({"motion": motion_results, "torch": torch_results}, indent=2),
        encoding="utf-8",
    )

    ok = motion_results.get("passed", False)
    if torch_results:
        ok = ok and torch_results.get("passed", False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
