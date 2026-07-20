"""Validation metrics for action-aware frame selection."""

from __future__ import annotations


def _matched(a: int, candidates: list[int], tolerance: int) -> bool:
    return any(abs(a - c) <= tolerance for c in candidates)


def change_point_precision_recall(
    predicted: list[int], truth: list[int], tolerance: int
) -> dict:
    """Precision/recall/F1 of predicted change points vs ground truth.

    A truth point is a TP when any prediction lands within `tolerance` frames.
    A prediction not near any truth point is an FP.
    """
    tp = sum(1 for t in truth if _matched(t, predicted, tolerance))
    fn = len(truth) - tp
    fp = sum(1 for p in predicted if not _matched(p, truth, tolerance))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def event_retention_rate(
    truth_changes: list[int], kept_indices: list[int], tolerance: int
) -> float:
    """Fraction of ground-truth change points with a kept frame nearby."""
    if not truth_changes:
        return 1.0
    kept = sorted(kept_indices)
    hit = sum(1 for t in truth_changes if _matched(t, kept, tolerance))
    return hit / len(truth_changes)


def action_label_stability(
    labels_orig: list[tuple[int, int]],
    labels_kept: list[tuple[int, int]],
    tolerance: int,
) -> float:
    """Fraction of original (frame, class) labels represented by a kept label
    of the same class within `tolerance` frames."""
    if not labels_orig:
        return 1.0
    stable = 0
    for frame, cls in labels_orig:
        if any(
            abs(frame - kf) <= tolerance and cls == kc
            for kf, kc in labels_kept
        ):
            stable += 1
    return stable / len(labels_orig)
