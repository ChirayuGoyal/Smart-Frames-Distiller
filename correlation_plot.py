"""Correlation score timeline and matplotlib plot for action-aware triggers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from action_model import ActionPrediction


def correlation_score(prev: ActionPrediction, curr: ActionPrediction) -> float:
    """Cosine similarity between consecutive prediction vectors (1.0 = stable)."""
    if prev.logits is not None and curr.logits is not None:
        a = prev.logits.astype(np.float64)
        b = curr.logits.astype(np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 1.0
        return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))

    if prev.class_id == curr.class_id:
        return float(np.clip(1.0 - abs(prev.confidence - curr.confidence), 0.0, 1.0))
    return 0.0


def build_correlation_timeline(
    predictions: list[ActionPrediction],
    fps: float,
    *,
    conf_delta: float,
) -> list[dict[str, Any]]:
    """Build per-sample correlation scores and trigger flags."""
    if len(predictions) < 2:
        return []

    timeline: list[dict[str, Any]] = []
    for i in range(1, len(predictions)):
        prev = predictions[i - 1]
        curr = predictions[i]
        score = correlation_score(prev, curr)
        label_changed = curr.class_id != prev.class_id
        conf_jump = abs(curr.confidence - prev.confidence) > conf_delta
        trigger = label_changed or conf_jump
        reasons: list[str] = []
        if label_changed:
            reasons.append(f"label {prev.top_label}->{curr.top_label}")
        if conf_jump:
            reasons.append(f"conf Δ={abs(curr.confidence - prev.confidence):.3f}")

        timeline.append(
            {
                "frame": curr.frame_index,
                "time_sec": round(curr.frame_index / fps, 3) if fps > 0 else 0.0,
                "correlation": round(score, 4),
                "trigger": trigger,
                "label_changed": label_changed,
                "confidence_delta": round(abs(curr.confidence - prev.confidence), 4),
                "prev_label": prev.top_label,
                "curr_label": curr.top_label,
                "detail": "; ".join(reasons) if reasons else None,
            }
        )
    return timeline


def plot_correlation_timeline(
    timeline: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str | None = None,
    trigger_threshold: float | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Save matplotlib plot of correlation score vs time with trigger markers."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not timeline:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Not enough samples for correlation plot", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return {"correlation_plot": str(output_path.resolve()), "points": 0, "triggers": 0}

    times = [p["time_sec"] for p in timeline]
    scores = [p["correlation"] for p in timeline]
    triggers = [p for p in timeline if p.get("trigger")]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, scores, color="#2563eb", linewidth=1.5, label="Correlation score")
    if triggers:
        ax.scatter(
            [p["time_sec"] for p in triggers],
            [p["correlation"] for p in triggers],
            color="#dc2626",
            s=48,
            zorder=5,
            label=f"Trigger ({len(triggers)})",
        )
        for point in triggers:
            ax.axvline(point["time_sec"], color="#dc2626", alpha=0.15, linewidth=1)

    if trigger_threshold is not None:
        ax.axhline(
            trigger_threshold,
            color="#f59e0b",
            linestyle="--",
            linewidth=1,
            label=f"Threshold ({trigger_threshold:.2f})",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Correlation score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "Action correlation score over time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return {
        "correlation_plot": str(output_path.resolve()),
        "points": len(timeline),
        "triggers": len(triggers),
        "min_correlation": round(min(scores), 4),
        "mean_correlation": round(float(np.mean(scores)), 4),
    }
