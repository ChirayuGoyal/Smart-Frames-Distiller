"""Helpers over frame-selection results."""

from __future__ import annotations

from common.types import FrameSelectionResult


def removed_indices(total_frames: int, kept: list[int]) -> list[int]:
    """Indices of frames NOT in the kept set, sorted ascending."""
    kept_set = set(kept)
    return [i for i in range(total_frames) if i not in kept_set]


def build_frame_reasons(result: FrameSelectionResult) -> dict[int, str]:
    """Map kept frame index → human-readable reason it was kept."""
    reasons: dict[int, str] = {}
    for ev in result.events:
        reasons[ev.frame_index] = ev.detail or ev.event_type
    for idx in result.kept_indices:
        reasons.setdefault(idx, "anchor/neighbor")
    return reasons
