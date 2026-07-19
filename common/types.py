"""Shared dataclasses for frame selection.

All types are plain picklable dataclasses — they cross the
ProcessPoolExecutor boundary in parallel_filter.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectedEvent:
    frame_index: int
    timestamp_sec: float
    event_type: str
    score: float
    detail: str = ""


@dataclass
class SelectionStats:
    total_frames: int
    kept_frames: int
    reduction_ratio: float
    processing_ms: float


@dataclass
class FrameSelectionResult:
    video_path: str
    kept_indices: list[int]
    events: list[DetectedEvent]
    stats: SelectionStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
