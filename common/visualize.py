"""Annotated-video rendering: dim/mark frames the filter would remove."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AnnotateStyle:
    remove_border_color: tuple[int, int, int] = (0, 0, 255)
    remove_border_thickness: int = 8
    remove_label: str = "REMOVE"
    remove_dim_factor: float = 0.45
    show_frame_number: bool = True
    show_stats_banner: bool = True


def write_annotated_video(
    video: str | Path,
    out_path: str | Path,
    kept_indices: list[int],
    style: AnnotateStyle,
    stats: dict | None = None,
) -> None:
    """Render source video with REMOVE overlay on dropped frames."""
    from common.video_io import iter_frames, read_video_meta

    meta = read_video_meta(video)
    kept = set(kept_indices)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        meta.fps,
        (meta.width, meta.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    try:
        for idx, frame in iter_frames(video):
            if idx not in kept:
                frame = (frame.astype(np.float32) * style.remove_dim_factor).astype(
                    np.uint8
                )
                th = style.remove_border_thickness
                cv2.rectangle(
                    frame,
                    (th // 2, th // 2),
                    (meta.width - th // 2, meta.height - th // 2),
                    style.remove_border_color,
                    th,
                )
                cv2.putText(
                    frame,
                    style.remove_label,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    style.remove_border_color,
                    3,
                )
            if style.show_frame_number:
                cv2.putText(
                    frame,
                    f"#{idx}",
                    (20, meta.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            if style.show_stats_banner and stats:
                banner = "  ".join(f"{k}={v}" for k, v in stats.items())
                cv2.putText(
                    frame,
                    banner,
                    (20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
            writer.write(frame)
    finally:
        writer.release()

    log.info("annotated video → %s", out_path)
