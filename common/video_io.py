"""Video I/O primitives: metadata probe, frame iteration, kept-frame writing,
H.264 re-encode, and audio muxing.

All cv2 handles are released in finally blocks; ffmpeg helpers degrade
gracefully (return False) when ffmpeg is not on PATH.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

_FFMPEG_TIMEOUT = 600  # seconds
_DEFAULT_FPS = 25.0


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def read_video_meta(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = _DEFAULT_FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return VideoMeta(
        path=str(path),
        width=width,
        height=height,
        fps=float(fps),
        frame_count=frame_count,
        duration_sec=frame_count / fps if fps > 0 else 0.0,
    )


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame). Capture released even on early exit."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def write_frame_index(
    path: str | Path, kept_indices: list[int], extra: dict[str, Any] | None = None
) -> None:
    payload: dict[str, Any] = {"kept_indices": list(kept_indices)}
    if extra:
        payload.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_kept_video(
    src: str | Path,
    dst: str | Path,
    kept_indices: list[int],
    *,
    fps: float | None = None,
    reencode_h264: bool = True,
    output_width: int | None = None,
    output_height: int | None = None,
) -> dict[str, Any]:
    """Write only the kept frames of `src` to `dst`.

    fps=None → keep the source frame rate.
    Returns {"reencoded_h264", "fps", "duration_sec", "output_resolution"}.
    """
    src, dst = Path(src), Path(dst)
    meta = read_video_meta(src)
    out_fps = float(fps) if fps else meta.fps
    kept = set(kept_indices)

    out_w = int(output_width) if output_width else meta.width
    out_h = int(output_height) if output_height else meta.height

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")

    writer = cv2.VideoWriter(
        str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h)
    )
    written = 0
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {tmp}")
        for idx, frame in iter_frames(src):
            if idx not in kept:
                continue
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        if written == 0 and tmp.exists():
            tmp.unlink(missing_ok=True)

    if written == 0:
        raise ValueError(f"No kept frames written from {src}")

    reencoded = False
    if reencode_h264:
        reencoded = try_reencode_h264(tmp, fps=None)  # fps already applied
    os.replace(tmp, dst)

    return {
        "reencoded_h264": reencoded,
        "fps": out_fps,
        "duration_sec": written / out_fps if out_fps > 0 else 0.0,
        "output_resolution": f"{out_w}x{out_h}",
    }


def try_reencode_h264(path: str | Path, fps: float | None = None) -> bool:
    """Re-encode `path` in place to H.264 + faststart. False if unavailable/failed."""
    if not ffmpeg_available():
        log.debug("ffmpeg not on PATH — skipping H.264 re-encode of %s", path)
        return False

    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)

    cmd = ["ffmpeg", "-y", "-i", str(path)]
    if fps:
        cmd += ["-vf", f"fps={fps}"]
    cmd += [
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        str(tmp),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT, check=False
        )
        if proc.returncode != 0:
            log.warning(
                "ffmpeg re-encode failed (%d): %s",
                proc.returncode,
                proc.stderr.decode(errors="replace")[-400:],
            )
            return False
        os.replace(tmp, path)
        return True
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg re-encode timed out for %s", path)
        return False
    finally:
        tmp.unlink(missing_ok=True)


def mux_audio_to_video(
    video: str | Path,
    audio_source: str | Path,
    windows: list[tuple[float, float]],
) -> bool:
    """Extract the audio window(s) from `audio_source` and mux into `video`
    in place. Safe no-op (False) when ffmpeg is missing or the source has
    no audio stream."""
    if not ffmpeg_available():
        return False
    if not windows:
        return False

    video, audio_source = Path(video), Path(audio_source)
    if not _has_audio_stream(audio_source):
        return False

    start, end = windows[0]
    duration = max(0.0, end - start)
    if duration <= 0:
        return False

    fd, tmp_name = tempfile.mkstemp(suffix=video.suffix, dir=str(video.parent))
    os.close(fd)
    tmp = Path(tmp_name)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        str(tmp),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT, check=False
        )
        if proc.returncode != 0:
            log.debug(
                "audio mux skipped (%d): %s",
                proc.returncode,
                proc.stderr.decode(errors="replace")[-200:],
            )
            return False
        os.replace(tmp, video)
        return True
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg audio mux timed out for %s", video)
        return False
    finally:
        tmp.unlink(missing_ok=True)


def _has_audio_stream(path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return True  # assume yes; mux will fail soft if wrong
    try:
        proc = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
        return bool(proc.stdout.strip())
    except subprocess.TimeoutExpired:
        return False
