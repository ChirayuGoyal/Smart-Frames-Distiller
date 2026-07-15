#!/usr/bin/env python3
"""
Test: mock an MP4 file as an RTSP stream and verify event-centric chunking.

The MockRTSPStreamProcessor subclass overrides _open() to return a
cv2.VideoCapture(mp4_path) instead of an RTSP URL.  When the file ends
(cap.read() → ok=False) it raises KeyboardInterrupt so the main loop
triggers the normal exit-flush path, exactly as a Ctrl-C would.

Usage:
    python test_rtsp_mock.py [mp4_path] [--chunk-dur N] [--out-dir DIR]

Output files written to --out-dir (default: test_rtsp_output/):
    {chunk_id}.mp4              video clip for each detected event cluster
    {chunk_id}.json             per-chunk metadata sidecar
    {run_id}_rtsp_summary.json  summary of all chunks (written by rtsp_stream.py)
    test_report.json            test runner result (written by this script)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import unittest.mock as mock
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from runner import RunOptions
from rtsp_stream import RTSPStreamProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_rtsp_mock")


# ── Mock processor ─────────────────────────────────────────────────────────────

class MockRTSPStreamProcessor(RTSPStreamProcessor):
    """
    Feeds an MP4 file as if it were an RTSP stream.

    On second call to _open() (i.e. when the file ends and the base class
    attempts to reconnect), we raise KeyboardInterrupt to trigger the
    normal exit-flush code path and end the loop cleanly.
    """

    def __init__(self, mp4_path: str, opts: RunOptions, **kwargs) -> None:
        self._mp4_path      = mp4_path
        self._open_calls    = 0
        super().__init__("rtsp://mock/test", opts, **kwargs)

    def _open(self) -> cv2.VideoCapture:
        self._open_calls += 1
        if self._open_calls > 1:
            raise KeyboardInterrupt("mock EOF — video file ended")
        cap = cv2.VideoCapture(self._mp4_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open test video: {self._mp4_path}")
        return cap


# ── Helpers ────────────────────────────────────────────────────────────────────

def _probe_video(path: str) -> tuple[float, int, int, int]:
    """Return (fps, width, height, frame_count) via OpenCV."""
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, n


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test RTSP event-centric chunking using a local MP4 file."
    )
    parser.add_argument(
        "mp4", nargs="?",
        default="0001_output_annotated.mp4",
        help="MP4 file to treat as a mock RTSP stream (default: 0001_output_annotated.mp4)",
    )
    parser.add_argument(
        "--chunk-dur", type=float, default=3.0,
        help="Chunk duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--out-dir", default="test_rtsp_output",
        help="Output directory for chunks and JSON files (default: test_rtsp_output)",
    )
    parser.add_argument(
        "--conf-delta", type=float, default=0.05,
        help="Trigger sensitivity — lower = more clusters (default: 0.05)",
    )
    parser.add_argument(
        "--run-id", default="test-rtsp-mock-001",
        help="run_id used for summary filename (default: test-rtsp-mock-001)",
    )
    args = parser.parse_args()

    mp4_path = str(Path(args.mp4).resolve())
    out_dir  = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(mp4_path).is_file():
        log.error("Video not found: %s", mp4_path)
        return 1

    fps, w, h, n_frames = _probe_video(mp4_path)
    duration_sec = n_frames / fps if fps > 0 else 0

    print(
        f"\n── Test Configuration ───────────────────────────────────────────\n"
        f"  Input video  : {mp4_path}\n"
        f"  Resolution   : {w}×{h}  @{fps:.1f} fps  (~{duration_sec:.1f} s, {n_frames} frames)\n"
        f"  Chunk dur    : {args.chunk_dur:.1f} s\n"
        f"  Conf delta   : {args.conf_delta:.2f}\n"
        f"  Output dir   : {out_dir}\n"
        f"  Run ID       : {args.run_id}\n"
        f"─────────────────────────────────────────────────────────────────\n"
    )

    opts = RunOptions(
        video=Path(mp4_path),
        run_id=args.run_id,
        camera_id="cam-mock-01",
        site_id="site-test",
        output_dir=out_dir,
        chunk_duration_sec=args.chunk_dur,
        conf_delta=args.conf_delta,
        clip_len=16,
        sample_stride=4,
        prefer_torch=False,    # MotionEnergy model — no GPU required
        ensemble=False,
        kafka_enabled=False,
        kafka_cfg={"enabled": False},
        filter_enabled=False,
        detection_enabled=False,
        chunk_enabled=True,
    )

    processor = MockRTSPStreamProcessor(
        mp4_path, opts,
        kafka_overrides={"enabled": False},
    )

    t0 = time.time()
    # Patch time.sleep to skip the 2-second reconnect delay when EOF is hit
    with mock.patch("rtsp_stream.time.sleep"):
        report = processor.run()
    elapsed = time.time() - t0

    # ── Collect and report results ─────────────────────────────────────────────
    report["test_video"]       = mp4_path
    report["elapsed_sec"]      = round(elapsed, 2)
    report["output_directory"] = str(out_dir)

    report_path = out_dir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    created = sorted(out_dir.iterdir())
    mp4s    = [f for f in created if f.suffix == ".mp4"]
    jsons   = [f for f in created if f.suffix == ".json"]

    print(f"\n── Output Files ─────────────────────────────────────────────────")
    for f in created:
        size_kb = f.stat().st_size / 1024
        tag = "VIDEO" if f.suffix == ".mp4" else "JSON "
        print(f"  [{tag}]  {f.name:<55}  {size_kb:>8.1f} KB")

    summary_path = out_dir / f"{args.run_id}_rtsp_summary.json"
    passed = report["total_chunks"] > 0 and summary_path.is_file()

    print(
        f"\n── Test Results ─────────────────────────────────────────────────\n"
        f"  Chunk MP4s created   : {len(mp4s)}\n"
        f"  JSON files created   : {len(jsons)}\n"
        f"  Total input frames   : {report['total_frames']}\n"
        f"  Total chunks flushed : {report['total_chunks']}\n"
        f"  Elapsed              : {elapsed:.1f} s\n"
        f"  Summary JSON         : {'✓ ' + str(summary_path) if summary_path.is_file() else '✗ NOT FOUND'}\n"
        f"  Test report          : {report_path}\n"
        f"\n  {'PASS' if passed else 'FAIL'} — "
        f"{'chunks and JSON files created successfully' if passed else 'no chunks were produced'}\n"
        f"─────────────────────────────────────────────────────────────────\n"
    )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
