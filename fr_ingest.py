"""
fr_ingest.py — Step 1: Ingest a video into the face recognition database (CLI wrapper).

Thin CLI wrapper over `faces.service.ingest_video`.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from faces.service import ingest_video

_BASE = Path(__file__).parent
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Ingest face embeddings from a video into Milvus.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-c", "--config", type=Path, default=_BASE / "config.json")
    parser.add_argument("--site-id", default=None, help="Site identifier override")
    parser.add_argument("--camera-id", default=None, help="Camera identifier override")
    parser.add_argument("--frame-skip", type=int, default=None, help="Process every Nth frame")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution provider")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8")).get("face_recognition", {})
    if args.site_id:
        cfg["site_id"] = args.site_id
    if args.camera_id:
        cfg["camera_id"] = args.camera_id
    if args.frame_skip:
        cfg["frame_skip"] = args.frame_skip
    if args.cpu:
        cfg["execution_provider"] = "CPUExecutionProvider"

    res = ingest_video(args.video, cfg)
    print("\n=== Ingestion Summary ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
