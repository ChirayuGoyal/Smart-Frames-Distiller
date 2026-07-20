"""
fr_merge.py — Step 2: Merge face embeddings within a site (CLI wrapper).

Thin CLI wrapper over `faces.service.merge_identities`.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from faces.service import merge_identities

_BASE = Path(__file__).parent
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Merge face embeddings within a site_id")
    parser.add_argument("-c", "--config", type=Path, default=_BASE / "config.json")
    parser.add_argument("--site-id", default=None, help="Site to merge (required)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to Milvus")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8")).get("face_recognition", {})
    if args.site_id is not None:
        cfg["site_id"] = args.site_id

    res = merge_identities(cfg, dry_run=args.dry_run)
    print("\n=== Merge Summary ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
