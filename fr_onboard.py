"""
fr_onboard.py — Manage people for a site's face recognition registry (CLI wrapper).

Thin CLI wrapper over `faces.service` (`onboard_identity`, `list_identities`, `delete_identity`)
and `faces.annotate` (`annotate_video` for verification).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from faces.annotate import annotate_video
from faces.service import delete_identity, list_identities, onboard_identity

_BASE = Path(__file__).parent
log = logging.getLogger(__name__)


def _load_cfg(args) -> dict:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")).get("face_recognition", {})
    if getattr(args, "site", None):
        cfg["site_id"] = args.site
    return cfg


def cmd_add(args) -> None:
    cfg = _load_cfg(args)
    src = args.video or args.image
    if not src:
        raise SystemExit("Must pass either --video or --image")
    res = onboard_identity(
        name=args.name,
        video_or_image=src,
        cfg=cfg,
        max_frames=int(args.frames),
        single_embedding=not args.multi,
        replace=args.replace,
        notes=args.notes or "",
    )
    print("\n=== Onboarded ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


def cmd_list(args) -> None:
    cfg = _load_cfg(args)
    site = args.site or cfg.get("site_id", "")
    if not site:
        raise SystemExit("--site is required")
    rows = list_identities(site, cfg=cfg, include_untagged=args.all)
    print(f"\nSite: {site} ({len(rows)} rows)")
    print(f"{'ID':<36}  {'Name':<24}  {'Role':<15}  {'Notes'}")
    print("-" * 90)
    for r in rows:
        print(f"{r.get('id',''):<36}  {r.get('name',''):<24}  {r.get('role',''):<15}  {r.get('notes','')}")


def cmd_delete(args) -> None:
    cfg = _load_cfg(args)
    if args.uuid:
        ok = delete_identity(args.uuid, cfg=cfg)
        print(f"Deleted UUID '{args.uuid}': {ok}")
    else:
        raise SystemExit("--uuid is required for deletion.")


def cmd_verify(args) -> None:
    cfg = _load_cfg(args)
    out = Path(args.output or "verify_output.mp4")
    res = annotate_video(args.source, out, cfg)
    print("\n=== Verify Complete ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Manage people for face recognition")
    parser.add_argument("-c", "--config", type=Path, default=_BASE / "config.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("add", help="Enroll a person from video/image")
    pa.add_argument("--site", help="Site ID override")
    pa.add_argument("--name", required=True, help="Person name")
    pa.add_argument("--video", help="Input video path")
    pa.add_argument("--image", help="Input image path")
    pa.add_argument("--frames", type=int, default=8, help="Max frames to sample from video")
    pa.add_argument("--multi", action="store_true", help="Store multiple embeddings instead of averaging")
    pa.add_argument("--replace", action="store_true", help="Replace existing entries for this name")
    pa.add_argument("--notes", default="", help="Optional notes")

    pl = sub.add_parser("list", help="List people registered at a site")
    pl.add_argument("--site", help="Site ID override")
    pl.add_argument("--all", action="store_true", help="Include untagged entries")

    pd = sub.add_parser("delete", help="Delete person/entry by UUID")
    pd.add_argument("--site", help="Site ID override")
    pd.add_argument("--uuid", required=True, help="UUID to delete")

    pv = sub.add_parser("verify", help="Run recognition on video to verify enrollments")
    pv.add_argument("--site", help="Site ID override")
    pv.add_argument("source", help="Input video file")
    pv.add_argument("-o", "--output", help="Output video path")

    args = parser.parse_args()
    {"add": cmd_add, "list": cmd_list, "delete": cmd_delete, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
