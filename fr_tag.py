"""
fr_tag.py — Step 3: Tag face UUIDs with identity metadata (CLI wrapper).

Thin CLI wrapper over `faces.service` (`tag_identity`, `list_identities`).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from faces.service import list_identities, tag_identity

_BASE = Path(__file__).parent
log = logging.getLogger(__name__)


def _load_cfg(args) -> dict:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")).get("face_recognition", {})
    if getattr(args, "site", None):
        cfg["site_id"] = args.site
    return cfg


def cmd_tag(args) -> None:
    cfg = _load_cfg(args)
    res = tag_identity(
        args.uuid,
        name=args.name or "",
        role=args.role or "",
        department=args.dept or "",
        notes=args.notes or "",
        cfg=cfg,
    )
    print("\n=== Tagged ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


def cmd_show(args) -> None:
    cfg = _load_cfg(args)
    site = cfg.get("site_id", "")
    rows = list_identities(site, cfg=cfg, include_untagged=True)
    match = next((r for r in rows if r["id"] == args.uuid), None)
    if not match:
        raise SystemExit(f"UUID '{args.uuid}' not found.")
    print("\n=== Entry Details ===")
    for k, v in match.items():
        if k != "embedding":
            print(f"  {k}: {v}")


def cmd_list(args) -> None:
    cfg = _load_cfg(args)
    site = args.site or cfg.get("site_id", "")
    rows = list_identities(site, cfg=cfg, include_untagged=args.all)
    print(f"\nSite: {site} ({len(rows)} rows)")
    print(f"{'ID':<36}  {'Name':<24}  {'Role':<15}  {'Notes'}")
    print("-" * 90)
    for r in rows:
        print(f"{r.get('id',''):<36}  {r.get('name',''):<24}  {r.get('role',''):<15}  {r.get('notes','')}")


def cmd_bulk(args) -> None:
    cfg = _load_cfg(args)
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV file not found: {path}")
    count = 0
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("uuid", "").strip()
            if not uid:
                continue
            tag_identity(
                uid,
                name=row.get("name", "").strip(),
                role=row.get("role", "").strip(),
                department=row.get("department", "").strip(),
                notes=row.get("notes", "").strip(),
                cfg=cfg,
            )
            count += 1
    print(f"Bulk import complete: {count} entries tagged.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Tag face entries with identity metadata")
    parser.add_argument("-c", "--config", type=Path, default=_BASE / "config.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("tag", help="Assign name/role/dept/notes to a UUID")
    pt.add_argument("--uuid", required=True, help="UUID to tag")
    pt.add_argument("--name", help="Person name")
    pt.add_argument("--role", default="", help="Job role")
    pt.add_argument("--dept", default="", help="Department")
    pt.add_argument("--notes", default="", help="Optional notes")
    pt.add_argument("--site", help="Site override")

    ps = sub.add_parser("show", help="Show current tags for a UUID")
    ps.add_argument("--uuid", required=True)
    ps.add_argument("--site", help="Site override")

    pl = sub.add_parser("list", help="List tagged face entries")
    pl.add_argument("--site", help="Site override")
    pl.add_argument("--all", action="store_true", help="Show untagged rows too")

    pb = sub.add_parser("bulk", help="Bulk tag entries from CSV")
    pb.add_argument("--csv", required=True, help="CSV file path")
    pb.add_argument("--site", help="Site override")

    args = parser.parse_args()
    {"tag": cmd_tag, "show": cmd_show, "list": cmd_list, "bulk": cmd_bulk}[args.cmd](args)


if __name__ == "__main__":
    main()
