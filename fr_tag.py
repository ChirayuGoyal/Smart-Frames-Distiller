"""
fr_tag.py — Step 3: Tag face UUIDs with identity metadata.

Subcommands:
    tag   — assign name/role/department/notes to a UUID
    show  — display current tags for a UUID
    list  — tabular listing of all tagged (or all) UUIDs
    bulk  — bulk-import tags from a CSV file

Usage:
    python fr_tag.py tag  --uuid UUID --name "Alice" [--role R] [--dept D] [--notes N]
    python fr_tag.py show --uuid UUID
    python fr_tag.py list [--all]
    python fr_tag.py bulk --csv tags.csv

CSV format (header required, role/department/notes optional):
    uuid,name,role,department,notes
    471e6646-...,Alice,Engineer,R&D,
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from fr_milvus import (
    escape_milvus_string,
    load_collection,
    query_paged,
    resolve_site_camera,
    site_expr,
    upsert_row,
)

_BASE = Path(__file__).parent
log = logging.getLogger(__name__)


def _upsert(col, uid: str, name: str, role: str, dept: str, notes: str) -> None:
    uid_esc = escape_milvus_string(uid)
    existing = col.query(
        expr=f'id == "{uid_esc}"',
        output_fields=["id", "embedding", "person_id", "site_id", "camera_id"],
    )
    if not existing:
        raise SystemExit(f"UUID '{uid}' not found in collection.")
    r = existing[0]
    upsert_row(
        col,
        uid=uid,
        person_id=r.get("person_id") or uid,
        embedding=r["embedding"],
        name=name,
        role=role,
        department=dept,
        notes=notes,
        site_id=r.get("site_id", ""),
        camera_id=r.get("camera_id", ""),
    )
    log.info("Tagged %s | name='%s' role='%s' dept='%s'", uid[:12], name, role, dept)


# ── Subcommand: tag ───────────────────────────────────────────────────────────

def cmd_tag(col, args: argparse.Namespace) -> None:
    if not any([args.name, args.role, args.dept, args.notes]):
        raise SystemExit("Provide at least one of --name / --role / --dept / --notes")

    existing = col.query(
        expr=f'id == "{args.uuid}"',
        output_fields=["name", "role", "department", "notes"],
    )
    cur = existing[0] if existing else {}
    _upsert(
        col, args.uuid,
        args.name  if args.name  is not None else cur.get("name", ""),
        args.role  if args.role  is not None else cur.get("role", ""),
        args.dept  if args.dept  is not None else cur.get("department", ""),
        args.notes if args.notes is not None else cur.get("notes", ""),
    )
    print(f"Tagged {args.uuid}")


# ── Subcommand: show ──────────────────────────────────────────────────────────

def cmd_show(col, args: argparse.Namespace) -> None:
    uid_esc = escape_milvus_string(args.uuid)
    res = col.query(
        expr=f'id == "{uid_esc}"',
        output_fields=["id", "person_id", "name", "role", "department", "notes", "site_id", "camera_id"],
    )
    if not res:
        print(f"UUID not found: {args.uuid}")
        return
    r = res[0]
    print(f"UUID:       {r['id']}")
    print(f"Person ID:  {r.get('person_id', '')}")
    print(f"Site:       {r.get('site_id', '')}")
    print(f"Camera:     {r.get('camera_id', '')}")
    print(f"Name:       {r.get('name', '')}")
    print(f"Role:       {r.get('role', '')}")
    print(f"Department: {r.get('department', '')}")
    print(f"Notes:      {r.get('notes', '')}")


# ── Subcommand: list ──────────────────────────────────────────────────────────

def cmd_list(col, args: argparse.Namespace) -> None:
    if args.all:
        expr = site_expr(args.site_id) if args.site_id else 'id != ""'
    else:
        base = 'name != ""'
        expr = f"({base}) and ({site_expr(args.site_id)})" if args.site_id else base

    res = query_paged(
        col,
        expr,
        ["id", "person_id", "name", "role", "department", "notes", "site_id", "camera_id"],
    )
    if not res:
        print("No entries found.")
        return
    print(f"{'UUID':<38}  {'Site':<12}  {'Name':<22}  {'Role':<12}  {'Department':<16}  Notes")
    print("-" * 130)
    for r in sorted(res, key=lambda x: (x.get("site_id", ""), x.get("name", ""))):
        print(
            f"{r['id']:<38}  {r.get('site_id',''):<12}  {r.get('name',''):<22}  "
            f"{r.get('role',''):<12}  {r.get('department',''):<16}  {r.get('notes','')}"
        )
    print(f"\nTotal: {len(res)}")


# ── Subcommand: bulk ──────────────────────────────────────────────────────────

def cmd_bulk(col, args: argparse.Namespace) -> None:
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("CSV is empty.")

    tagged = skipped = 0
    for row in rows:
        uid  = row.get("uuid", "").strip()
        name = row.get("name", "").strip()
        if not uid or not name:
            log.warning("Skipping row (missing uuid or name): %s", row)
            skipped += 1
            continue
        try:
            _upsert(col, uid, name,
                    row.get("role", "").strip(),
                    row.get("department", "").strip(),
                    row.get("notes", "").strip())
            tagged += 1
        except SystemExit as e:
            log.warning("%s", e)
            skipped += 1

    log.info("Bulk tag done | tagged=%d | skipped=%d", tagged, skipped)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", type=Path, default=_BASE / "config.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Tag face UUIDs with identity metadata")
    _add_config_arg(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # tag
    p_tag = sub.add_parser("tag", help="Assign tags to a UUID")
    _add_config_arg(p_tag)
    p_tag.add_argument("--uuid", required=True)
    p_tag.add_argument("--name",  default=None)
    p_tag.add_argument("--role",  default=None)
    p_tag.add_argument("--dept",  default=None, help="Department")
    p_tag.add_argument("--notes", default=None)

    # show
    p_show = sub.add_parser("show", help="Show tags for a UUID")
    _add_config_arg(p_show)
    p_show.add_argument("--uuid", required=True)

    # list
    p_list = sub.add_parser("list", help="List tagged UUIDs")
    _add_config_arg(p_list)
    p_list.add_argument("--all", action="store_true", help="Include untagged UUIDs")
    p_list.add_argument("--site-id", default=None, help="Filter by site_id")

    # bulk
    p_bulk = sub.add_parser("bulk", help="Bulk tag from CSV")
    _add_config_arg(p_bulk)
    p_bulk.add_argument("--csv", required=True, help="Path to CSV (uuid,name,role,department,notes)")

    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8")).get("face_recognition", {})
    if getattr(args, "site_id", None):
        cfg["site_id"] = args.site_id
    site_id, _ = resolve_site_camera(cfg)
    if args.cmd == "list" and args.site_id is None and site_id:
        args.site_id = site_id

    col = load_collection(cfg.get("milvus", {}))

    {"tag": cmd_tag, "show": cmd_show, "list": cmd_list, "bulk": cmd_bulk}[args.cmd](col, args)


if __name__ == "__main__":
    main()
