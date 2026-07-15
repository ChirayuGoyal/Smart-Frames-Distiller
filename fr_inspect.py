"""
fr_inspect.py — Step 3a: Visually review untagged face UUIDs.

Subcommands:
    crops  — scan a video and save one face crop per untagged UUID
    sheet  — combine crops in a folder into a single contact-sheet grid image
    both   — run crops then sheet in sequence (default)

Usage:
    python fr_inspect.py both  video.mp4
    python fr_inspect.py crops video.mp4 [--per-uuid 2] [--out-dir uuid_crops]
    python fr_inspect.py sheet            [--in-dir  uuid_crops] [--out contact_sheet.jpg]
    python fr_inspect.py both  video.mp4  --per-uuid 2 --cols 12
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from face_recognizer import SCRFD, ArcFace, align_face, resolve_model_path
from fr_milvus import (
    load_collection,
    query_paged,
    require_site_id,
    resolve_site_camera,
    untagged_expr,
)

_BASE = Path(__file__).parent
log = logging.getLogger(__name__)


def make_crops(video: str, cfg: dict, out_dir: Path, per_uuid: int = 1) -> None:
    """
    Scan video once, save up to per_uuid face crops per untagged UUID.
    File names: <uuid_prefix>_f<frame>_face<idx>.jpg
    """
    provider  = cfg.get("execution_provider", "CPUExecutionProvider")
    device_id = int(cfg.get("gpu_device_id", 0))
    det_high  = float(cfg.get("det_thresh_high", 0.5))
    det_low   = float(cfg.get("det_thresh_low", 0.35))
    skip      = int(cfg.get("frame_skip", 2))
    match_thr = float(cfg.get("dedup_thresh", 0.95))
    site_id, _camera_id = resolve_site_camera(cfg)
    require_site_id(site_id, tool="fr_inspect")

    col = load_collection(cfg.get("milvus", {}))
    rows = query_paged(col, untagged_expr(site_id), ["id", "embedding"])
    if not rows:
        log.info("No untagged UUIDs for site=%s — everything is already tagged.", site_id)
        return

    uids = [r["id"] for r in rows]
    embs = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
    log.info("%d untagged UUIDs for site=%s. Scanning %s...", len(uids), site_id, video)

    saves  = {u: 0 for u in uids}
    remain = len(uids)

    detector = SCRFD(resolve_model_path(cfg["detector_model"]), provider, device_id)
    arcface  = ArcFace(resolve_model_path(cfg["embedding_model"]), provider, device_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    idx = 0
    while remain > 0:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip != 0:
            idx += 1
            continue

        boxes, kps = detector.detect(frame, det_high)
        if len(boxes) == 0:
            boxes, kps = detector.detect(frame, det_low)

        for fi in range(len(boxes)):
            crop = align_face(frame, kps[fi])
            emb  = arcface.embed(crop)

            scores   = embs @ emb
            best_idx = int(np.argmax(scores))
            best_sc  = float(scores[best_idx])

            if best_sc < match_thr:
                continue

            uid = uids[best_idx]
            if saves[uid] >= per_uuid:
                continue

            x1, y1, x2, y2 = (max(0, int(v)) for v in boxes[fi])
            raw_crop = frame[y1:y2, x1:x2]
            if raw_crop.size == 0:
                continue

            out_path = out_dir / f"{uid[:8]}_f{idx}_face{fi}.jpg"
            cv2.imwrite(str(out_path), raw_crop)
            saves[uid] += 1
            log.info("Saved %s | score=%.4f", out_path.name, best_sc)

            if saves[uid] >= per_uuid:
                remain -= 1

        idx += 1

    cap.release()
    done    = sum(1 for v in saves.values() if v > 0)
    missing = [u for u, v in saves.items() if v == 0]
    log.info("Crops done: %d/%d UUIDs got at least 1 crop in %s/", done, len(uids), out_dir)
    if missing:
        log.info("%d UUIDs had no match in video (low quality / edge detections):", len(missing))
        for u in missing[:20]:
            print(f"  {u}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


# ── Contact sheet ─────────────────────────────────────────────────────────────

def make_sheet(in_dir: Path, out_path: Path, cols: int = 10, cell_size: int = 150,
               label_h: int = 24) -> None:
    """
    Combine all crop images in in_dir into a single grid image.
    Each tile is labelled with the first 8 chars of the UUID.
    """
    files = sorted(f for f in in_dir.glob("*.jpg") if "_full" not in f.name)
    if not files:
        raise SystemExit(f"No crop images found in {in_dir}")

    log.info("Building contact sheet: %d crops, %d cols...", len(files), cols)
    cell_h = cell_size + label_h
    rows   = (len(files) + cols - 1) // cols
    sheet  = np.full((rows * cell_h, cols * cell_size, 3), 255, dtype=np.uint8)

    for i, fpath in enumerate(files):
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        h, w   = img.shape[:2]
        scale  = cell_size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        rsz    = cv2.resize(img, (nw, nh))
        canvas = np.full((cell_size, cell_size, 3), 220, dtype=np.uint8)
        yo     = (cell_size - nh) // 2
        xo     = (cell_size - nw) // 2
        canvas[yo:yo + nh, xo:xo + nw] = rsz

        r, c = i // cols, i % cols
        y0, x0 = r * cell_h, c * cell_size
        sheet[y0:y0 + cell_size, x0:x0 + cell_size] = canvas

        short = fpath.stem.split("_")[0]
        cv2.putText(sheet, short, (x0 + 4, y0 + cell_size + label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), sheet)
    log.info("Contact sheet saved: %s  (%d cols × %d rows)", out_path, cols, rows)
    print(f"\nContact sheet: {out_path}")
    print("Each tile is labelled with the first 8 chars of the UUID.")
    print("Match to full UUID via filename in the crops folder or: python fr_tag.py list --all")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", type=Path, default=_BASE / "config.json")


def _add_site_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--site-id", default=None, help="Site identifier (required for crops)")
    p.add_argument("--camera-id", default=None, help="Camera identifier (optional metadata)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Inspect untagged face UUIDs")
    _add_config_arg(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # both (default workflow)
    p_both = sub.add_parser("both", help="Generate crops then contact sheet (recommended)")
    _add_config_arg(p_both)
    p_both.add_argument("video", help="Input video path")
    p_both.add_argument("--per-uuid",  type=int,  default=1,                help="Crops per UUID")
    p_both.add_argument("--out-dir",   type=Path, default=Path("uuid_crops"), help="Folder for crops")
    p_both.add_argument("--sheet-out", type=Path, default=Path("contact_sheet.jpg"))
    p_both.add_argument("--cols",      type=int,  default=10)
    p_both.add_argument("--cell-size", type=int,  default=150)
    _add_site_args(p_both)

    # crops only
    p_crops = sub.add_parser("crops", help="Save face crops for untagged UUIDs")
    _add_config_arg(p_crops)
    p_crops.add_argument("video",       help="Input video path")
    p_crops.add_argument("--per-uuid",  type=int,  default=1)
    p_crops.add_argument("--out-dir",   type=Path, default=Path("uuid_crops"))
    _add_site_args(p_crops)

    # sheet only
    p_sheet = sub.add_parser("sheet", help="Build contact sheet from existing crops")
    _add_config_arg(p_sheet)
    p_sheet.add_argument("--in-dir",   type=Path, default=Path("uuid_crops"))
    p_sheet.add_argument("--out",      type=Path, default=Path("contact_sheet.jpg"))
    p_sheet.add_argument("--cols",     type=int,  default=10)
    p_sheet.add_argument("--cell-size",type=int,  default=150)

    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8")).get("face_recognition", {})
    if getattr(args, "site_id", None):
        cfg["site_id"] = args.site_id
    if getattr(args, "camera_id", None):
        cfg["camera_id"] = args.camera_id

    if args.cmd in ("both", "crops"):
        make_crops(args.video, cfg, args.out_dir, per_uuid=args.per_uuid)

    if args.cmd in ("both", "sheet"):
        in_dir  = args.out_dir if args.cmd == "both" else args.in_dir
        out_p   = args.sheet_out if args.cmd == "both" else args.out
        cols    = args.cols
        cell_sz = args.cell_size
        make_sheet(in_dir, out_p, cols=cols, cell_size=cell_sz)


if __name__ == "__main__":
    main()
