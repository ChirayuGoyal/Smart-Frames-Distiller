"""
fr_discover.py — Discover unique people in a video, then register them by UUID.

Step 1:  scan      — find every unique person, save crops + embeddings, print UUID table
Step 2:  (review)  — open contact_sheet.jpg to see who each UUID prefix belongs to
Step 3:  register  — attach a name + site to a discovered UUID, store in Milvus

Usage:
    python3 fr_discover.py scan     video.mp4 [--out-dir uuid_crops] [--thresh 0.50]
    python3 fr_discover.py sheet              [--in-dir  uuid_crops] [--cols 8]
    python3 fr_discover.py register --uuid abc12345 --name "Alice" --site site-001
    python3 fr_discover.py list     --site site-001

The threshold controls how similar two face embeddings must be to be grouped as the
same person (cosine similarity, 0–1).  Lower = more clusters; higher = fewer.
Typical range: 0.45 (strict) – 0.60 (lenient).  Default 0.50 works well for one video.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from fr_core import (
    PersonDetector, create_detector, FaceEmbedder, FaceDB,
    align_face, preprocess_face, resolve_model,
)

_BASE = Path(__file__).parent
log   = logging.getLogger(__name__)


# ── Person clustering ─────────────────────────────────────────────────────────

class _Clusters:
    """
    Online clustering of face embeddings into unique-person groups.
    Each cluster represents one person found in the video.
    """

    def __init__(self, merge_thresh: float = 0.50):
        self.thresh   = merge_thresh
        self._uids:   list[str]         = []
        self._cents:  list[np.ndarray]  = []   # unit-norm centroid per cluster
        self._embs:   list[list]        = []   # all embeddings per cluster
        self._crops:  list[np.ndarray]  = []   # best (sharpest) crop
        self._quals:  list[float]       = []   # sharpness of best crop
        self._counts: list[int]         = []   # frames seen

    def add(self, emb: np.ndarray, raw_crop: np.ndarray, quality: float) -> str:
        """
        Add one face embedding. Returns the UUID of the cluster it was assigned to.
        """
        if not self._uids:
            return self._new(emb, raw_crop, quality)

        cents = np.stack(self._cents)   # (N, 512)
        sims  = cents @ emb             # (N,) cosine similarity (both unit-norm)
        best  = int(np.argmax(sims))

        if float(sims[best]) >= self.thresh:
            self._embs[best].append(emb)
            self._counts[best] += 1
            if quality > self._quals[best]:
                self._crops[best] = raw_crop
                self._quals[best] = quality
            # Update running centroid
            c    = np.mean(self._embs[best], axis=0)
            norm = np.linalg.norm(c)
            self._cents[best] = c / norm if norm > 0 else c
            return self._uids[best]
        else:
            return self._new(emb, raw_crop, quality)

    def _new(self, emb, crop, qual) -> str:
        import uuid as _u
        uid = str(_u.uuid4())
        self._uids.append(uid)
        self._cents.append(emb.copy())
        self._embs.append([emb])
        self._crops.append(crop)
        self._quals.append(qual)
        self._counts.append(1)
        return uid

    def result(self) -> dict:
        """Returns {uuid: {"centroid": ndarray, "count": int, "crop": ndarray}}"""
        return {
            uid: {
                "centroid": self._cents[i],
                "count":    self._counts[i],
                "crop":     self._crops[i],
            }
            for i, uid in enumerate(self._uids)
        }


# ── Contact sheet ─────────────────────────────────────────────────────────────

def _build_sheet(out_dir: Path, cols: int = 8, cell: int = 140, label_h: int = 24) -> Path:
    files = sorted(out_dir.glob("*_best.jpg"))
    if not files:
        raise SystemExit(f"No *_best.jpg files found in {out_dir}")

    rows   = (len(files) + cols - 1) // cols
    cell_h = cell + label_h
    sheet  = np.full((rows * cell_h, cols * cell, 3), 245, dtype=np.uint8)

    for i, fpath in enumerate(files):
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        h, w   = img.shape[:2]
        sc     = cell / max(h, w, 1)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        rsz    = cv2.resize(img, (nw, nh))
        tile   = np.full((cell, cell, 3), 210, dtype=np.uint8)
        yo, xo = (cell - nh) // 2, (cell - nw) // 2
        tile[yo:yo+nh, xo:xo+nw] = rsz

        r, c = divmod(i, cols)
        y0, x0 = r * cell_h, c * cell
        sheet[y0:y0+cell, x0:x0+cell] = tile

        uid8 = fpath.stem[:8]
        cv2.putText(sheet, uid8,
                    (x0 + 3, y0 + cell + label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)

    out = out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(out), sheet)
    return out


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_scan(args) -> None:
    cfg         = json.loads(Path(args.config).read_text(encoding="utf-8")).get("face_recognition", {})
    out         = Path(args.out_dir)
    stride      = int(cfg.get("onboard_stride", 20))
    thresh      = float(args.thresh or cfg.get("cluster_thresh", 0.50))
    face_thresh = float(cfg.get("det_threshold", 0.4))
    person_conf = float(cfg.get("person_conf", 0.4))
    cols        = int(args.cols)

    person_det = PersonDetector(cfg["person_model"])
    face_det   = create_detector(cfg)
    embedder   = FaceEmbedder(cfg["embedding_model"])
    clusters   = _Clusters(merge_thresh=thresh)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")

    oh         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx  = 0
    detections = 0

    log.info("Scanning %s  (stride=%d  cluster_thresh=%.2f) ...", args.video, stride, thresh)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        p_boxes = person_det.detect(frame, conf=person_conf)

        for pbox in p_boxes:
            px1, py1, px2, py2 = (int(v) for v in pbox)
            pad  = int(max(px2 - px1, py2 - py1) * 0.05)
            rx1  = max(0, px1 - pad);  ry1 = max(0, py1 - pad)
            rx2  = min(ow, px2 + pad); ry2 = min(oh, py2 + pad)
            roi  = frame[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue

            f_boxes, f_kps = face_det.detect(roi, thresh=face_thresh)
            if len(f_boxes) == 0:
                continue

            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in f_boxes]
            fi    = int(np.argmax(areas))

            aligned = align_face(roi, f_kps[fi])
            cleaned = preprocess_face(aligned)
            emb     = embedder.embed(cleaned)

            # Quality score on the face crop (sharpness)
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            qual = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Save the person crop for contact sheet (more recognisable than face-only)
            person_crop = frame[py1:py2, px1:px2]

            clusters.add(emb, person_crop, qual)
            detections += 1

        frame_idx += 1
        if frame_idx % 200 == 0:
            log.info("  frame %d / %d  |  clusters so far: %d", frame_idx, total, len(clusters._uids))

    cap.release()

    result = clusters.result()
    if not result:
        print("No faces detected in video.")
        return

    out.mkdir(parents=True, exist_ok=True)
    for uid, data in result.items():
        # Save centroid embedding (used later for registration)
        np.save(str(out / f"{uid}.npy"), data["centroid"])
        # Save best crop image
        crop = data["crop"]
        if crop is not None and crop.size > 0:
            cv2.imwrite(str(out / f"{uid[:8]}_best.jpg"), crop)

    sheet_path = _build_sheet(out, cols=cols)

    print(f"\n{'─'*55}")
    print(f"  Video : {args.video}")
    print(f"  Frames: {frame_idx}  |  Face detections: {detections}")
    print(f"  Unique people found: {len(result)}")
    print(f"{'─'*55}")
    print(f"\n  {'UUID (prefix)':<14}  {'Seen (frames)':>14}")
    print(f"  {'─'*14}  {'─'*14}")
    for uid, data in sorted(result.items(), key=lambda x: -x[1]["count"]):
        print(f"  {uid[:8]}...     {data['count']:>14}")
    print(f"\n  Crops      : {out}/")
    print(f"  Sheet      : {sheet_path}")
    print(f"\n  To register a person:")
    print(f"    python3 fr_discover.py register --uuid <UUID_PREFIX> --name \"Name\" --site <site-id>")


def cmd_sheet(args) -> None:
    in_dir = Path(args.in_dir)
    path   = _build_sheet(in_dir, cols=int(args.cols))
    print(f"Contact sheet saved: {path}")


def cmd_register(args) -> None:
    cfg  = json.loads(Path(args.config).read_text(encoding="utf-8")).get("face_recognition", {})
    site = args.site or cfg.get("site_id", "")
    if not site:
        raise SystemExit("--site is required (or set face_recognition.site_id in config.json)")

    from_dir  = Path(args.from_dir)
    uid_input = args.uuid.strip()

    # Match UUID by full string or 8-char prefix
    npy_files = list(from_dir.glob("*.npy"))
    match_file = None
    for f in npy_files:
        if f.stem == uid_input or f.stem.startswith(uid_input):
            match_file = f
            break

    if not match_file:
        available = [f.stem[:8] for f in npy_files]
        raise SystemExit(
            f"No embedding found for '{uid_input}' in {from_dir}/\n"
            f"Available prefixes: {available}"
        )

    full_uid = match_file.stem
    emb      = np.load(str(match_file))

    mc = cfg.get("milvus", {})
    db = FaceDB(mc["host"], int(mc.get("port", 19530)), mc.get("collection", "face_registry"))

    if args.replace:
        removed = db.delete_person(site, args.name)
        if removed:
            log.info("Removed %d old embeddings for '%s' @ '%s'", removed, args.name, site)

    stored_uid = db.add(site, args.name, emb, notes=args.notes or "", uid=full_uid)

    print(f"\nRegistered  : {args.name}")
    print(f"Site        : {site}")
    print(f"UUID        : {stored_uid}")
    print(f"Source      : {match_file.name}")


def cmd_list(args) -> None:
    cfg  = json.loads(Path(args.config).read_text(encoding="utf-8")).get("face_recognition", {})
    site = args.site or cfg.get("site_id", "")
    if not site:
        raise SystemExit("--site is required")

    mc     = cfg.get("milvus", {})
    db     = FaceDB(mc["host"], int(mc.get("port", 19530)), mc.get("collection", "face_registry"))
    people = db.list_people(site)

    if not people:
        print(f"No people registered at site '{site}'.")
        return

    print(f"\nSite: {site}  ({len(people)} people)\n")
    print(f"  {'Name':<30}  {'Embeddings':>10}  Notes")
    print(f"  {'─'*30}  {'─'*10}  {'─'*20}")
    for p in people:
        print(f"  {p['name']:<30}  {p['embeddings']:>10}  {p.get('notes','')}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(
        description="Discover unique people in a video, then register them by UUID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-c", "--config", default=str(_BASE / "config.json"))
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── scan ──────────────────────────────────────────────────────────────────
    ps = sub.add_parser("scan", help="Find unique people in a video")
    ps.add_argument("video",                           help="Input video")
    ps.add_argument("--out-dir", default="uuid_crops", help="Where to save crops and embeddings")
    ps.add_argument("--thresh",  default=None, type=float,
                    help="Clustering threshold 0–1 (default 0.50: higher = fewer clusters)")
    ps.add_argument("--cols",    default=8,    type=int,  help="Contact sheet columns")

    # ── sheet ─────────────────────────────────────────────────────────────────
    ph = sub.add_parser("sheet", help="Rebuild contact sheet from existing crops")
    ph.add_argument("--in-dir", default="uuid_crops")
    ph.add_argument("--cols",   default=8, type=int)

    # ── register ──────────────────────────────────────────────────────────────
    pr = sub.add_parser("register", help="Tag a discovered UUID with a name and site")
    pr.add_argument("--uuid",     required=True, help="UUID or 8-char prefix from scan")
    pr.add_argument("--name",     required=True, help="Person name")
    pr.add_argument("--site",     default=None,  help="Site ID")
    pr.add_argument("--notes",    default="")
    pr.add_argument("--replace",  action="store_true",
                    help="Remove existing embeddings for this name first")
    pr.add_argument("--from-dir", default="uuid_crops",
                    help="Folder containing .npy files from scan (default: uuid_crops)")

    # ── list ──────────────────────────────────────────────────────────────────
    pl = sub.add_parser("list", help="List registered people at a site")
    pl.add_argument("--site", default=None)

    args = p.parse_args()
    {"scan": cmd_scan, "sheet": cmd_sheet, "register": cmd_register, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    main()
