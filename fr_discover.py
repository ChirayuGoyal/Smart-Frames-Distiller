"""
fr_discover.py — Discover unique people in a video, then register them by UUID (Stack B).

Step 1:  scan      — find every unique person, save crops + embeddings, print UUID table
Step 2:  (review)  — open contact_sheet.jpg to see who each UUID prefix belongs to
Step 3:  register  — attach a name + site to a discovered UUID, store in Milvus
"""
from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np

from faces.engine import SCRFD, ArcFace, align_face, resolve_model_path
from faces.persons import PersonDetector
from faces.service import list_identities
from faces.store import MilvusFaceStore, escape_milvus_string, resolve_site_camera

_BASE = Path(__file__).parent
log = logging.getLogger(__name__)


# ── Person clustering ─────────────────────────────────────────────────────────

class _Clusters:
    """Online clustering of face embeddings into unique-person groups."""

    def __init__(self, merge_thresh: float = 0.50):
        self.thresh = merge_thresh
        self._uids: list[str] = []
        self._cents: list[np.ndarray] = []
        self._embs: list[list] = []
        self._crops: list[np.ndarray] = []
        self._quals: list[float] = []
        self._counts: list[int] = []

    def add(self, emb: np.ndarray, raw_crop: np.ndarray, quality: float) -> str:
        if not self._uids:
            return self._new(emb, raw_crop, quality)

        cents = np.stack(self._cents)
        sims = cents @ emb
        best = int(np.argmax(sims))

        if float(sims[best]) >= self.thresh:
            self._embs[best].append(emb)
            self._counts[best] += 1
            if quality > self._quals[best]:
                self._crops[best] = raw_crop
                self._quals[best] = quality
            c = np.mean(self._embs[best], axis=0)
            norm = np.linalg.norm(c)
            self._cents[best] = c / norm if norm > 0 else c
            return self._uids[best]
        return self._new(emb, raw_crop, quality)

    def _new(self, emb, crop, qual) -> str:
        uid = str(uuid.uuid4())
        self._uids.append(uid)
        self._cents.append(emb.copy())
        self._embs.append([emb])
        self._crops.append(crop)
        self._quals.append(qual)
        self._counts.append(1)
        return uid

    def result(self) -> dict:
        return {
            uid: {
                "centroid": self._cents[i],
                "count": self._counts[i],
                "crop": self._crops[i],
            }
            for i, uid in enumerate(self._uids)
        }


def _build_sheet(out_dir: Path, cols: int = 8, cell: int = 140, label_h: int = 24) -> Path:
    files = sorted(out_dir.glob("*_best.jpg"))
    if not files:
        raise SystemExit(f"No *_best.jpg files found in {out_dir}")

    rows = (len(files) + cols - 1) // cols
    cell_h = cell + label_h
    sheet = np.full((rows * cell_h, cols * cell, 3), 245, dtype=np.uint8)

    for i, fpath in enumerate(files):
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        h, w = img.shape[:2]
        sc = cell / max(h, w, 1)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        rsz = cv2.resize(img, (nw, nh))
        tile = np.full((cell, cell, 3), 210, dtype=np.uint8)
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


def _store_from_cfg(cfg: dict) -> MilvusFaceStore:
    mc = cfg.get("milvus", {}) if isinstance(cfg.get("milvus"), dict) else {}
    return MilvusFaceStore(
        host=str(mc.get("host", "localhost")),
        port=int(mc.get("port", 19530)),
        collection=str(mc.get("collection", "face_registry")),
    )


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_scan(args) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")).get("face_recognition", {})
    out = Path(args.out_dir)
    stride = int(cfg.get("onboard_stride", 20))
    thresh = float(args.thresh or cfg.get("cluster_thresh", 0.50))
    face_thresh = float(cfg.get("det_threshold", 0.4))
    person_conf = float(cfg.get("person_conf", 0.4))
    cols = int(args.cols)
    provider = cfg.get("execution_provider", "CPUExecutionProvider")
    device_id = int(cfg.get("gpu_device_id", 0))

    person_det = PersonDetector(resolve_model_path(cfg["person_model"]), device=cfg.get("device", "auto"))
    face_det = SCRFD(resolve_model_path(cfg["detector_model"]), provider, device_id)
    embedder = ArcFace(resolve_model_path(cfg["embedding_model"]), provider, device_id)
    clusters = _Clusters(merge_thresh=thresh)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")

    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx = 0
    detections = 0

    log.info("Scanning %s  (stride=%d  cluster_thresh=%.2f) ...", args.video, stride, thresh)

    try:
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
                pad = int(max(px2 - px1, py2 - py1) * 0.05)
                rx1 = max(0, px1 - pad); ry1 = max(0, py1 - pad)
                rx2 = min(ow, px2 + pad); ry2 = min(oh, py2 + pad)
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size == 0:
                    continue

                f_boxes, f_kps = face_det.detect(roi, thresh=face_thresh)
                if len(f_boxes) == 0:
                    continue

                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in f_boxes]
                fi = int(np.argmax(areas))

                aligned = align_face(roi, f_kps[fi])
                emb = embedder.embed(aligned)

                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                qual = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                person_crop = frame[py1:py2, px1:px2]
                clusters.add(emb, person_crop, qual)
                detections += 1

            frame_idx += 1
            if frame_idx % 200 == 0:
                log.info("  frame %d / %d  |  clusters so far: %d", frame_idx, total, len(clusters._uids))
    finally:
        cap.release()

    result = clusters.result()
    if not result:
        print("No faces detected in video.")
        return

    out.mkdir(parents=True, exist_ok=True)
    for uid, data in result.items():
        np.save(str(out / f"{uid}.npy"), data["centroid"])
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
    path = _build_sheet(in_dir, cols=int(args.cols))
    print(f"Contact sheet saved: {path}")


def cmd_register(args) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")).get("face_recognition", {})
    site = args.site or cfg.get("site_id", "")
    if not site:
        raise SystemExit("--site is required (or set face_recognition.site_id in config.json)")

    site_id, camera_id = resolve_site_camera(cfg)
    if args.site:
        site_id = args.site

    from_dir = Path(args.from_dir)
    uid_input = args.uuid.strip()

    npy_files = list(from_dir.glob("*.npy"))
    match_file = None
    for f in npy_files:
        if f.stem == uid_input or f.stem.startswith(uid_input):
            match_file = f
            break

    if not match_file:
        available = [f.stem[:8] for f in npy_files]
        raise SystemExit(f"No embedding found for '{uid_input}' in {from_dir}/\nAvailable prefixes: {available}")

    full_uid = match_file.stem
    emb = np.load(str(match_file))

    store = _store_from_cfg(cfg)
    store.connect()
    try:
        store.get_or_create_collection(dim=512)
        if args.replace:
            site_esc = escape_milvus_string(site_id)
            name_esc = escape_milvus_string(args.name)
            removed = store.delete_by_expr(f'site_id == "{site_esc}" and name == "{name_esc}"')
            if removed:
                log.info("Removed %d old embeddings for '%s' @ '%s'", removed, args.name, site_id)

        store.upsert_row(
            uid=full_uid,
            person_id=full_uid,
            embedding=emb.tolist(),
            name=args.name,
            role="",
            department="",
            notes=args.notes or "",
            site_id=site_id,
            camera_id=camera_id,
        )
        print(f"\nRegistered  : {args.name}")
        print(f"Site        : {site_id}")
        print(f"UUID        : {full_uid}")
        print(f"Source      : {match_file.name}")
    finally:
        store.close()


def cmd_list(args) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")).get("face_recognition", {})
    site = args.site or cfg.get("site_id", "")
    if not site:
        raise SystemExit("--site is required")

    people = list_identities(site, cfg=cfg, include_untagged=False)
    if not people:
        print(f"No people registered at site '{site}'.")
        return

    print(f"\nSite: {site}  ({len(people)} people)\n")
    print(f"  {'ID':<36}  {'Name':<30}  {'Notes'}")
    print(f"  {'─'*36}  {'─'*30}  {'─'*20}")
    for p in people:
        print(f"  {p['id']:<36}  {p['name']:<30}  {p.get('notes','')}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Discover unique people in a video, then register them by UUID")
    p.add_argument("-c", "--config", default=str(_BASE / "config.json"))
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("scan", help="Find unique people in a video")
    ps.add_argument("video", help="Input video")
    ps.add_argument("--out-dir", default="uuid_crops", help="Where to save crops and embeddings")
    ps.add_argument("--thresh", default=None, type=float, help="Clustering threshold 0–1")
    ps.add_argument("--cols", default=8, type=int, help="Contact sheet columns")

    ph = sub.add_parser("sheet", help="Rebuild contact sheet from existing crops")
    ph.add_argument("--in-dir", default="uuid_crops")
    ph.add_argument("--cols", default=8, type=int)

    pr = sub.add_parser("register", help="Tag a discovered UUID with a name and site")
    pr.add_argument("--uuid", required=True, help="UUID or 8-char prefix from scan")
    pr.add_argument("--name", required=True, help="Person name")
    pr.add_argument("--site", default=None, help="Site ID")
    pr.add_argument("--notes", default="")
    pr.add_argument("--replace", action="store_true", help="Remove existing embeddings first")
    pr.add_argument("--from-dir", default="uuid_crops", help="Folder containing .npy files")

    pl = sub.add_parser("list", help="List registered people at a site")
    pl.add_argument("--site", default=None)

    args = p.parse_args()
    {"scan": cmd_scan, "sheet": cmd_sheet, "register": cmd_register, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    main()
