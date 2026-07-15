"""
fr_onboard.py — Manage people for a site's face recognition registry.

Subcommands:
    add     — Enroll a person from a video or image
    list    — Show all people registered at a site
    delete  — Remove a person from a site
    verify  — Run recognition on a video to test enrollments

Usage:
    python3 fr_onboard.py add    --site site-001 --name "Alice" video.mp4 [--frames 8]
    python3 fr_onboard.py add    --site site-001 --name "Alice" --image alice.jpg
    python3 fr_onboard.py list   --site site-001
    python3 fr_onboard.py delete --site site-001 --name "Alice"
    python3 fr_onboard.py verify --site site-001 video.mp4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from fr_core import create_detector, FaceEmbedder, FaceDB, align_face, resolve_model

_BASE = Path(__file__).parent
log   = logging.getLogger(__name__)


# ── Config helper ─────────────────────────────────────────────────────────────

def _load(args) -> dict:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    return cfg.get("face_recognition", {})


def _make_db(cfg: dict) -> FaceDB:
    mc = cfg.get("milvus", {})
    return FaceDB(mc["host"], int(mc.get("port", 19530)), mc.get("collection", "face_registry"))


def _make_models(cfg: dict):
    det = create_detector(cfg)
    emb = FaceEmbedder(cfg["embedding_model"])
    return det, emb


# ── Face sampling helpers ─────────────────────────────────────────────────────

def _quality(crop: np.ndarray) -> float:
    """Sharpness score via Laplacian variance. Higher = sharper."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _best_face(frame: np.ndarray, boxes, kps):
    """Return (aligned_crop, quality_score) for the largest face in the frame."""
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    i     = int(np.argmax(areas))
    x1, y1, x2, y2 = (max(0, int(v)) for v in boxes[i])
    crop  = frame[y1:y2, x1:x2]
    qual  = _quality(crop) if crop.size else 0.0
    aligned = align_face(frame, kps[i])
    return aligned, qual


def _collect_from_video(video_path: str, detector, embedder: FaceEmbedder,
                        det_thresh: float, max_frames: int, stride: int) -> list[np.ndarray]:
    """
    Sample up to max_frames face embeddings from video.
    Returns list of embedding vectors sorted by face sharpness (best first).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    samples: list[tuple[np.ndarray, float]] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        boxes, kps = detector.detect(frame, det_thresh)
        if len(boxes):
            aligned, qual = _best_face(frame, boxes, kps)
            emb = embedder.embed(aligned)
            samples.append((emb, qual))
            if len(samples) >= max_frames * 3:
                break

        frame_idx += 1

    cap.release()

    if not samples:
        return []

    samples.sort(key=lambda x: -x[1])           # best sharpness first
    return [s[0] for s in samples[:max_frames]]


def _collect_from_image(image_path: str, detector, embedder: FaceEmbedder,
                        det_thresh: float) -> list[np.ndarray]:
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    boxes, kps = detector.detect(frame, det_thresh)
    if not len(boxes):
        raise RuntimeError("No face detected in image.")
    aligned, _ = _best_face(frame, boxes, kps)
    return [embedder.embed(aligned)]


def _average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    avg  = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_add(args) -> None:
    cfg  = _load(args)
    site = args.site or cfg.get("site_id", "")
    if not site:
        raise SystemExit("--site is required (or set face_recognition.site_id in config.json)")

    det_thresh = float(cfg.get("det_threshold", 0.5))
    det, emb   = _make_models(cfg)

    if args.image:
        embeddings = _collect_from_image(args.image, det, emb, det_thresh)
        source     = f"image {args.image}"
    elif args.source:
        stride = max(1, int(cfg.get("onboard_stride", 20)))
        embeddings = _collect_from_video(args.source, det, emb, det_thresh,
                                         max_frames=args.frames, stride=stride)
        source     = f"video {args.source}"
    else:
        raise SystemExit("Provide a video path or --image")

    if not embeddings:
        raise SystemExit("No usable faces found — try a different source.")

    # One averaged embedding per person is usually enough for recognition.
    # To get multiple embeddings (better coverage), pass --frames > 1.
    if args.single_embedding:
        final_embs = [_average_embeddings(embeddings)]
        log.info("Averaged %d frames → 1 embedding", len(embeddings))
    else:
        final_embs = embeddings
        log.info("Storing %d separate embeddings (from %s)", len(final_embs), source)

    db = _make_db(cfg)

    if args.replace:
        removed = db.delete_person(site, args.name)
        if removed:
            log.info("Removed %d old embeddings for '%s'", removed, args.name)

    notes = args.notes or ""
    for i, e in enumerate(final_embs):
        uid = db.add(site, args.name, e, notes)
        log.info("Stored embedding %d/%d for '%s' @ site '%s' [%s]",
                 i + 1, len(final_embs), args.name, site, uid[:8])

    total = db.count(site)
    print(f"\nEnrolled: {args.name} @ {site}")
    print(f"Embeddings stored: {len(final_embs)}")
    print(f"Total embeddings at site: {total}")


def cmd_list(args) -> None:
    cfg  = _load(args)
    site = args.site or cfg.get("site_id", "")
    if not site:
        raise SystemExit("--site is required")

    db      = _make_db(cfg)
    people  = db.list_people(site)

    if not people:
        print(f"No people enrolled at site '{site}'.")
        return

    print(f"\nSite: {site}  ({len(people)} people)\n")
    print(f"{'Name':<30}  {'Embeddings':>10}  Notes")
    print("-" * 60)
    for p in people:
        print(f"{p['name']:<30}  {p['embeddings']:>10}  {p['notes']}")


def cmd_delete(args) -> None:
    cfg  = _load(args)
    site = args.site or cfg.get("site_id", "")
    if not site:
        raise SystemExit("--site is required")

    db = _make_db(cfg)
    n  = db.delete_person(site, args.name)
    if n:
        print(f"Deleted {n} embedding(s) for '{args.name}' @ site '{site}'.")
    else:
        print(f"'{args.name}' not found at site '{site}'.")


def cmd_verify(args) -> None:
    """Run face recognition on a video to see who is identified."""
    cfg  = _load(args)
    site = args.site or cfg.get("site_id", "")
    if not site:
        raise SystemExit("--site is required")

    det_thresh = float(cfg.get("det_threshold", 0.5))
    sim_thresh = float(cfg.get("similarity_threshold", 0.45))
    stride     = int(cfg.get("frame_skip", 3))

    det, emb   = _make_models(cfg)
    db         = _make_db(cfg)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.source}")

    seen: dict[str, float] = {}   # name → best score
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        boxes, kps = det.detect(frame, det_thresh)
        for i in range(len(boxes)):
            aligned = align_face(frame, kps[i])
            vec     = emb.embed(aligned)
            match   = db.search(site, vec, sim_thresh)
            name    = match["name"] if match else "<unknown>"
            sc      = match["score"] if match else 0.0
            if sc > seen.get(name, 0.0):
                seen[name] = sc

        frame_idx += 1

    cap.release()

    print(f"\nVerify results for site '{site}'  ({frame_idx} frames scanned):\n")
    if not seen:
        print("  No faces detected.")
        return
    for name, score in sorted(seen.items(), key=lambda x: -x[1]):
        flag = "✓" if name != "<unknown>" else " "
        print(f"  {flag}  {name:<30}  best={int(score*100)}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Face recognition person management")
    p.add_argument("-c", "--config", default=str(_BASE / "config.json"))
    p.add_argument("--site", default=None, help="Site ID (overrides config)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # add
    pa = sub.add_parser("add", help="Enroll a person")
    pa.add_argument("source", nargs="?", help="Video file path")
    pa.add_argument("--image",            help="Image file instead of video")
    pa.add_argument("--name",   required=True, help="Person name")
    pa.add_argument("--notes",  default="",   help="Optional notes")
    pa.add_argument("--frames", type=int, default=8,
                    help="Max face samples to collect from video (default 8)")
    pa.add_argument("--replace", action="store_true",
                    help="Delete existing embeddings for this person before adding")
    pa.add_argument("--single-embedding", action="store_true",
                    help="Average all samples into one embedding (default: store separately)")

    # list
    pl = sub.add_parser("list", help="List enrolled people at a site")

    # delete
    pd = sub.add_parser("delete", help="Remove a person from a site")
    pd.add_argument("--name", required=True)

    # verify
    pv = sub.add_parser("verify", help="Test recognition on a video")
    pv.add_argument("source", help="Video file path")

    args = p.parse_args()
    {"add": cmd_add, "list": cmd_list, "delete": cmd_delete, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
