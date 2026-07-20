"""
faces/service.py — High-level, library-friendly face operations without CLI/argparse/SystemExit.

Exports:
    ingest_video     — Extract, deduplicate, and ingest embeddings from a video clip.
    merge_identities — Cluster and merge similar embeddings into canonical person_ids.
    onboard_identity — Enroll a person from a video or image.
    tag_identity     — Set or update metadata (`name`, `role`, `department`, `notes`) for a UUID.
    list_identities  — Query and list identities stored for a site.
    delete_identity  — Delete a person or specific embedding by ID/name.
    search_face      — Search for a face image or embedding vector across a site.
"""
from __future__ import annotations

import logging
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from faces.engine import SCRFD, ArcFace, align_face, resolve_model_path
from faces.store import (
    FaceStore,
    FaceStoreError,
    MilvusFaceStore,
    SiteIdRequiredError,
    escape_milvus_string,
    require_site_id,
    resolve_site_camera,
    site_expr,
)

log = logging.getLogger(__name__)


def _store_from_cfg(cfg: dict) -> FaceStore:
    milvus_cfg = cfg.get("milvus", {}) if isinstance(cfg.get("milvus"), dict) else {}
    host = str(milvus_cfg.get("host", "localhost"))
    port = int(milvus_cfg.get("port", 19530))
    collection = str(milvus_cfg.get("collection", "face_registry"))
    return MilvusFaceStore(host=host, port=port, collection=collection)


# ── Dedup & Quality Helpers ───────────────────────────────────────────────────

class _HybridDedup:
    """Gate 1: in-memory cosine cache. Gate 2: store ANN search within site_id."""

    def __init__(self, store: FaceStore, site_id: str, thresh: float) -> None:
        self._store = store
        self._site_id = site_id
        self._thresh = thresh
        self._cache = np.empty((0, 0), dtype=np.float32)
        self._ids: list[str] = []
        self.g1_hits = self.g2_hits = self.g2_calls = 0

    def is_duplicate(self, emb: np.ndarray) -> bool:
        if len(self._ids) > 0 and float(np.max(self._cache @ emb)) >= self._thresh:
            self.g1_hits += 1
            return True
        self.g2_calls += 1
        res = self._store.search(emb, self._site_id, limit=1, output_fields=["id"])
        if res and res[0]:
            score = float(res[0].get("score", 0.0))
            if score >= self._thresh:
                self.g2_hits += 1
                return True
        return False

    def register(self, emb: np.ndarray, uid: str) -> None:
        self._cache = emb[np.newaxis] if len(self._ids) == 0 else np.vstack([self._cache, emb[np.newaxis]])
        self._ids.append(uid)

    def summary(self) -> str:
        return f"Gate1 skipped: {self.g1_hits} | Gate2 called: {self.g2_calls} | Gate2 skipped: {self.g2_hits}"


def _check_quality(frame: np.ndarray, box, min_size: int, blur_thresh: float, min_asp: float, max_asp: float) -> tuple[bool, str]:
    x1, y1, x2, y2 = (max(0, int(v)) for v in box)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False, "invalid_box"
    if w < min_size or h < min_size:
        return False, "too_small"
    asp = w / h
    if asp < min_asp or asp > max_asp:
        return False, "bad_aspect"
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "empty_crop"
    blur = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if blur < blur_thresh:
        return False, "too_blurry"
    return True, "ok"


def _calibrate(video: str | Path, detector: SCRFD, det_high: float, det_low: float, stride: int, size_pct: float, blur_pct: float, min_samples: int) -> tuple[int, float]:
    sizes, blurs = [], []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 20, 30.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, stride) == 0:
                boxes, _ = detector.detect(frame, det_high)
                if len(boxes) == 0:
                    boxes, _ = detector.detect(frame, det_low)
                for box in boxes:
                    x1, y1, x2, y2 = (max(0, int(v)) for v in box)
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    sizes.append(min(w, h))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        blurs.append(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            idx += 1
    finally:
        cap.release()

    if len(sizes) < min_samples:
        log.warning("Calibration: only %d samples (need %d) — using fallbacks (size=20, blur=30).", len(sizes), min_samples)
        return 20, 30.0

    min_size = int(np.percentile(sizes, size_pct))
    blur_floor = float(np.percentile(blurs, blur_pct))
    log.info("Calibration: %d samples | min_face_size=%d | blur_thresh=%.1f", len(sizes), min_size, blur_floor)
    return min_size, blur_floor


# ── Centroid Clustering Helpers ───────────────────────────────────────────────

class _CentroidStore:
    def __init__(self, thresh: float) -> None:
        self.thresh = thresh
        self.centroids = np.empty((0, 0), dtype=np.float32)
        self.counts: list[int] = []
        self.members: list[list[str]] = []

    def add(self, uid: str, emb: np.ndarray) -> None:
        if len(self.counts) == 0:
            self._new(uid, emb)
            return
        sims = self.centroids @ emb
        best = int(np.argmax(sims))
        if float(sims[best]) >= self.thresh:
            self._update(best, uid, emb)
        else:
            self._new(uid, emb)

    def _new(self, uid: str, emb: np.ndarray) -> None:
        self.centroids = emb[np.newaxis] if len(self.counts) == 0 else np.vstack([self.centroids, emb[np.newaxis]])
        self.counts.append(1)
        self.members.append([uid])

    def _update(self, idx: int, uid: str, emb: np.ndarray) -> None:
        n = self.counts[idx]
        c = (self.centroids[idx] * n + emb) / (n + 1)
        norm = np.linalg.norm(c)
        self.centroids[idx] = c / norm if norm > 0 else c
        self.counts[idx] += 1
        self.members[idx].append(uid)


def _merge_centroids(centroids: np.ndarray, counts: list, members: list, thresh: float) -> tuple[np.ndarray, list, list]:
    centroids = centroids.copy()
    counts, members = list(counts), [list(m) for m in members]
    changed = True
    while changed and len(centroids) > 1:
        changed = False
        sim = centroids @ centroids.T
        np.fill_diagonal(sim, -1.0)
        i, j = np.unravel_index(int(np.argmax(sim)), sim.shape)
        if sim[i, j] < thresh:
            break
        ni, nj = counts[i], counts[j]
        c = (centroids[i] * ni + centroids[j] * nj) / (ni + nj)
        norm = np.linalg.norm(c)
        centroids[i] = c / norm if norm > 0 else c
        counts[i] = ni + nj
        members[i].extend(members[j])
        counts.pop(j)
        members.pop(j)
        centroids = np.delete(centroids, j, axis=0)
        changed = True
    return centroids, counts, members


# ── Service Operations ────────────────────────────────────────────────────────

def ingest_video(video: str | Path, cfg: dict, *, store: FaceStore | None = None) -> dict:
    """Extract, filter, deduplicate, and ingest face embeddings from a video."""
    site_id, camera_id = resolve_site_camera(cfg)
    require_site_id(site_id, tool="ingest_video")

    provider = cfg.get("execution_provider") or cfg.get("device", "auto")
    device_id = int(cfg.get("gpu_device_id", 0))
    det_high = float(cfg.get("det_thresh_high", 0.5))
    det_low = float(cfg.get("det_thresh_low", 0.35))
    dedup_thr = float(cfg.get("dedup_thresh", 0.95))
    skip = max(1, int(cfg.get("frame_skip", 2)))
    batch_sz = int(cfg.get("milvus", {}).get("batch_size", 100))

    q = cfg.get("quality", {})
    size_pct = float(q.get("face_size_percentile", 10))
    blur_pct = float(q.get("blur_percentile", 10))
    min_asp = float(q.get("min_aspect", 0.6))
    max_asp = float(q.get("max_aspect", 1.8))
    cal_stride = max(1, int(q.get("sample_frame_stride", 5)))
    min_samp = int(q.get("min_samples_required", 30))

    detector = SCRFD(resolve_model_path(cfg["detector_model"]), provider, device_id)
    arcface = ArcFace(resolve_model_path(cfg["embedding_model"]), provider, device_id)

    min_size, blur_thresh = _calibrate(video, detector, det_high, det_low, cal_stride, size_pct, blur_pct, min_samp)

    owns_store = False
    if store is None:
        store = _store_from_cfg(cfg)
        store.connect()
        owns_store = True

    try:
        store.get_or_create_collection(dim=512)
        dedup = _HybridDedup(store, site_id, dedup_thr)

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        batch_ids, batch_embs = [], []
        counts = Counter()
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % skip != 0:
                    frame_idx += 1
                    continue

                boxes, kps = detector.detect(frame, det_high)
                if len(boxes) == 0:
                    boxes, kps = detector.detect(frame, det_low)

                for i in range(len(boxes)):
                    counts["detected"] += 1
                    ok_q, reason = _check_quality(frame, boxes[i], min_size, blur_thresh, min_asp, max_asp)
                    if not ok_q:
                        counts[f"rejected_{reason}"] += 1
                        continue

                    crop = align_face(frame, kps[i])
                    emb = arcface.embed(crop)

                    if dedup.is_duplicate(emb):
                        counts["dedup_skipped"] += 1
                        continue

                    uid = str(uuid.uuid4())
                    dedup.register(emb, uid)
                    batch_ids.append(uid)
                    batch_embs.append(emb.tolist())
                    counts["inserted"] += 1

                    if len(batch_ids) >= batch_sz:
                        store.insert_batch(batch_ids, batch_embs, site_id, camera_id)
                        batch_ids, batch_embs = [], []

                frame_idx += 1
        finally:
            cap.release()

        if batch_ids:
            store.insert_batch(batch_ids, batch_embs, site_id, camera_id)

        return {
            "site_id": site_id,
            "camera_id": camera_id,
            "processed_frames": frame_idx,
            "detected": counts["detected"],
            "inserted": counts["inserted"],
            "dedup_skipped": counts["dedup_skipped"],
            "rejections": {k: v for k, v in counts.items() if k.startswith("rejected_")},
            "dedup_summary": dedup.summary(),
        }
    finally:
        if owns_store:
            store.close()


def merge_identities(cfg: dict, *, store: FaceStore | None = None, dry_run: bool = False) -> dict:
    """Cluster and merge close face embeddings across a site into canonical person_ids."""
    site_id, _ = resolve_site_camera(cfg)
    require_site_id(site_id, tool="merge_identities")

    # Default to 0.55 for cosine similarity (`centroids @ emb` dot product of unit vectors)
    thresh = float(cfg.get("merge_thresh", 0.55))

    owns_store = False
    if store is None:
        store = _store_from_cfg(cfg)
        store.connect()
        owns_store = True

    try:
        store.require_collection()
        rows = store.fetch_site_rows(site_id)
        if not rows:
            return {"site_id": site_id, "clusters": 0, "updates": 0, "conflicts": 0, "dry_run": dry_run}

        rows = sorted(rows, key=lambda r: str(r.get("id", "")))
        by_id = {str(r["id"]): r for r in rows if r.get("id")}

        centroid_store = _CentroidStore(thresh)
        for r in rows:
            if r.get("id") and r.get("embedding"):
                centroid_store.add(str(r["id"]), np.array(r["embedding"], dtype=np.float32))

        _, _, clusters = _merge_centroids(centroid_store.centroids, centroid_store.counts, centroid_store.members, thresh)

        updates, conflicts = [], []
        for comp in clusters:
            pids = {str(by_id[u]["person_id"]) for u in comp if by_id[u].get("person_id") and str(by_id[u]["person_id"]) != u}
            names = {str(by_id[u]["name"]) for u in comp if by_id[u].get("name")}

            if len(pids) > 1 or len(names) > 1:
                conflicts.append({"ids": comp, "person_ids": list(pids), "names": list(names)})
                continue

            canon_pid = next(iter(pids)) if pids else sorted(comp)[0]
            canon_name = next(iter(names)) if names else ""

            for uid in comp:
                r = by_id[uid]
                if str(r.get("person_id", "")) != canon_pid or (canon_name and str(r.get("name", "")) != canon_name):
                    updates.append({
                        "uid": uid,
                        "person_id": canon_pid,
                        "embedding": r["embedding"],
                        "name": canon_name or str(r.get("name", "")),
                        "role": str(r.get("role", "")),
                        "department": str(r.get("department", "")),
                        "notes": str(r.get("notes", "")),
                        "site_id": str(r.get("site_id", site_id)),
                        "camera_id": str(r.get("camera_id", "")),
                    })

        if not dry_run:
            for u in updates:
                store.upsert_row(**u)

        return {
            "site_id": site_id,
            "clusters": len(clusters),
            "updates": len(updates),
            "conflicts": len(conflicts),
            "conflict_details": conflicts,
            "dry_run": dry_run,
        }
    finally:
        if owns_store:
            store.close()


def onboard_identity(
    name: str,
    video_or_image: str | Path,
    cfg: dict,
    *,
    store: FaceStore | None = None,
    max_frames: int = 8,
    single_embedding: bool = True,
    replace: bool = False,
    notes: str = "",
) -> dict:
    """Enroll a person by extracting quality face crops from an image or video."""
    site_id, camera_id = resolve_site_camera(cfg)
    require_site_id(site_id, tool="onboard_identity")
    if not name or not name.strip():
        raise ValueError("Person name is required for onboarding.")
    name = name.strip()

    provider = cfg.get("execution_provider") or cfg.get("device", "auto")
    device_id = int(cfg.get("gpu_device_id", 0))
    det_thresh = float(cfg.get("det_threshold", 0.5))

    detector = SCRFD(resolve_model_path(cfg["detector_model"]), provider, device_id)
    arcface = ArcFace(resolve_model_path(cfg["embedding_model"]), provider, device_id)

    path = Path(video_or_image)
    if not path.exists():
        raise FileNotFoundError(f"Source path not found: {path}")

    embeddings: list[np.ndarray] = []
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {path}")
        boxes, kps = detector.detect(frame, det_thresh)
        if not len(boxes):
            raise RuntimeError("No face detected in image.")
        # Best face
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        i = int(np.argmax(areas))
        aligned = align_face(frame, kps[i])
        embeddings.append(arcface.embed(aligned))
    else:
        stride = max(1, int(cfg.get("onboard_stride", 20)))
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        samples: list[tuple[np.ndarray, float]] = []
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    boxes, kps = detector.detect(frame, det_thresh)
                    if len(boxes):
                        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                        i = int(np.argmax(areas))
                        x1, y1, x2, y2 = (max(0, int(v)) for v in boxes[i])
                        crop = frame[y1:y2, x1:x2]
                        qual = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size else 0.0
                        aligned = align_face(frame, kps[i])
                        samples.append((arcface.embed(aligned), qual))
                        if len(samples) >= max_frames * 3:
                            break
                frame_idx += 1
        finally:
            cap.release()

        if not samples:
            raise RuntimeError(f"No usable faces found in video: {path}")
        samples.sort(key=lambda x: -x[1])
        embeddings = [s[0] for s in samples[:max_frames]]

    if single_embedding:
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        final_embs = [avg / norm if norm > 0 else avg]
    else:
        final_embs = embeddings

    owns_store = False
    if store is None:
        store = _store_from_cfg(cfg)
        store.connect()
        owns_store = True

    try:
        store.get_or_create_collection(dim=512)
        if replace:
            site_esc = escape_milvus_string(site_id)
            name_esc = escape_milvus_string(name)
            store.delete_by_expr(f'site_id == "{site_esc}" and name == "{name_esc}"')

        uids = []
        for e in final_embs:
            uid = str(uuid.uuid4())
            store.upsert_row(
                uid=uid,
                person_id=uid,
                embedding=e.tolist(),
                name=name,
                role="",
                department="",
                notes=notes,
                site_id=site_id,
                camera_id=camera_id,
            )
            uids.append(uid)

        return {
            "name": name,
            "site_id": site_id,
            "camera_id": camera_id,
            "enrolled_uids": uids,
            "embeddings_stored": len(uids),
        }
    finally:
        if owns_store:
            store.close()


def tag_identity(
    uid: str,
    *,
    name: str | None = None,
    role: str | None = None,
    department: str | None = None,
    notes: str | None = None,
    store: FaceStore | None = None,
    cfg: dict | None = None,
) -> dict:
    """Update metadata for an existing row by UUID.

    Field semantics: None = keep the stored value, "" = clear it.
    """
    owns_store = False
    if store is None:
        if not cfg:
            raise ValueError("Must provide either `store` or `cfg` to tag_identity.")
        store = _store_from_cfg(cfg)
        store.connect()
        owns_store = True

    try:
        store.require_collection()
        uid_esc = escape_milvus_string(uid)
        rows = store.query_paged(f'id == "{uid_esc}"', ["id", "person_id", "embedding", "name", "role", "department", "notes", "site_id", "camera_id"])
        if not rows:
            raise KeyError(f"UUID '{uid}' not found in face registry.")

        r = rows[0]
        new_name = name if name is not None else r.get("name", "")
        new_role = role if role is not None else r.get("role", "")
        new_dept = department if department is not None else r.get("department", "")
        new_notes = notes if notes is not None else r.get("notes", "")

        store.upsert_row(
            uid=uid,
            person_id=str(r.get("person_id") or uid),
            embedding=r["embedding"],
            name=new_name,
            role=new_role,
            department=new_dept,
            notes=new_notes,
            site_id=str(r.get("site_id", "")),
            camera_id=str(r.get("camera_id", "")),
        )
        return {
            "id": uid,
            "person_id": str(r.get("person_id") or uid),
            "name": new_name,
            "role": new_role,
            "department": new_dept,
            "notes": new_notes,
            "site_id": str(r.get("site_id", "")),
        }
    finally:
        if owns_store:
            store.close()


def list_identities(
    site_id: str,
    *,
    store: FaceStore | None = None,
    cfg: dict | None = None,
    include_untagged: bool = False,
) -> list[dict]:
    """Return stored face identities for *site_id*."""
    owns_store = False
    if store is None:
        if not cfg:
            raise ValueError("Must provide either `store` or `cfg` to list_identities.")
        store = _store_from_cfg(cfg)
        store.connect()
        owns_store = True

    try:
        store.require_collection()
        if include_untagged:
            expr = site_expr(site_id) if site_id else 'id != ""'
        else:
            base = 'name != ""'
            expr = f"({base}) and ({site_expr(site_id)})" if site_id else base

        output_fields = ["id", "person_id", "name", "role", "department", "notes", "site_id", "camera_id"]
        rows = store.query_paged(expr, output_fields)
        return sorted(rows, key=lambda x: (str(x.get("site_id", "")), str(x.get("name", ""))))
    finally:
        if owns_store:
            store.close()


def delete_identity(
    uid: str,
    *,
    store: FaceStore | None = None,
    cfg: dict | None = None,
) -> bool:
    """Delete an identity/embedding by UUID."""
    owns_store = False
    if store is None:
        if not cfg:
            raise ValueError("Must provide either `store` or `cfg` to delete_identity.")
        store = _store_from_cfg(cfg)
        store.connect()
        owns_store = True

    try:
        store.require_collection()
        return store.delete_identity(uid)
    finally:
        if owns_store:
            store.close()


def search_face(
    image_or_emb: str | Path | np.ndarray,
    site_id: str,
    *,
    store: FaceStore | None = None,
    cfg: dict | None = None,
    limit: int = 5,
) -> list[dict]:
    """Search for matching identities given a face image file or embedding vector."""
    require_site_id(site_id, tool="search_face")

    if isinstance(image_or_emb, np.ndarray):
        emb = image_or_emb
    else:
        if not cfg:
            raise ValueError("Must provide `cfg` when searching from an image path.")
        provider = cfg.get("execution_provider") or cfg.get("device", "auto")
        device_id = int(cfg.get("gpu_device_id", 0))
        det_thresh = float(cfg.get("det_threshold", 0.5))
        detector = SCRFD(resolve_model_path(cfg["detector_model"]), provider, device_id)
        arcface = ArcFace(resolve_model_path(cfg["embedding_model"]), provider, device_id)

        path = Path(image_or_emb)
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {path}")
        boxes, kps = detector.detect(frame, det_thresh)
        if not len(boxes):
            raise RuntimeError("No face detected in query image.")
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        i = int(np.argmax(areas))
        aligned = align_face(frame, kps[i])
        emb = arcface.embed(aligned)

    owns_store = False
    if store is None:
        if not cfg:
            raise ValueError("Must provide either `store` or `cfg` to search_face.")
        store = _store_from_cfg(cfg)
        store.connect()
        owns_store = True

    try:
        store.require_collection()
        return store.search(
            emb, site_id, limit=limit, output_fields=["id", "person_id", "name", "role", "department", "notes"]
        )
    finally:
        if owns_store:
            store.close()
