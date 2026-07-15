"""
fr_merge.py — Step 2: Consolidate ingested face embeddings into person identities.

Merging is scoped to a single site_id — faces from different sites are never
clustered together.

Usage:
    python fr_merge.py --site-id site-001
    python fr_merge.py -c config.json
    python fr_merge.py --site-id site-001 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from fr_milvus import (
    fetch_site_rows,
    load_collection,
    require_site_id,
    resolve_site_camera,
    site_expr,
    upsert_row,
)

_BASE = Path(__file__).parent
log = logging.getLogger(__name__)


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
        self.centroids = (
            emb[np.newaxis] if len(self.counts) == 0 else np.vstack([self.centroids, emb[np.newaxis]])
        )
        self.counts.append(1)
        self.members.append([uid])

    def _update(self, idx: int, uid: str, emb: np.ndarray) -> None:
        n = self.counts[idx]
        c = (self.centroids[idx] * n + emb) / (n + 1)
        norm = np.linalg.norm(c)
        self.centroids[idx] = c / norm if norm > 0 else c
        self.counts[idx] += 1
        self.members[idx].append(uid)


def _merge_centroids(centroids: np.ndarray, counts: list, members: list, thresh: float):
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


def merge(cfg: dict, dry_run: bool = False) -> None:
    site_id, _camera_id = resolve_site_camera(cfg)
    require_site_id(site_id, tool="fr_merge")
    thresh = float(cfg.get("merge_thresh", 0.25))

    col = load_collection(cfg.get("milvus", {}))
    rows = fetch_site_rows(col, site_id)
    if not rows:
        log.info("No rows found for site_id=%s — nothing to merge.", site_id)
        return

    rows = sorted(rows, key=lambda r: r["id"])
    by_id = {r["id"]: r for r in rows}
    log.info(
        "%d rows for site=%s. Stage 1: centroid clustering (thresh=%.3f)...",
        len(rows), site_id, thresh,
    )

    store = _CentroidStore(thresh)
    for r in rows:
        store.add(r["id"], np.array(r["embedding"], dtype=np.float32))

    log.info(
        "Stage 1 done — %d clusters | sizes: %s",
        len(store.members),
        sorted((len(m) for m in store.members), reverse=True),
    )

    log.info("Stage 2: merging close centroids...")
    _, _, clusters = _merge_centroids(store.centroids, store.counts, store.members, thresh)

    log.info(
        "Stage 2 done — %d clusters | sizes: %s",
        len(clusters),
        sorted((len(m) for m in clusters), reverse=True),
    )

    updates, conflicts = [], []
    for comp in clusters:
        pids = {by_id[u]["person_id"] for u in comp if by_id[u]["person_id"] and by_id[u]["person_id"] != u}
        names = {by_id[u]["name"] for u in comp if by_id[u]["name"]}

        if len(pids) > 1 or len(names) > 1:
            conflicts.append({"ids": comp, "person_ids": pids, "names": names})
            continue

        canon_pid = next(iter(pids)) if pids else sorted(comp)[0]
        canon_name = next(iter(names)) if names else ""

        for uid in comp:
            r = by_id[uid]
            if r["person_id"] != canon_pid or (canon_name and r["name"] != canon_name):
                updates.append({
                    "id": uid,
                    "person_id": canon_pid,
                    "embedding": r["embedding"],
                    "name": canon_name or r["name"],
                    "role": r["role"],
                    "department": r["department"],
                    "notes": r["notes"],
                    "site_id": r["site_id"],
                    "camera_id": r.get("camera_id", ""),
                })

    log.info("Updates needed: %d | Conflicts: %d", len(updates), len(conflicts))

    if conflicts:
        log.warning("=== CONFLICTS (not auto-merged — review manually) ===")
        for c in conflicts:
            log.warning(
                "Cluster of %d | person_ids=%s | names=%s | ids=%s",
                len(c["ids"]), c["person_ids"], c["names"], [i[:8] for i in c["ids"]],
            )

    if dry_run:
        log.info("DRY RUN — no writes.")
        return

    for u in updates:
        upsert_row(
            col,
            uid=u["id"],
            person_id=u["person_id"],
            embedding=u["embedding"],
            name=u["name"],
            role=u["role"],
            department=u["department"],
            notes=u["notes"],
            site_id=u["site_id"],
            camera_id=u["camera_id"],
        )
    if updates:
        log.info("Applied %d updates for site=%s.", len(updates), site_id)
    else:
        log.info("Already consolidated for site=%s — no updates needed.", site_id)

    total = col.query(expr=site_expr(site_id), output_fields=["person_id"])
    unique = len({r["person_id"] for r in total})
    log.info("Merge complete | site=%s | rows=%d | unique person_ids=%d", site_id, len(total), unique)


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

    merge(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
