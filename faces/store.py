"""
Face-store abstraction layer over Milvus.

Provides a ``FaceStore`` protocol, a concrete ``MilvusFaceStore`` that wraps
the raw pymilvus operations from the legacy ``fr_milvus`` module, and an
``InMemoryFaceStore`` suitable for unit tests (no Milvus dependency).

Embeddings are logically partitioned by *site_id*: ingest, merge, and
recognition searches all filter by site_id.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

# ── Schema constants ──────────────────────────────────────────────────────────

SCALAR_FIELDS: tuple[str, ...] = (
    "id", "person_id", "name", "role", "department", "notes",
    "site_id", "camera_id",
)
VECTOR_FIELD: str = "embedding"


# ── Typed exceptions ─────────────────────────────────────────────────────────

class FaceStoreError(Exception):
    """Base exception for face-store operations."""


class CollectionNotFoundError(FaceStoreError):
    """Raised when the expected Milvus collection does not exist."""


class SiteIdRequiredError(FaceStoreError):
    """Raised when a *site_id* is required but missing or empty."""


# ── Module-level utility helpers ──────────────────────────────────────────────

def escape_milvus_string(value: str) -> str:
    """Escape backslashes and double-quotes for Milvus filter expressions."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def site_expr(site_id: str) -> str:
    """Return a Milvus boolean expression matching *site_id*."""
    return f'site_id == "{escape_milvus_string(site_id)}"'


def untagged_expr(site_id: str | None = None) -> str:
    """Return an expression for rows whose ``name`` is empty (untagged).

    Optionally also filtered to a single *site_id*.
    """
    base = 'name == ""'
    if site_id:
        return f"({base}) and ({site_expr(site_id)})"
    return base


def require_site_id(site_id: str, *, tool: str) -> str:
    """Validate that *site_id* is non-empty, raising on failure.

    Parameters
    ----------
    site_id:
        The site identifier to validate.
    tool:
        A human-readable tool/command name used in the error message.

    Raises
    ------
    SiteIdRequiredError
        If *site_id* is falsy.
    """
    if not site_id:
        raise SiteIdRequiredError(
            f"{tool}: site_id is required — set face_recognition.site_id in "
            "config.json or pass --site-id"
        )
    return site_id


def resolve_site_camera(cfg: dict) -> tuple[str, str]:
    """Read ``site_id`` / ``camera_id`` from a face-recognition config dict."""
    milvus = cfg.get("milvus", {})
    site_id = str(cfg.get("site_id") or milvus.get("site_id") or "").strip()
    camera_id = str(cfg.get("camera_id") or milvus.get("camera_id") or "").strip()
    return site_id, camera_id


# ── FaceStore protocol ───────────────────────────────────────────────────────

@runtime_checkable
class FaceStore(Protocol):
    """Abstract interface for face-embedding storage back-ends."""

    def connect(self) -> None:
        """Establish a connection to the backing store."""
        ...

    def close(self) -> None:
        """Disconnect / release resources."""
        ...

    def search(
        self,
        emb: np.ndarray,
        site_id: str,
        *,
        limit: int = 1,
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """ANN search restricted to one *site_id*.

        Returns a list of dicts, each containing at minimum ``id`` and
        ``score``; additional keys depend on *output_fields*.
        """
        ...

    def insert_batch(
        self,
        ids: list[str],
        embeddings: list,
        site_id: str,
        camera_id: str,
    ) -> None:
        """Insert a batch of embeddings in a single round-trip."""
        ...

    def upsert_row(
        self,
        *,
        uid: str,
        person_id: str,
        embedding: list,
        name: str,
        role: str,
        department: str,
        notes: str,
        site_id: str,
        camera_id: str,
    ) -> None:
        """Insert-or-update a single row."""
        ...

    def query_paged(
        self,
        expr: str,
        output_fields: list[str],
    ) -> list[dict]:
        """Return all rows matching *expr*, with automatic pagination."""
        ...

    def fetch_site_rows(self, site_id: str) -> list[dict]:
        """Fetch every row (scalars + embedding) belonging to *site_id*."""
        ...

    def get_or_create_collection(self, dim: int = 512) -> None:
        """Ensure the target collection exists, creating it if necessary."""
        ...

    def require_collection(self) -> None:
        """Raise :class:`CollectionNotFoundError` if the collection is absent."""
        ...

    def untagged_expr(self, site_id: str | None = None) -> str:
        """Return a filter expression for untagged rows."""
        ...

    def delete_by_expr(self, expr: str) -> int:
        """Delete all rows matching *expr* and return count of deleted rows."""
        ...

    def delete_identity(self, uid: str) -> bool:
        """Delete identity by *uid* and return whether row existed."""
        ...


# ── Milvus concrete implementation ───────────────────────────────────────────

class MilvusFaceStore:
    """Concrete :class:`FaceStore` backed by a Milvus instance.

    Parameters
    ----------
    host:
        Milvus gRPC host.
    port:
        Milvus gRPC port.
    collection:
        Collection name to operate on.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection: str = "face_registry",
    ) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection
        self._alias = f"milvus_{host}_{port}"
        self._col: Any | None = None  # lazily set after connect()

    # ── connection lifecycle ──────────────────────────────────────────────

    def connect(self) -> None:
        from pymilvus import connections

        # ``connect`` is idempotent; pymilvus silently reuses existing aliases.
        connections.connect(alias=self._alias, host=self._host, port=self._port)
        log.info("Milvus connected via alias '%s': %s:%s",
                 self._alias, self._host, self._port)

    def close(self) -> None:
        from pymilvus import connections

        try:
            connections.disconnect(alias=self._alias)
        except Exception:  # noqa: BLE001
            pass
        self._col = None

    # ── collection management ─────────────────────────────────────────────

    def _load_existing(self) -> None:
        """Load the collection object, assuming it already exists."""
        from pymilvus import Collection

        col = Collection(self._collection_name, using=self._alias)
        col.load()
        self._col = col

    @staticmethod
    def _has_site_fields(col: Any) -> bool:
        names = {f.name for f in col.schema.fields}
        return "site_id" in names and "camera_id" in names

    def get_or_create_collection(self, dim: int = 512) -> None:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        if utility.has_collection(self._collection_name, using=self._alias):
            col = Collection(self._collection_name, using=self._alias)
            col.load()
            if not self._has_site_fields(col):
                raise FaceStoreError(
                    f"Collection '{self._collection_name}' exists but is missing "
                    "site_id/camera_id fields. Drop and recreate the collection, "
                    "or ingest into a new collection name "
                    "(set face_recognition.milvus.collection in config.json)."
                )
            log.info("Collection '%s' loaded (site-scoped schema).",
                     self._collection_name)
            self._col = col
            return

        fields = [
            FieldSchema("id", DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema("person_id", DataType.VARCHAR, max_length=64),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("name", DataType.VARCHAR, max_length=256),
            FieldSchema("role", DataType.VARCHAR, max_length=64),
            FieldSchema("department", DataType.VARCHAR, max_length=256),
            FieldSchema("notes", DataType.VARCHAR, max_length=256),
            FieldSchema("site_id", DataType.VARCHAR, max_length=64),
            FieldSchema("camera_id", DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(
            fields, description="Face embeddings (site-scoped)",
        )
        col = Collection(
            name=self._collection_name, schema=schema, using=self._alias,
        )
        col.create_index(
            "embedding",
            {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        col.load()
        log.info("Collection '%s' created with site_id + camera_id fields.",
                 self._collection_name)
        self._col = col

    def require_collection(self) -> None:
        from pymilvus import utility

        if not utility.has_collection(self._collection_name, using=self._alias):
            raise CollectionNotFoundError(
                f"Collection '{self._collection_name}' not found — run ingest first."
            )
        self._load_existing()
        if not self._has_site_fields(self._col):
            raise FaceStoreError(
                f"Collection '{self._collection_name}' is missing site_id/camera_id "
                "— migrate or use a new collection."
            )

    # ── data operations ───────────────────────────────────────────────────

    def _ensure_col(self) -> Any:
        if self._col is None:
            raise FaceStoreError(
                "No collection loaded. Call connect() + get_or_create_collection() "
                "or require_collection() first."
            )
        return self._col

    def search(
        self,
        emb: np.ndarray,
        site_id: str,
        *,
        limit: int = 1,
        output_fields: list[str] | None = None,
        extra_expr: str | None = None,
    ) -> list[dict]:
        """ANN search restricted to *site_id*, returning flat dicts."""
        col = self._ensure_col()
        expr = site_expr(site_id)
        if extra_expr:
            expr = f"({expr}) and ({extra_expr})"

        results = col.search(
            data=[emb.tolist()],
            anns_field=VECTOR_FIELD,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=output_fields or ["id"],
            expr=expr,
        )
        hits: list[dict] = []
        for hit_list in results:
            for hit in hit_list:
                row: dict[str, Any] = {"id": hit.id, "score": hit.score}
                for field in (output_fields or ["id"]):
                    if field != "id" and hasattr(hit.entity, "get"):
                        row[field] = hit.entity.get(field)
                hits.append(row)
        return hits

    def insert_batch(
        self,
        ids: list[str],
        embeddings: list,
        site_id: str,
        camera_id: str,
    ) -> None:
        col = self._ensure_col()
        n = len(ids)
        col.insert([
            ids,
            ids,             # person_id defaults to self until merge
            embeddings,
            [""] * n,        # name
            [""] * n,        # role
            [""] * n,        # department
            [""] * n,        # notes
            [site_id] * n,
            [camera_id] * n,
        ])
        col.flush()
        log.info("Inserted batch of %d embeddings (site=%s camera=%s).",
                 n, site_id, camera_id)

    def upsert_row(
        self,
        *,
        uid: str,
        person_id: str,
        embedding: list,
        name: str,
        role: str,
        department: str,
        notes: str,
        site_id: str,
        camera_id: str,
    ) -> None:
        col = self._ensure_col()
        col.upsert([
            [uid],
            [person_id],
            [embedding],
            [name],
            [role],
            [department],
            [notes],
            [site_id],
            [camera_id],
        ])
        col.flush()

    def query_paged(
        self,
        expr: str,
        output_fields: list[str],
        *,
        page_size: int = 10_000,
    ) -> list[dict]:
        col = self._ensure_col()
        # Prefer query_iterator: offset-based paging hits Milvus's
        # offset+limit <= 16384 cap on large collections.
        if hasattr(col, "query_iterator"):
            rows: list[dict] = []
            it = col.query_iterator(
                expr=expr, output_fields=output_fields, batch_size=page_size
            )
            try:
                while True:
                    page = it.next()
                    if not page:
                        break
                    rows.extend(page)
            finally:
                it.close()
            return rows

        rows = []
        offset = 0
        while True:
            page = col.query(
                expr=expr,
                output_fields=output_fields,
                offset=offset,
                limit=page_size,
            )
            if not page:
                break
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return rows

    def fetch_site_rows(self, site_id: str) -> list[dict]:
        fields = list(SCALAR_FIELDS) + [VECTOR_FIELD]
        return self.query_paged(site_expr(site_id), fields)

    # ── convenience ───────────────────────────────────────────────────────

    def untagged_expr(self, site_id: str | None = None) -> str:  # noqa: PLR6301
        return untagged_expr(site_id)

    def delete_by_expr(self, expr: str) -> int:
        col = self._ensure_col()
        res = col.delete(expr)
        col.flush()
        if hasattr(res, "delete_count"):
            return int(res.delete_count)
        if hasattr(res, "delete_cnt"):
            return int(res.delete_cnt)
        return 1

    def delete_identity(self, uid: str) -> bool:
        uid_esc = escape_milvus_string(uid)
        expr = f'id == "{uid_esc}"'
        existing = self.query_paged(expr, ["id"])
        if not existing:
            return False
        self.delete_by_expr(expr)
        return True


# ── In-memory implementation (testing) ────────────────────────────────────────

class InMemoryFaceStore:
    """Lightweight in-memory :class:`FaceStore` for unit tests.

    Embeddings are stored as numpy arrays and search uses brute-force
    cosine similarity — no Milvus dependency required.
    """

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._connected: bool = False

    # ── connection lifecycle ──────────────────────────────────────────────

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    # ── collection management ─────────────────────────────────────────────

    def get_or_create_collection(self, dim: int = 512) -> None:  # noqa: ARG002
        pass  # nothing to do — in-memory store is schema-less

    def require_collection(self) -> None:
        if not self._rows:
            raise CollectionNotFoundError(
                "InMemoryFaceStore has no data — insert some rows first."
            )

    # ── data operations ───────────────────────────────────────────────────

    def search(
        self,
        emb: np.ndarray,
        site_id: str,
        *,
        limit: int = 1,
        output_fields: list[str] | None = None,
        extra_expr: str | None = None,  # noqa: ARG002 — not used in-memory
    ) -> list[dict]:
        """Brute-force cosine similarity search within *site_id*."""
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        scored: list[tuple[float, str]] = []

        for uid, row in self._rows.items():
            if row.get("site_id") != site_id:
                continue
            row_emb = row["embedding"]
            row_norm = row_emb / (np.linalg.norm(row_emb) + 1e-10)
            score = float(np.dot(emb_norm, row_norm))
            scored.append((score, uid))

        scored.sort(key=lambda t: t[0], reverse=True)
        out_fields = output_fields or ["id"]
        results: list[dict] = []
        for score, uid in scored[:limit]:
            hit: dict[str, Any] = {"id": uid, "score": score}
            for f in out_fields:
                if f != "id" and f in self._rows[uid]:
                    hit[f] = self._rows[uid][f]
            results.append(hit)
        return results

    def insert_batch(
        self,
        ids: list[str],
        embeddings: list,
        site_id: str,
        camera_id: str,
    ) -> None:
        for uid, emb in zip(ids, embeddings):
            arr = np.asarray(emb, dtype=np.float32)
            self._rows[uid] = {
                "id": uid,
                "person_id": uid,
                "embedding": arr,
                "name": "",
                "role": "",
                "department": "",
                "notes": "",
                "site_id": site_id,
                "camera_id": camera_id,
            }

    def upsert_row(
        self,
        *,
        uid: str,
        person_id: str,
        embedding: list,
        name: str,
        role: str,
        department: str,
        notes: str,
        site_id: str,
        camera_id: str,
    ) -> None:
        self._rows[uid] = {
            "id": uid,
            "person_id": person_id,
            "embedding": np.asarray(embedding, dtype=np.float32),
            "name": name,
            "role": role,
            "department": department,
            "notes": notes,
            "site_id": site_id,
            "camera_id": camera_id,
        }

    def query_paged(
        self,
        expr: str,
        output_fields: list[str],
    ) -> list[dict]:
        """Very basic expression matcher — supports ``site_id == "…"``
        and ``name == ""`` filters only, which covers the common cases."""
        results: list[dict] = []
        for row in self._rows.values():
            if self._matches_expr(row, expr):
                results.append({f: row.get(f) for f in output_fields if f in row})
        return results

    def fetch_site_rows(self, site_id: str) -> list[dict]:
        out: list[dict] = []
        for row in self._rows.values():
            if row.get("site_id") == site_id:
                out.append(dict(row))
        return out

    def untagged_expr(self, site_id: str | None = None) -> str:  # noqa: PLR6301
        return untagged_expr(site_id)

    def delete_by_expr(self, expr: str) -> int:
        to_remove = [uid for uid, row in self._rows.items() if self._matches_expr(row, expr)]
        for uid in to_remove:
            del self._rows[uid]
        return len(to_remove)

    def delete_identity(self, uid: str) -> bool:
        if uid in self._rows:
            del self._rows[uid]
            return True
        return False

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _matches_expr(row: dict, expr: str) -> bool:
        """Minimal expression evaluator for test usage.

        Handles conjunctions (``and``) of equality predicates such as
        ``site_id == "foo"`` and ``name == ""``.
        """
        parts = [p.strip() for p in expr.replace("(", "").replace(")", "").split(" and ")]
        for part in parts:
            if "==" not in part:
                continue
            lhs, rhs = part.split("==", 1)
            field = lhs.strip()
            value = rhs.strip().strip('"')
            if row.get(field) != value:
                return False
        return True
