"""
Shared Milvus helpers for the face-recognition ingest / merge / tag pipeline.

Embeddings are partitioned logically by site_id:
  - ingest dedup searches only within the same site_id
  - fr_merge clusters only rows for the requested site_id
  - recognition search filters by site_id
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

log = logging.getLogger(__name__)

SCALAR_FIELDS = ("id", "person_id", "name", "role", "department", "notes", "site_id", "camera_id")
VECTOR_FIELD = "embedding"


def escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def site_expr(site_id: str) -> str:
    return f'site_id == "{escape_milvus_string(site_id)}"'


def untagged_expr(site_id: str | None = None) -> str:
    base = 'name == ""'
    if site_id:
        return f"({base}) and ({site_expr(site_id)})"
    return base


def resolve_site_camera(cfg: dict) -> tuple[str, str]:
    """Read site_id / camera_id from face_recognition config (CLI may override)."""
    milvus = cfg.get("milvus", {})
    site_id = str(cfg.get("site_id") or milvus.get("site_id") or "").strip()
    camera_id = str(cfg.get("camera_id") or milvus.get("camera_id") or "").strip()
    return site_id, camera_id


def require_site_id(site_id: str, *, tool: str) -> str:
    if not site_id:
        raise SystemExit(
            f"{tool}: site_id is required — set face_recognition.site_id in config.json "
            "or pass --site-id"
        )
    return site_id


def connect(mc: dict) -> None:
    connections.connect(alias="default", host=mc["host"], port=int(mc.get("port", 19530)))
    log.info("Milvus connected: %s:%s", mc["host"], mc.get("port", 19530))


def collection_field_names(col: Collection) -> set[str]:
    return {f.name for f in col.schema.fields}


def has_site_fields(col: Collection) -> bool:
    names = collection_field_names(col)
    return "site_id" in names and "camera_id" in names


def get_or_create_collection(mc: dict, dim: int = 512) -> Collection:
    name = mc.get("collection", "face_embeddings")
    if utility.has_collection(name):
        col = Collection(name)
        col.load()
        if not has_site_fields(col):
            raise RuntimeError(
                f"Collection '{name}' exists but is missing site_id/camera_id fields. "
                "Drop and recreate the collection, or ingest into a new collection name "
                "(set face_recognition.milvus.collection in config.json)."
            )
        log.info("Collection '%s' loaded (site-scoped schema).", name)
        return col

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
    schema = CollectionSchema(fields, description="Face embeddings (site-scoped)")
    col = Collection(name=name, schema=schema)
    col.create_index(
        "embedding",
        {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
    )
    col.load()
    log.info("Collection '%s' created with site_id + camera_id fields.", name)
    return col


def load_collection(mc: dict) -> Collection:
    connect(mc)
    name = mc.get("collection", "face_embeddings")
    if not utility.has_collection(name):
        raise SystemExit(f"Collection '{name}' not found — run fr_ingest.py first.")
    col = Collection(name)
    col.load()
    if not has_site_fields(col):
        raise RuntimeError(
            f"Collection '{name}' is missing site_id/camera_id — migrate or use a new collection."
        )
    return col


def search_same_site(
    col: Collection,
    emb: np.ndarray,
    site_id: str,
    *,
    limit: int = 1,
    output_fields: list[str] | None = None,
    extra_expr: str | None = None,
) -> list[Any]:
    """ANN search restricted to one site_id."""
    expr = site_expr(site_id)
    if extra_expr:
        expr = f"({expr}) and ({extra_expr})"
    return col.search(
        data=[emb.tolist()],
        anns_field=VECTOR_FIELD,
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=limit,
        output_fields=output_fields or ["id"],
        expr=expr,
    )


def insert_batch(
    col: Collection,
    ids: list[str],
    embs: list,
    site_id: str,
    camera_id: str,
) -> None:
    n = len(ids)
    col.insert([
        ids,
        ids,  # person_id defaults to self until merge
        embs,
        [""] * n,
        [""] * n,
        [""] * n,
        [""] * n,
        [site_id] * n,
        [camera_id] * n,
    ])
    col.flush()
    log.info("Inserted batch of %d embeddings (site=%s camera=%s).", n, site_id, camera_id)


def upsert_row(
    col: Collection,
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
    col: Collection,
    expr: str,
    output_fields: list[str],
    *,
    page_size: int = 10_000,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        page = col.query(expr=expr, output_fields=output_fields, offset=offset, limit=page_size)
        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


def fetch_site_rows(col: Collection, site_id: str) -> list[dict]:
    fields = list(SCALAR_FIELDS) + [VECTOR_FIELD]
    return query_paged(col, site_expr(site_id), fields)
