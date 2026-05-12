"""Voyage embeddings + Pinecone upsert for PFAF chunk pipelines."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

import voyageai
from pinecone import Pinecone

from extract_normalize import PfafStructured

_DEFAULT_VOYAGE_MODEL = "voyage-3-large"
_DEFAULT_VOYAGE_BATCH = 128
_DEFAULT_PINECONE_BATCH = 100
# Pinecone metadata total size is limited; keep chunk text bounded.
_MAX_TEXT_META_CHARS = 32000
_MAX_META_VALUE_CHARS = 4000

# Fact-table keys not copied into Pinecone metadata (available elsewhere / reduce noise).
_SKIP_METADATA_TABLE_KEYS = frozenset(
    {
        "care",
        "common_name",
        "family",
        "habitats",
        "habitat",
        "range",
        "usda_hardiness",
    }
)


@dataclass(frozen=True)
class PineconeUpsertResult:
    vectors_upserted: int
    index_name: str
    namespace: str
    embedding_model: str


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is not None and str(v).strip() != "":
        return str(v).strip().strip('"').strip("'")
    return default


def _chunk_vector_id(
    *,
    plant_id: int | str | None,
    latin_name: str | None,
    source_url: str,
    chunk_index: int,
) -> str:
    """Opaque id ``pfaf_<hex>``. ``plant_id`` is only mixed into the hash (not visible in the id)."""
    seed = f"{plant_id if plant_id is not None else ''}|{latin_name or ''}|{source_url}|{chunk_index}"
    h = hashlib.sha256(seed.encode()).hexdigest()[:32]
    return f"pfaf_{h}"


def _plant_id_for_metadata(pid: int | str) -> int | str:
    if isinstance(pid, int):
        return pid
    s = str(pid).strip()
    return int(s) if s.isdigit() else s


def _flatten_plant_metadata(meta: dict[str, str]) -> dict[str, str]:
    """Prefix table keys so they stay distinct from built-in fields."""
    out: dict[str, str] = {}
    for k, v in meta.items():
        if k in _SKIP_METADATA_TABLE_KEYS:
            continue
        key = f"meta_{k}" if not k.startswith("meta_") else k
        s = str(v).strip()
        if len(s) > _MAX_META_VALUE_CHARS:
            s = s[: _MAX_META_VALUE_CHARS] + "…"
        out[key] = s
    return out


def _pinecone_sanitize_metadata(md: dict[str, Any]) -> dict[str, Any]:
    """Pinecone allows string, number, bool, list of strings."""
    out: dict[str, Any] = {}
    for k, v in md.items():
        key = str(k)[:512]
        if isinstance(v, bool):
            out[key] = v
        elif isinstance(v, (int, float)):
            out[key] = v
        elif isinstance(v, str):
            out[key] = v[:_MAX_TEXT_META_CHARS] if len(v) > _MAX_TEXT_META_CHARS else v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[key] = v[:100]
        else:
            out[key] = json.dumps(v, ensure_ascii=False)[:_MAX_META_VALUE_CHARS]
    return out


def _embed_voyage_batches(
    texts: list[str],
    *,
    model: str,
    batch_size: int,
    voyage_client: voyageai.Client,
) -> list[list[float]]:
    all_emb: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = voyage_client.embed(texts=batch, model=model, input_type="document")
        all_emb.extend(resp.embeddings)
    return all_emb


def _pinecone_index_dimension(pc: Pinecone, index_name: str) -> int:
    """Vector dimension for ``index_name`` (from describe_index, or ``PINECONE_INDEX_DIMENSION``)."""
    raw = _env("PINECONE_INDEX_DIMENSION")
    if raw is not None and raw.isdigit():
        return int(raw)
    desc = pc.describe_index(index_name)
    dim = getattr(desc, "dimension", None)
    if dim is not None:
        return int(dim)
    spec = getattr(desc, "spec", None)
    if spec is not None:
        dim = getattr(spec, "dimension", None)
        if dim is not None:
            return int(dim)
    raise ValueError(
        "Could not read index dimension from describe_index; set PINECONE_INDEX_DIMENSION in .env"
    )


def pinecone_has_latin_name(
    index: Any,
    *,
    latin_name: str,
    namespace: str,
    dimension: int,
) -> bool:
    """
    Return True if at least one vector exists with metadata ``latin_name`` equal to ``latin_name``
    (same field as upserts). Uses a cheap filtered query with a zero vector.
    """
    name = (latin_name or "").strip()
    if not name:
        return False
    filt: dict[str, Any] = {"latin_name": {"$eq": name}}
    q_kw: dict[str, Any] = {
        "vector": [0.0] * dimension,
        "top_k": 1,
        "filter": filt,
        "include_metadata": False,
    }
    if namespace:
        q_kw["namespace"] = namespace
    resp = index.query(**q_kw)
    matches = getattr(resp, "matches", None)
    if matches is None and isinstance(resp, dict):
        matches = resp.get("matches")
    return bool(matches)


def make_pfaf_embed_clients_with_dimension() -> tuple[voyageai.Client, Any, str, int]:
    """
    Build one Voyage client, one Pinecone ``Index``, namespace string, and index vector dimension.

    Requires ``VOYAGE_API_KEY``, ``PINECONE_API_KEY``, and ``PINECONE_INDEX`` in the environment.
    """
    voyage_key = _env("VOYAGE_API_KEY")
    if not voyage_key:
        raise ValueError("Missing VOYAGE_API_KEY in environment")
    pc_key = _env("PINECONE_API_KEY")
    if not pc_key:
        raise ValueError("Missing PINECONE_API_KEY in environment")
    index_name = _env("PINECONE_INDEX")
    if not index_name:
        raise ValueError("Set PINECONE_INDEX in environment")
    namespace = _env("PINECONE_NAMESPACE") or ""
    pc = Pinecone(api_key=pc_key)
    idx = pc.Index(index_name)
    dim = _pinecone_index_dimension(pc, index_name)
    voyage_client = voyageai.Client(api_key=voyage_key)
    return voyage_client, idx, namespace, dim


def make_pfaf_embed_clients() -> tuple[voyageai.Client, Any]:
    """Same as :func:`make_pfaf_embed_clients_with_dimension` but without returning namespace/dimension."""
    v, idx, _, _ = make_pfaf_embed_clients_with_dimension()
    return v, idx


def delete_all_pfaf_index_vectors() -> None:
    """
    Delete every vector in ``PINECONE_INDEX`` using ``delete_all=True``.

    Uses the same namespace rules as :func:`embed_and_upsert_pfaf_chunks`: if ``PINECONE_NAMESPACE``
    is set, only that namespace is cleared; otherwise the index default namespace is cleared.
    """
    pc_key = _env("PINECONE_API_KEY")
    if not pc_key:
        raise ValueError("Missing PINECONE_API_KEY in environment")
    index_name = _env("PINECONE_INDEX")
    if not index_name:
        raise ValueError("Set PINECONE_INDEX in environment")
    namespace = _env("PINECONE_NAMESPACE") or ""

    pc = Pinecone(api_key=pc_key)
    index = pc.Index(index_name)
    if namespace:
        index.delete(delete_all=True, namespace=namespace)
    else:
        index.delete(delete_all=True)


def embed_and_upsert_pfaf_chunks(
    chunks: Sequence[str],
    structured: PfafStructured,
    source_url: str,
    *,
    plant_id: int | str | None = None,
    voyage_model: str | None = None,
    pinecone_index: str | None = None,
    pinecone_namespace: str | None = None,
    voyage_batch_size: int = _DEFAULT_VOYAGE_BATCH,
    pinecone_batch_size: int = _DEFAULT_PINECONE_BATCH,
    voyage_client: voyageai.Client | None = None,
    pinecone_index_obj: Any | None = None,
    metadata_latin_name: str | None = None,
) -> PineconeUpsertResult:
    """
    Embed each chunk with Voyage (``input_type=document``) and upsert to Pinecone with metadata.

    Environment:
        - ``VOYAGE_API_KEY`` (required)
        - ``PINECONE_API_KEY`` (required)
        - ``PINECONE_INDEX`` — index name (required)
        - ``VOYAGE_EMBED_MODEL`` — defaults to ``voyage-3-large`` (must match index dimension)
        - ``PINECONE_NAMESPACE`` — optional namespace string
        - ``PLANT_ID`` — optional; if set and ``plant_id`` arg omitted, stored in metadata only

    Vector ids are always opaque ``pfaf_<hex>`` (plant id is **not** encoded as ``4_0`` etc.; it may be
    mixed into the hash seed so different plants stay distinct). When ``plant_id`` is set, metadata
    includes ``plant_id`` (int when numeric) for filtering.

    Metadata per vector includes ``latin_name``, ``source_url``, ``chunk_index``, ``text`` (chunk body),
    ``source=pfaf``, and flattened fact-table fields as ``meta_*`` from ``structured.metadata``.

    Pass ``voyage_client`` and ``pinecone_index_obj`` from :func:`make_pfaf_embed_clients` to avoid
    creating a new client and index handle on every call (recommended for batch pipelines).

    When ``metadata_latin_name`` is set (e.g. Mongo ``scientific_name``), the Pinecone ``latin_name``
    field is filled **only** from that string—not from parsed HTML. The vector id hash uses it when
    non-empty; otherwise it falls back to ``structured.latin_name`` so ids stay distinct.
    """
    index_name = pinecone_index or _env("PINECONE_INDEX")
    if not index_name:
        raise ValueError("Set pinecone_index= or PINECONE_INDEX in environment")

    model = voyage_model or _env("VOYAGE_EMBED_MODEL", _DEFAULT_VOYAGE_MODEL) or _DEFAULT_VOYAGE_MODEL
    namespace = pinecone_namespace if pinecone_namespace is not None else (_env("PINECONE_NAMESPACE") or "")

    resolved_plant_id: int | str | None = plant_id
    if resolved_plant_id is None:
        raw_pid = _env("PLANT_ID")
        if raw_pid is not None:
            resolved_plant_id = int(raw_pid) if raw_pid.isdigit() else raw_pid

    texts = [c.strip() for c in chunks if c and c.strip()]
    if not texts:
        return PineconeUpsertResult(0, index_name, namespace, model)

    if voyage_client is None:
        voyage_key = _env("VOYAGE_API_KEY")
        if not voyage_key:
            raise ValueError("Missing VOYAGE_API_KEY in environment")
        voyage_client = voyageai.Client(api_key=voyage_key)

    embeddings = _embed_voyage_batches(
        texts,
        model=model,
        batch_size=voyage_batch_size,
        voyage_client=voyage_client,
    )

    if len(embeddings) != len(texts):
        raise RuntimeError(f"Voyage returned {len(embeddings)} embeddings for {len(texts)} texts")

    if metadata_latin_name is not None:
        latin = metadata_latin_name.strip()
    else:
        latin = structured.latin_name or ""
    latin_for_id = latin if latin else (structured.latin_name or "")
    flat_meta = _flatten_plant_metadata(structured.metadata)

    vectors: list[dict[str, Any]] = []
    for idx, (text, values) in enumerate(zip(texts, embeddings)):
        vid = _chunk_vector_id(
            plant_id=resolved_plant_id,
            latin_name=latin_for_id,
            source_url=source_url,
            chunk_index=idx,
        )
        meta: dict[str, Any] = {
            "latin_name": latin,
            "source_url": source_url[:2000],
            "chunk_index": idx,
            "text": text[:_MAX_TEXT_META_CHARS],
            "source": "pfaf",
            **flat_meta,
        }
        if resolved_plant_id is not None:
            meta["plant_id"] = _plant_id_for_metadata(resolved_plant_id)
        meta = _pinecone_sanitize_metadata(meta)
        vectors.append({"id": vid, "values": values, "metadata": meta})

    if pinecone_index_obj is None:
        pc_key = _env("PINECONE_API_KEY")
        if not pc_key:
            raise ValueError("Missing PINECONE_API_KEY in environment")
        pc = Pinecone(api_key=pc_key)
        index = pc.Index(index_name)
    else:
        index = pinecone_index_obj

    upserted = 0
    upsert_kw: dict[str, Any] = {}
    if namespace:
        upsert_kw["namespace"] = namespace
    for i in range(0, len(vectors), pinecone_batch_size):
        batch = vectors[i : i + pinecone_batch_size]
        index.upsert(vectors=batch, **upsert_kw)
        upserted += len(batch)

    return PineconeUpsertResult(
        vectors_upserted=upserted,
        index_name=index_name,
        namespace=namespace,
        embedding_model=model,
    )
