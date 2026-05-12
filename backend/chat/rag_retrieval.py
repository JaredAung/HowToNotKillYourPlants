"""
Pinecone + Voyage RAG for PFAF plant chunks. Mirrors ``resources/RAG/test.py`` retrieval path.
Requires VOYAGE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX (and optional PINECONE_NAMESPACE, VOYAGE_EMBED_MODEL).

Query/embed behavior matches :mod:`chat.rag_eval` (same ``index.query`` kwargs, match normalization).

Plant-scoped search uses metadata ``latin_name`` (scientific binomial); pass it as
:func:`query_plant_knowledge_rag`'s ``plant_name`` or ``latin_name``, or use :func:`plant_name_filter`
with :func:`voyage_pinecone_search`.
"""
from __future__ import annotations

import logging
import os
import sys
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Callable, TypedDict

logger = logging.getLogger(__name__)


class PfafChunkMetadata(TypedDict, total=False):
    """
    Metadata on PFAF Pinecone vectors (see ``embed_and_upsert_pfaf_chunks`` in ``embed_pinecone``).
    Additional ``meta_*`` keys come from the plant fact table.
    """

    chunk_index: int
    latin_name: str
    plant_id: int
    source: str
    source_url: str
    text: str


def _metadata_as_dict(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return dict(meta)
    if hasattr(meta, "items"):
        return {str(k): v for k, v in meta.items()}
    return {}


def pinecone_matches_to_rows(
    matches: Any,
    *,
    include_values: bool = False,
) -> list[dict[str, Any]]:
    """
    Normalize Pinecone ``query`` match objects to the same row shape as
    :func:`chat.rag_eval.voyage_pinecone_search`: ``id``, ``score``, ``metadata``, optional ``values``.
    """
    if not matches:
        matches = []
    rows: list[dict[str, Any]] = []
    for m in matches:
        raw_meta = getattr(m, "metadata", None)
        if raw_meta is None and isinstance(m, dict):
            raw_meta = m.get("metadata")
        meta = _metadata_as_dict(raw_meta)
        score = getattr(m, "score", None)
        if score is None and isinstance(m, dict):
            score = m.get("score")
        vid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else "")
        row: dict[str, Any] = {"id": vid, "score": score, "metadata": meta}
        if include_values:
            vals = getattr(m, "values", None)
            if vals is None and isinstance(m, dict):
                vals = m.get("values")
            if vals is not None:
                row["values"] = list(vals) if not isinstance(vals, list) else vals
        rows.append(row)
    return rows

_clients_cache: tuple[Any, ...] | None = None
_make_pfaf_embed_clients: Callable[..., Any] | None = None


def _find_embed_pinecone_path() -> Path | None:
    """Resolve resources/RAG/embed_pinecone.py by walking upward from this file."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / "resources" / "RAG" / "embed_pinecone.py"
        if candidate.is_file():
            return candidate
    return None


def _load_make_pfaf_embed_clients() -> Callable[..., Any]:
    """
    Lazy-load embed_pinecone (not at rag_retrieval import time) so the API can start
    without RAG files. Uses explicit path so it does not depend on cwd.
    """
    global _make_pfaf_embed_clients
    if _make_pfaf_embed_clients is not None:
        return _make_pfaf_embed_clients

    ep = _find_embed_pinecone_path()
    if ep is None:
        raise ImportError(
            "Could not find resources/RAG/embed_pinecone.py relative to the backend. "
            "Clone the full repo including resources/RAG."
        )
    rag_dir = str(ep.parent)
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)

    spec = importlib_util.spec_from_file_location("embed_pinecone", ep)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {ep}")
    mod = importlib_util.module_from_spec(spec)
    # Required before exec_module so dataclasses / typing can resolve sys.modules[mod.__name__]
    # (Python 3.13 dataclass decorator touches sys.modules during class creation).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "make_pfaf_embed_clients_with_dimension", None)
    if fn is None:
        raise ImportError("embed_pinecone missing make_pfaf_embed_clients_with_dimension")
    _make_pfaf_embed_clients = fn
    return fn


def _get_clients():
    global _clients_cache
    if _clients_cache is None:
        make = _load_make_pfaf_embed_clients()
        _clients_cache = make()
    return _clients_cache


def plant_name_filter(plant_name: str) -> dict[str, Any]:
    """
    Pinecone metadata filter for a single species. Vectors store the binomial in ``latin_name``
    (see ``embed_pinecone``); pass the same spelling you use in the index (e.g. ``Populus trichocarpa``).
    """
    name = (plant_name or "").strip()
    if not name:
        raise ValueError("plant_name must be non-empty")
    return {"latin_name": {"$eq": name}}


def _build_filter(
    *,
    plant_name: str | None = None,
    latin_name: str | None = None,
    plant_id: int | None = None,
) -> dict[str, Any]:
    """Prefer ``plant_id``; otherwise filter by scientific name via ``plant_name`` or ``latin_name``."""
    if plant_id is not None:
        return {"plant_id": {"$eq": int(plant_id)}}
    name = (plant_name or latin_name or "").strip()
    if not name:
        raise ValueError(
            "Provide plant_name (scientific/Latin name as stored in chunk metadata), latin_name, or plant_id "
            "to scope retrieval to one plant."
        )
    return {"latin_name": {"$eq": name}}


def _voyage_embed_query_text(voyage_client: Any, q: str) -> list[float]:
    raw_model = (os.environ.get("VOYAGE_EMBED_MODEL") or "").strip()
    model = raw_model or "voyage-3-large"
    resp = voyage_client.embed(texts=[q], model=model, input_type="query")
    return resp.embeddings[0]


def voyage_pinecone_search(
    query: str,
    *,
    top_k: int = 10,
    filter: dict[str, Any] | None = None,
    include_metadata: bool = True,
    include_values: bool = False,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """
    Embed ``query`` with Voyage (``input_type="query"``), then Pinecone ``index.query``.

    Same pipeline as :func:`query_plant_knowledge_rag`, optional metadata ``filter``, ``top_k`` up to
    10000. Returns rows from :func:`pinecone_matches_to_rows`.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be non-empty")

    voyage_client, index, default_ns, _dim = _get_clients()
    vector = _voyage_embed_query_text(voyage_client, q)

    ns = default_ns if namespace is None else namespace
    k = max(1, min(int(top_k), 10000))
    q_kw: dict[str, Any] = {
        "vector": vector,
        "top_k": k,
        "include_metadata": include_metadata,
        "include_values": include_values,
    }
    if filter is not None:
        q_kw["filter"] = filter
    if ns:
        q_kw["namespace"] = ns

    out = index.query(**q_kw)
    matches = getattr(out, "matches", None) or (out.get("matches") if isinstance(out, dict) else []) or []

    return pinecone_matches_to_rows(matches, include_values=include_values)


def query_plant_knowledge_rag(
    query: str,
    *,
    plant_name: str | None = None,
    plant_id: int | None = None,
    latin_name: str | None = None,
    top_k: int = 5,
) -> str:
    """
    Embed ``query`` with Voyage (input_type=query) and retrieve top_k chunks from Pinecone
    filtered to one plant.

    Use **plant_name** for the scientific (Latin) binomial as stored in chunk metadata—same field as
    ``latin_name`` in Pinecone. ``latin_name`` is accepted as an alias. **plant_id** filters on
    numeric ``plant_id`` metadata when set instead.
    """
    q = (query or "").strip()
    if not q:
        return "Provide a non-empty question or search query for plant knowledge."

    try:
        voyage_client, index, namespace, _dim = _get_clients()
    except Exception as e:
        logger.warning("RAG client init failed: %s", e)
        return (
            "Plant knowledge search is unavailable (check VOYAGE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX). "
            f"Details: {e}"
        )

    try:
        filt = _build_filter(plant_name=plant_name, latin_name=latin_name, plant_id=plant_id)
    except ValueError as e:
        return str(e)

    try:
        vector = _voyage_embed_query_text(voyage_client, q)
    except Exception as e:
        logger.exception("Voyage embed failed")
        return f"Embedding failed: {e}"

    k = max(1, min(int(top_k), 20))
    q_kw: dict[str, Any] = {
        "vector": vector,
        "top_k": k,
        "filter": filt,
        "include_metadata": True,
        "include_values": False,
    }
    if namespace:
        q_kw["namespace"] = namespace

    try:
        out = index.query(**q_kw)
    except Exception as e:
        logger.exception("Pinecone query failed")
        return f"Pinecone query failed: {e}"

    matches = getattr(out, "matches", None) or (out.get("matches") if isinstance(out, dict) else []) or []

    rows = pinecone_matches_to_rows(matches, include_values=False)

    if not rows:
        return (
            "No knowledge-base chunks found for this plant and query "
            "(empty index, wrong filter, or chunks not embedded for this species)."
        )

    lines: list[str] = [
        f"Pinecone retrieval (top {len(rows)} chunks; filter={filt}):",
        "",
    ]
    for i, row in enumerate(rows, 1):
        meta = row.get("metadata") or {}
        score = row.get("score")
        vid = row.get("id")
        text = meta.get("text", "") if isinstance(meta, dict) else ""
        lines.append(f"--- Chunk {i} (score={score!r}, id={vid!r}) ---")
        lines.append(text if text else "(no text in metadata)")
        lines.append("")

    return "\n".join(lines).rstrip()


__all__ = [
    "PfafChunkMetadata",
    "pinecone_matches_to_rows",
    "plant_name_filter",
    "query_plant_knowledge_rag",
    "voyage_pinecone_search",
]
