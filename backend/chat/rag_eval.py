"""
RAG eval helpers: Pinecone index wiring and **Voyage query embed → Pinecone search**.

Loads repo-root ``.env``. Env names align with ``resources/RAG/embed_pinecone.py`` and
``chat.rag_retrieval``:

- ``PINECONE_API_KEY`` (required)
- ``PINECONE_INDEX`` — index name for ``pc.Index(name=...)`` (default connection path)
- ``PINECONE_INDEX_HOST`` — optional; if set, ``pc.Index(host=...)`` bypasses control-plane name lookup
- ``PINECONE_NAMESPACE`` — optional query/upsert namespace
- ``VOYAGE_API_KEY`` (required for :func:`voyage_pinecone_search`)
- ``VOYAGE_EMBED_MODEL`` — optional; defaults to ``voyage-3-large``

Use :func:`chat.rag_retrieval.voyage_pinecone_search` (re-exported below) for Voyage embed +
Pinecone query. For a pre-computed vector, use ``get_pinecone_index().query(...)`` directly.

CLI from ``backend``: ``python -m chat.rag_eval search "your question"`` or ``python -m chat.rag_eval info``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from chat.rag_retrieval import PfafChunkMetadata, voyage_pinecone_search

_pc: Any | None = None
_index: Any | None = None


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is not None and str(v).strip() != "":
        return str(v).strip().strip('"').strip("'")
    return default


def get_pinecone() -> Any:
    """Singleton ``Pinecone`` client (control plane + factory for index handles)."""
    global _pc
    if _pc is None:
        from pinecone import Pinecone

        key = _env("PINECONE_API_KEY")
        if not key:
            raise ValueError("PINECONE_API_KEY is required in environment or .env")
        _pc = Pinecone(api_key=key)
    return _pc


def get_pinecone_index() -> Any:
    """
    Singleton Pinecone **Index** handle for vector ops (query, upsert, …).

    Prefer ``PINECONE_INDEX_HOST`` when set (direct data-plane URL from the Pinecone console);
    otherwise ``PINECONE_INDEX`` (index name; client resolves host via control plane).
    """
    global _index
    if _index is None:
        pc = get_pinecone()
        host = (_env("PINECONE_INDEX_HOST") or "").strip()
        name = _env("PINECONE_INDEX")
        if host:
            _index = pc.Index(host=host)
        elif name:
            _index = pc.Index(name)
        else:
            raise ValueError(
                "Set PINECONE_INDEX (index name) or PINECONE_INDEX_HOST (data-plane URL) in .env"
            )
    return _index


def get_pinecone_namespace() -> str:
    """Namespace segment for this project’s vectors; empty string = index default namespace."""
    return _env("PINECONE_NAMESPACE") or ""


def reset_pinecone_clients() -> None:
    """Drop cached client/index (e.g. after env change in tests)."""
    global _pc, _index
    _pc = None
    _index = None


def main(argv: list[str] | None = None) -> int:
    """CLI: ``search`` (Voyage + Pinecone) or ``info`` (env-backed index smoke check)."""
    parser = argparse.ArgumentParser(description="RAG eval — Voyage query embed + Pinecone search")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Embed query with Voyage and query Pinecone")
    p_search.add_argument("query", help="Natural-language search string")
    p_search.add_argument("-k", "--top-k", type=int, default=10, help="Number of matches (default 10)")
    p_search.add_argument(
        "--json",
        action="store_true",
        help="Print full hits as JSON (id, score, metadata; and values if --include-values)",
    )
    p_search.add_argument(
        "--include-values",
        action="store_true",
        help="Request embedding vectors from Pinecone (very large output with --json)",
    )

    sub.add_parser("info", help="Print Pinecone index handle and PINECONE_NAMESPACE from env")

    args = parser.parse_args(argv)

    if args.command == "info":
        idx = get_pinecone_index()
        ns = get_pinecone_namespace()
        print(f"index: {idx!r}")
        print(f"PINECONE_NAMESPACE: {ns!r}")
        return 0

    if args.command == "search":
        try:
            hits = voyage_pinecone_search(
                args.query,
                top_k=args.top_k,
                include_values=bool(args.include_values),
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if not hits:
            print("(no matches)")
            return 0
        if args.json:
            print(json.dumps(hits, ensure_ascii=False, indent=2))
        else:
            for i, h in enumerate(hits, 1):
                meta = h.get("metadata") or {}
                print(f"--- {i} score={h.get('score')} id={h.get('id')!r} ---")
                print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "get_pinecone",
    "get_pinecone_index",
    "get_pinecone_namespace",
    "PfafChunkMetadata",
    "reset_pinecone_clients",
    "voyage_pinecone_search",
]
