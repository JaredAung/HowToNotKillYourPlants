"""
Structured RAG tracing for observability (log tail, aggregators).

Set ``RAG_TRACE_MAX_CHARS`` (default 8000) to cap stored ``result_full``.
Set ``RAG_TRACE_STDOUT=0`` to disable duplicate ``print`` (logging still emits).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_PREFIX = "[RAG_TRACE]"


def _truncate(s: str, max_len: int) -> str:
    s = s or ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def log_rag_retrieval(
    *,
    phase: str,
    user_query: str,
    rag_result: str,
    plant_name: str | None = None,
    top_k: int | None = None,
    user_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Emit one JSON line tagged ``[RAG_TRACE]`` with the RAG payload for tracing.

    ``rag_result`` is the full string returned from PFAF retrieval (or tool output).
    """
    max_full = int((os.environ.get("RAG_TRACE_MAX_CHARS") or "8000").strip() or "8000")
    payload: dict[str, Any] = {
        "phase": phase,
        "plant_name": plant_name,
        "user_query": _truncate(user_query, 800),
        "top_k": top_k,
        "result_chars": len(rag_result or ""),
        "result_full": _truncate(rag_result or "", max_full),
    }
    if user_id:
        payload["user_id"] = user_id
    if extra:
        payload.update(extra)

    line = f"{_PREFIX} {json.dumps(payload, ensure_ascii=False, default=str)}"
    logger.info("%s", line)

    if (os.environ.get("RAG_TRACE_STDOUT") or "1").strip().lower() in ("1", "true", "yes"):
        print(line, flush=True)
