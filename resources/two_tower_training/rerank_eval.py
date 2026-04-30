"""
Cohere rerank path for offline eval.

1) Two-tower scores all plants (Feast 21-d → model).
2) Take top candidates; resolve Mongo plant docs from an **eval-wide prefetch** (one query, slim projection).
3) Build text documents from Mongo (aligned with ``backend/recommend/recommend.py``) and call
   ``cohere.ClientV2().rerank``. Set ``COHERE_API_KEY``; install ``cohere``.
4) Merge: reranked prefix (up to ``rerank_top_m``) + remaining catalog in original two-tower order.

Requires MongoDB (same collection as backend).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"

# Same as backend/recommend/recommend.py
RERANK_MODEL = "rerank-v3.5"

# Exclude large blobs (e.g. plant_tower_embedding) — fusion only needs profile_embedding + text fields.
MONGO_RERANK_PROJECTION = {
    "_id": 0,
    "plant_id": 1,
    "Info": 1,
    "Care": 1,
    "img_url": 1,
    "profile_embedding": 1,
}


def _ensure_backend_on_path() -> None:
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))


def fetch_mongo_plants_by_ids(
    plant_ids: list[int],
    projection: dict | None = None,
) -> dict[int, dict]:
    """plant_id -> document slice from NEW_PLANT_COLLECTION (default: slim fields for rerank)."""
    if not plant_ids:
        return {}
    _ensure_backend_on_path()
    try:
        from database import get_plant_collection
    except Exception:
        return {}

    proj = projection if projection is not None else MONGO_RERANK_PROJECTION
    coll = get_plant_collection()
    cursor = coll.find({"plant_id": {"$in": plant_ids}}, proj)
    out: dict[int, dict] = {}
    for doc in cursor:
        pid = doc.get("plant_id")
        if pid is None:
            continue
        out[int(pid)] = doc
    return out


def prefetch_mongo_plants_for_rerank(plant_ids: list[int]) -> dict[int, dict]:
    """
    One Mongo query for the whole scored catalog (Feast plant ids).
    Avoids per-user ``$in`` round-trips and full-document payloads.
    """
    return fetch_mongo_plants_by_ids(plant_ids, projection=MONGO_RERANK_PROJECTION)


def _light_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("sunlight_type") or v.get("sunlight_bucket") or str(v)
    return str(v)


def mongo_doc_to_flat_for_cohere(doc: dict) -> dict:
    """Map Mongo plant doc to flat keys expected by ``_flat_plant_to_document``."""
    info = doc.get("Info") or {}
    care = doc.get("Care") or {}
    lr = care.get("light_req") or {}
    ideal = lr.get("ideal_light")
    tol = lr.get("tolerated_light")
    temp_req = care.get("temp_req") or {}
    desc = info.get("desc") if isinstance(info.get("desc"), dict) else {}
    if not isinstance(desc, dict):
        desc = {}
    return {
        "common_name": info.get("common_name"),
        "latin": info.get("latin"),
        "ideal_light": _light_str(ideal),
        "tolerated_light": _light_str(tol),
        "sunlight_type": _light_str(ideal) or _light_str(tol),
        "humidity": care.get("humidity_req_bucket") or care.get("humidity_req"),
        "care_level": care.get("care_level"),
        "water_req": care.get("water_req_bucket") or care.get("water_req"),
        "temp_min": temp_req.get("min_temp"),
        "temp_max": temp_req.get("max_temp"),
        "climate": care.get("climate"),
        "size": info.get("size"),
        "category": info.get("category"),
        "physical_desc": desc.get("physical_desc"),
        "symbolism": desc.get("symbolism"),
    }


def _flat_plant_to_document(p: dict) -> str:
    """Same string shape as recommend._plant_to_document (flat plant dict)."""
    parts: list[str] = []
    if p.get("common_name"):
        parts.append(str(p["common_name"]))
    if p.get("latin"):
        parts.append(f"({p['latin']})")
    if p.get("ideal_light"):
        parts.append(f"Ideal light: {p['ideal_light']}")
    if p.get("tolerated_light"):
        parts.append(f"Tolerated light: {p['tolerated_light']}")
    if not p.get("ideal_light") and not p.get("tolerated_light") and p.get("sunlight_type"):
        parts.append(f"Sunlight: {p['sunlight_type']}")
    if p.get("humidity"):
        parts.append(f"Humidity: {p['humidity']}")
    if p.get("care_level"):
        parts.append(f"Care level: {p['care_level']}")
    if p.get("water_req"):
        parts.append(f"Water: {p['water_req']}")
    if p.get("temp_min") is not None and p.get("temp_max") is not None:
        parts.append(f"Temp: {p['temp_min']}-{p['temp_max']}°F")
    if p.get("climate"):
        parts.append(f"Climate: {p['climate']}")
    if p.get("size"):
        parts.append(f"Size: {p['size']}")
    if p.get("category"):
        parts.append(f"Category: {p['category']}")
    if p.get("physical_desc"):
        desc = str(p["physical_desc"])
        if len(desc) > 150:
            desc = desc[:150] + "..."
        parts.append(desc)
    if p.get("symbolism"):
        parts.append(f"Symbolism: {p['symbolism']}")
    return " | ".join(str(x) for x in parts)


def mongo_plant_to_cohere_document(doc: dict) -> str:
    """Build Cohere document text from Mongo plant doc."""
    if not doc:
        return ""
    text = _flat_plant_to_document(mongo_doc_to_flat_for_cohere(doc))
    if len(text.strip()) >= 12:
        return text
    info = doc.get("Info") or {}
    pid = doc.get("plant_id")
    nm = info.get("common_name") or info.get("latin") or ""
    return f"plant_id={pid} {nm}".strip() or f"plant_id={pid}"


def cohere_query_from_eval_user(user: dict) -> str:
    """Query string for Cohere reranker (synthetic eval users)."""
    from resources.two_tower_training.baseline_semantic import synthetic_user_semantic_text

    pref = synthetic_user_semantic_text(user)
    return (
        "Task: rank plants for this user.\n"
        f"User preferences: {pref}\n"
        "Rank higher: plants that best match the user's care level, light, water, and climate."
    )


def order_after_rerank(
    ranked_tt: list[tuple[int, float]],
    reranked_front: list[int],
) -> list[int]:
    """Place ``reranked_front`` first (in that order), then all other plants in two-tower order."""
    front = set(reranked_front)
    tail = [pid for pid, _ in ranked_tt if pid not in front]
    return list(reranked_front) + tail


def collect_candidates_for_cohere(
    ranked_tt: list[tuple[int, float]],
    retrieve_k: int,
    max_docs: int,
    mongo_cache: dict[int, dict] | None = None,
) -> list[tuple[int, dict, str]]:
    """
    Top of two-tower list; Mongo docs with non-empty Cohere document text.
    Returns up to ``max_docs`` (pid, doc, text).
    If the first window yields fewer than 2 candidates, widens the pool up to full ``ranked_tt``.
    """
    limits = [
        min(len(ranked_tt), max(retrieve_k, max_docs, 48)),
        min(len(ranked_tt), 200),
        len(ranked_tt),
    ]
    seen_limits: set[int] = set()
    for pool_limit in limits:
        if pool_limit in seen_limits or pool_limit <= 0:
            continue
        seen_limits.add(pool_limit)
        pool = ranked_tt[:pool_limit]
        if mongo_cache is not None:
            fetched = mongo_cache
        else:
            fetched = fetch_mongo_plants_by_ids([pid for pid, _ in pool])
        out: list[tuple[int, dict, str]] = []
        for pid, _sc in pool:
            doc = fetched.get(pid)
            if not doc:
                continue
            text = mongo_plant_to_cohere_document(doc)
            if len(text.strip()) < 12:
                continue
            out.append((pid, doc, text))
            if len(out) >= max_docs:
                break
        if len(out) >= 2:
            return out
    return out


def _cohere_exception_summary(e: BaseException) -> str:
    """
    Short summary for logs. Cohere SDK ``str(exc)`` often starts with huge ``headers: {...}``;
    prefer status_code / body / JSON ``message`` when present.
    """
    parts: list[str] = []
    sc = getattr(e, "status_code", None)
    if sc is None:
        resp = getattr(e, "response", None)
        if resp is not None:
            sc = getattr(resp, "status_code", None)
    body = getattr(e, "body", None)
    if body is None:
        body = getattr(e, "body_json", None)

    if sc is not None:
        parts.append(f"HTTP {int(sc)}")

    if isinstance(body, dict):
        for key in ("message", "msg"):
            v = body.get(key)
            if v is not None:
                parts.append(str(v)[:400])
                break
    elif isinstance(body, str) and body.strip():
        try:
            j = json.loads(body)
            if isinstance(j, dict) and j.get("message") is not None:
                parts.append(str(j["message"])[:400])
            else:
                parts.append(body[:400])
        except json.JSONDecodeError:
            parts.append(body[:400])

    if len(parts) >= 2 or (len(parts) == 1 and parts[0].startswith("HTTP")):
        return " | ".join(parts)

    raw = str(e)
    mj = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if mj:
        return (parts[0] + " | " if parts else "") + mj.group(1).replace("\\n", " ")[:400]
    msc = re.search(r"'status_code':\s*(\d+)", raw)
    if msc:
        line = f"HTTP {msc.group(1)}"
        if mj := re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', raw):
            return f"{line} | {mj.group(1)[:400]}"
        return line
    if len(raw) <= 500:
        return raw
    return raw[:400] + "..."


def _cohere_error_should_retry(e: BaseException) -> bool:
    sc = getattr(e, "status_code", None)
    if sc is None:
        resp = getattr(e, "response", None)
        if resp is not None:
            sc = getattr(resp, "status_code", None)
    if sc in (429, 502, 503, 504):
        return True
    s = str(e).lower()
    return any(
        x in s
        for x in ("429", "503", "502", "504", "timeout", "rate", "overloaded", "temporarily unavailable")
    )


def two_tower_then_cohere_rerank(
    ranked_tt: list[tuple[int, float]],
    user: dict,
    *,
    retrieve_k: int = 36,
    rerank_top_m: int = 20,
    mongo_cache: dict[int, dict] | None = None,
) -> tuple[list[int], dict]:
    """
    Rerank top Mongo-backed text documents with Cohere (``backend`` uses same model).
    Returns (full_order_plant_ids, debug_info).
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return [pid for pid, _ in ranked_tt], {"cohere": "skipped_no_api_key", "api_ms": 0.0}

    pool_limit = min(len(ranked_tt), max(retrieve_k, rerank_top_m * 2, 48))
    max_docs = min(pool_limit, max(rerank_top_m * 2, 32))
    candidates = collect_candidates_for_cohere(
        ranked_tt, retrieve_k, max_docs=max_docs, mongo_cache=mongo_cache
    )
    if len(candidates) < 2:
        return [pid for pid, _ in ranked_tt], {"cohere": "skipped_too_few_documents", "api_ms": 0.0}

    try:
        import cohere
    except ImportError:
        return [pid for pid, _ in ranked_tt], {"cohere": "skipped_no_cohere_package", "api_ms": 0.0}

    documents = [c[2] for c in candidates]
    query = cohere_query_from_eval_user(user)
    top_n = min(rerank_top_m, len(documents))
    resp = None
    last_err: str | None = None
    api_ms = 0.0
    try:
        co = cohere.ClientV2(api_key=api_key)
        for attempt in range(3):
            try:
                t_call = time.perf_counter()
                resp = co.rerank(
                    model=RERANK_MODEL,
                    query=query,
                    documents=documents,
                    top_n=top_n,
                )
                api_ms += (time.perf_counter() - t_call) * 1000.0
                break
            except Exception as e:
                api_ms += (time.perf_counter() - t_call) * 1000.0
                last_err = _cohere_exception_summary(e)
                if attempt < 2 and _cohere_error_should_retry(e):
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return [pid for pid, _ in ranked_tt], {
                    "cohere": "error",
                    "message": last_err,
                    "api_ms": round(api_ms, 3),
                }
    except Exception as e:
        return [pid for pid, _ in ranked_tt], {
            "cohere": "error",
            "message": _cohere_exception_summary(e),
            "api_ms": round(api_ms, 3),
        }

    if resp is None:
        return [pid for pid, _ in ranked_tt], {
            "cohere": "error",
            "message": last_err or "unknown",
            "api_ms": round(api_ms, 3),
        }

    front = [candidates[r.index][0] for r in resp.results]
    full_order = order_after_rerank(ranked_tt, front)
    return full_order, {
        "cohere": "ok",
        "n_candidates": len(candidates),
        "n_reranked": len(front),
        "model": RERANK_MODEL,
        "api_ms": round(api_ms, 3),
    }
