"""
Map MongoDB catalog documents (Permapeople-style) to flat API fields.

Expected document shape (subset):
- ``plant_id``, ``name``, ``scientific_name``, ``description``
- ``info`` — nested metadata (optional ``desc`` dict with ``physical_desc`` / ``symbolism``)
- ``environment_care`` — lighting, water, soil, size, climate strings
- ``images`` — ``thumb``, ``title``, etc.
- ``care_level`` (top-level or under ``environment_care``)
"""
from __future__ import annotations

import copy
from datetime import date, datetime

from bson import ObjectId
from bson.decimal128 import Decimal128

try:
    from bson import Int64 as BSONInt64
except ImportError:
    BSONInt64 = None  # type: ignore

_SKIP_KEYS = frozenset({"_id", "plant_id", "id"})
_SKIP_NAMES = frozenset({"profile_embedding", "plant_tower_embedding"})


def pick_image_url(p: dict) -> str | None:
    imgs = p.get("images")
    if isinstance(imgs, dict):
        for key in ("thumb", "title", "primary", "url"):
            v = imgs.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return None


def _as_light_side(raw: object) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return {"sunlight_type": raw}
    return {}


def _light_req_from_doc(p: dict) -> dict:
    ec = p.get("environment_care") or {}
    lighting = ec.get("lighting")
    if isinstance(lighting, dict) and (lighting.get("ideal_light") or lighting.get("tolerated_light")):
        return {
            "ideal_light": lighting.get("ideal_light"),
            "tolerated_light": lighting.get("tolerated_light"),
        }
    lr_txt = ec.get("Light requirement")
    if isinstance(lr_txt, str) and lr_txt.strip():
        return {"ideal_light": lr_txt, "tolerated_light": lr_txt}
    return {}


def _water_from_doc(p: dict):
    ec = p.get("environment_care") or {}
    water = ec.get("water")
    if isinstance(water, dict):
        x = water.get("ideal_water") or water.get("tolerated_water")
        if x is not None:
            return x
    wr = ec.get("Water requirement")
    if isinstance(wr, str) and wr.strip():
        return wr
    return None


def _humidity_from_doc(p: dict):
    ec = p.get("environment_care") or {}
    h = ec.get("Humidity requirement") or ec.get("humidity_req")
    return h


def _temp_from_doc(p: dict) -> tuple:
    """Return (min, max) in °F when present; catalog may omit numeric temps."""
    ec = p.get("environment_care") or {}
    tr = ec.get("temp_req") or ec.get("temperature_pref")
    if isinstance(tr, dict):
        lo = tr.get("min_temp")
        hi = tr.get("max_temp")
        if lo is not None or hi is not None:
            return lo, hi
    return None, None


def _climate_from_doc(p: dict):
    ec = p.get("environment_care") or {}
    return ec.get("origin_climate")


def _care_level_from_doc(p: dict):
    if p.get("care_level"):
        return p.get("care_level")
    ec = p.get("environment_care") or {}
    return ec.get("care_level")


def _coerce_float_scalar(v: object) -> float | None:
    """Parse Mongo BSON scalars (Decimal128, Int64, etc.) for JSON-safe floats."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, Decimal128):
        try:
            return float(v.to_decimal())
        except Exception:
            return None
    if isinstance(v, (int, float)):
        return float(v)
    if BSONInt64 is not None and isinstance(v, BSONInt64):
        try:
            return float(int(v))
        except Exception:
            return None
    if isinstance(v, str) and v.strip():
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _infer_difficulty_score_from_care_level(care: str | None) -> float | None:
    """When numeric difficulty is missing, map common care labels to a 0–100 scale (display hint)."""
    if not care or not isinstance(care, str):
        return None
    c = care.strip().lower()
    if c in ("easy", "beginner", "low"):
        return 25.0
    if c in ("medium", "moderate"):
        return 50.0
    if c in ("hard", "difficult", "high"):
        return 75.0
    return None


def _difficulty_score_from_doc(p: dict) -> float | None:
    """Read numeric difficulty from catalog docs (Permapeople: top-level ``difficulty_score``)."""
    candidates = []
    v = p.get("difficulty_score")
    if v is not None:
        candidates.append(v)
    info = p.get("info")
    if isinstance(info, dict):
        for key in ("difficulty_score", "Difficulty"):
            x = info.get(key)
            if x is not None:
                candidates.append(x)
                break
        desc = info.get("desc")
        if isinstance(desc, dict):
            x = desc.get("difficulty_score")
            if x is not None:
                candidates.append(x)

    out: float | None = None
    for raw in candidates:
        coerced = _coerce_float_scalar(raw)
        if coerced is not None:
            out = coerced
            break

    if out is None:
        out = _infer_difficulty_score_from_care_level(_care_level_from_doc(p))

    if out is None:
        return None
    return float(out)


def _info_dict(p: dict) -> dict:
    return p.get("info") or {}


def flatten_catalog_plant_for_api(p: dict) -> dict:
    """
    Map one Mongo catalog document to the flat shape used by ``/recommend``, rerank, and the web UI.
    """
    info = _info_dict(p)
    light_req = _light_req_from_doc(p)
    ideal_raw = light_req.get("ideal_light")
    tol_raw = light_req.get("tolerated_light")
    ideal = _as_light_side(ideal_raw)
    tolerated = _as_light_side(tol_raw)

    sunlight_type = ideal.get("sunlight_type") or tolerated.get("sunlight_type")
    ideal_out = ideal.get("sunlight_type") or ideal.get("sunlight_bucket")
    tol_out = tolerated.get("sunlight_type") or tolerated.get("sunlight_bucket")
    if not ideal_out and isinstance(ideal_raw, str):
        ideal_out = ideal_raw
    if not tol_out and isinstance(tol_raw, str):
        tol_out = tol_raw
    if not sunlight_type:
        sunlight_type = ideal_out or tol_out

    temp_min, temp_max = _temp_from_doc(p)
    desc = info.get("desc") if isinstance(info.get("desc"), dict) else {}
    physical_desc = desc.get("physical_desc") if desc else None
    if physical_desc is None and isinstance(p.get("description"), str):
        physical_desc = p.get("description")

    ec = p.get("environment_care") or {}
    size = info.get("size")
    if size is None:
        size = ec.get("size")

    return {
        "plant_id": p["plant_id"],
        "score": round(float(p.get("score", 0)), 4),
        "img_url": pick_image_url(p),
        "latin": info.get("latin") or p.get("scientific_name"),
        "common_name": info.get("common_name") or p.get("name"),
        "sunlight_type": sunlight_type,
        "ideal_light": ideal_out,
        "tolerated_light": tol_out,
        "humidity": _humidity_from_doc(p),
        "care_level": _care_level_from_doc(p),
        "difficulty_score": _difficulty_score_from_doc(p),
        "water_req": _water_from_doc(p),
        "temp_min": temp_min,
        "temp_max": temp_max,
        "climate": _climate_from_doc(p),
        "size": size,
        "category": info.get("category") or info.get("Family"),
        "physical_desc": physical_desc,
        "symbolism": desc.get("symbolism") if desc else None,
    }


def sanitize_catalog_document(doc: dict) -> dict:
    """Copy a Mongo plant document for API JSON: strip ids and embedding vectors; normalize BSON types."""

    def walk(x: object) -> object:
        if isinstance(x, dict):
            out: dict = {}
            for k, v in x.items():
                if k in _SKIP_KEYS or k in _SKIP_NAMES:
                    continue
                if isinstance(k, str) and k.endswith("_embedding"):
                    continue
                out[k] = walk(v)
            return out
        if isinstance(x, list):
            return [walk(i) for i in x]
        if isinstance(x, ObjectId):
            return None
        if isinstance(x, (datetime, date)):
            return x.isoformat()
        if isinstance(x, Decimal128):
            return float(x.to_decimal())
        return x

    return walk(copy.deepcopy(doc))
