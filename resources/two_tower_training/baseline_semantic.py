"""
Semantic baseline for offline two-tower evaluation.

Ranks plants by cosine similarity between:
  - the user's ``profile_embedding`` (Voyage over a built profile string, or precomputed), and
  - each plant's ``profile_embedding`` vector in the Mongo plant catalog
    (``NEW_PLANT_COLLECTION``, default ``NewPlantCollection``).

Used when Mongo plants have ``profile_embedding`` and baseline is not disabled.
Requires ``backend`` on ``sys.path`` for ``database.get_plant_collection``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"

VOYAGE_EMBED_MODEL = "voyage-4-lite"
VOYAGE_BATCH = 128


def _ensure_backend_on_path() -> None:
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))


def synthetic_user_semantic_text(user: dict) -> str:
    """Build a single string for Voyage user embedding (aligned with plant profile embedding space)."""
    parts: list[str] = []
    if user.get("persona"):
        parts.append(f"persona: {user['persona']}")
    if user.get("care_level"):
        parts.append(f"care_level: {user['care_level']}")
    if user.get("light"):
        parts.append(f"light: {user['light']}")
    if user.get("water_freq") is not None:
        parts.append(f"watering_days_between: {user['water_freq']}")
    if user.get("soil"):
        parts.append(f"soil: {user['soil']}")
    if user.get("size"):
        parts.append(f"plant_size_preference: {user['size']}")
    if user.get("growth_pref"):
        parts.append(f"growth: {user['growth_pref']}")
    if user.get("temp"):
        parts.append(f"temp: {user['temp']}")
    if user.get("climate"):
        parts.append(f"climate: {user['climate']}")
    if user.get("usda_zone") is not None:
        parts.append(f"usda_zone: {user['usda_zone']}")
    env = user.get("environment") or {}
    if env.get("light_level"):
        parts.append(f"light_level: {env['light_level']}")
    if env.get("humidity_level"):
        parts.append(f"humidity: {env['humidity_level']}")
    tp = env.get("temperature_pref") or {}
    if tp.get("min_f") is not None or tp.get("max_f") is not None:
        parts.append(f"temp_f: {tp.get('min_f')}-{tp.get('max_f')}")
    pref = user.get("preferences") or {}
    if pref.get("care_level"):
        parts.append(f"declared_care: {pref['care_level']}")
    cp = pref.get("care_preferences") or {}
    if cp.get("watering_freq") is not None:
        parts.append(f"watering_freq: {cp['watering_freq']}")
    cons = user.get("constraints") or {}
    if cons.get("preferred_size"):
        parts.append(f"preferred_size: {cons['preferred_size']}")
    return " | ".join(parts) if parts else "indoor plant grower"


def ensure_user_semantic_embeddings(users: list[dict]) -> None:
    """Set profile_embedding via Voyage for users that do not have it."""
    missing = [u for u in users if not u.get("profile_embedding")]
    if not missing:
        return
    import voyageai

    vo = voyageai.Client()
    texts = [synthetic_user_semantic_text(u) for u in missing]
    all_emb: list[list[float]] = []
    for i in range(0, len(texts), VOYAGE_BATCH):
        batch = texts[i : i + VOYAGE_BATCH]
        result = vo.embed(batch, model=VOYAGE_EMBED_MODEL, input_type="document")
        all_emb.extend(result.embeddings)
    for u, emb in zip(missing, all_emb):
        u["profile_embedding"] = emb


def load_semantic_plants_from_mongo() -> tuple[list[dict], str | None]:
    """
    Plants with non-empty ``profile_embedding`` on the document.

    Returns ``(plants, err)``. On success, ``err`` is None. On Mongo/import failure,
    ``plants`` is empty and ``err`` explains why (so callers can distinguish from an
    empty catalog).
    """
    _ensure_backend_on_path()
    try:
        from database import get_plant_collection

        coll = get_plant_collection()
        cursor = coll.find(
            {"profile_embedding": {"$exists": True, "$ne": []}},
            {"plant_id": 1, "profile_embedding": 1},
        )
        return list(cursor), None
    except Exception as e:
        return [], str(e)


def _plant_profile_embedding_vector(doc: dict) -> list[float] | None:
    raw = doc.get("profile_embedding")
    if not raw:
        return None
    if isinstance(raw[0], (int, float)):
        return [float(x) for x in raw]
    if isinstance(raw[0], list):
        return [float(x) for x in raw[0]]
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dp = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dp / (na * nb)


def baseline_semantic_rank(
    user: dict,
    mongo_plants: list[dict],
    k: int = 20,
) -> list[int]:
    """Cosine similarity — user profile_embedding vs each plant profile_embedding in MongoDB."""
    user_emb = user.get("profile_embedding")
    if not user_emb:
        return []

    scored: list[tuple[int, float]] = []
    for doc in mongo_plants:
        pid = doc.get("plant_id")
        if pid is None:
            continue
        p_vec = _plant_profile_embedding_vector(doc)
        if not p_vec or len(p_vec) != len(user_emb):
            continue
        scored.append((pid, cosine_similarity(user_emb, p_vec)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:k]]
