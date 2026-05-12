"""
Two-tower inference aligned with ``resources/two_tower_training/two_tower_model.py`` and
``training_script.py``.

User tower input is **149-d**: 21-d ``categorical_embedding`` (Feast ``user_features`` when
``feast_user_id`` is set, otherwise derived from the Mongo profile via
``resources/ETL/feature_engineer.apply_user_embeddings``), plus **64-d** mean pooled
``plant_tower_embedding`` for plants currently in the user's garden (grown) and **64-d** for plants
in recent death records (killed)—matching training aggregates built from interaction labels.

Plant vectors in MongoDB must come from the same ``PlantTower`` / checkpoint as this model.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from resources.ETL.feature_engineer import apply_user_embeddings  # noqa: E402
from resources.two_tower_training.two_tower_model import (  # noqa: E402
    OUTPUT_DIM,
    TAU,
    TwoTowerModel,
    create_two_tower_model,
    load_features_from_feast,
)

# --- Profile normalization (API / search reuse) ---------------------------------

# Matches ``resources/ETL/feature_engineer.ORDINAL_ORDERS["light"]`` / ``vocabs.json`` ``light``.
LIGHT_VOCAB = ["full shade", "partial sun/shade", "full sun"]
CARE_LEVEL_VOCAB = ["easy", "medium", "hard"]
SIZE_VOCAB = ["small", "medium", "large"]
# Same tokens as ``resources/ETL/vocabs.json`` ``origin_climate`` (two-tower climate embedding).
CLIMATE_VOCAB = ["alpine", "arid", "mediterranean", "temperate", "tropical"]
SOIL_VOCAB = ["light", "medium", "heavy"]
GROWTH_VOCAB = ["slow", "med", "fast"]
WATER_VOCAB = ["low", "medium", "high"]

VALUE_NORM: dict[str, dict[str, str]] = {
    "light": {
        # Canonical (two-tower / plant ordinal scale)
        "full shade": "full shade",
        "partial sun/shade": "partial sun/shade",
        "full sun": "full sun",
        "partial shade": "partial sun/shade",
        "partial sun": "partial sun/shade",
        # Legacy onboarding / search UI (map into same three ordinals)
        "direct": "full sun",
        "bright_light": "full sun",
        "bright light": "full sun",
        "bright_indirect": "partial sun/shade",
        "bright indirect": "partial sun/shade",
        "indirect": "partial sun/shade",
        "diffused": "full shade",
    },
    "care": {"easy": "easy", "medium": "medium", "hard": "hard", "moderate": "medium"},
    "size": {"small": "small", "medium": "medium", "large": "large"},
    # Canonical values match ``vocabs.json`` origin_climate (+ backward compat for old UI).
    "climate": {
        "alpine": "alpine",
        "arid": "arid",
        "mediterranean": "mediterranean",
        "temperate": "temperate",
        "tropical": "tropical",
        "arid tropical": "arid",
        "subtropical": "temperate",
        "subtropical arid": "arid",
        "tropical humid": "tropical",
    },
    "soil": {"light": "light", "medium": "medium", "heavy": "heavy"},
    "growth": {"slow": "slow", "med": "med", "medium": "med", "fast": "fast"},
    "water": {"low": "low", "medium": "medium", "high": "high", "moderate": "medium"},
}


def _normalize(val: str | None, key: str) -> str | None:
    if val is None or not str(val).strip():
        return None
    v = str(val).strip()
    v_lower = v.lower().replace(" ", "_")
    norm_map = VALUE_NORM.get(key, {})
    for k, canonical in norm_map.items():
        if k.lower().replace(" ", "_") == v_lower:
            return canonical
    return v


def normalize_profile(profile: dict) -> dict:
    """Normalize structured string fields to canonical vocab (Mongo onboarding ↔ training)."""
    import copy

    result = copy.deepcopy(profile)
    env = result.get("environment") or {}
    pref = result.get("preferences") or {}
    care_pref = pref.get("care_preferences") or {}
    constraints = result.get("constraints") or {}

    if env.get("light_level"):
        n = _normalize(env["light_level"], "light")
        if n:
            env["light_level"] = n
    if pref.get("care_level"):
        n = _normalize(pref["care_level"], "care")
        if n:
            pref["care_level"] = n
    if constraints.get("preferred_size"):
        n = _normalize(constraints["preferred_size"], "size")
        if n:
            constraints["preferred_size"] = n
    if env.get("soil_preference"):
        n = _normalize(env["soil_preference"], "soil")
        if n:
            env["soil_preference"] = n
    if result.get("climate"):
        n = _normalize(result["climate"], "climate")
        if n:
            result["climate"] = n
    if care_pref.get("watering_freq"):
        n = _normalize(care_pref["watering_freq"], "water")
        if n:
            care_pref["watering_freq"] = n
    if pref.get("growth_pref"):
        n = _normalize(pref["growth_pref"], "growth")
        if n:
            pref["growth_pref"] = n

    pref["care_preferences"] = care_pref
    result["environment"] = env
    result["preferences"] = pref
    result["constraints"] = constraints
    return result


# Mongo watering_freq low/medium/high → days-between-watering for user tower water norms
_WATER_BUCKET_TO_DAYS = {"low": 7.0, "medium": 2.0, "high": 1.0}


def _temp_mid_to_bucket(mid_f: float | None) -> str | None:
    if mid_f is None:
        return None
    if mid_f < 55:
        return "cold"
    if mid_f < 65:
        return "cool"
    if mid_f < 75:
        return "warm"
    return "hot"


def _username_from_user_doc(user: dict) -> str | None:
    auth = user.get("auth") or {}
    u = (auth.get("username") or auth.get("email") or "").strip()
    return u or None


def _mongo_user_to_flat_fe_dict(user: dict) -> dict:
    """Flatten normalized Mongo user doc for ``apply_user_embeddings``."""
    env = user.get("environment") or {}
    pref = user.get("preferences") or {}
    care_pref = pref.get("care_preferences") or {}
    constraints = user.get("constraints") or {}
    temp_pref = env.get("temperature_pref") or {}

    min_f = temp_pref.get("min_f")
    max_f = temp_pref.get("max_f")
    mid = None
    if min_f is not None and max_f is not None:
        mid = (float(min_f) + float(max_f)) / 2.0

    light_raw = env.get("light_level")
    light_mapped = None
    if light_raw and str(light_raw).strip():
        n = _normalize(str(light_raw).strip(), "light")
        if n in ("full shade", "partial sun/shade", "full sun"):
            light_mapped = n

    wf = care_pref.get("watering_freq")
    wf_key = str(wf).strip().lower() if wf else ""
    water_days = _WATER_BUCKET_TO_DAYS.get(wf_key)

    soil_raw = env.get("soil_preference")
    soil_key = str(soil_raw).strip().lower() if soil_raw else ""
    soil_mapped = soil_key if soil_key in ("light", "medium", "heavy") else None

    return {
        "climate": user.get("climate"),
        "light": light_mapped,
        "soil": soil_mapped,
        "size": constraints.get("preferred_size"),
        "growth_pref": pref.get("growth_pref"),
        "temp": _temp_mid_to_bucket(mid),
        "care_level": pref.get("care_level"),
        "water_freq": water_days,
        "usda_zone_min": user.get("usda_zone_min"),
        "usda_zone_max": user.get("usda_zone_max"),
    }


def _twenty_one_d_from_feast_or_mongo(user_doc: dict) -> list[float]:
    """21-d categorical_embedding: Feast row when configured, else feature_engineer on Mongo."""
    feast_uid = user_doc.get("feast_user_id")
    if feast_uid is None:
        training = user_doc.get("training") or {}
        if isinstance(training, dict) and training.get("user_id") is not None:
            feast_uid = training["user_id"]

    repo_env = os.getenv("FEAST_REPO_PATH", "").strip()
    repo_path = Path(repo_env).expanduser() if repo_env else None

    if feast_uid is not None:
        try:
            uid_int = int(feast_uid)
            user_by_id, _ = load_features_from_feast([uid_int], [], repo_path=repo_path)
            row = user_by_id.get(uid_int, {})
            cat = row.get("categorical_embedding")
            if cat and len(cat) == 21:
                return [float(x) for x in cat]
        except Exception:
            pass

    normalized = normalize_profile(user_doc)
    flat = _mongo_user_to_flat_fe_dict(normalized)
    embedded = apply_user_embeddings([flat])
    cat = embedded[0].get("categorical_embedding") if embedded else None
    if cat and len(cat) == 21:
        return [float(x) for x in cat]
    return [0.0] * 21


def _mean_pool_embeddings(plant_coll, plant_ids: list[int]) -> list[float]:
    """Mean ``plant_tower_embedding`` (64-d) for distinct plant_ids."""
    zero = [0.0] * OUTPUT_DIM
    if not plant_ids:
        return zero
    uniq = sorted(set(int(x) for x in plant_ids))
    docs = list(
        plant_coll.find(
            {"plant_id": {"$in": uniq}},
            {"plant_id": 1, "plant_tower_embedding": 1},
        )
    )
    embs: list[list[float]] = []
    for p in docs:
        emb = p.get("plant_tower_embedding")
        if emb and len(emb) == OUTPUT_DIM:
            embs.append([float(emb[i]) for i in range(OUTPUT_DIM)])
    if not embs:
        return zero
    n = len(embs)
    return [sum(e[i] for e in embs) / n for i in range(OUTPUT_DIM)]


def _grown_killed_aggregate_embeddings(username: str | None) -> tuple[list[float], list[float]]:
    """Mirror training aggregate dims using Mongo garden + deaths + catalog embeddings."""
    zero = [0.0] * OUTPUT_DIM
    if not username:
        return zero, zero

    from database import get_death_collection, get_garden_collection, get_plant_collection

    garden_coll = get_garden_collection()
    death_coll = get_death_collection()
    plant_coll = get_plant_collection()

    grown_ids = [d["plant_id"] for d in garden_coll.find({"username": username}, {"plant_id": 1})]
    killed_ids = [d["plant_id"] for d in death_coll.find({"username": username}, {"plant_id": 1})]

    return _mean_pool_embeddings(plant_coll, grown_ids), _mean_pool_embeddings(plant_coll, killed_ids)


def build_user_input_vector(user_doc: dict) -> torch.Tensor:
    """Shape ``(1, 149)`` float tensor for ``TwoTowerModel.user_tower``."""
    cat21 = _twenty_one_d_from_feast_or_mongo(user_doc)
    uname = _username_from_user_doc(user_doc)
    grown64, killed64 = _grown_killed_aggregate_embeddings(uname)
    vec = cat21 + grown64 + killed64
    if len(vec) != 149:
        raise RuntimeError(f"Expected 149-d user input, got {len(vec)}")
    return torch.tensor([vec], dtype=torch.float32)


# --- Checkpoint & model -----------------------------------------------------------

_model: TwoTowerModel | None = None


def get_two_tower_checkpoint_path() -> Path:
    """Resolve ``two_tower.pt``: ``TWO_TOWER_MODEL_PATH``, then standard repo paths."""
    env = os.getenv("TWO_TOWER_MODEL_PATH", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            _ROOT / "resources" / "two_tower_training" / "two_tower.pt",
            _ROOT / "resources" / "two_tower_training" / "output" / "two_tower.pt",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return (_ROOT / "resources" / "two_tower_training" / "two_tower.pt").resolve()


def _unwrap_checkpoint_state(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(raw).__name__}")
    inner = raw.get("model_state")
    if isinstance(inner, dict) and inner:
        return inner
    for key in ("state_dict", "model_state_dict"):
        inner = raw.get(key)
        if isinstance(inner, dict) and inner and any(
            isinstance(k, str) and k.startswith("user_tower.") for k in inner
        ):
            return inner
    return raw


def get_two_tower_model() -> TwoTowerModel:
    """Lazy-load ``TwoTowerModel`` weights from checkpoint."""
    global _model
    if _model is None:
        ckpt_path = get_two_tower_checkpoint_path()
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Two-tower checkpoint not found at {ckpt_path}. "
                "Train with resources/two_tower_training/training_script.py or set TWO_TOWER_MODEL_PATH."
            )
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = _unwrap_checkpoint_state(raw)
        model = create_two_tower_model()
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TwoTowerModel from {ckpt_path}: {e}. "
                "Checkpoint must match two_tower_model.TwoTowerModel (user_tower + plant_tower MLPs)."
            ) from e
        model.eval()
        _model = model
    return _model


def compute_user_embedding(user_doc: dict) -> list[float]:
    """
    L2-normalized **64-d** user embedding for Mongo ``$vectorSearch`` against ``plant_tower_embedding``.
    """
    x = build_user_input_vector(user_doc)
    model = get_two_tower_model()
    with torch.no_grad():
        emb = model.encode_user(x)
    return emb[0].tolist()


def score_plants(
    user_embedding: list[float],
    plant_embeddings: list[tuple[int, list[float]]],
) -> list[tuple[int, float]]:
    """
    Dot-product scores / ``TAU`` (embeddings assumed L2-normalized like training logits).
    """
    u = torch.tensor([user_embedding], dtype=torch.float32)
    scores: list[tuple[int, float]] = []
    for pid, p_emb in plant_embeddings:
        p = torch.tensor([p_emb], dtype=torch.float32)
        s = (u * p).sum().item() / TAU
        scores.append((pid, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def __getattr__(name: str) -> Any:
    """Backward-compat ``MODEL_PATH`` as Path (some callers expect module attribute)."""
    if name == "MODEL_PATH":
        return get_two_tower_checkpoint_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
