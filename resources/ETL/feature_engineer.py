"""
Plant feature embeddings: ordinal features as numeric (0-1), origin_climate as categorical.
Water (1,2,7) and USDA zone (min,max) as numeric. Ordinal: light, soil, size, growth, temperature, care_level.

Missing data handling:
- ideal/tolerated pairs (light, soil, water): if one missing, use the other for both; if both missing see below
- light (ideal, tolerated): fallback -> 0.5 each
- soil (ideal, tolerated): fallback -> 0.5 each
- water (ideal, tolerated): fallback -> 2 (moist) each, or raw Water requirement, or 2
- size, growth, temperature, care_level: 0.5 for unknown
- origin_climate: index 0 (padding) for unknown
- difficulty_score: 0.0 for None
- USDA zone: (3, 9) for null
"""
import json
import re
from pathlib import Path

import torch
import torch.nn as nn

ETL_DIR = Path(__file__).resolve().parent
VOCABS_PATH = ETL_DIR / "vocabs.json"
EMBED_DIM = 8  # per categorical feature (origin_climate only)

# Default water mapping (used if vocabs.json water_to_days missing) - dry->7, moist->2, wet->1
WATER_TO_DAYS_DEFAULT = {"dry": 7, "moist": 2, "wet": 1, "water": 1}

# USDA zones 1-13. Default when null.
USDA_ZONE_DEFAULT_MIN = 3
USDA_ZONE_DEFAULT_MAX = 9

# Ordinal orders for numeric encoding (index / (len-1) -> [0, 1]). Aligned with generate_interactions.
ORDINAL_ORDERS = {
    "light": ["full shade", "partial sun/shade", "full sun"],
    "soil": ["light", "medium", "heavy"],
    "size": ["small", "medium", "large"],
    "growth": ["slow", "med", "fast"],
    "temperature": ["cold", "cool", "warm", "hot"],
    "care_level": ["easy", "medium", "hard"],
}


def _load_vocabs() -> dict:
    if not VOCABS_PATH.exists():
        return {}
    with open(VOCABS_PATH) as f:
        return json.load(f)


def _vocab_to_idx_map(vocab: list[str]) -> dict[str, int]:
    """Map value -> index. 0 reserved for unknown/padding."""
    return {v.lower(): i + 1 for i, v in enumerate(vocab)}


def _encode(val: str | None, idx_map: dict[str, int]) -> int:
    """Encode value to index. 0 if unknown or null."""
    if val is None or not str(val).strip():
        return 0
    return idx_map.get(str(val).strip().lower(), 0)


def _ideal_tolerated_fallback(ideal: str | None, tolerated: str | None) -> tuple[str | None, str | None]:
    """
    Resolve ideal/tolerated pair: if one is missing, use the other for both.
    If ideal present and tolerated missing -> tolerated = ideal.
    If tolerated present and ideal missing -> ideal = tolerated.
    If both missing -> (None, None).
    """
    ideal_ok = ideal and str(ideal).strip()
    tolerated_ok = tolerated and str(tolerated).strip()
    if ideal_ok and tolerated_ok:
        return (ideal, tolerated)
    if ideal_ok:
        return (ideal, ideal)
    if tolerated_ok:
        return (tolerated, tolerated)
    return (None, None)


def _ordinal_to_norm(val: str | None, order: list[str]) -> float:
    """Map ordinal value to [0, 1]. Returns 0.5 for unknown."""
    if not val or not str(val).strip():
        return 0.5
    v = str(val).strip().lower()
    rank_map = {o.lower(): i for i, o in enumerate(order)}
    idx = rank_map.get(v)
    if idx is None:
        return 0.5
    n = len(order) - 1
    return float(idx) / n if n > 0 else 0.5


def _water_to_days(val: str | None, water_map: dict[str, int] | None = None) -> int:
    """Map categorical water (dry/moist/wet) to days 1, 2, 7. Uses water_to_days from vocabs; default 2 (moist)."""
    if not val or not str(val).strip():
        return 2
    m = water_map if water_map else WATER_TO_DAYS_DEFAULT
    return m.get(str(val).strip().lower(), 2)


def water_freq_to_days(water_freq: str | None) -> int:
    """Map categorical water need (dry/moist/wet) to days between watering. Default 2 (moist)."""
    return _water_to_days(water_freq)


def _extract_water_days(plant: dict, water_map: dict[str, int] | None = None) -> tuple[float, float]:
    """
    Extract ideal_water_days and tolerated_water_days (1, 2, or 7) from plant.
    Uses _ideal_tolerated_fallback: if one missing, use the other. Default 2 (moist) when both missing.
    """
    ec = plant.get("environment_care") or {}
    wr = ec.get("water") or {}

    # Prefer numeric days if present
    ideal_days = wr.get("ideal_water_days")
    tolerated_days = wr.get("tolerated_water_days")
    if ideal_days is not None and tolerated_days is not None:
        ideal = int(ideal_days) if isinstance(ideal_days, (int, float)) else 2
        tolerated = int(tolerated_days) if isinstance(tolerated_days, (int, float)) else ideal
    elif ideal_days is not None:
        ideal = int(ideal_days) if isinstance(ideal_days, (int, float)) else 2
        tolerated = ideal
    elif tolerated_days is not None:
        tolerated = int(tolerated_days) if isinstance(tolerated_days, (int, float)) else 2
        ideal = tolerated
    else:
        # Resolve ideal/tolerated from categorical; if one missing, use the other
        ideal_cat, tolerated_cat = _ideal_tolerated_fallback(wr.get("ideal_water"), wr.get("tolerated_water"))
        if ideal_cat is not None:
            ideal = _water_to_days(ideal_cat, water_map)
            tolerated = _water_to_days(tolerated_cat, water_map)
        else:
            # Fallback: raw Water requirement from plant schema
            raw = ec.get("Water requirement")
            if raw and str(raw).strip():
                parts = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
                if parts:
                    ideal = _water_to_days(parts[0], water_map)
                    tolerated = _water_to_days(parts[-1], water_map) if len(parts) > 1 else ideal
                else:
                    ideal = tolerated = 2
            else:
                ideal = tolerated = 2

    return float(ideal) / 7.0, float(tolerated) / 7.0


def _parse_usda_zone(raw: str | None) -> tuple[int | None, int | None]:
    """
    Parse USDA Hardiness zone to (min_zone, max_zone). Zones 1-13.
    - "3-9" -> (3, 9)
    - "10+" -> (10, 13)
    - "5" -> (5, 5)
    """
    if not raw or not str(raw).strip():
        return (None, None)
    s = str(raw).strip()
    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (max(1, min(13, lo)), max(1, min(13, hi)))
    m = re.match(r"(\d+)\s*\+", s)
    if m:
        lo = int(m.group(1))
        return (max(1, min(13, lo)), 13)
    m = re.match(r"(\d+)", s)
    if m:
        z = max(1, min(13, int(m.group(1))))
        return (z, z)
    return (None, None)


def _extract_usda_zone(plant: dict) -> tuple[float, float]:
    """
    Extract USDA Hardiness zone min, max (1-13) from plant.
    Handles: USDA Hardiness zone as {min, max}, usda_zone_min/max, or raw string.
    Returns (min_norm, max_norm) in [0, 1] for numeric embedding.
    """
    ec = plant.get("environment_care") or {}
    usda = ec.get("USDA Hardiness zone")

    if isinstance(usda, dict):
        min_z = usda.get("min")
        max_z = usda.get("max")
    else:
        min_z = ec.get("usda_zone_min")
        max_z = ec.get("usda_zone_max")
        if min_z is None or max_z is None:
            min_z, max_z = _parse_usda_zone(usda)

    min_z = min_z if min_z is not None else USDA_ZONE_DEFAULT_MIN
    max_z = max_z if max_z is not None else USDA_ZONE_DEFAULT_MAX
    min_z = max(1, min(13, int(min_z)))
    max_z = max(1, min(13, int(max_z)))
    if min_z > max_z:
        min_z, max_z = max_z, min_z

    # Normalize to [0, 1]: zone 1 -> 0, zone 13 -> 1
    min_norm = (min_z - 1) / 12.0
    max_norm = (max_z - 1) / 12.0
    return min_norm, max_norm


def _extract_ordinal_numerics(plant: dict) -> list[float]:
    """
    Extract ordinal features as normalized [0,1] values.
    ideal/tolerated pairs use _ideal_tolerated_fallback (if one missing, use the other).
    Single-value features use 0.5 for unknown.
    """
    ec = plant.get("environment_care") or {}
    lr = ec.get("lighting") or {}
    sr = ec.get("soil") or {}

    ideal_light, tolerated_light = _ideal_tolerated_fallback(lr.get("ideal_light"), lr.get("tolerated_light"))
    ideal_soil, tolerated_soil = _ideal_tolerated_fallback(sr.get("ideal_soil"), sr.get("tolerated_soil"))

    gr = ec.get("growth_req")
    if isinstance(gr, str):
        growth_val = gr
    elif isinstance(gr, dict) and gr.get("growths"):
        growth_val = gr["growths"][0] if gr["growths"] else None
    else:
        growth_val = None

    return [
        _ordinal_to_norm(ideal_light, ORDINAL_ORDERS["light"]),
        _ordinal_to_norm(tolerated_light, ORDINAL_ORDERS["light"]),
        _ordinal_to_norm(ideal_soil, ORDINAL_ORDERS["soil"]),
        _ordinal_to_norm(tolerated_soil, ORDINAL_ORDERS["soil"]),
        _ordinal_to_norm(ec.get("size"), ORDINAL_ORDERS["size"]),
        _ordinal_to_norm(growth_val, ORDINAL_ORDERS["growth"]),
        _ordinal_to_norm(ec.get("Temperature"), ORDINAL_ORDERS["temperature"]),
        _ordinal_to_norm(plant.get("care_level"), ORDINAL_ORDERS["care_level"]),
    ]


def _extract_origin_climate_idx(plant: dict, vocabs: dict) -> int:
    """Extract origin_climate as index for categorical embedding. 0 if unknown."""
    climate_v = _vocab_to_idx_map(vocabs.get("origin_climate", []))
    return _encode(plant.get("environment_care", {}).get("origin_climate"), climate_v)


def _build_origin_climate_embedding(vocabs: dict, embed_dim: int = EMBED_DIM) -> nn.Embedding | None:
    """Build embedding layer for origin_climate (only categorical feature)."""
    vocab_list = vocabs.get("origin_climate", [])
    if not vocab_list:
        return None
    size = len(vocab_list) + 1
    emb = nn.Embedding(size, embed_dim, padding_idx=0)
    nn.init.xavier_uniform_(emb.weight)
    return emb


def apply_categorical_embeddings(plants: list[dict], embed_dim: int = EMBED_DIM) -> list[dict]:
    """
    Add categorical_embedding to each plant.
    - origin_climate: categorical embedding (8 dims)
    - Ordinal numeric [0,1]: ideal_light, tolerated_light, ideal_soil, tolerated_soil, size, growth, temperature, care_level
    - Numeric: difficulty_score (0-1), ideal_water_norm, tolerated_water_norm (1/7, 2/7, 7/7), usda_min_norm, usda_max_norm
    """
    vocabs = _load_vocabs()
    if not vocabs:
        return plants

    water_map = vocabs.get("water_to_days") or WATER_TO_DAYS_DEFAULT
    if isinstance(water_map, dict):
        water_map = {k.lower(): int(v) for k, v in water_map.items()}
    else:
        water_map = WATER_TO_DAYS_DEFAULT

    climate_emb = _build_origin_climate_embedding(vocabs, embed_dim)
    if climate_emb is not None:
        climate_emb.eval()

    for plant in plants:
        emb_list = []

        # origin_climate: categorical embedding (8 dims)
        if climate_emb is not None:
            idx = _extract_origin_climate_idx(plant, vocabs)
            t = torch.tensor([[idx]], dtype=torch.long)
            emb = climate_emb(t)
            emb_list.extend(emb.flatten().tolist())

        # Ordinal numeric [0,1]: light, soil, size, growth, temperature, care_level
        emb_list.extend(_extract_ordinal_numerics(plant))

        # Numeric: difficulty_score, water, USDA zone
        score = plant.get("difficulty_score")
        norm = max(0.0, min(1.0, float(score) / 100.0)) if score is not None else 0.0
        emb_list.append(norm)
        ideal_norm, tolerated_norm = _extract_water_days(plant, water_map)
        emb_list.append(ideal_norm)
        emb_list.append(tolerated_norm)
        usda_min_norm, usda_max_norm = _extract_usda_zone(plant)
        emb_list.append(usda_min_norm)
        emb_list.append(usda_max_norm)

        plant["categorical_embedding"] = emb_list
    return plants


def _extract_user_ordinal_numerics(user: dict) -> list[float]:
    """
    Extract user ordinal features as normalized [0,1] values.
    User has single light/soil preference -> use for both ideal and tolerated.
    """
    light_val = user.get("light")
    soil_val = user.get("soil")
    return [
        _ordinal_to_norm(light_val, ORDINAL_ORDERS["light"]),
        _ordinal_to_norm(light_val, ORDINAL_ORDERS["light"]),
        _ordinal_to_norm(soil_val, ORDINAL_ORDERS["soil"]),
        _ordinal_to_norm(soil_val, ORDINAL_ORDERS["soil"]),
        _ordinal_to_norm(user.get("size"), ORDINAL_ORDERS["size"]),
        _ordinal_to_norm(user.get("growth_pref"), ORDINAL_ORDERS["growth"]),
        _ordinal_to_norm(user.get("temp"), ORDINAL_ORDERS["temperature"]),
        _ordinal_to_norm(user.get("care_level"), ORDINAL_ORDERS["care_level"]),
    ]


def _extract_user_water_norm(user: dict) -> tuple[float, float]:
    """User water_freq is days between watering. Normalize to [0,1] as freq/7."""
    wf = user.get("water_freq")
    if wf is None:
        return 2.0 / 7.0, 2.0 / 7.0  # moist default
    val = max(1.0, min(7.0, float(wf)))
    norm = val / 7.0
    return norm, norm


def _extract_user_usda_zone(user: dict) -> tuple[float, float]:
    """User has usda_zone_min, usda_zone_max. Normalize to [0,1]."""
    min_z = user.get("usda_zone_min")
    max_z = user.get("usda_zone_max")
    min_z = min_z if min_z is not None else USDA_ZONE_DEFAULT_MIN
    max_z = max_z if max_z is not None else USDA_ZONE_DEFAULT_MAX
    min_z = max(1, min(13, int(min_z)))
    max_z = max(1, min(13, int(max_z)))
    if min_z > max_z:
        min_z, max_z = max_z, min_z
    return (min_z - 1) / 12.0, (max_z - 1) / 12.0


def _extract_user_climate_idx(user: dict, vocabs: dict) -> int:
    """Extract user climate as index for categorical embedding. 0 if unknown."""
    climate_v = _vocab_to_idx_map(vocabs.get("origin_climate", []))
    return _encode(user.get("climate"), climate_v)


def apply_user_embeddings(users: list[dict], embed_dim: int = EMBED_DIM) -> list[dict]:
    """
    Add categorical_embedding to each user. Same structure as plants for two-tower compatibility.
    - origin_climate (climate): categorical embedding (8 dims)
    - ideal_light, tolerated_light, ideal_soil, tolerated_soil, size, growth, temperature, care_level
    - difficulty_score: 0.5 (neutral, users don't have this)
    - ideal_water_norm, tolerated_water_norm
    - usda_min_norm, usda_max_norm
    """
    vocabs = _load_vocabs()
    if not vocabs:
        return users

    climate_emb = _build_origin_climate_embedding(vocabs, embed_dim)
    if climate_emb is not None:
        climate_emb.eval()

    for user in users:
        emb_list = []

        # climate: categorical embedding (8 dims), same as plant origin_climate
        if climate_emb is not None:
            idx = _extract_user_climate_idx(user, vocabs)
            t = torch.tensor([[idx]], dtype=torch.long)
            emb = climate_emb(t)
            emb_list.extend(emb.flatten().tolist())

        # Ordinal numeric [0,1]
        emb_list.extend(_extract_user_ordinal_numerics(user))

        # Numeric: difficulty_score (0.5 neutral), water, USDA zone
        emb_list.append(0.5)  # users don't have difficulty_score
        ideal_norm, tolerated_norm = _extract_user_water_norm(user)
        emb_list.append(ideal_norm)
        emb_list.append(tolerated_norm)
        usda_min_norm, usda_max_norm = _extract_user_usda_zone(user)
        emb_list.append(usda_min_norm)
        emb_list.append(usda_max_norm)

        user["categorical_embedding"] = emb_list
    return users
