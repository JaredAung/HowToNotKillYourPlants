"""
Mapping logic for Permapeople plant data to two-tower features.
Standardizes values to lowercase. Writes vocabs to vocabs.json as we go.
Height/width: no unit = meters; "ft" = convert to meters.
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from resources.ETL.score import score_plant

ETL_DIR = Path(__file__).resolve().parent
FT_TO_M = 0.3048
VOCABS_PATH = ETL_DIR / "vocabs.json"
REGION_CLIMATE_MAP_PATH = ETL_DIR / "region_climate_map.json"


def _load_vocabs() -> dict:
    if VOCABS_PATH.exists():
        with open(VOCABS_PATH) as f:
            return json.load(f)
    return {}


def _save_vocabs(vocabs: dict) -> None:
    with open(VOCABS_PATH, "w") as f:
        json.dump(vocabs, f, indent=2)


def _normalize(s: str | None) -> str | None:
    """Standardize to lowercase; return None if empty."""
    if s is None or not str(s).strip():
        return None
    return str(s).strip().lower()


def _parse_length_meters(raw: str | None) -> tuple[float | None, float | None]:
    """
    Parse height or width to meters. Returns (min_m, max_m).
    - No unit -> assume meters.
    - Contains 'ft' -> convert to meters (1 ft = 0.3048 m).
    """
    if not raw or not str(raw).strip():
        return (None, None)
    s = str(raw).strip().lower()
    is_ft = "ft" in s
    # Range: "0.1-0.3m", "15-32m", "1-1.5m"
    m = re.match(r"([\d.]+)\s*-\s*([\d.]+)", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if is_ft:
            lo, hi = lo * FT_TO_M, hi * FT_TO_M
        return (round(lo, 4), round(hi, 4))
    # Single value: "0.6", "15 ft", "10.0"
    m = re.search(r"([\d.]+)", s)
    if m:
        val = float(m.group(1))
        if is_ft:
            val = val * FT_TO_M
        val = round(val, 4)
        return (val, val)
    return (None, None)


# USDA zones 1-13. Default when null: 3-9 (widely adaptable, covers temperate regions).
USDA_ZONE_DEFAULT_MIN = 3
USDA_ZONE_DEFAULT_MAX = 9

_REGION_CLIMATE_MAP_CACHE: dict[str, str] | None = None


def _load_region_climate_map() -> dict[str, str]:
    """Load region -> climate mapping from JSON. Cached."""
    global _REGION_CLIMATE_MAP_CACHE
    if _REGION_CLIMATE_MAP_CACHE is not None:
        return _REGION_CLIMATE_MAP_CACHE
    if not REGION_CLIMATE_MAP_PATH.exists():
        _REGION_CLIMATE_MAP_CACHE = {}
        return {}
    with open(REGION_CLIMATE_MAP_PATH) as f:
        _REGION_CLIMATE_MAP_CACHE = json.load(f)
    return _REGION_CLIMATE_MAP_CACHE


def infer_origin_climate(native_to_str: str | None) -> str:
    """Infer origin climate from Native to. Returns most common climate; default temperate."""
    if not native_to_str or not str(native_to_str).strip():
        return "temperate"
    region_map = _load_region_climate_map()
    regions = [r.strip() for r in str(native_to_str).split(",") if r.strip()]
    climates = [region_map.get(r) for r in regions]
    climates = [c for c in climates if c is not None]
    if not climates:
        return "temperate"
    return Counter(climates).most_common(1)[0][0]


def _parse_usda_zone(raw: str | None) -> tuple[int | None, int | None]:
    """
    Parse USDA Hardiness zone to (min_zone, max_zone). Zones 1-13.
    - "3-9" -> (3, 9)
    - "10+" -> (10, 13)
    - "5" -> (5, 5)
    - "4a" / "4b" -> (4, 4)  # half-zones collapse to zone number
    """
    if not raw or not str(raw).strip():
        return (None, None)
    s = str(raw).strip()
    # Range: "3-9", "2-11"
    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (max(1, min(13, lo)), max(1, min(13, hi)))
    # Single with +: "10+"
    m = re.match(r"(\d+)\s*\+", s)
    if m:
        lo = int(m.group(1))
        return (max(1, min(13, lo)), 13)
    # Single: "5", "4a", "4b"
    m = re.match(r"(\d+)", s)
    if m:
        z = int(m.group(1))
        z = max(1, min(13, z))
        return (z, z)
    return (None, None)


def _normalize_water(val: str | None) -> str | None:
    """Normalize water value. Maps 'water' -> 'wet'."""
    v = _normalize(val)
    if not v:
        return None
    if v == "water":
        return "wet"
    return v


def map_light_to_ideal_tolerated(light_raw: str | None) -> dict[str, str | None]:
    """
    Convert Light requirement into ideal_light and tolerated_light.
    Standardizes to lowercase.

    - Split by comma.
    - If 1 value: ideal_light = first, tolerated_light = null.
    - If 2+ values: ideal_light = first, tolerated_light = last.
    """
    if not light_raw or not str(light_raw).strip():
        return {"ideal_light": None, "tolerated_light": None}

    parts = [_normalize(p) for p in str(light_raw).split(",") if p and p.strip()]
    parts = [p for p in parts if p]
    if not parts:
        return {"ideal_light": None, "tolerated_light": None}

    ideal = parts[0]
    tolerated = parts[-1] if len(parts) > 1 else None
    return {"ideal_light": ideal, "tolerated_light": tolerated}


def _normalize_soil(val: str | None) -> str | None:
    """Lowercase, remove (sandy)/(clay), filter out acidic."""
    v = _normalize(val)
    if not v:
        return None
    v = re.sub(r"\s*\(sandy\)", "", v, flags=re.I)
    v = re.sub(r"\s*\(clay\)", "", v, flags=re.I)
    v = v.strip()
    if not v or v == "acidic":
        return None
    return v


def map_soil_to_ideal_tolerated(soil_raw: str | None) -> dict[str, str | None]:
    """
    Convert Soil type into ideal_soil and tolerated_soil.
    Split by comma. Remove (sandy), (clay). Exclude acidic.
    - If 1 value: ideal_soil = first, tolerated_soil = null.
    - If 2+ values: ideal_soil = first, tolerated_soil = last.
    """
    if not soil_raw or not str(soil_raw).strip():
        return {"ideal_soil": None, "tolerated_soil": None}

    parts = [_normalize_soil(p) for p in str(soil_raw).split(",") if p and p.strip()]
    parts = [p for p in parts if p]
    if not parts:
        return {"ideal_soil": None, "tolerated_soil": None}

    ideal = parts[0]
    tolerated = parts[-1] if len(parts) > 1 else None
    return {"ideal_soil": ideal, "tolerated_soil": tolerated}


def map_water_to_ideal_tolerated(water_raw: str | None) -> dict[str, str | None]:
    """
    Convert Water requirement into ideal_water and tolerated_water.
    Standardizes to lowercase.

    - Split by comma.
    - If 1 value: ideal_water = first, tolerated_water = null.
    - If 2+ values: ideal_water = first, tolerated_water = last.
    """
    if not water_raw or not str(water_raw).strip():
        return {"ideal_water": None, "tolerated_water": None}

    parts = [_normalize_water(p) for p in str(water_raw).split(",") if p and p.strip()]
    parts = [p for p in parts if p]
    if not parts:
        return {"ideal_water": None, "tolerated_water": None}

    ideal = parts[0]
    tolerated = parts[-1] if len(parts) > 1 else None
    return {"ideal_water": ideal, "tolerated_water": tolerated}


def apply_light_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Apply light mapping. Replaces with lighting.ideal_light, lighting.tolerated_light."""
    ec = plant.get("environment_care") or {}
    light_raw = ec.get("Light requirement")
    mapped = map_light_to_ideal_tolerated(light_raw)
    ec.pop("light_req", None)
    ideal = mapped["ideal_light"] or "full sun"
    ec["lighting"] = {
        "ideal_light": ideal,
        "tolerated_light": mapped["tolerated_light"] or ideal,
    }
    plant["environment_care"] = ec
    return plant


def _size_from_height_width(height_m: float | None, width_m: float | None) -> str:
    """
    Size from combined height and width (1 m threshold).
    - small: height < 1 and width < 1
    - medium: (height < 1 and width >= 1) or (height >= 1 and width < 1), or if either is None
    - large: height >= 1 and width >= 1
    """
    if height_m is None or width_m is None:
        return "medium"
    if height_m < 1 and width_m < 1:
        return "small"
    if height_m >= 1 and width_m >= 1:
        return "large"
    return "medium"


def apply_height_width_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Parse Height and Width to meters. If range, use max. Adds height_meters, width_meters, size."""
    ec = plant.get("environment_care") or {}
    h_lo, h_hi = _parse_length_meters(ec.get("Height"))
    w_lo, w_hi = _parse_length_meters(ec.get("Width"))
    if h_hi is not None:
        ec["height_meters"] = h_hi  # use max for ranges
    if w_hi is not None:
        ec["width_meters"] = w_hi  # use max for ranges
    ec["size"] = _size_from_height_width(ec.get("height_meters"), ec.get("width_meters"))
    plant["environment_care"] = ec
    return plant


def _zone_to_temperature(zone: int) -> str:
    """Map USDA zone to temperature: 1-4 cold, 5-7 cool, 8-10 warm, 11+ hot."""
    if zone <= 4:
        return "cold"
    if zone <= 7:
        return "cool"
    if zone <= 10:
        return "warm"
    return "hot"


def apply_usda_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Parse USDA Hardiness zone: replace with {min, max} subfields. Append Temperature."""
    ec = plant.get("environment_care") or {}
    raw = ec.get("USDA Hardiness zone")
    min_z, max_z = _parse_usda_zone(raw)
    min_z = min_z if min_z is not None else USDA_ZONE_DEFAULT_MIN
    max_z = max_z if max_z is not None else USDA_ZONE_DEFAULT_MAX
    ec["USDA Hardiness zone"] = {"min": min_z, "max": max_z}
    ec.pop("usda_zone_min", None)
    ec.pop("usda_zone_max", None)
    mid = (min_z + max_z) // 2
    ec["Temperature"] = _zone_to_temperature(mid)
    ec.pop("temperature", None)
    plant["environment_care"] = ec
    return plant


def apply_soil_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Apply soil mapping. Replaces with soil.ideal_soil, soil.tolerated_soil."""
    ec = plant.get("environment_care") or {}
    soil_raw = ec.get("Soil type")
    mapped = map_soil_to_ideal_tolerated(soil_raw)
    ec.pop("soil_req", None)
    ideal = mapped["ideal_soil"] or "medium"
    ec["soil"] = {
        "ideal_soil": ideal,
        "tolerated_soil": mapped["tolerated_soil"] or ideal,
    }
    plant["environment_care"] = ec
    return plant


def map_layer_to_list(layer_raw: str | None) -> list[str]:
    """
    Convert Layer into a list of layers. Split by comma, lowercase each.
    If null/empty, return ["herbs"].
    """
    if not layer_raw or not str(layer_raw).strip():
        return ["herbs"]

    parts = [_normalize(p) for p in str(layer_raw).split(",") if p and p.strip()]
    parts = [p for p in parts if p]
    return parts if parts else ["herbs"]


def map_life_cycle_to_list(life_cycle_raw: str | None) -> list[str]:
    """
    Convert Life cycle into a list. Split by comma, lowercase each.
    If null/empty, return ["annual"].
    """
    if not life_cycle_raw or not str(life_cycle_raw).strip():
        return ["annual"]

    parts = [_normalize(p) for p in str(life_cycle_raw).split(",") if p and p.strip()]
    parts = [p for p in parts if p]
    return parts if parts else ["annual"]


def apply_life_cycle_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Apply life cycle mapping. Adds life_cycle_req.life_cycles as list. Keeps Life cycle as comma-joined string."""
    ec = plant.get("environment_care") or {}
    raw = ec.get("Life cycle")
    life_cycles = map_life_cycle_to_list(raw)

    if "life_cycle_req" not in ec:
        ec["life_cycle_req"] = {}
    ec["life_cycle_req"]["life_cycles"] = life_cycles
    ec["Life cycle"] = ", ".join(life_cycles)

    plant["environment_care"] = ec
    return plant


def map_growth_to_list(growth_raw: str | None) -> list[str]:
    """
    Convert Growth into a list. Split by comma, lowercase each.
    If null/empty, return ["medium"].
    """
    if not growth_raw or not str(growth_raw).strip():
        return ["medium"]

    parts = [_normalize(p) for p in str(growth_raw).split(",") if p and p.strip()]
    parts = [p for p in parts if p]
    return parts if parts else ["medium"]


def _growth_to_short(val: str) -> str:
    """Map growth to fast/med/slow. Uses first value if list."""
    if not val:
        return "med"
    v = str(val).strip().lower()
    if v == "medium":
        return "med"
    if v in ("fast", "slow"):
        return v
    return "med"


def apply_growth_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Apply growth mapping. Replaces with growth_req as single value fast/med/slow."""
    ec = plant.get("environment_care") or {}
    raw = ec.get("Growth")
    growths = map_growth_to_list(raw)
    first = growths[0] if growths else "medium"
    ec.pop("growth_req", None)
    ec["growth_req"] = _growth_to_short(first)
    ec["Growth"] = ", ".join(growths)

    plant["environment_care"] = ec
    return plant


def apply_layer_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Apply layer mapping. Adds layer_req.layers as list. Keeps Layer as comma-joined string."""
    ec = plant.get("environment_care") or {}
    raw = ec.get("Layer")
    layers = map_layer_to_list(raw)

    if "layer_req" not in ec:
        ec["layer_req"] = {}
    ec["layer_req"]["layers"] = layers
    ec["Layer"] = ", ".join(layers)

    plant["environment_care"] = ec
    return plant


def apply_water_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Apply water mapping. Adds water.ideal_water, water.tolerated_water, ideal_water_days, tolerated_water_days."""
    from resources.ETL.feature_engineer import water_freq_to_days

    ec = plant.get("environment_care") or {}
    water_raw = ec.get("Water requirement")
    mapped = map_water_to_ideal_tolerated(water_raw)
    ec.pop("water_req", None)
    ideal = mapped["ideal_water"] or "moist"
    tolerated = mapped["tolerated_water"] or ideal
    ec["water"] = {
        "ideal_water": ideal,
        "tolerated_water": tolerated,
        "ideal_water_days": water_freq_to_days(ideal),
        "tolerated_water_days": water_freq_to_days(tolerated),
    }
    plant["environment_care"] = ec
    return plant


def apply_climate_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Infer origin_climate from Native to. Adds to environment_care."""
    info = plant.get("info") or {}
    native_to = info.get("Native to")
    climate = infer_origin_climate(native_to)
    ec = plant.get("environment_care") or {}
    ec["origin_climate"] = climate
    plant["environment_care"] = ec
    return plant


def _score_to_care_level(score: float) -> str:
    """Convert difficulty_score to care_level: easy (<40), medium (40-59), hard (60+)."""
    if score < 40:
        return "easy"
    if score < 60:
        return "medium"
    return "hard"


def apply_care_level_mapping(plant: dict[str, Any]) -> dict[str, Any]:
    """Add care_level and difficulty_score. If null/absent, compute via scoring formula."""
    care = plant.get("care_level")
    score = plant.get("difficulty_score")
    if care is not None and care != "" and score is not None:
        return plant  # already set

    difficulty_score, _ = score_plant(plant)
    plant["difficulty_score"] = difficulty_score
    plant["care_level"] = _score_to_care_level(difficulty_score)
    return plant


def apply_mappings(plants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply light and water mappings to all plants. Collects unique values
    and writes vocabs.json at the end.
    """
    light_vals: set[str] = set()
    water_vals: set[str] = set()
    size_vals: set[str] = set()

    soil_vals: set[str] = set()
    layer_vals: set[str] = set()
    life_cycle_vals: set[str] = set()
    growth_vals: set[str] = set()
    temperature_vals: set[str] = set()
    climate_vals: set[str] = set()

    for p in plants:
        apply_light_mapping(p)
        apply_water_mapping(p)
        apply_soil_mapping(p)
        apply_life_cycle_mapping(p)
        apply_growth_mapping(p)
        apply_layer_mapping(p)
        apply_climate_mapping(p)
        apply_height_width_mapping(p)
        apply_usda_mapping(p)
        apply_care_level_mapping(p)
        ec = p.get("environment_care") or {}
        lr = ec.get("lighting") or {}
        wr = ec.get("water") or {}
        sr = ec.get("soil") or {}
        layr = ec.get("layer_req") or {}
        if lr.get("ideal_light"):
            light_vals.add(lr["ideal_light"])
        if lr.get("tolerated_light"):
            light_vals.add(lr["tolerated_light"])
        if wr.get("ideal_water"):
            water_vals.add(wr["ideal_water"])
        if wr.get("tolerated_water"):
            water_vals.add(wr["tolerated_water"])
        if sr.get("ideal_soil"):
            soil_vals.add(sr["ideal_soil"])
        if sr.get("tolerated_soil"):
            soil_vals.add(sr["tolerated_soil"])
        for layer in layr.get("layers") or []:
            layer_vals.add(layer)
        for lc in (ec.get("life_cycle_req") or {}).get("life_cycles") or []:
            life_cycle_vals.add(lc)
        gr = ec.get("growth_req")
        if gr and isinstance(gr, str):
            growth_vals.add(gr)
        if ec.get("Temperature"):
            temperature_vals.add(ec["Temperature"])
        if ec.get("origin_climate"):
            climate_vals.add(ec["origin_climate"])
        if ec.get("size"):
            size_vals.add(ec["size"])

    vocabs = _load_vocabs()
    vocabs["light"] = sorted(light_vals)
    vocabs["water_freq"] = [1, 2, 7]
    vocabs["soil"] = sorted(soil_vals)
    vocabs["layer"] = sorted(layer_vals)
    vocabs["life_cycle"] = sorted(life_cycle_vals)
    vocabs["growth"] = sorted(growth_vals)
    vocabs["temperature"] = sorted(temperature_vals)
    vocabs["origin_climate"] = sorted(climate_vals)
    vocabs["size"] = sorted(size_vals)
    _save_vocabs(vocabs)
    return plants
