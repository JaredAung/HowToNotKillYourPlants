"""
Generate user-plant interactions. Maps plant water (dry/moist/wet) to days (1, 2, 7).
Loads users, balanced-samples 12 plants per user, guarantees each plant accessed at least 4 times.
Adds oracle_match (environment compatibility) to each interaction.
Run from project root: python -m resources.synthetic_user.generate_interactions
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from resources.ETL.feature_engineer import water_freq_to_days

SCRIPT_DIR = Path(__file__).resolve().parent
RESOURCES = SCRIPT_DIR.parent
USERS_PATH = RESOURCES / "data" / "synthetic_users.json"
PLANTS_PATH = RESOURCES / "data_creating" / "permapeople_plants_mapped_normalized.json"
OUTPUT_PATH = RESOURCES / "two_tower_training" / "synthetic_interactions.json"
PERSONA_INTERACTIONS_PATH = SCRIPT_DIR / "persona_interactions.json"

PLANTS_PER_USER_DEFAULT = 12
MIN_ACCESSES_PER_PLANT = 4

# Feature weights for oracle_score. Climate reduced so plants can be raised in different climates.
ORACLE_WEIGHTS = {
    "light": 0.22,
    "water": 0.27,
    "soil": 0.18,
    "care_level": 0.13,
    "climate": 0.10,
    "zone": 0.10,
}
# climate_mismatch users: climate is the dominant killer (alpine user + tropical plant)
CLIMATE_MISMATCH_ORACLE_WEIGHTS = {
    "light": 0.18,
    "water": 0.22,
    "soil": 0.12,
    "care_level": 0.08,
    "climate": 0.25,
    "zone": 0.15,
}
# Penalty for mismatches: each additional mismatch multiplies score by this factor.
# 0.85 (softer) so 2–3 mismatches don't kill the score; 0.7 was too aggressive.
MISMATCH_DECAY = 0.87

# Label threshold per care level (oracle_match uses this).
MATCH_THRESHOLDS = {"easy": 0.30, "medium": 0.40, "hard": 0.50}

# Baseline boost so overall positive rate lands in 65-80% (oracle is compatibility, not survival)
SURVIVAL_BASELINE_BOOST = 0.55

# Label modifiers: overload = total difficulty_score of user's plants above threshold
OVERLOAD_THRESHOLDS = {"easy": 100, "medium": 200, "hard": 400}
OVERLOAD_PENALTY_MAX = 0.18

# Ordinal order for categorical features (used for gradual penalty: 1/(1+distance))
LIGHT_ORDER = ["full shade", "partial sun/shade", "full sun"]
SOIL_ORDER = ["light", "medium", "heavy"]
CARE_ORDER = ["easy", "medium", "hard"]
CLIMATE_ORDER = ["alpine", "arid", "mediterranean", "temperate", "tropical"]


def _extract_plant_usda_zone(plant: dict) -> tuple[int, int]:
    """
    Extract plant USDA zone (min, max) from environment_care. Zones 1-13.
    Handles: USDA Hardiness zone as {min, max}, usda_zone_min/max, or raw string.
    """
    ec = plant.get("environment_care") or {}
    usda = ec.get("USDA Hardiness zone")
    min_z = ec.get("usda_zone_min")
    max_z = ec.get("usda_zone_max")
    if min_z is not None and max_z is not None:
        return max(1, min(13, int(min_z))), max(1, min(13, int(max_z)))
    if isinstance(usda, dict):
        min_z = usda.get("min")
        max_z = usda.get("max")
    else:
        min_z, max_z = _parse_usda_zone_raw(usda)
    min_z = min_z if min_z is not None else 3
    max_z = max_z if max_z is not None else 9
    return max(1, min(13, int(min_z))), max(1, min(13, int(max_z)))


def _parse_usda_zone_raw(raw: str | None) -> tuple[int | None, int | None]:
    """Parse USDA zone string: '3-9', '10+', '5' -> (min, max)."""
    if not raw or not str(raw).strip():
        return (None, None)
    s = str(raw).strip()
    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"(\d+)\s*\+", s)
    if m:
        return int(m.group(1)), 13
    m = re.match(r"(\d+)", s)
    if m:
        z = int(m.group(1))
        return (z, z)
    return (None, None)


def _zone_overlap_score(user_min: int, user_max: int, plant_min: int, plant_max: int) -> float:
    """
    Partial credit for USDA zone overlap. Returns 0-1.
    overlap / user_span = fraction of user's zone where plant can survive.
    Full overlap -> 1.0, partial -> 0 < x < 1, no overlap -> 0.
    """
    overlap = max(0, min(user_max, plant_max) - max(user_min, plant_min) + 1)
    user_span = user_max - user_min + 1
    if user_span <= 0:
        return 1.0 if overlap > 0 else 0.0
    return overlap / user_span


def _ordinal_distance(user_val: str | None, plant_vals: list[str], order: list[str]) -> float:
    """
    Distance in ordinal space. Returns 0 if user matches any plant value, else min distance
    from user to any plant value. Used for gradual penalty: score = 1/(1+distance).
    """
    if not user_val or not plant_vals:
        return 0.0
    uv = _norm(user_val)
    if not uv:
        return 0.0
    rank_map = {_norm(v): i for i, v in enumerate(order)}
    user_rank = rank_map.get(uv)
    if user_rank is None:
        return 0.0  # unknown value, treat as no penalty
    plant_ranks = [rank_map.get(_norm(p)) for p in plant_vals if p]
    plant_ranks = [r for r in plant_ranks if r is not None]
    if not plant_ranks:
        return 0.0
    if user_rank in plant_ranks:
        return 0.0
    return min(abs(user_rank - r) for r in plant_ranks)


def _plant_ec(plant: dict, key: str, alt_key: str | None = None) -> dict:
    """Get sub-dict from environment_care, handling normalized vs uniform format."""
    ec = plant.get("environment_care") or {}
    return ec.get(key) or ec.get(alt_key or key.replace("_req", "")) or {}


def _plant_val(plant: dict, key: str, alt_key: str | None = None):
    """Get value from environment_care."""
    ec = plant.get("environment_care") or {}
    return ec.get(key) or ec.get(alt_key or key)


def oracle_score(user: dict, plant: dict) -> float:
    """
    Weighted compatibility score 0-1. Higher = better match.
    climate_mismatch users get higher climate weight (climate is the killer).
    """
    ec = plant.get("environment_care") or {}
    weights = (
        CLIMATE_MISMATCH_ORACLE_WEIGHTS
        if _norm(user.get("persona")) == "climate_mismatch"
        else ORACLE_WEIGHTS
    )
    score = 0.0
    total_weight = 0.0

    # Light: gradual penalty by ordinal distance from plant's ideal/tolerated
    lr = _plant_ec(plant, "lighting", "light_req")
    ideal_light = lr.get("ideal_light")
    tolerated_light = lr.get("tolerated_light") or ideal_light
    plant_lights = [v for v in (ideal_light, tolerated_light) if v]
    dist = _ordinal_distance(user.get("light"), plant_lights, LIGHT_ORDER)
    light_match = 1.0 / (1.0 + dist)
    score += light_match * weights["light"]
    total_weight += weights["light"]

    # Water: gradual penalty - further from [plant_min, plant_max], worse the score
    wr = _plant_ec(plant, "water", "water_req")
    ideal_days = wr.get("ideal_water_days") or water_freq_to_days(wr.get("ideal_water"))
    tolerated_days = wr.get("tolerated_water_days") or water_freq_to_days(wr.get("tolerated_water"))
    plant_min = min(ideal_days, tolerated_days)
    plant_max = max(ideal_days, tolerated_days)
    user_freq = user.get("water_freq")
    if user_freq is None:
        water_match = 1.0
    else:
        uf = float(user_freq)
        if plant_min <= uf <= plant_max:
            distance = 0.0
        elif uf < plant_min:
            distance = plant_min - uf
        else:
            distance = uf - plant_max
        water_match = 1.0 / (1.0 + distance)  # 1.0 in range, decays with distance
    score += water_match * weights["water"]
    total_weight += weights["water"]

    # Climate: gradual penalty by ordinal distance (plants can be raised in different climates)
    plant_climate = _plant_val(plant, "origin_climate")
    dist = _ordinal_distance(user.get("climate"), [plant_climate] if plant_climate else [], CLIMATE_ORDER)
    climate_match = 1.0 / (1.0 + dist)
    score += climate_match * weights["climate"]
    total_weight += weights["climate"]

    # Soil: gradual penalty by ordinal distance from plant's ideal/tolerated
    sr = _plant_ec(plant, "soil", "soil_req")
    ideal_soil = sr.get("ideal_soil")
    tolerated_soil = sr.get("tolerated_soil") or ideal_soil
    plant_soils = [v for v in (ideal_soil, tolerated_soil) if v]
    dist = _ordinal_distance(user.get("soil"), plant_soils, SOIL_ORDER)
    soil_match = 1.0 / (1.0 + dist)
    score += soil_match * weights["soil"]
    total_weight += weights["soil"]

    # Care level: one-sided ordinal - user must be >= plant. Gradual penalty when user < plant.
    plant_care = _norm(plant.get("care_level")) or _norm(ec.get("care_level"))
    user_care = _norm(user.get("care_level"))
    rank_map = {_norm(v): i for i, v in enumerate(CARE_ORDER)}
    user_rank = rank_map.get(user_care) if user_care else None
    plant_rank = rank_map.get(plant_care) if plant_care else None
    if user_rank is None or plant_rank is None:
        care_match = 1.0
    elif user_rank >= plant_rank:
        care_match = 1.0
    else:
        distance = plant_rank - user_rank
        care_match = 1.0 / (1.0 + distance)
    score += care_match * weights["care_level"]
    total_weight += weights["care_level"]

    # Zone: partial credit by overlap of user zone [min,max] with plant zone [min,max]
    u_min = user.get("usda_zone_min")
    u_max = user.get("usda_zone_max")
    if u_min is not None and u_max is not None:
        p_min, p_max = _extract_plant_usda_zone(plant)
        zone_match = _zone_overlap_score(int(u_min), int(u_max), p_min, p_max)
    else:
        zone_match = 1.0  # no user zone -> no penalty
    score += zone_match * weights["zone"]
    total_weight += weights["zone"]

    base_score = score / total_weight if total_weight > 0 else 0.0
    num_mismatches = sum(
        1 for m in (light_match, water_match, climate_match, soil_match, care_match, zone_match) if m < 1.0
    )
    final_score = base_score * (MISMATCH_DECAY ** num_mismatches)
    return round(final_score, 4)


def oracle_match(user: dict, plant: dict, oracle_scr: float | None = None) -> bool:
    """True if oracle_score >= care-level threshold."""
    care = _norm(user.get("care_level")) or "medium"
    threshold = MATCH_THRESHOLDS.get(care, 0.40)
    if oracle_scr is not None:
        return oracle_scr >= threshold
    return oracle_score(user, plant) >= threshold


def compute_label(
    user: dict,
    plant: dict,
    oracle_scr: float,
    total_difficulty_score: float,
    persona_config: dict,
    rng: random.Random,
) -> tuple[int, float]:
    """
    Survival probability from oracle + behavior modifiers. Returns (label, health_score).
    Stochastic sampling so model learns patterns not thresholds.
    total_difficulty_score: sum of difficulty_score of all plants this user has.
    """
    survival_prob = float(oracle_scr)

    persona = user.get("persona", "")
    cfg = persona_config.get(persona, {})
    care_level = _norm(user.get("care_level")) or "medium"

    # Overload penalty: total difficulty_score above threshold dilutes care
    total_diff = cfg.get("difficulty_override")
    if total_diff is not None:
        total_diff = float(total_diff)
    else:
        total_diff = total_difficulty_score
    threshold = OVERLOAD_THRESHOLDS.get(care_level, 200)
    excess = max(0.0, total_diff - threshold)
    if excess > 0:
        overload = excess / 100.0  # scale to ~plant-count magnitude
        overload_penalty = 1.0 - (1.0 / (1.0 + 0.20 * overload))
        survival_prob -= overload_penalty * OVERLOAD_PENALTY_MAX

    # Persona survival bonus (fine-tune)
    survival_prob += cfg.get("survival_bonus", 0.0)

    # Baseline boost
    survival_prob += SURVIVAL_BASELINE_BOOST

    # Clip to valid probability range
    survival_prob = max(0.05, min(0.95, survival_prob))

    # Stochastic label (binomial — 0.7 prob → label 1 ~70% of time)
    label = 1 if rng.random() < survival_prob else 0
    health_score = round(survival_prob * rng.uniform(0.85, 1.0), 3)

    return label, health_score


def _norm(val: str | None) -> str | None:
    """Normalize string for comparison."""
    if val is None:
        return None
    return str(val).strip().lower() or None


def plant_water_to_days(plant: dict) -> tuple[int, int]:
    """
    Map plant water requirements to ideal_water_days, tolerated_water_days (1, 2, 7).
    Uses ideal_water_days/tolerated_water_days if present, else converts from ideal_water/tolerated_water.
    """
    ec = plant.get("environment_care") or {}
    wr = ec.get("water") or {}
    ideal_days = wr.get("ideal_water_days")
    tolerated_days = wr.get("tolerated_water_days")
    if ideal_days is None:
        ideal_days = water_freq_to_days(wr.get("ideal_water"))
    if tolerated_days is None:
        tolerated_days = water_freq_to_days(wr.get("tolerated_water"))
    return int(ideal_days), int(tolerated_days)


def load_users() -> list[dict]:
    with open(USERS_PATH) as f:
        return json.load(f)


def load_persona_interactions_config() -> dict:
    """Load persona interaction config (plants_min/max, selection, biases)."""
    if not PERSONA_INTERACTIONS_PATH.exists():
        return {}
    with open(PERSONA_INTERACTIONS_PATH) as f:
        return json.load(f)


def _plant_care_level(plant: dict) -> str | None:
    ec = plant.get("environment_care") or {}
    return _norm(plant.get("care_level")) or _norm(ec.get("care_level"))


def _plant_ideal_water(plant: dict) -> str | None:
    wr = _plant_ec(plant, "water", "water_req")
    return _norm(wr.get("ideal_water"))


def _plant_origin_climate(plant: dict) -> str | None:
    return _norm(_plant_val(plant, "origin_climate"))


def _plant_layer(plant: dict) -> str | None:
    lr = _plant_ec(plant, "layer_req", "layer")
    layers = lr.get("layers") or []
    return _norm(layers[0]) if layers else None


def _plant_family(plant: dict) -> str | None:
    info = plant.get("info") or {}
    return _norm(info.get("Family"))


def _cached_oracle(user_by_id: dict, plant_by_id: dict):
    """Return oracle_score function with (user_id, plant_id) -> score cache."""
    cache: dict[tuple[int, int], float] = {}

    def fn(user: dict, plant: dict) -> float:
        uid = user.get("user_id")
        pid = plant.get("id")
        if uid is not None and pid is not None:
            key = (uid, pid)
            if key not in cache:
                cache[key] = oracle_score(user, plant)
            return cache[key]
        return oracle_score(user, plant)
    return fn


def _plant_selection_weight(
    plant_id: int,
    plant: dict,
    user: dict,
    user_plants: set[int],
    plant_by_id: dict,
    persona_config: dict,
    oracle_fn,
    rng: random.Random,
) -> float:
    """
    Compute selection weight for a plant given user persona. Higher = more likely to be selected.
    """
    selection = persona_config.get("selection", "random")
    oracle_bias = persona_config.get("oracle_bias", 0.0)
    base = 1.0

    if selection == "random":
        w = base
    elif selection in ("oracle_positive", "oracle_negative", "oracle_trend"):
        score = oracle_fn(user, plant)
        if selection == "oracle_positive":
            w = base + oracle_bias * score
        elif selection == "oracle_negative":
            w = base + oracle_bias * (1.0 - score)
        else:
            w = base + oracle_bias * score
    elif selection == "care_hard":
        care = _plant_care_level(plant)
        w = base * 2.0 if care == "hard" else base
    elif selection == "care_bias":
        target = persona_config.get("care_bias", "easy")
        care = _plant_care_level(plant)
        w = base * 2.0 if care == target else base
    elif selection == "niche":
        niche = persona_config.get("niche", "dry")
        if niche == "dry":
            water = _plant_ideal_water(plant)
            w = base * 3.0 if water == "dry" else base * 0.3
        else:
            w = base
    elif selection == "climate_bias":
        target = persona_config.get("climate_bias", "tropical")
        climate = _plant_origin_climate(plant)
        w = base * 2.0 if climate == target else base
    elif selection == "diversity":
        diversity_weight = persona_config.get("diversity_weight", 2.0)
        selected_plants = [plant_by_id[pid] for pid in user_plants if pid in plant_by_id]
        my_layer = _plant_layer(plant)
        my_climate = _plant_origin_climate(plant)
        my_family = _plant_family(plant)
        overlap = 0
        for p in selected_plants:
            if _plant_layer(p) == my_layer:
                overlap += 1
            if _plant_origin_climate(p) == my_climate:
                overlap += 1
            if _plant_family(p) == my_family and my_family:
                overlap += 2
        w = base * (diversity_weight ** (1.0 / (1.0 + overlap)))
    else:
        w = base

    # care_bias: specialist/researcher attempt harder plants (adds failure signal)
    care_bias = persona_config.get("care_bias")
    if care_bias:
        care = _plant_care_level(plant)
        w *= 2.0 if care == care_bias else 0.4

    return w


def load_plants() -> list[dict]:
    path = PLANTS_PATH if PLANTS_PATH.exists() else SCRIPT_DIR.parent / "data_creating" / "permapeople_plants_mapped_uniform.json"
    with open(path) as f:
        data = json.load(f)
    return data["plants"]


def generate_interactions(
    users: list[dict],
    plants: list[dict],
    user_by_id: dict[int, dict],
    plant_by_id: dict[int, dict],
    persona_config: dict,
    seed: int = 42,
) -> list[dict]:
    """
    Balanced sample plants per user. Phase 1: each plant accessed at least MIN_ACCESSES_PER_PLANT.
    Phase 2: fill each user to persona-specific plants_per_user with persona-driven selection.
    """
    rng = random.Random(seed)
    user_ids = [u["user_id"] for u in users]
    plant_ids = [p["id"] for p in plants]
    n_users = len(user_ids)
    n_plants = len(plant_ids)

    user_plants: dict[int, set[int]] = defaultdict(set)
    plant_count: dict[int, int] = defaultdict(int)
    interactions: list[dict] = []
    cached_oracle = _cached_oracle(user_by_id, plant_by_id)

    # Phase 1: Each plant accessed at least MIN_ACCESSES_PER_PLANT times (round-robin for balance)
    for i, plant_id in enumerate(plant_ids):
        for j in range(MIN_ACCESSES_PER_PLANT):
            user_id = user_ids[(i * MIN_ACCESSES_PER_PLANT + j) % n_users]
            user_plants[user_id].add(plant_id)
            plant_count[plant_id] += 1
            interactions.append({"user_id": user_id, "plant_id": plant_id})

    # Phase 2: Fill each user to persona-specific plants_per_user with persona-driven selection
    for user_id in user_ids:
        user = user_by_id.get(user_id)
        persona = user.get("persona", "") if user else ""
        cfg = persona_config.get(persona, {})
        plants_min = cfg.get("plants_min", PLANTS_PER_USER_DEFAULT - 2)
        plants_max = cfg.get("plants_max", PLANTS_PER_USER_DEFAULT + 2)
        target = rng.randint(plants_min, plants_max)
        needed = target - len(user_plants[user_id])
        if needed <= 0:
            continue

        selection = cfg.get("selection", "random")
        early_bias = cfg.get("early_oracle_bias", 0.0)
        late_bias = cfg.get("late_oracle_bias", 0.0)

        for fill_idx in range(needed):
            candidates = [p for p in plant_ids if p not in user_plants[user_id]]
            if not candidates:
                break

            # Persona-driven weights (oracle_trend: early picks low-oracle, late picks high-oracle)
            fill_cfg = dict(cfg)
            if selection == "oracle_trend" and needed > 1:
                is_early = fill_idx < needed / 2
                fill_cfg["oracle_bias"] = early_bias if is_early else late_bias
                fill_cfg["selection"] = "oracle_positive" if not is_early else "oracle_negative"

            weights = []
            for plant_id in candidates:
                plant = plant_by_id.get(plant_id)
                if not plant:
                    weights.append(1.0)
                    continue
                w = _plant_selection_weight(
                    plant_id, plant, user or {}, user_plants[user_id],
                    plant_by_id, fill_cfg, cached_oracle, rng,
                )
                balance = 1.0 / (plant_count[plant_id] + 1)
                weights.append(max(0.01, w * balance))

            total_w = sum(weights)
            if total_w <= 0:
                probs = [1.0 / len(candidates)] * len(candidates)
            else:
                probs = [w / total_w for w in weights]
            plant_id = rng.choices(candidates, weights=probs, k=1)[0]
            user_plants[user_id].add(plant_id)
            plant_count[plant_id] += 1
            interactions.append({"user_id": user_id, "plant_id": plant_id})

    return interactions


def main(seed: int = 42) -> int:
    rng = random.Random(seed)
    users = load_users()
    plants = load_plants()
    user_by_id = {u["user_id"]: u for u in users}
    plant_by_id = {p["id"]: p for p in plants}
    persona_config = load_persona_interactions_config()

    interactions = generate_interactions(
        users, plants, user_by_id, plant_by_id, persona_config, seed=seed
    )

    user_plant_counts = Counter(i["user_id"] for i in interactions)
    user_total_difficulty: dict[int, float] = {}
    for i in interactions:
        uid = i["user_id"]
        plant = plant_by_id.get(i["plant_id"])
        diff = float(plant.get("difficulty_score") or 0) if plant else 0
        user_total_difficulty[uid] = user_total_difficulty.get(uid, 0) + diff

    for i in interactions:
        user = user_by_id.get(i["user_id"])
        plant = plant_by_id.get(i["plant_id"])
        if user and plant:
            oracle_scr = oracle_score(user, plant)
            i["oracle_score"] = oracle_scr
            i["oracle_match"] = oracle_match(user, plant, oracle_scr=oracle_scr)
            total_diff = user_total_difficulty.get(i["user_id"], 0)
            label, health = compute_label(
                user, plant, oracle_scr, total_diff, persona_config, rng
            )
            i["label"] = label
            i["health_score"] = health
        else:
            i["oracle_score"] = 0.0
            i["oracle_match"] = False
            i["label"] = 0
            i["health_score"] = 0.0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(interactions, f, indent=2)

    # Validation
    plant_accesses = Counter(i["plant_id"] for i in interactions)
    oracle_matches = sum(1 for i in interactions if i["oracle_match"])
    labels = [i["label"] for i in interactions]
    positive_rate = sum(labels) / len(labels) if labels else 0
    avg_score = sum(i["oracle_score"] for i in interactions) / len(interactions)
    min_access = min(plant_accesses.values())
    avg_plants = sum(user_plant_counts.values()) / len(user_plant_counts) if user_plant_counts else 0

    print(f"Generated {len(interactions)} interactions for {len(users)} users, {len(plants)} plants")
    print(f"Plants per user: avg {avg_plants:.1f}, min accesses per plant: {min_access} (required >= {MIN_ACCESSES_PER_PLANT})")
    print(f"Oracle score avg: {avg_score:.3f}, oracle_match: {oracle_matches} ({100 * oracle_matches / len(interactions):.1f}%)")
    print(f"Label positive rate: {positive_rate:.1%}  (target: 65-80%)")

    persona_labels: dict[str, list[int]] = defaultdict(list)
    for i in interactions:
        user = user_by_id.get(i["user_id"])
        persona = user.get("persona", "unknown") if user else "unknown"
        persona_labels[persona].append(i["label"])
    print("Per persona:")
    for persona in sorted(persona_labels.keys()):
        lbls = persona_labels[persona]
        rate = sum(lbls) / len(lbls) if lbls else 0
        print(f"  {persona:30s}: {rate:.1%}")

    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
