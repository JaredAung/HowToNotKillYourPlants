"""
Generate synthetic users from personas.json with PERSONA_POPULATION distribution.
Balanced randomization of vocab values within each persona's user pool.
Run from project root: python -m resources.synthetic_user.generate_users
"""
import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PERSONAS_PATH = SCRIPT_DIR / "personas.json"
VOCABS_PATH = SCRIPT_DIR / "vocabs.json"
OUTPUT_PATH = SCRIPT_DIR.parent / "data" / "synthetic_users.json"

VOCAB_KEYS = ["light", "water_freq", "care_level", "soil", "size", "growth_pref", "temp", "climate"]

# care_level = easy always: overconfident_beginner, nervous_nurturer, impulsive_buyer
CARE_LEVEL_EASY = {"overconfident_beginner", "nervous_nurturer", "impulsive_buyer"}
# care_level = hard always: researcher, serial_experimenter, specialist
CARE_LEVEL_HARD = {"researcher", "serial_experimenter", "specialist"}

# water_freq category -> (min, max) days for decimal sampling
WATER_FREQ_RANGES = {1: (0, 3), 2: (1, 5), 7: (4, 7)}

# Climate -> (zone_min, zone_max) for USDA hardiness. User's zone sampled within this range.
CLIMATE_TO_ZONE_RANGE = {
    "alpine": (1, 4),
    "arid": (5, 10),
    "mediterranean": (8, 10),
    "temperate": (4, 8),
    "tropical": (10, 13),
}

PERSONA_POPULATION = {
    "overconfident_beginner": 120,
    "nervous_nurturer": 100,
    "serial_experimenter": 80,
    "specialist": 100,
    "recovering_killer": 70,
    "collector": 70,
    "climate_mismatch": 70,
    "impulsive_buyer": 100,
    "researcher": 100,
}


def load_vocabs() -> dict:
    """Load vocab lists (exclude non-list values like 'embeddings', 'scalar')."""
    with open(VOCABS_PATH) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k in VOCAB_KEYS and isinstance(v, list)}


def load_personas() -> dict[str, dict]:
    """Load personas keyed by persona name."""
    with open(PERSONAS_PATH) as f:
        personas = json.load(f)
    return {p["persona"]: p for p in personas}


def balanced_sample(vocab_values: list, n: int, seed: int | None = None) -> list:
    """Return n values from vocab, each option appearing ~n/len(vocab) times. Shuffled."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    size = len(vocab_values)
    # Build list: each value appears floor(n/size) times, first n%size values get +1
    counts = [n // size] * size
    for i in range(n % size):
        counts[i] += 1
    result = []
    for i, v in enumerate(vocab_values):
        result.extend([v] * counts[i])
    rng.shuffle(result)
    return result


def sample_water_freq_decimal(category: int, rng: random.Random) -> float:
    """Given water_freq category 1, 2, or 7, return decimal in the appropriate range."""
    lo, hi = WATER_FREQ_RANGES[category]
    return round(rng.uniform(lo, hi), 2)


def get_care_level(persona_name: str, variant_value: str | None) -> str:
    """Apply care_level rules: easy/hard for specific personas, else use variant."""
    if persona_name in CARE_LEVEL_EASY:
        return "easy"
    if persona_name in CARE_LEVEL_HARD:
        return "hard"
    return variant_value or "medium"


def generate_user(
    user_id: int,
    persona: dict,
    vocabs: dict,
    persona_name: str,
    variant_assignments: dict[str, list],
    water_freq_values: list[float],
    user_index: int,
    rng: random.Random,
) -> dict:
    """Create one user with persona base + balanced vocab variation."""
    user = {
        "user_id": user_id,
        "persona": persona_name,
    }
    for key in VOCAB_KEYS:
        if key == "care_level":
            variant_val = variant_assignments[key][user_index] if key in variant_assignments and user_index < len(variant_assignments.get(key, [])) else "medium"
            user[key] = get_care_level(persona_name, variant_val)
        elif key == "water_freq":
            user[key] = water_freq_values[user_index] if user_index < len(water_freq_values) else 2.0
        elif key in variant_assignments and user_index < len(variant_assignments[key]):
            user[key] = variant_assignments[key][user_index]
        elif key in persona:
            user[key] = persona[key]
        elif key in vocabs:
            user[key] = vocabs[key][user_index % len(vocabs[key])]

    # USDA zone: based on user's climate. Sample min within climate range, max from min to climate max
    climate = (user.get("climate") or "temperate").strip().lower()
    zone_lo, zone_hi = CLIMATE_TO_ZONE_RANGE.get(climate, (4, 8))
    min_z = rng.randint(zone_lo, zone_hi)
    max_z = rng.randint(min_z, zone_hi)
    user["usda_zone_min"] = min_z
    user["usda_zone_max"] = max_z

    return user


def main(seed: int = 42) -> int:
    rng = random.Random(seed)
    vocabs = load_vocabs()
    personas = load_personas()

    if not vocabs:
        print("No valid vocabs found. Ensure vocabs.json has list values for:", VOCAB_KEYS)
        return 1

    users: list[dict] = []
    user_id = 1

    for persona_name, count in PERSONA_POPULATION.items():
        if persona_name not in personas:
            print(f"Warning: persona '{persona_name}' not in personas.json, skipping")
            continue

        persona = personas[persona_name]
        persona_seed = seed + hash(persona_name) % 10000

        # Build balanced variant assignments (exclude water_freq - handled separately)
        variant_assignments = {}
        for key in VOCAB_KEYS:
            if key == "water_freq":
                continue
            if key not in vocabs:
                continue
            vocab_values = vocabs[key]
            variant_assignments[key] = balanced_sample(vocab_values, count, seed=persona_seed)

        # water_freq: sample category 1, 2, 7 balanced, then decimal in range
        water_freq_categories = balanced_sample([1, 2, 7], count, seed=persona_seed + 1)
        water_freq_values = [
            sample_water_freq_decimal(cat, random.Random(persona_seed + 2 + i))
            for i, cat in enumerate(water_freq_categories)
        ]

        for i in range(count):
            user = generate_user(
                user_id, persona, vocabs, persona_name,
                variant_assignments, water_freq_values, i, rng
            )
            users.append(user)
            user_id += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(users, f, indent=2)

    total = len(users)
    print(f"Generated {total} users across {len(PERSONA_POPULATION)} personas")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
