"""
Generate synthetic user profiles with realistic distributions for two-tower training.
Uses weighted sampling (not uniform) to match real-world preferences.
"""
import json
import random
from collections import Counter
from pathlib import Path
INTERACTIONS_PATH = Path(__file__).parent / "synthetic_interactions.json"

PLANT_PROFILES_PATH = Path(__file__).parent / "plant_profiles.json"
OUTPUT_PATH = Path(__file__).parent / "synthetic_users.json"

# Per-user interaction counts
N_POSITIVES_PER_USER = 20
N_NEGATIVES_PER_USER = 80
SCORE_POSITIVE_THRESHOLD = 0.75
SCORE_NEGATIVE_THRESHOLD = 0.35

# Weighted distributions (realistic, not uniform)
LIGHT_LEVEL_WEIGHTS = {
    "indirect": 0.40,
    "bright_indirect": 0.30,
    "diffused": 0.15,
    "bright_light": 0.10,
    "direct": 0.05,
}

HUMIDITY_LEVEL_WEIGHTS = {
    "medium": 0.55,
    "low": 0.30,
    "high": 0.15,
}

WATERING_FREQ_WEIGHTS = {
    "medium": 0.45,
    "low": 0.35,
    "high": 0.20,
}

# care_level: most people want easy/medium
CARE_LEVEL_WEIGHTS = {
    "easy": 0.45,
    "medium": 0.40,
    "hard": 0.15,
}

# preferred_size: small/medium more common for indoor
PREFERRED_SIZE_WEIGHTS = {
    "small": 0.45,
    "medium": 0.40,
    "large": 0.15,
}

# care_freq: correlates with care_level
CARE_FREQ_WEIGHTS = {
    "low": 0.40,
    "medium": 0.45,
    "high": 0.15,
}

# Temperature ranges by climate (Fahrenheit)
CLIMATE_TEMP_RANGES = {
    "Arid Tropical": (65, 95),
    "Subtropical": (55, 85),
    "Subtropical arid": (60, 90),
    "Tropical": (65, 90),
    "Tropical humid": (68, 88),
}


def load_climate_weights_from_catalog() -> dict[str, float]:
    """Derive climate weights from plant catalog distribution."""
    with open(PLANT_PROFILES_PATH) as f:
        plants = json.load(f)
    counts = Counter(p.get("Care", {}).get("climate") for p in plants if p.get("Care", {}).get("climate"))
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def weighted_choice(weights: dict[str, float]) -> str:
    """Sample one key according to weights."""
    keys = list(weights.keys())
    probs = [weights[k] for k in keys]
    return random.choices(keys, weights=probs, k=1)[0]


def generate_user(user_id: int, climate_weights: dict[str, float]) -> dict:
    """Generate one synthetic user with realistic distributions."""
    climate = weighted_choice(climate_weights)
    temp_min, temp_max = CLIMATE_TEMP_RANGES.get(climate, (65, 80))
    # Add some variance to temp range (user's actual room may vary)
    span = temp_max - temp_min
    min_f = round(temp_min + random.uniform(0, span * 0.3), 1)
    max_f = round(temp_max - random.uniform(0, span * 0.3), 1)
    if min_f >= max_f:
        max_f = min_f + 5

    return {
        "user_id": user_id,
        "climate": climate,
        "environment": {
            "light_level": weighted_choice(LIGHT_LEVEL_WEIGHTS),
            "humidity_level": weighted_choice(HUMIDITY_LEVEL_WEIGHTS),
            "temperature_pref": {
                "min_f": min_f,
                "max_f": max_f,
            },
        },
        "constraints": {
            "preferred_size": weighted_choice(PREFERRED_SIZE_WEIGHTS),
        },
        "preferences": {
            "care_level": weighted_choice(CARE_LEVEL_WEIGHTS),
            "care_preferences": {
                "watering_freq": weighted_choice(WATERING_FREQ_WEIGHTS),
                "care_freq": weighted_choice(CARE_FREQ_WEIGHTS),
            },
        },
    }


def main(n_users: int = 1000, seed: int | None = 42):
    """Generate n_users synthetic profiles."""
    if seed is not None:
        random.seed(seed)

    climate_weights = load_climate_weights_from_catalog()
    users = [generate_user(i, climate_weights) for i in range(n_users)]

    with open(OUTPUT_PATH, "w") as f:
        json.dump(users, f, indent=2)

    # Print distribution summary
    light = Counter(u["environment"]["light_level"] for u in users)
    humidity = Counter(u["environment"]["humidity_level"] for u in users)
    climate = Counter(u["climate"] for u in users)
    print(f"Generated {n_users} users -> {OUTPUT_PATH}")
    print("light_level:", dict(light))
    print("humidity_level:", dict(humidity))
    print("climate:", dict(climate))
    return users


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(n_users=n)
