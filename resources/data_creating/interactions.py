"""
Generate user-plant interactions for two-tower training.
Per-user: 20 positives (top-N by score), 80 negatives (bottom 40 + 40 hard negatives).
Missing plant data treated as neutral (0.5) to avoid unfair penalization.
"""
import json
import random
from collections import Counter
from pathlib import Path

USERS_PATH = Path(__file__).parent / "synthetic_users.json"
PLANTS_PATH = Path(__file__).parent / "plant_profiles.json"
OUTPUT_PATH = Path(__file__).parent / "interactions.json"

N_POSITIVES_PER_USER = 20
N_NEGATIVES_BOTTOM = 40
N_NEGATIVES_HARD = 40
NEGATIVE_BOUNDARY_LOW = 0.35  # hard negatives: scores in (boundary_low, boundary_high)
NEGATIVE_BOUNDARY_HIGH = 0.74  # just below positive threshold

MISSING_NEUTRAL = 0.5  # treat missing bucket/temp as neutral

BUCKET_ORDER = ["low", "medium", "high"]


def bucket_distance(a: str, b: str) -> int:
    """Return 0=same, 1=adjacent, 2=opposite for low/medium/high."""
    if a not in BUCKET_ORDER or b not in BUCKET_ORDER:
        return 2
    i, j = BUCKET_ORDER.index(a), BUCKET_ORDER.index(b)
    return abs(i - j)


def light_match(user_light: str | None, plant_ideal: str | None, plant_tolerated: str | None) -> float:
    """1.0 if ideal match, 0.6 if tolerated match, 0.0 otherwise. Missing -> neutral."""
    if not user_light:
        return MISSING_NEUTRAL
    if not plant_ideal and not plant_tolerated:
        return MISSING_NEUTRAL
    if plant_ideal and user_light == plant_ideal:
        return 1.0
    if plant_tolerated and user_light == plant_tolerated:
        return 0.6
    return 0.0


def water_match(user_watering: str | None, plant_water: str | None) -> float:
    """1.0 exact, 0.6 off by 1, 0.0 low↔high. Missing -> neutral."""
    if not user_watering or not plant_water:
        return MISSING_NEUTRAL
    d = bucket_distance(user_watering, plant_water)
    if d == 0:
        return 1.0
    if d == 1:
        return 0.6
    return 0.0


def humidity_match(user_humidity: str | None, plant_humidity: str | None) -> float:
    """1.0 exact, 0.6 off by 1, 0.0 opposite. Missing -> neutral."""
    if not user_humidity:
        return MISSING_NEUTRAL
    if not plant_humidity:
        return MISSING_NEUTRAL
    d = bucket_distance(user_humidity, plant_humidity)
    if d == 0:
        return 1.0
    if d == 1:
        return 0.6
    return 0.0


def temp_overlap(
    user_min: float | None, user_max: float | None,
    plant_min: float | None, plant_max: float | None,
) -> float:
    """Overlap ratio if data present. Missing -> neutral."""
    if user_min is None or user_max is None:
        return MISSING_NEUTRAL
    if plant_min is None or plant_max is None:
        return MISSING_NEUTRAL
    overlap_start = max(user_min, plant_min)
    overlap_end = min(user_max, plant_max)
    overlap_len = max(0.0, overlap_end - overlap_start)
    if overlap_len <= 0:
        return 0.0
    user_range = user_max - user_min
    if user_range <= 0:
        return 0.0
    return min(1.0, overlap_len / user_range)


def size_match(user_preferred: str | None, plant_size: str | None) -> float:
    """1.0 if match, 0.5 otherwise. Missing -> neutral."""
    if not user_preferred or not plant_size:
        return MISSING_NEUTRAL
    return 1.0 if user_preferred == plant_size else 0.5


def hard_filter(user: dict, plant: dict) -> bool:
    """True if pair should be forced negative (score=0)."""
    hard_no = user.get("constraints", {}).get("hard_no") or []
    if "frequent_watering" in hard_no:
        plant_water = plant.get("Care", {}).get("water_req_bucket")
        if plant_water == "high":
            return True
    return False


def compute_score(user: dict, plant: dict) -> float:
    """Oracle compatibility score 0..1. Missing data = neutral (0.5)."""
    if hard_filter(user, plant):
        return 0.0

    env = user.get("environment", {}) or {}
    pref = user.get("preferences", {}) or {}
    care_pref = pref.get("care_preferences", {}) or {}
    constraints = user.get("constraints", {}) or {}
    plant_care = plant.get("Care", {}) or {}
    plant_info = plant.get("Info", {}) or {}
    light_req = plant_care.get("light_req", {}) or {}
    ideal = light_req.get("ideal_light", {}) or {}
    tolerated = light_req.get("tolerated_light", {}) or {}

    lm = light_match(
        env.get("light_level"),
        ideal.get("sunlight_bucket"),
        tolerated.get("sunlight_bucket"),
    )
    wm = water_match(
        care_pref.get("watering_freq"),
        plant_care.get("water_req_bucket"),
    )
    hm = humidity_match(
        env.get("humidity_level"),
        plant_care.get("humidity_req_bucket"),
    )
    temp_pref = env.get("temperature_pref", {}) or {}
    tm = temp_overlap(
        temp_pref.get("min_f"),
        temp_pref.get("max_f"),
        plant_care.get("temp_req", {}).get("min_temp"),
        plant_care.get("temp_req", {}).get("max_temp"),
    )
    sm = size_match(
        constraints.get("preferred_size"),
        plant_info.get("size"),
    )

    return (lm + wm + hm + tm + sm) / 5.0


def sample_interactions(user: dict, plants: list[dict]) -> list[dict]:
    """
    Per-user: top 20 positives, 80 negatives (bottom 40 + 40 hard negatives near boundary).
    Ensures every user gets exactly 20 positives and 80 negatives when possible.
    """
    scores = []
    for plant in plants:
        score = compute_score(user, plant)
        scores.append((plant["plant_id"], score))

    # Sort by score descending
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    # Positives: top N by score (exclude hard-filtered with score 0)
    candidates_pos = [(pid, s) for pid, s in scores_sorted if s > 0]
    pos_sample = candidates_pos[:N_POSITIVES_PER_USER]

    # Negatives: bottom 40 (lowest scores) + 40 hard negatives (scores in boundary band)
    bottom = [(pid, s) for pid, s in scores_sorted if s <= NEGATIVE_BOUNDARY_LOW]
    hard_negs = [(pid, s) for pid, s in scores_sorted
                 if NEGATIVE_BOUNDARY_LOW < s < NEGATIVE_BOUNDARY_HIGH]

    neg_bottom = random.sample(bottom, min(N_NEGATIVES_BOTTOM, len(bottom))) if bottom else []
    neg_hard = random.sample(hard_negs, min(N_NEGATIVES_HARD, len(hard_negs))) if hard_negs else []
    # If we need more: fill from remaining negatives
    combined = neg_bottom + neg_hard
    remaining = [(pid, s) for pid, s in scores_sorted
                 if s < NEGATIVE_BOUNDARY_HIGH and (pid, s) not in combined]
    need = N_NEGATIVES_BOTTOM + N_NEGATIVES_HARD - len(combined)
    if need > 0 and remaining:
        extra = random.sample(remaining, min(need, len(remaining)))
        combined.extend(extra)
    neg_sample = combined[: N_NEGATIVES_BOTTOM + N_NEGATIVES_HARD]

    interactions = []
    for pid, score in pos_sample:
        interactions.append({
            "user_id": user["user_id"],
            "plant_id": pid,
            "label": 1,
            "score": round(score, 4),
        })
    for pid, score in neg_sample:
        interactions.append({
            "user_id": user["user_id"],
            "plant_id": pid,
            "label": 0,
            "score": round(score, 4),
        })
    return interactions


def sanity_report(interactions: list[dict]) -> None:
    """Print data sanity report before training."""
    n_total = len(interactions)
    n_pos = sum(1 for i in interactions if i["label"] == 1)
    n_neg = sum(1 for i in interactions if i["label"] == 0)
    ratio = n_pos / n_neg if n_neg else 0

    print("\n" + "=" * 60)
    print("DATA SANITY REPORT")
    print("=" * 60)
    print(f"Total interactions: {n_total}")
    print(f"Positives: {n_pos}, Negatives: {n_neg}")
    print(f"Pos/Neg ratio: {ratio:.3f}")

    # Per-user distribution
    pos_per_user = Counter()
    neg_per_user = Counter()
    for i in interactions:
        uid = i["user_id"]
        if i["label"] == 1:
            pos_per_user[uid] += 1
        else:
            neg_per_user[uid] += 1

    users_with_pos = set(pos_per_user.keys())
    pos_counts = list(pos_per_user.values())
    neg_counts = list(neg_per_user.values())

    if pos_counts:
        print(f"\nPer-user positives: min={min(pos_counts)}, max={max(pos_counts)}, "
              f"mean={sum(pos_counts)/len(pos_counts):.1f}")
    if neg_counts:
        print(f"Per-user negatives: min={min(neg_counts)}, max={max(neg_counts)}, "
              f"mean={sum(neg_counts)/len(neg_counts):.1f}")

    users_few_pos = sum(1 for c in pos_counts if c < 5)
    pct_few = 100 * users_few_pos / len(pos_counts) if pos_counts else 0
    print(f"\nUsers with <5 positives: {users_few_pos} ({pct_few:.1f}%)")

    # Score histograms
    pos_scores = [i["score"] for i in interactions if i["label"] == 1]
    neg_scores = [i["score"] for i in interactions if i["label"] == 0]

    def hist(scores: list[float], bins: list[float], name: str) -> None:
        counts = [0] * (len(bins) - 1)
        for s in scores:
            for i in range(len(bins) - 1):
                if i < len(bins) - 2:
                    if bins[i] <= s < bins[i + 1]:
                        counts[i] += 1
                        break
                else:
                    if bins[i] <= s <= bins[i + 1]:
                        counts[i] += 1
                        break
        print(f"\n{name} score histogram:")
        for i in range(len(bins) - 1):
            sep = "]" if i == len(bins) - 2 else ")"
            print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f}{sep}: {counts[i]}")

    bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.75, 1.0]
    if pos_scores:
        hist(pos_scores, bins, "Positive")
    if neg_scores:
        hist(neg_scores, bins, "Negative")
    print("=" * 60 + "\n")


def main(seed: int | None = 42, report: bool = True):
    """Generate interactions: per-user 20 positives, 80 negatives."""
    if seed is not None:
        random.seed(seed)

    with open(USERS_PATH) as f:
        users = json.load(f)
    with open(PLANTS_PATH) as f:
        plants = json.load(f)

    interactions = []
    for user in users:
        interactions.extend(sample_interactions(user, plants))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(interactions, f, indent=2)

    if report:
        sanity_report(interactions)

    print(f"Generated {len(interactions)} interactions -> {OUTPUT_PATH}")
    return interactions


if __name__ == "__main__":
    import sys
    report = "--no-report" not in sys.argv
    main(report=report)
