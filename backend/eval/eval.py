"""
Eval script: baseline (cosine similarity on profile_embedding) vs rec pipeline (extract + two-tower + rerank).

Ground truth: Oracle-based via compute_oracle_score (light, water, humidity, temp, size).
Positives: score >= 0.75; negatives: score < 0.3 or hard-filtered.

Baseline: Cosine similarity between user profile_embedding and plant profile_embedding -> rank.
Rec pipeline: Extract profile from query (LLM) -> merge with user -> recommend_for_profile.

Metrics: Recall@K, NDCG@K, Hit Rate@K (K=5, 10, 20).
Latency: Mean and p95 for both pipelines.

Run from project root: python -m backend.eval.eval
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure backend is on path and .env is loaded
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Backend imports (after path setup)
from llm import chat_simple
from recommend.feature_loader import normalize_profile
from recommend.recommend import recommend_for_profile
from search.search import EXTRACT_SCHEMA, _merge_profile, _parse_extracted

# Paths
USERS_PATH = ROOT / "resources" / "synthetic_users.json"
PLANTS_PATH = ROOT / "resources" / "plant_profiles.json"
if not PLANTS_PATH.exists():
    PLANTS_PATH = ROOT / "resources" / "data_creating" / "plant_profiles.json"

# Oracle (from two_tower_training)
ORACLE_POS_THRESHOLD = 0.75
ORACLE_NEG_THRESHOLD = 0.3

# Eval K values
EVAL_KS = [5, 10, 20]

# Voyage (match plant desc_embeddings: voyage-4-lite)
VOYAGE_EMBED_MODEL = "voyage-4-lite"


def _syn_user_to_existing(syn_user: dict) -> dict:
    """Convert synthetic user to format expected by _merge_profile."""
    return {
        "username": f"eval_user_{syn_user['user_id']}",
        "environment": syn_user.get("environment", {}),
        "preferences": syn_user.get("preferences", {}),
        "constraints": syn_user.get("constraints", {}),
        "climate": syn_user.get("climate"),
    }


def _query_from_user(syn_user: dict) -> str:
    """Generate a query from synthetic user profile (for eval)."""
    env = syn_user.get("environment", {}) or {}
    pref = syn_user.get("preferences", {}) or {}
    care_pref = pref.get("care_preferences", {}) or {}
    constraints = syn_user.get("constraints", {}) or {}
    parts = []
    if env.get("light_level"):
        parts.append(f"I have {env['light_level'].replace('_', ' ')} light")
    if env.get("humidity_level"):
        parts.append(f"{env['humidity_level']} humidity")
    if constraints.get("preferred_size"):
        parts.append(f"I want a {constraints['preferred_size']} plant")
    if pref.get("care_level"):
        parts.append(f"prefer {pref['care_level']} care")
    if care_pref.get("watering_freq"):
        parts.append(f"{care_pref['watering_freq']} watering")
    return ". ".join(parts) if parts else "I want an easy low-light plant"


def extract_profile_from_query(query: str, existing: dict) -> dict:
    """Extract profile from query using LLM, merge with existing. No auth."""
    system = (
        "You extract plant preferences from user descriptions. "
        "The text may describe: (1) environment/care needs (light, humidity, etc.), "
        "or (2) what kind of plant they want - physical appearance and symbolism. "
        "Extract both structured fields and free-text physical_desc/symbolism when present. "
        "Infer only what is clearly stated or strongly implied. "
        + EXTRACT_SCHEMA
    )
    user_msg = (
        f"User's current profile (for context): {json.dumps(existing, default=str)}\n\n"
        f"User's new text:\n{query}\n\n"
        "Extract any plant preference fields from the new text. Return JSON."
    )
    try:
        out = chat_simple(user_message=user_msg, system=system)
        extracted = _parse_extracted(out or "")
    except Exception:
        extracted = {}
    merged = _merge_profile(existing, extracted)
    return normalize_profile(merged)


# --- Oracle (same logic as two_tower_training) ---
_BUCKET_ORDER = ["low", "medium", "high"]
_MISSING_NEUTRAL = 0.5


def _bucket_distance(a: str | None, b: str | None) -> int:
    if a not in _BUCKET_ORDER or b not in _BUCKET_ORDER:
        return 2
    return abs(_BUCKET_ORDER.index(a) - _BUCKET_ORDER.index(b))


def _light_match(user_light: str | None, plant_ideal: str | None, plant_tolerated: str | None) -> float:
    if not user_light:
        return _MISSING_NEUTRAL
    if not plant_ideal and not plant_tolerated:
        return _MISSING_NEUTRAL
    if plant_ideal and user_light == plant_ideal:
        return 1.0
    if plant_tolerated and user_light == plant_tolerated:
        return 0.6
    return 0.0


def _water_match(user_watering: str | None, plant_water: str | None) -> float:
    if not user_watering or not plant_water:
        return _MISSING_NEUTRAL
    d = _bucket_distance(user_watering, plant_water)
    return 1.0 if d == 0 else (0.6 if d == 1 else 0.0)


def _humidity_match(user_humidity: str | None, plant_humidity: str | None) -> float:
    if not user_humidity or not plant_humidity:
        return _MISSING_NEUTRAL
    d = _bucket_distance(user_humidity, plant_humidity)
    return 1.0 if d == 0 else (0.6 if d == 1 else 0.0)


def _temp_overlap(
    user_min: float | None, user_max: float | None,
    plant_min: float | None, plant_max: float | None,
) -> float:
    if user_min is None or user_max is None or plant_min is None or plant_max is None:
        return _MISSING_NEUTRAL
    overlap_start = max(user_min, plant_min)
    overlap_end = min(user_max, plant_max)
    overlap_len = max(0.0, overlap_end - overlap_start)
    if overlap_len <= 0:
        return 0.0
    user_range = user_max - user_min
    return min(1.0, overlap_len / user_range) if user_range > 0 else 0.0


def _size_match(user_preferred: str | None, plant_size: str | None) -> float:
    if not user_preferred or not plant_size:
        return _MISSING_NEUTRAL
    return 1.0 if user_preferred == plant_size else 0.5


def _hard_filter(user: dict, plant: dict) -> bool:
    hard_no = user.get("constraints", {}).get("hard_no") or []
    if "frequent_watering" in hard_no:
        plant_water = (plant.get("Care", {}) or {}).get("water_req_bucket")
        if plant_water == "high":
            return True
    return False


def compute_oracle_score(user: dict, plant: dict) -> float:
    """Oracle compatibility 0..1 (same logic as two_tower_training)."""
    if _hard_filter(user, plant):
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
    lm = _light_match(env.get("light_level"), ideal.get("sunlight_bucket"), tolerated.get("sunlight_bucket"))
    wm = _water_match(care_pref.get("watering_freq"), plant_care.get("water_req_bucket"))
    hm = _humidity_match(env.get("humidity_level"), plant_care.get("humidity_req_bucket"))
    temp_pref = env.get("temperature_pref", {}) or {}
    tm = _temp_overlap(
        temp_pref.get("min_f"), temp_pref.get("max_f"),
        plant_care.get("temp_req", {}).get("min_temp"),
        plant_care.get("temp_req", {}).get("max_temp"),
    )
    sm = _size_match(constraints.get("preferred_size"), plant_info.get("size"))
    return (lm + wm + hm + tm + sm) / 5.0


def load_plants_from_json() -> list[dict]:
    """Load plants from plant_profiles.json (has desc_embeddings)."""
    with open(PLANTS_PATH) as f:
        return json.load(f)


def load_plants_from_mongo() -> list[dict] | None:
    """Load plants from MongoDB. Returns None if Mongo unavailable."""
    try:
        from database import get_plant_collection

        coll = get_plant_collection()
        cursor = coll.find(
            {"plant_tower_embedding": {"$exists": True}},
            {"plant_id": 1, "Info": 1, "Care": 1, "desc_embeddings": 1},
        )
        return list(cursor)
    except Exception:
        return None


def voyage_embed_query(query: str) -> list[float]:
    """Embed query with Voyage (for baseline)."""
    import voyageai

    vo = voyageai.Client()
    result = vo.embed([query], model=VOYAGE_EMBED_MODEL, input_type="query")
    return result.embeddings[0]


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dp = dot_product(a, b)
    na = math.sqrt(dot_product(a, a))
    nb = math.sqrt(dot_product(b, b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dp / (na * nb)


def baseline_rank(user: dict, plants: list[dict], k: int = 20) -> list[int]:
    """Baseline: cosine similarity between user profile_embedding and plant profile_embedding -> rank."""
    user_emb = user.get("profile_embedding") or []
    if not user_emb:
        return []
    scored = []
    for p in plants:
        plant_emb = p.get("profile_embedding") or []
        if not plant_emb:
            continue
        score = cosine_similarity(user_emb, plant_emb)
        scored.append((p["plant_id"], score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:k]]


def rec_pipeline_rank(profile: dict, username: str, k: int = 20, use_rerank: bool = True) -> list[int]:
    """Rec pipeline: recommend_for_profile -> plant_ids in order."""
    out = recommend_for_profile(profile, username, k=k, use_rerank=use_rerank)
    return [p["plant_id"] for p in out.get("plants", [])]


def recall_at_k(relevant: set[int], predicted: list[int], k: int) -> float:
    if not relevant:
        return 1.0
    top_k = set(predicted[:k])
    hits = len(relevant & top_k)
    return hits / len(relevant)


def ndcg_at_k(relevant: set[int], predicted: list[int], k: int) -> float:
    if not relevant:
        return 1.0
    top_k = predicted[:k]
    dcg = 0.0
    for i, pid in enumerate(top_k):
        if pid in relevant:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(relevant: set[int], predicted: list[int], k: int) -> float:
    if not relevant:
        return 1.0
    top_k = set(predicted[:k])
    return 1.0 if (relevant & top_k) else 0.0


def run_eval(
    users: list[dict],
    plants: list[dict],
    use_mongo: bool,
    max_users: int = 50,
    use_rerank: bool = False,
) -> dict:
    """Run full eval. Returns metrics and latency."""
    users = users[:max_users]
    plant_by_id = {p["plant_id"]: p for p in plants}
    plant_ids = list(plant_by_id.keys())

    baseline_recalls = {k: [] for k in EVAL_KS}
    baseline_ndcgs = {k: [] for k in EVAL_KS}
    baseline_hits = {k: [] for k in EVAL_KS}
    rec_recalls = {k: [] for k in EVAL_KS}
    rec_ndcgs = {k: [] for k in EVAL_KS}
    rec_hits = {k: [] for k in EVAL_KS}
    baseline_latencies = []
    rec_latencies = []

    for syn_user in users:
        existing = _syn_user_to_existing(syn_user)
        query = _query_from_user(syn_user)
        username = existing["username"]

        # Ground truth: oracle positives
        relevant = set()
        for pid in plant_ids:
            plant = plant_by_id.get(pid)
            if not plant:
                continue
            score = compute_oracle_score(syn_user, plant)
            if score >= ORACLE_POS_THRESHOLD:
                relevant.add(pid)

        if not relevant:
            continue  # skip users with no positives

        # Baseline: cosine similarity between user profile_embedding and plant profile_embedding
        t0 = time.perf_counter()
        baseline_pred = baseline_rank(syn_user, plants, k=max(EVAL_KS))
        baseline_latencies.append((time.perf_counter() - t0) * 1000)

        for k in EVAL_KS:
            baseline_recalls[k].append(recall_at_k(relevant, baseline_pred, k))
            baseline_ndcgs[k].append(ndcg_at_k(relevant, baseline_pred, k))
            baseline_hits[k].append(hit_rate_at_k(relevant, baseline_pred, k))

        # Rec pipeline (extract + recommend)
        if use_mongo:
            t0 = time.perf_counter()
            merged = extract_profile_from_query(query, existing)
            pred = rec_pipeline_rank(merged, username, k=max(EVAL_KS), use_rerank=use_rerank)
            rec_latencies.append((time.perf_counter() - t0) * 1000)

            for k in EVAL_KS:
                rec_recalls[k].append(recall_at_k(relevant, pred, k))
                rec_ndcgs[k].append(ndcg_at_k(relevant, pred, k))
                rec_hits[k].append(hit_rate_at_k(relevant, pred, k))

            # Cohere Trial: 10 calls/min. Throttle to avoid 429.
            if use_rerank:
                delay = float(os.environ.get("EVAL_RERANK_DELAY_SEC", "6"))
                if delay > 0:
                    time.sleep(delay)

    def mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    def p95(lst: list[float]) -> float:
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        idx = int(len(sorted_lst) * 0.95) - 1
        return sorted_lst[max(0, idx)]

    results = {
        "n_users": len(users),
        "n_plants": len(plants),
        "baseline": {
            "recall": {k: mean(baseline_recalls[k]) for k in EVAL_KS},
            "ndcg": {k: mean(baseline_ndcgs[k]) for k in EVAL_KS},
            "hit_rate": {k: mean(baseline_hits[k]) for k in EVAL_KS},
            "latency_ms_mean": mean(baseline_latencies),
            "latency_ms_p95": p95(baseline_latencies),
        },
    }
    if use_mongo and rec_latencies:
        results["rec_pipeline"] = {
            "recall": {k: mean(rec_recalls[k]) for k in EVAL_KS},
            "ndcg": {k: mean(rec_ndcgs[k]) for k in EVAL_KS},
            "hit_rate": {k: mean(rec_hits[k]) for k in EVAL_KS},
            "latency_ms_mean": mean(rec_latencies),
            "latency_ms_p95": p95(rec_latencies),
        }
    else:
        results["rec_pipeline"] = None

    return results


def main():
    print("Loading synthetic users...")
    with open(USERS_PATH) as f:
        users = json.load(f)
    print(f"  {len(users)} users")

    print("Loading plants...")
    plants = load_plants_from_json()
    plants_with_emb = [p for p in plants if p.get("profile_embedding")]
    print(f"  {len(plants)} plants, {len(plants_with_emb)} with profile_embedding")

    if not plants_with_emb:
        print("ERROR: No plants with profile_embedding. Run: python -m backend.eval.plant_embed")
        sys.exit(1)

    users_with_emb = [u for u in users if u.get("profile_embedding")]
    if not users_with_emb:
        print("ERROR: No users with profile_embedding. Run: python -m backend.eval.embed")
        sys.exit(1)
    users = users_with_emb

    use_mongo = load_plants_from_mongo() is not None
    if not use_mongo:
        print("WARNING: MongoDB unavailable. Rec pipeline will be skipped.")
        print("  Set MONGO_URI in .env to enable rec pipeline.")

    use_rerank = os.environ.get("EVAL_USE_RERANK", "true").lower() in ("true", "1", "yes")
    print("\nRunning eval (max 50 users)...")
    results = run_eval(users, plants_with_emb, use_mongo=use_mongo, max_users=50, use_rerank=use_rerank)

    print("\n--- Results ---")
    print(json.dumps(results, indent=2))

    print("\n--- Summary ---")
    print("Baseline:")
    for k in EVAL_KS:
        print(f"  Recall@{k}={results['baseline']['recall'][k]:.4f}  NDCG@{k}={results['baseline']['ndcg'][k]:.4f}  Hit@{k}={results['baseline']['hit_rate'][k]:.4f}")
    print(f"  Latency: mean={results['baseline']['latency_ms_mean']:.0f}ms  p95={results['baseline']['latency_ms_p95']:.0f}ms")

    if results.get("rec_pipeline"):
        print("Rec pipeline:")
        for k in EVAL_KS:
            print(f"  Recall@{k}={results['rec_pipeline']['recall'][k]:.4f}  NDCG@{k}={results['rec_pipeline']['ndcg'][k]:.4f}  Hit@{k}={results['rec_pipeline']['hit_rate'][k]:.4f}")
        print(f"  Latency: mean={results['rec_pipeline']['latency_ms_mean']:.0f}ms  p95={results['rec_pipeline']['latency_ms_p95']:.0f}ms")


if __name__ == "__main__":
    main()
