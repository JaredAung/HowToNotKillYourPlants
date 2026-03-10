"""
Plant recommendation API. Two-tower inference + Cohere reranker.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Ensure .env is loaded (project root)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from fastapi import APIRouter, Depends, HTTPException

import logging

from auth.jwt import get_current_username
from database import get_plant_collection, get_user_collection
from garden.death import get_dead_plant_ids
from llm import gemini_generate
from recommend.cache import get_cached, inspect_cache, set_cached
from recommend.feature_loader import compute_user_embedding, score_plants

router = APIRouter(prefix="/recommend", tags=["recommend"])

DEFAULT_TOP_K = 20
VECTOR_INDEX = os.getenv("VECTOR_SEARCH_INDEX", "vector_index")
DEATH_PENALTY_LAMBDA = float(os.getenv("DEATH_PENALTY_LAMBDA", "0.5"))
USE_DEATH_PENALTY = os.getenv("USE_DEATH_PENALTY", "true").lower() in ("true", "1", "yes")


def _vector_search_plants(plant_coll, user_emb: list[float], k: int) -> list[dict]:
    """
    Use MongoDB $vectorSearch (dot product) to get top-k plants.
    Requires a vector index on plant_tower_embedding with similarity: dotProduct.
    Returns list of plant docs with score from $meta vectorSearchScore.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "plant_tower_embedding",
                "queryVector": user_emb,
                "numCandidates": min(200, max(k * 20, 150)),  # cap for small catalogs (~159 plants)
                "limit": k,
            }
        },
        {
            "$project": {
                "plant_id": 1,
                "Info": 1,
                "Care": 1,
                "img_url": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(plant_coll.aggregate(pipeline))


def recommend_for_profile(
    profile: dict,
    username: str,
    k: int = DEFAULT_TOP_K,
    use_rerank: bool | None = None,
    use_death_penalty: bool | None = None,
) -> dict:
    """
    Run recommendation pipeline for a given profile.
    Uses ONLY the profile dict passed in—does NOT fetch from MongoDB.
    Use for search extraction: pass the merged+normalized profile (existing + extracted).
    Returns {"username": str, "plants": list}. No LLM explanation.
    """
    plant_coll = get_plant_collection()
    user_emb = compute_user_embedding(profile)  # Uses profile param only, not MongoDB

    try:
        plants = _vector_search_plants(plant_coll, user_emb, k)
    except Exception as e:
        logging.warning("MongoDB $vectorSearch failed (%s), falling back to Python scoring", e)
        all_plants = list(plant_coll.find(
            {"plant_tower_embedding": {"$exists": True}},
            {"plant_id": 1, "plant_tower_embedding": 1, "Info": 1, "Care": 1, "img_url": 1}
        ))
        if not all_plants:
            return {"username": username, "plants": []}
        plant_embs = [(p["plant_id"], p["plant_tower_embedding"]) for p in all_plants]
        scored = score_plants(user_emb, plant_embs)[:k]
        plant_by_id = {p["plant_id"]: p for p in all_plants}
        plants = []
        for pid, score in scored:
            p = plant_by_id.get(pid, {})
            p["score"] = score
            plants.append(p)

    if not plants:
        return {"username": username, "plants": []}

    results = []
    for p in plants:
        info = p.get("Info", {}) or {}
        care = p.get("Care", {}) or {}
        light_req = care.get("light_req", {}) or {}
        ideal = light_req.get("ideal_light", {}) or {}
        tolerated = light_req.get("tolerated_light", {}) or {}
        temp_req = care.get("temp_req", {}) or {}
        desc = info.get("desc", {}) or {}
        results.append({
            "plant_id": p["plant_id"],
            "score": round(float(p.get("score", 0)), 4),
            "img_url": p.get("img_url"),
            "latin": info.get("latin"),
            "common_name": info.get("common_name"),
            "sunlight_type": ideal.get("sunlight_type") or tolerated.get("sunlight_type"),
            "ideal_light": ideal.get("sunlight_type") or ideal.get("sunlight_bucket"),
            "tolerated_light": tolerated.get("sunlight_type") or tolerated.get("sunlight_bucket"),
            "humidity": care.get("humidity_req_bucket") or care.get("humidity_req"),
            "care_level": care.get("care_level"),
            "water_req": care.get("water_req_bucket") or care.get("water_req"),
            "temp_min": temp_req.get("min_temp"),
            "temp_max": temp_req.get("max_temp"),
            "climate": care.get("climate"),
            "size": info.get("size"),
            "category": info.get("category"),
            "physical_desc": desc.get("physical_desc"),
            "symbolism": desc.get("symbolism"),
        })

    if use_rerank is None:
        use_rerank = os.getenv("USE_RERANK", "true").lower() in ("true", "1", "yes")
    if use_rerank and results:
        query = _user_profile_to_query(profile)  # Uses profile param only, not MongoDB
        results = _rerank_with_cohere(query, results, k)

    apply_death = use_death_penalty if use_death_penalty is not None else USE_DEATH_PENALTY
    if apply_death and results:
        results = _apply_death_penalty(plant_coll, results, username, k)

    return {"username": username, "plants": results}
RERANK_MODEL = "rerank-v3.5"


def _user_profile_to_query(user: dict) -> str:
    """Build a text query from user profile for Cohere reranking.
    Uses schema fields: environment, climate, safety, constraints, preferences.
    """
    env = user.get("environment", {}) or {}
    pref = user.get("preferences", {}) or {}
    constraints = user.get("constraints", {}) or {}
    care_pref = pref.get("care_preferences", {}) or {}
    temp_pref = env.get("temperature_pref", {}) or {}

    hard = []
    soft = []

    # Environment
    if env.get("light_level"):
        hard.append(f"must tolerate light={env['light_level']}")
    if env.get("humidity_level"):
        hard.append(f"humidity should match {env['humidity_level']}")
    min_f = temp_pref.get("min_f")
    max_f = temp_pref.get("max_f")
    if min_f is not None and max_f is not None:
        hard.append(f"plant temp range must overlap with {min_f}–{max_f}°F")

    # Constraints
    if constraints.get("preferred_size"):
        soft.append(f"prefer size={constraints['preferred_size']}")

    # Preferences
    if care_pref.get("watering_freq"):
        hard.append(f"watering should match {care_pref['watering_freq']}")
    if care_pref.get("care_freq"):
        soft.append(f"prefer care frequency={care_pref['care_freq']}")
    if pref.get("care_level"):
        soft.append(f"prefer care level={pref['care_level']}")

    # Climate
    if user.get("climate"):
        soft.append(f"prefer climate={user['climate']}")

    # Physical description and symbolism (from search extraction)
    if user.get("physical_desc"):
        soft.append(f"user wants: {user['physical_desc']}")
    if user.get("symbolism"):
        soft.append(f"user wants symbolism: {user['symbolism']}")

    hard_txt = "; ".join(hard) if hard else "no hard constraints"
    soft_txt = "; ".join(soft) if soft else "no soft preferences"
    return (
        "Task: rank plants for this user.\n"
        f"Hard constraints: {hard_txt}.\n"
        f"Soft preferences: {soft_txt}.\n"
        "Rank higher: plants meeting all hard constraints and most soft preferences. "
        "Rank lower: any plant violating hard constraints."
    )


def _plant_to_document(p: dict) -> str:
    """Build a text document from plant dict for Cohere reranking."""
    parts = []
    if p.get("common_name"):
        parts.append(p["common_name"])
    if p.get("latin"):
        parts.append(f"({p['latin']})")
    if p.get("ideal_light"):
        parts.append(f"Ideal light: {p['ideal_light']}")
    if p.get("tolerated_light"):
        parts.append(f"Tolerated light: {p['tolerated_light']}")
    if not p.get("ideal_light") and not p.get("tolerated_light") and p.get("sunlight_type"):
        parts.append(f"Sunlight: {p['sunlight_type']}")
    if p.get("humidity"):
        parts.append(f"Humidity: {p['humidity']}")
    if p.get("care_level"):
        parts.append(f"Care level: {p['care_level']}")
    if p.get("water_req"):
        parts.append(f"Water: {p['water_req']}")
    if p.get("temp_min") is not None and p.get("temp_max") is not None:
        parts.append(f"Temp: {p['temp_min']}-{p['temp_max']}°F")
    if p.get("climate"):
        parts.append(f"Climate: {p['climate']}")
    if p.get("size"):
        parts.append(f"Size: {p['size']}")
    if p.get("category"):
        parts.append(f"Category: {p['category']}")
    if p.get("physical_desc"):
        desc = str(p["physical_desc"])
        if len(desc) > 150:
            desc = desc[:150] + "..."
        parts.append(desc)
    if p.get("symbolism"):
        parts.append(f"Symbolism: {p['symbolism']}")
    return " | ".join(str(x) for x in parts)


def _apply_death_penalty(
    plant_coll,
    results: list[dict],
    username: str,
    k: int,
) -> list[dict]:
    """
    Apply death penalty: final_score = base_score - λ * similarity_penalty.
    similarity_penalty = max dot product between candidate and any dead plant embedding.
    """
    dead_ids = get_dead_plant_ids(username)
    if not dead_ids:
        return results

    candidate_ids = [r["plant_id"] for r in results]
    all_ids = list(set(candidate_ids) | set(dead_ids))
    plant_docs = list(
        plant_coll.find(
            {"plant_id": {"$in": all_ids}, "plant_tower_embedding": {"$exists": True}},
            {"plant_id": 1, "plant_tower_embedding": 1},
        )
    )
    emb_by_id = {p["plant_id"]: p["plant_tower_embedding"] for p in plant_docs}
    dead_embs = [(i, emb_by_id[i]) for i in dead_ids if i in emb_by_id]
    if not dead_embs:
        return results

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    for r in results:
        base_score = r.get("rerank_score") if r.get("rerank_score") is not None else r.get("score", 0)
        cand_emb = emb_by_id.get(r["plant_id"])
        if cand_emb is None:
            r["final_score"] = base_score
            r["score"] = base_score
            continue
        similarity_penalty = 0.0
        for _, dead_emb in dead_embs:
            sim = max(0.0, dot(cand_emb, dead_emb))
            similarity_penalty = max(similarity_penalty, sim)
        final_score = base_score - DEATH_PENALTY_LAMBDA * similarity_penalty
        r["final_score"] = round(final_score, 4)
        r["score"] = r["final_score"]

    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return results[:k]


def _rerank_with_cohere(query: str, results: list[dict], top_n: int) -> list[dict]:
    """Rerank results using Cohere. Returns reordered list with rerank_score. Raises if rerank fails."""
    if not results:
        return results
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="COHERE_API_KEY not configured. Add it to .env to enable recommendations.",
        )
    import cohere

    co = cohere.ClientV2(api_key=api_key)
    documents = [_plant_to_document(p) for p in results]
    rerank_resp = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
    )
    out = []
    for r in rerank_resp.results:
        p = dict(results[r.index])
        p["rerank_score"] = getattr(r, "relevance_score", None) or getattr(r, "score", None)
        out.append(p)
    return out


def _format_plant_for_llm(p: dict) -> str:
    """Format a plant dict for LLM context."""
    name = p.get("common_name") or p.get("latin") or f"Plant #{p.get('plant_id')}"
    parts = [f"- {name}"]
    if p.get("latin") and p.get("common_name"):
        parts[0] += f" ({p['latin']})"
    if p.get("sunlight_type"):
        parts.append(f"  Sunlight: {p['sunlight_type']}")
    if p.get("humidity"):
        parts.append(f"  Humidity: {p['humidity']}")
    if p.get("care_level"):
        parts.append(f"  Care level: {p['care_level']}")
    if p.get("water_req"):
        parts.append(f"  Water: {p['water_req']}")
    if p.get("temp_min") is not None and p.get("temp_max") is not None:
        parts.append(f"  Temp: {p['temp_min']}-{p['temp_max']}°F")
    return "\n".join(parts)


def _generate_explanation(user: dict, top_plants: list[dict]) -> str:
    """Use Google GenAI (Gemini) via LangChain to explain why these plants match the user. Returns empty string on error."""
    if not top_plants:
        return ""
    profile_text = _user_profile_to_query(user).replace(" ", ", ")
    plants_text = "\n\n".join(_format_plant_for_llm(p) for p in top_plants)
    user_name = (user.get("profile") or {}).get("name") or (user.get("auth") or {}).get("username") or "you"
    system = (
        "You are a friendly plant care expert. For each plant, write 2-4 short sentences explaining why it matches the user. "
        "Use this exact format for each plant (one per line):\n"
        "• **Plant common name (Latin name)**: Your explanation here.\n"
        "Example: • **Sago palm (Cycas revoluta)**: This plant thrives in bright indirect light...\n"
        "IMPORTANT: Keep each plant's explanation focused ONLY on that plant. Do NOT add generic advice "
        "(e.g. 'Remember, every plant is unique...', 'For this user, every plant has unique characteristics...') to each plant—add any generic closing only ONCE at the very end. "
        "Start with a brief intro if you like, then list all plants, then one closing sentence."
    )
    user_msg = (
        f"User's name: {user_name}\n\n"
        f"User preferences: {profile_text}\n\n"
        f"Top recommended plants:\n{plants_text}\n\n"
        f"Explain why each plant is a good match for {user_name}. Use the format: • **Name (Latin)**: explanation"
    )
    try:
        return gemini_generate(system=system, user_message=user_msg)
    except Exception as e:
        logging.warning("Gemini explanation failed: %s", e)
        return ""


@router.get("/")
def get_recommendations(
    username: str = Depends(get_current_username),
    k: int = DEFAULT_TOP_K,
    use_rerank: bool = True,
):
    """
    Get plant recommendations for the logged-in user.
    Fetches user profile from MongoDB, computes user embedding, scores against plant_tower_embedding.
    use_rerank: if False, skip Cohere reranking (vector search order only).
    """
    user_coll = get_user_collection()
    user = user_coll.find_one({"auth.username": username}) or user_coll.find_one(
        {"auth.email": username.lower()}
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    use_redis = os.getenv("USE_REDIS_CACHE", "false").lower() in ("true", "1", "yes")
    if use_redis:
        cached = get_cached(username, user, k, use_rerank)
        if cached is not None:
            out = cached
        else:
            out = recommend_for_profile(user, username, k, use_rerank=use_rerank)
            set_cached(username, user, out, k, use_rerank)
    else:
        out = recommend_for_profile(user, username, k, use_rerank=use_rerank)
    if not out["plants"]:
        out["message"] = "No plants with embeddings in database. Run upload.py first."
    return out


@router.get("/cache")
def get_cache_status(username: str = Depends(get_current_username)):
    """
    Inspect Redis cache for recommendations (keys, count, sample).
    Requires auth. For debugging.
    """
    return inspect_cache()


@router.get("/explanation")
def get_explanation(
    plant_ids: str,
    username: str = Depends(get_current_username),
):
    """
    Generate LLM explanation for top plants. Call after displaying recommendations.
    plant_ids: comma-separated, e.g. "0,1,2,3,4"
    """
    user_coll = get_user_collection()
    plant_coll = get_plant_collection()

    user = user_coll.find_one({"auth.username": username}) or user_coll.find_one(
        {"auth.email": username.lower()}
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        pids = [int(x.strip()) for x in plant_ids.split(",") if x.strip()][:5]
    except ValueError:
        raise HTTPException(status_code=400, detail="plant_ids must be comma-separated integers")

    if not pids:
        return {"explanation": ""}

    plants = list(plant_coll.find({"plant_id": {"$in": pids}}, {"plant_id": 1, "Info": 1, "Care": 1}))
    plant_by_id = {p["plant_id"]: p for p in plants}

    top_plants = []
    for pid in pids:
        p = plant_by_id.get(pid, {})
        if not p:
            continue
        info = p.get("Info", {}) or {}
        care = p.get("Care", {}) or {}
        light_req = care.get("light_req", {}) or {}
        ideal = light_req.get("ideal_light", {}) or {}
        tolerated = light_req.get("tolerated_light", {}) or {}
        temp_req = care.get("temp_req", {}) or {}
        top_plants.append({
            "plant_id": pid,
            "latin": info.get("latin"),
            "common_name": info.get("common_name"),
            "sunlight_type": ideal.get("sunlight_type") or tolerated.get("sunlight_type"),
            "humidity": care.get("humidity_req_bucket") or care.get("humidity_req"),
            "care_level": care.get("care_level"),
            "water_req": care.get("water_req_bucket") or care.get("water_req"),
            "temp_min": temp_req.get("min_temp"),
            "temp_max": temp_req.get("max_temp"),
        })

    explanation = _generate_explanation(user, top_plants)
    return {"explanation": explanation}
