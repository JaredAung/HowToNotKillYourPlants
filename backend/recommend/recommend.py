"""
Plant recommendation API — scores catalog plants for a user and exposes REST endpoints.

**Scope & features**

- 
**Entry points**

- ``recommend_for_profile()`` — core pipeline (used by this router, chat, search, eval).
- ``GET /recommend/``, ``GET /recommend/cache``, ``GET /recommend/explanation`` — authenticated
  HTTP surface (JWT).

**Layout in this file** — vector helpers → ``recommend_for_profile`` → Cohere helpers → Gemini
formatting → FastAPI routes.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the repository root (.env).
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from fastapi import APIRouter, Depends, HTTPException

import logging

from auth.jwt import get_current_username # get the logged in user's username
from database import get_plant_collection, get_user_collection # get the MongoDB collections
from llm import gemini_generate # generate a natural-language explanation for the plants
from recommend.cache import get_cached, inspect_cache, set_cached # cache the recommendation results
from recommend.feature_loader import compute_user_embedding, score_plants # compute the user's embedding and score the plants using the two-tower model

router = APIRouter(prefix="/recommend", tags=["recommend"])

#CONSTRAINTS AND CONFIGURATIONS
DEFAULT_TOP_K = 20 # the default number of plants to return from the two-tower model
RERANK_MODEL = "rerank-v3.5" # the Cohere rerank model to use

VECTOR_INDEX = os.getenv("VECTOR_SEARCH_INDEX", "vector_index") # MongoDB vector search index

# =============================================================================
# Vector retrieval (MongoDB Atlas $vectorSearch)
# =============================================================================


def _vector_search_plants(plant_coll, user_emb: list[float], k: int) -> list[dict]:
    """Find the ``k`` most similar plants to the user using Atlas vector search.

    The plant embeddings are generated using the two-tower model, L2-normalized and stored in MongoDB.
    The user's embedding vector is compared against each plant's ``plant_tower_embedding``
    using dot-product similarity. Requires a vector search index (see ``VECTOR_SEARCH_INDEX``).

    Args:
        plant_coll: MongoDB plants collection.
        user_emb: User embedding vector (same length as plant embeddings; L2-normalized, 1024-dimensional).
        k: Maximum number of plants to return.

    Returns:
        A list of MongoDB documents. Each document includes ``plant_id``, nested ``Info`` /
        ``Care``, ``img_url``, and a ``score`` field from the vector search stage.

    Note:
        If this query fails (missing index, Atlas error), the caller falls back to scoring
        every embedded plant in Python instead—slower, but keeps the API usable.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "plant_tower_embedding",
                "queryVector": user_emb,
                "numCandidates": min(200, max(k * 20, 150)),
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


# =============================================================================
# Main recommendation pipeline
# =============================================================================


def recommend_for_profile(
    profile: dict,
    username: str,
    k: int = DEFAULT_TOP_K,
    use_rerank: bool | None = None,
) -> dict:
    """Score and rank plants for a single user profile (the heart of the recommender).

    Execution order: 
    (1) Embed the profile with the two-tower user model     
    (2) Retrieve similar plants from MongoDB
    (3) Rerank with Cohere using a text query

    Args:
        profile: User data used for embedding and rerank text (e.g. environment, preferences).
        username: Included in the response payload for traceability.
        k: How many plants to keep in the final list.
        use_rerank: ``None`` → read default from ``USE_RERANK`` env. ``True`` → call Cohere.

    Returns:
        Dictionary with keys ``username`` and ``plants``. Each plant entry is a flat dict with
        identifiers, care summary fields, and numeric scores. If embedding fails or no plants
        qualify, ``plants`` is empty.
    """
    plant_coll = get_plant_collection()

    # Step A: vector for this profile (loads UserTower weights from feature_loader / two_tower.pt).
    try:
        user_emb = compute_user_embedding(profile)
    except Exception as e:
        logging.warning("compute_user_embedding failed, returning no plants: %s", e)
        return {"username": username, "plants": []}

    try:
        # Two-tower model retrieval
        plants = _vector_search_plants(plant_coll, user_emb, k)
    except Exception as e:
        # Fallback: dot-product in app code when Atlas vector search is unavailable.
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

    # Step B: flatten nested Mongo ``Info`` / ``Care`` into stable keys for API and rerankers.
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

    # Step C: rerank with Cohere 
    if use_rerank is None:
        use_rerank = os.getenv("USE_RERANK", "true").lower() in ("true", "1", "yes")
    if use_rerank and results:
        query = _user_profile_to_query(profile)
        results = _rerank_with_cohere(query, results, k)

    return {"username": username, "plants": results}


# =============================================================================
# Cohere rerank — turn profile + plants into text the rerank API understands
# =============================================================================


def _user_profile_to_query(user: dict) -> str:
    """Build one ranking instruction string from a user document for Cohere's rerank API.

    Splits preferences into "hard" (must match) and "soft" (nice to match) so the model can
    prioritize realistically. 
    
    Used only when reranking is enabled.

    Args:
        user: Typically a Mongo-style nested user document with ``environment``, ``preferences``,
            ``constraints``, optional ``climate``, and optional search fields like ``physical_desc``.

    Returns:
        A multi-sentence prompt assigned to the reranker's ``query`` argument.
    """
    env = user.get("environment", {}) or {}
    pref = user.get("preferences", {}) or {}
    constraints = user.get("constraints", {}) or {}
    care_pref = pref.get("care_preferences", {}) or {}
    temp_pref = env.get("temperature_pref", {}) or {}

    hard = []
    soft = []

    # Hard / soft buckets mirror how we describe constraints to the rerank model.
    if env.get("light_level"):
        hard.append(f"must tolerate light={env['light_level']}")
    if env.get("humidity_level"):
        hard.append(f"humidity should match {env['humidity_level']}")
    min_f = temp_pref.get("min_f")
    max_f = temp_pref.get("max_f")
    if min_f is not None and max_f is not None:
        hard.append(f"plant temp range must overlap with {min_f}–{max_f}°F")

    if constraints.get("preferred_size"):
        soft.append(f"prefer size={constraints['preferred_size']}")

    if care_pref.get("watering_freq"):
        hard.append(f"watering should match {care_pref['watering_freq']}")
    if care_pref.get("care_freq"):
        soft.append(f"prefer care frequency={care_pref['care_freq']}")
    if pref.get("care_level"):
        soft.append(f"prefer care level={pref['care_level']}")

    if user.get("climate"):
        soft.append(f"prefer climate={user['climate']}")

    # Optional fields from semantic search or onboarding flows.
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
    """Serialize one plant record into a single line of plain text for Cohere reranking.

    The rerank API compares each line against the user query; keep fields readable and concise.

    Args:
        p: Flattened plant dict (names, light, water, temp range, description fields).

    Returns:
        One string of plant profile for the reranker to compare against the user query.
    """
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


def _rerank_with_cohere(query: str, results: list[dict], top_n: int) -> list[dict]:
    """Reorder candidate plants using Cohere's hosted rerank model (semantic relevance to the query).

    Args:
        query: Instructions built by :func:`_user_profile_to_query`.
        results: Plants from vector search, already in flattened dict form.
        top_n: Upper bound on how many items to return.

    Returns:
        A new list sorted by the reranker. Each dict includes a ``rerank_score`` field.

    Raises:
        HTTPException: Status 503 when ``COHERE_API_KEY`` is not set (service not configured).

    Note:
        Requires network access to Cohere's API at request time.
    """
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

    # list of flattened plant profiles for the reranker to compare against the user query
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


# =============================================================================
# Gemini — natural-language "why these plants?" (not used in numeric ranking)
# =============================================================================


def _format_plant_for_llm(p: dict) -> str:
    """Format one plant as short bullet text with relevant fields for the explanation LLM prompt.

    Args:
        p: Summary dict with display name and care fields.

    Returns:
        A few lines of human-readable text suitable to paste into a Gemini user message.
    """
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
    """Generate a multi-plant write-up explaining why the recommendations are good fits for the user.

    The LLM can be Google Gemini (deployed) or Ollama (local/testing).

    Args:
        user: Complete user document from Mongo (profile + auth blocks as stored).
        top_plants: Ordered list of simplified plant dicts (typically up to five ids).

    Returns:
        Plain-text explanation from the model, or an empty string if the call fails or the
        input list is empty.
    """
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


# =============================================================================
# HTTP routes (all require a valid JWT)
# =============================================================================


@router.get("/")
def get_recommendations(
    username: str = Depends(get_current_username),
    k: int = DEFAULT_TOP_K,
    use_rerank: bool = True,
):
    """
    API endpoint to the recommendation pipeline.

    - Runs the entire pipeline (Two-Tower + Reranker)  
    - Returns ranked plant recommendations for the signed-in user.
    - Recommendations cached in Redis for 1 hour.
    
    Args: 
        username: Resolved from the JWT by FastAPI dependency injection.
        k: How many plants to return (defaults to ``DEFAULT_TOP_K``).
        use_rerank: Default to ``True`` but Set ``False`` to skip Cohere (useful in testing/local development).

    Returns:
        JSON object: ``username``, ``plants`` (list of scored dicts), and optionally ``message``
        if the catalog has no embedded plants. 

    Raises:
        HTTPException: 404 if no user matches the JWT identity.
    """
    user_coll = get_user_collection()
    user = user_coll.find_one({"auth.username": username}) or user_coll.find_one(
        {"auth.email": username.lower()}
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if Redis caching is enabled
    use_redis = os.getenv("USE_REDIS_CACHE", "false").lower() in ("true", "1", "yes")
    if use_redis: 
        # Try to get cached recommendations from Redis
        cached = get_cached(username, user, k, use_rerank) 
        if cached is not None:
            out = cached
        else:
            # If no cached recommendations, run the pipeline and cache the results
            out = recommend_for_profile(user, username, k, use_rerank=use_rerank)
            set_cached(username, user, out, k, use_rerank)
    else:
        out = recommend_for_profile(user, username, k, use_rerank=use_rerank)
    if not out["plants"]: # if no plants are found, return a message
        out["message"] = "No plants with embeddings in database."
    return out

@router.get("/explanation")
def get_explanation(
    plant_ids: str,
    username: str = Depends(get_current_username),
):
    """
    API endpoint to generate a narrative explaining why the listed plants suit this user.

    - Fetches fresh plant metadata from Mongo for the given ids
    - Generates a narrative explaining why the listed plants suit this user

    Args:
        plant_ids: Comma-separated Mongo ``plant_id`` integers (e.g. ``"3,7,12"``). Only the
            first five ids are processed.
        username: JWT identity; used to load the user document for personalization context.

    Returns:
        JSON mapping ``explanation`` to the model string, which may be empty if inputs are invalid.

    Raises:
        HTTPException: 404 when the user record is missing; 400 when ``plant_ids`` is malformed.
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
