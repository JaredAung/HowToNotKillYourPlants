"""
Tools for the chat agent. Bind these to the LLM for plant retrieval and actions.
"""
import logging

from database import get_plant_collection, get_user_collection
from plant.mongo_plant import flatten_catalog_plant_for_api

logger = logging.getLogger(__name__)


def _flatten_plant(p: dict) -> dict:
    """Flatten MongoDB plant doc to a readable profile. Handles partial projections."""
    merged = dict(p)
    merged.setdefault("score", 0)
    row = flatten_catalog_plant_for_api(merged)
    if "plant_id" in p and "plant_id" not in row:
        row["plant_id"] = p["plant_id"]
    return {k: v for k, v in row.items() if v is not None}


def retrieve_plant_profile(
    search_query: str,
    limit: int = 1,
    include_plant_id: bool = True,
    include_info: bool = True,
    include_care: bool = True,
    include_img_url: bool = True,
) -> str:
    """
    Retrieve a single plant profile by search from the plant collection.
    Searches by name, scientific name, description, slug, and select ``info`` fields.
    Use this when the user asks about a specific plant or needs plant information.

    Choose which sections to include based on what the user needs:
    - include_plant_id: plant identifier (for linking, adding to garden)
    - include_info: names, category, physical_desc, symbolism (from ``info`` + top-level text)
    - include_care: sunlight, humidity, care_level, water_req, temp, climate (from ``environment_care``)
    - include_img_url: image URL for display (from ``images``)
    """
    if not search_query or not search_query.strip():
        return "Please provide a search query (e.g. plant name, appearance, or care needs)."

    # Always load plant_id: ``flatten_catalog_plant_for_api`` expects it; strip below when omitted from output.
    projection: dict[str, int] = {"plant_id": 1}
    if include_info:
        projection["info"] = 1
        projection["name"] = 1
        projection["scientific_name"] = 1
        projection["description"] = 1
        projection["slug"] = 1
    if include_care:
        projection["environment_care"] = 1
        projection["care_level"] = 1
    if include_img_url:
        projection["images"] = 1

    if not (include_plant_id or include_info or include_care or include_img_url):
        return "At least one section (plant_id, info, environment_care, images) must be included."

    plant_coll = get_plant_collection()
    query = search_query.strip()
    regex = {"$regex": query, "$options": "i"}

    cursor = plant_coll.find(
        {
            "$or": [
                {"name": regex},
                {"scientific_name": regex},
                {"description": regex},
                {"slug": regex},
                {"info.Family": regex},
                {"info.Genus": regex},
                {"info.latin": regex},
                {"info.common_name": regex},
            ]
        },
        projection,
    ).limit(limit)

    plants = list(cursor)
    if not plants:
        return f"No plants found matching '{query}'."

    def _format_plant_block(doc: dict, idx: int) -> str:
        row = _flatten_plant(doc)
        if not include_plant_id:
            row.pop("plant_id", None)
        lines = "\n".join(f"  {k}: {v}" for k, v in row.items() if v is not None)
        return f"Plant {idx + 1}:\n{lines}"

    return "\n\n".join(_format_plant_block(p, i) for i, p in enumerate(plants))


def _flatten_user(u: dict) -> dict:
    """Two-tower / onboarding fields only (matches ``feature_loader`` flat user dict)."""
    auth = u.get("auth", {}) or {}
    env = u.get("environment", {}) or {}
    temp_pref = env.get("temperature_pref", {}) or {}
    constraints = u.get("constraints", {}) or {}
    prefs = u.get("preferences", {}) or {}
    care_prefs = prefs.get("care_preferences", {}) or {}

    result: dict = {}
    result["username"] = auth.get("username") or auth.get("email")
    if u.get("climate") is not None:
        result["climate"] = u.get("climate")
    if u.get("usda_zone_min") is not None:
        result["usda_zone_min"] = u.get("usda_zone_min")
    if u.get("usda_zone_max") is not None:
        result["usda_zone_max"] = u.get("usda_zone_max")
    if env.get("light_level") is not None:
        result["light_level"] = env.get("light_level")
    if env.get("soil_preference") is not None:
        result["soil_preference"] = env.get("soil_preference")
    if temp_pref.get("min_f") is not None:
        result["temp_min_f"] = temp_pref.get("min_f")
    if temp_pref.get("max_f") is not None:
        result["temp_max_f"] = temp_pref.get("max_f")
    if constraints.get("preferred_size") is not None:
        result["preferred_size"] = constraints.get("preferred_size")
    if prefs.get("care_level") is not None:
        result["care_level"] = prefs.get("care_level")
    if prefs.get("growth_pref") is not None:
        result["growth_pref"] = prefs.get("growth_pref")
    if care_prefs.get("watering_freq") is not None:
        result["watering_freq"] = care_prefs.get("watering_freq")
    return result


def format_tower_profile_for_llm(profile: dict | None) -> str:
    """
    Format the logged-in user's profile for LLM system/human prompts.

    Expects the same nested shape as ``GET /profile`` / :func:`profile.profile.tower_profile_response`
    (and :class:`schemas.profile.ProfileUpdate` fields): ``profile.name``, ``climate``, USDA zones,
    ``environment`` (light_level, soil_preference, temperature_pref), ``constraints.preferred_size``,
    ``preferences`` (care_level, growth_pref, care_preferences.watering_freq).
    """
    if not profile:
        return "(No user profile)"
    parts: list[str] = []
    un = profile.get("username")
    if un:
        parts.append(f"  username: {un}")
    p = profile.get("profile") or {}
    if p.get("name"):
        parts.append(f"  name: {p['name']}")
    if profile.get("climate") is not None:
        parts.append(f"  climate: {profile.get('climate')}")
    zmin, zmax = profile.get("usda_zone_min"), profile.get("usda_zone_max")
    if zmin is not None or zmax is not None:
        parts.append(f"  usda_zones: {zmin}–{zmax}")

    env = profile.get("environment") or {}
    if env.get("light_level") is not None:
        parts.append(f"  light_level: {env.get('light_level')}")
    if env.get("soil_preference") is not None:
        parts.append(f"  soil_preference: {env.get('soil_preference')}")
    temp = env.get("temperature_pref") or {}
    if temp.get("min_f") is not None or temp.get("max_f") is not None:
        parts.append(f"  temp_pref_f: {temp.get('min_f')}–{temp.get('max_f')}")

    constraints = profile.get("constraints") or {}
    if constraints.get("preferred_size") is not None:
        parts.append(f"  preferred_size: {constraints.get('preferred_size')}")

    prefs = profile.get("preferences") or {}
    if prefs.get("care_level") is not None:
        parts.append(f"  care_level: {prefs.get('care_level')}")
    if prefs.get("growth_pref") is not None:
        parts.append(f"  growth_pref: {prefs.get('growth_pref')}")
    care_prefs = prefs.get("care_preferences") or {}
    if care_prefs.get("watering_freq") is not None:
        parts.append(f"  watering_freq: {care_prefs.get('watering_freq')}")

    return "\n".join(parts) if parts else "(No user profile)"


def retrieve_user_profile(
    username: str,
    include_profile: bool = False,
    include_location: bool = False,
    include_environment: bool = True,
    include_climate: bool = True,
    include_constraints: bool = True,
    include_preferences: bool = True,
    include_gamification: bool = False,
    include_history: bool = False,
) -> str:
    """
    Retrieve user fields used by the two-tower recommender (same as training / onboarding).

    Optional extras (off by default): ``include_profile`` (name, avatar), ``include_location`` (city only),
    ``include_gamification``, ``include_history``.
    """
    if not username or not username.strip():
        return "Please provide a username to look up."

    projection = {"auth.username": 1, "auth.email": 1}
    if include_environment or include_climate:
        projection["climate"] = 1
        projection["usda_zone_min"] = 1
        projection["usda_zone_max"] = 1
        projection["environment"] = 1
    if include_constraints:
        projection["constraints"] = 1
    if include_preferences:
        projection["preferences"] = 1
    if include_profile:
        projection["profile"] = 1
    if include_location:
        projection["location"] = 1
    if include_gamification:
        projection["gamification"] = 1
    if include_history:
        projection["history"] = 1

    user_coll = get_user_collection()
    uname = username.strip()
    user = user_coll.find_one(
        {"$or": [{"auth.username": uname}, {"auth.email": uname.lower()}]},
        projection,
    )

    if not user:
        return f"User '{uname}' not found."

    flat = _flatten_user(user)
    if include_profile:
        prof = user.get("profile") or {}
        if prof.get("name") is not None:
            flat["name"] = prof.get("name")
        if prof.get("avatar_url") is not None:
            flat["avatar_url"] = prof.get("avatar_url")
    if include_location:
        loc = user.get("location") or {}
        if loc.get("city") is not None:
            flat["city"] = loc.get("city")
        if loc.get("state") is not None:
            flat["state"] = loc.get("state")
        if loc.get("postal_code") is not None:
            flat["postal_code"] = loc.get("postal_code")
        if loc.get("country") is not None:
            flat["country"] = loc.get("country")
    if include_gamification:
        gamification = user.get("gamification") or {}
        flat["care_points"] = gamification.get("care_points")
        flat["streak_days"] = gamification.get("streak_days")
        flat["badges"] = gamification.get("badges")
    if include_history:
        history = user.get("history") or {}
        flat["owned_plants_count"] = history.get("owned_plants_count")
        flat["deaths_count"] = history.get("deaths_count")
        flat["average_health_score"] = history.get("average_health_score")

    return "\n".join(f"  {k}: {v}" for k, v in flat.items() if v is not None)


def retrieve_pfaff_plant_knowledge(question: str, plant_name: str, top_k: int = 5) -> str:
    """
    Search the Plants For A Future (PFAF) knowledge base (embedded in Pinecone) for this species.

    Use when the user wants cultivation, edibility, habitat, or other PFAF-style detail beyond our catalog.
    **plant_name** must be the scientific Latin binomial exactly as stored in the index (e.g.
    ``Populus trichocarpa``), same spelling as ``latin_name`` in chunk metadata—not the common name.

    **question** is the user's natural-language query (what to retrieve about that plant).
    """
    name = (plant_name or "").strip()
    if not name:
        return "Provide plant_name: the scientific (Latin) binomial for the species (e.g. Solanum lycopersicum)."

    try:
        from chat.rag_retrieval import query_plant_knowledge_rag

        return query_plant_knowledge_rag(question, plant_name=name, top_k=max(1, min(int(top_k), 20)))
    except Exception as e:
        logger.warning("PFAF RAG retrieval failed: %s", e)
        return f"PFAF knowledge lookup failed: {e}"


# All tools for the agent. Bind with: llm.bind_tools(AGENT_TOOLS)
AGENT_TOOLS = [
    retrieve_plant_profile,
    retrieve_pfaff_plant_knowledge,
    retrieve_user_profile,
]
