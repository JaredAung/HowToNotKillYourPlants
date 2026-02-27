"""
Tools for the chat agent. Bind these to the LLM for plant retrieval and actions.
"""
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from database import get_plant_collection, get_user_collection

# Tavily internet search tool. Uses TAVILY_API_KEY from env.
tavily_search = TavilySearch(max_results=5, search_depth="basic", topic="general")


def _flatten_plant(p: dict) -> dict:
    """Flatten MongoDB plant doc to a readable profile. Handles partial projections."""
    info = p.get("Info", {}) or {}
    care = p.get("Care", {}) or {}
    light_req = care.get("light_req", {}) or {}
    ideal = light_req.get("ideal_light", {}) or {}
    tolerated = light_req.get("tolerated_light", {}) or {}
    temp_req = care.get("temp_req", {}) or {}
    desc = info.get("desc", {}) or {}
    result: dict = {}
    if "plant_id" in p:
        result["plant_id"] = p["plant_id"]
    if "img_url" in p:
        result["img_url"] = p["img_url"]
    if "Info" in p:
        result["latin"] = info.get("latin")
        result["common_name"] = info.get("common_name")
        result["category"] = info.get("category")
        result["physical_desc"] = desc.get("physical_desc")
        result["symbolism"] = desc.get("symbolism")
    if "Care" in p:
        result["sunlight_type"] = ideal.get("sunlight_type") or tolerated.get("sunlight_type")
        result["humidity"] = care.get("humidity_req_bucket") or care.get("humidity_req")
        result["care_level"] = care.get("care_level")
        result["water_req"] = care.get("water_req_bucket") or care.get("water_req")
        result["temp_min"] = temp_req.get("min_temp")
        result["temp_max"] = temp_req.get("max_temp")
        result["climate"] = care.get("climate")
    return result

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
    Searches by latin name, common name, physical description, and symbolism.
    Use this when the user asks about a specific plant or needs plant information.

    Choose which sections to include based on what the user needs:
    - include_plant_id: plant identifier (for linking, adding to garden)
    - include_info: latin name, common name, category, physical_desc, symbolism
    - include_care: sunlight, humidity, care_level, water_req, temp, climate
    - include_img_url: image URL for display
    """
    if not search_query or not search_query.strip():
        return "Please provide a search query (e.g. plant name, appearance, or care needs)."

    projection = {}
    if include_plant_id:
        projection["plant_id"] = 1
    if include_info:
        projection["Info"] = 1
    if include_care:
        projection["Care"] = 1
    if include_img_url:
        projection["img_url"] = 1

    if not projection:
        return "At least one section (plant_id, Info, Care, img_url) must be included."

    plant_coll = get_plant_collection()
    query = search_query.strip()
    regex = {"$regex": query, "$options": "i"}

    cursor = plant_coll.find(
        {
            "$or": [
                {"Info.latin": regex},
                {"Info.common_name": regex},
                {"Info.desc.physical_desc": regex},
                {"Info.desc.symbolism": regex},
                {"Info.category": regex},
            ]
        },
        projection,
    ).limit(limit)

    plants = list(cursor)
    if not plants:
        return f"No plants found matching '{query}'."

    return "\n\n".join(
        f"Plant {i + 1}:\n" + "\n".join(f"  {k}: {v}" for k, v in _flatten_plant(p).items() if v is not None)
        for i, p in enumerate(plants)
    )


def _flatten_user(u: dict) -> dict:
    """Flatten MongoDB user doc to a readable profile. Handles partial projections. Excludes password_hash."""
    auth = u.get("auth", {}) or {}
    profile = u.get("profile", {}) or {}
    location = u.get("location", {}) or {}
    env = u.get("environment", {}) or {}
    temp_pref = env.get("temperature_pref", {}) or {}
    constraints = u.get("constraints", {}) or {}
    prefs = u.get("preferences", {}) or {}
    care_prefs = prefs.get("care_preferences", {}) or {}
    gamification = u.get("gamification", {}) or {}
    history = u.get("history", {}) or {}
    result: dict = {}
    if "auth" in u:
        result["username"] = auth.get("username") or auth.get("email")
    if "profile" in u:
        result["name"] = profile.get("name")
        result["avatar_url"] = profile.get("avatar_url")
    if "location" in u:
        result["city"] = location.get("city")
        result["state"] = location.get("state")
        result["postal_code"] = location.get("postal_code")
        result["country"] = location.get("country")
    if "environment" in u:
        result["light_level"] = env.get("light_level")
        result["humidity_level"] = env.get("humidity_level")
        result["temp_min_f"] = temp_pref.get("min_f")
        result["temp_max_f"] = temp_pref.get("max_f")
    if "climate" in u:
        result["climate"] = u.get("climate")
    if "constraints" in u:
        result["preferred_size"] = constraints.get("preferred_size")
    if "preferences" in u:
        result["care_level"] = prefs.get("care_level")
        result["watering_freq"] = care_prefs.get("watering_freq")
        result["care_freq"] = care_prefs.get("care_freq")
    if "gamification" in u:
        result["care_points"] = gamification.get("care_points")
        result["streak_days"] = gamification.get("streak_days")
        result["badges"] = gamification.get("badges")
    if "history" in u:
        result["owned_plants_count"] = history.get("owned_plants_count")
        result["deaths_count"] = history.get("deaths_count")
        result["average_health_score"] = history.get("average_health_score")
    return result


def retrieve_user_profile(
    username: str,
    include_profile: bool = True,
    include_location: bool = True,
    include_environment: bool = True,
    include_climate: bool = True,
    include_constraints: bool = True,
    include_preferences: bool = True,
    include_gamification: bool = False,
    include_history: bool = False,
) -> str:
    """
    Retrieve a user profile from the user collection by username.
    Use this when you need the current user's preferences, environment, constraints, or location to personalize recommendations or advice.

    Choose which sections to include based on what you need:
    - include_profile: name, avatar_url
    - include_location: city, state, postal_code, country
    - include_environment: light_level, humidity_level, temp_min_f, temp_max_f
    - include_climate: climate zone
    - include_constraints: preferred_size
    - include_preferences: care_level, watering_freq, care_freq
    - include_gamification: care_points, streak_days, badges
    - include_history: owned_plants_count, deaths_count, average_health_score
    """
    if not username or not username.strip():
        return "Please provide a username to look up."

    projection = {"auth.username": 1, "auth.email": 1}
    if include_profile:
        projection["profile"] = 1
    if include_location:
        projection["location"] = 1
    if include_environment:
        projection["environment"] = 1
    if include_climate:
        projection["climate"] = 1
    if include_constraints:
        projection["constraints"] = 1
    if include_preferences:
        projection["preferences"] = 1
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
    return "\n".join(f"  {k}: {v}" for k, v in flat.items() if v is not None)


# All tools for the agent. Bind with: llm.bind_tools(AGENT_TOOLS)
AGENT_TOOLS = [retrieve_plant_profile, retrieve_user_profile, tavily_search]
