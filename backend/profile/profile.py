"""
Profile/onboarding API. Saves user tower fields to UserCollection.
Requires Authorization: Bearer <token>.
"""
from fastapi import APIRouter, Depends, HTTPException

from auth.jwt import get_current_username
from database import get_user_collection
from schemas import ProfileUpdate

router = APIRouter(prefix="/profile", tags=["profile"])


def tower_profile_response(user: dict, username: str) -> dict:
    """Only keys used by two-tower / ``feature_loader._mongo_user_to_flat_fe_dict``."""
    auth = user.get("auth", {}) or {}
    env = user.get("environment", {}) or {}
    pref = user.get("preferences", {}) or {}
    care_pref = pref.get("care_preferences", {}) or {}
    constraints = user.get("constraints", {}) or {}
    profile = user.get("profile", {}) or {}
    return {
        "username": auth.get("username") or username,
        "profile": {"name": profile.get("name")},
        "climate": user.get("climate"),
        "usda_zone_min": user.get("usda_zone_min"),
        "usda_zone_max": user.get("usda_zone_max"),
        "environment": {
            "light_level": env.get("light_level"),
            "soil_preference": env.get("soil_preference"),
            "temperature_pref": env.get("temperature_pref"),
        },
        "constraints": {
            "preferred_size": constraints.get("preferred_size"),
        },
        "preferences": {
            "care_level": pref.get("care_level"),
            "growth_pref": pref.get("growth_pref"),
            "care_preferences": {
                "watering_freq": care_pref.get("watering_freq"),
            },
        },
    }


@router.get("/")
def get_profile(username: str = Depends(get_current_username)):
    """Return tower-aligned profile fields for the logged-in user."""
    collection = get_user_collection()
    user = collection.find_one({"auth.username": username}) or collection.find_one(
        {"auth.email": username.lower()}
    )
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User not found. Sign up first at /auth.",
        )
    return tower_profile_response(user, username)


@router.post("/update")
def update_profile(data: ProfileUpdate, username: str = Depends(get_current_username)):
    """Update tower profile fields for the logged-in user."""
    collection = get_user_collection()
    user = collection.find_one({"auth.username": username}) or collection.find_one(
        {"auth.email": username.lower()}
    )
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User not found. Sign up first at /auth, or ensure username '{username}' exists.",
        )

    query = (
        {"auth.username": username}
        if user.get("auth", {}).get("username")
        else {"auth.email": username.lower()}
    )

    update = {}
    if data.name is not None:
        update["profile.name"] = data.name
    if data.light_level is not None:
        update["environment.light_level"] = data.light_level
    if data.soil_preference is not None:
        update["environment.soil_preference"] = data.soil_preference
    if data.temp_min_f is not None or data.temp_max_f is not None:
        update["environment.temperature_pref"] = {
            "min_f": data.temp_min_f,
            "max_f": data.temp_max_f,
        }
    if data.climate is not None:
        update["climate"] = data.climate
    if data.preferred_size is not None:
        update["constraints.preferred_size"] = data.preferred_size
    if data.care_level is not None:
        update["preferences.care_level"] = data.care_level
    if data.growth_pref is not None:
        update["preferences.growth_pref"] = data.growth_pref
    if data.usda_zone_min is not None:
        update["usda_zone_min"] = data.usda_zone_min
    if data.usda_zone_max is not None:
        update["usda_zone_max"] = data.usda_zone_max
    if data.watering_freq is not None:
        care = dict((user.get("preferences") or {}).get("care_preferences") or {})
        care["watering_freq"] = data.watering_freq
        update["preferences.care_preferences"] = care

    if update:
        collection.update_one(query, {"$set": update})

    refreshed = collection.find_one(query) or user
    return {"message": "Profile updated", **tower_profile_response(refreshed, username)}
