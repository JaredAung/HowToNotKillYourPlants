"""
Profile/onboarding API. Saves user profile data to UserCollection.
Requires Authorization: Bearer <token>.
"""
from fastapi import APIRouter, Depends, HTTPException

from auth.jwt import get_current_username
from database import get_user_collection
from schemas import ProfileUpdate

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/")
def get_profile(username: str = Depends(get_current_username)):
    """Return the logged-in user's full profile."""
    collection = get_user_collection()
    user = collection.find_one({"auth.username": username}) or collection.find_one(
        {"auth.email": username.lower()}
    )
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User not found. Sign up first at /auth.",
        )
    # Return profile data (exclude password_hash)
    auth = user.get("auth", {}) or {}
    profile = user.get("profile", {}) or {}
    location = user.get("location", {}) or {}
    environment = user.get("environment", {}) or {}
    climate = user.get("climate")
    safety = user.get("safety", {}) or {}
    constraints = user.get("constraints", {}) or {}
    preferences = user.get("preferences", {}) or {}
    return {
        "username": auth.get("username") or username,
        "profile": profile,
        "location": location,
        "environment": environment,
        "climate": climate,
        "safety": safety,
        "constraints": constraints,
        "preferences": preferences,
    }


@router.post("/update")
def update_profile(data: ProfileUpdate, username: str = Depends(get_current_username)):
    """Update profile for the logged-in user."""
    collection = get_user_collection()
    user = collection.find_one({"auth.username": username}) or collection.find_one(
        {"auth.email": username.lower()}
    )
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User not found. Sign up first at /auth, or ensure username '{username}' exists.",
        )

    update = {}
    if data.name is not None:
        update["profile.name"] = data.name
    if data.avatar_url is not None:
        update["profile.avatar_url"] = data.avatar_url
    if data.city is not None:
        update["location.city"] = data.city
    if data.state is not None:
        update["location.state"] = data.state
    if data.postal_code is not None:
        update["location.postal_code"] = data.postal_code
    if data.country is not None:
        update["location.country"] = data.country
    if data.light_level is not None:
        update["environment.light_level"] = data.light_level
    if data.humidity_level is not None:
        update["environment.humidity_level"] = data.humidity_level
    if data.temp_min_f is not None or data.temp_max_f is not None:
        update["environment.temperature_pref"] = {
            "min_f": data.temp_min_f,
            "max_f": data.temp_max_f,
        }
    if data.climate is not None:
        update["climate"] = data.climate
    if data.has_kids is not None:
        update["safety.has_kids"] = data.has_kids
    if data.preferred_size is not None:
        update["constraints.preferred_size"] = data.preferred_size
    if data.hard_no is not None:
        update["constraints.hard_no"] = data.hard_no
    if data.care_level is not None:
        update["preferences.care_level"] = data.care_level
    if data.watering_freq is not None or data.care_freq is not None:
        care = dict((user.get("preferences") or {}).get("care_preferences") or {})
        if data.watering_freq is not None:
            care["watering_freq"] = data.watering_freq
        if data.care_freq is not None:
            care["care_freq"] = data.care_freq
        update["preferences.care_preferences"] = care

    if update:
        query = (
            {"auth.username": username}
            if user.get("auth", {}).get("username")
            else {"auth.email": username.lower()}
        )
        collection.update_one(query, {"$set": update})

    return {"message": "Profile updated", "username": username}
