"""
Mark plants in the user's garden as dead.
Maps form values to two-tower training vocab (LIGHT_VOCAB, WATER_VOCAB).
"""
from datetime import datetime, timedelta, timezone

DEATH_TTL_DAYS = 30

from fastapi import APIRouter, Depends, HTTPException

from auth.jwt import get_current_username
from database import get_death_collection, get_garden_collection
from schemas import DeathReport

router = APIRouter(tags=["garden"])

# Map death form display values -> two-tower vocab (feature_loader.py)
WATERING_TO_BUCKET = {
    "Every day": "high",
    "Every 2 days": "medium",
    "Weekly": "low",
}
# LIGHT_VOCAB = ["direct", "bright_light", "bright_indirect", "indirect", "diffused"]
PLANT_LOCATION_TO_BUCKET = {
    "Direct sunlight": "direct",
    "Bright light": "bright_light",
    "Bright indirect light": "bright_indirect",
    "Medium indirect light": "indirect",
    "Low light": "diffused",
}
# HUMIDITY_VOCAB = ["low", "medium", "high"]
HUMIDITY_TO_BUCKET = {
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}
# Room temp: no two-tower vocab; use low/medium/high for consistency
ROOM_TEMP_TO_BUCKET = {
    "Cold": "low",
    "Comfortable": "medium",
    "Hot": "high",
}


@router.post("/death")
def mark_plant_dead(
    body: DeathReport,
    username: str = Depends(get_current_username),
):
    """
    Record a plant death and remove it from the user's garden.
    Inserts into Death_Collection, then deletes from User_Garden_Collection.
    """
    garden_coll = get_garden_collection()
    death_coll = get_death_collection()

    garden_doc = garden_coll.find_one({"username": username, "plant_id": body.plant_id})
    if not garden_doc:
        raise HTTPException(
            status_code=404,
            detail="Plant not found in your garden or already marked dead",
        )

    died_at = datetime.now(timezone.utc)
    expires_at = died_at + timedelta(days=DEATH_TTL_DAYS)
    watering_bucket = WATERING_TO_BUCKET.get(body.watering_frequency) if body.watering_frequency else None
    plant_location_bucket = PLANT_LOCATION_TO_BUCKET.get(body.plant_location) if body.plant_location else None
    humidity_bucket = HUMIDITY_TO_BUCKET.get(body.humidity_level) if body.humidity_level else None
    room_temp_bucket = ROOM_TEMP_TO_BUCKET.get(body.room_temperature) if body.room_temperature else None

    death_doc = {
        "username": username,
        "plant_id": body.plant_id,
        "custom_name": garden_doc.get("custom_name"),
        "added_at": garden_doc.get("added_at"),
        "died_at": died_at,
        "expires_at": expires_at,
        "what_happened": body.what_happened,
        "watering_frequency": body.watering_frequency,
        "watering_frequency_bucket": watering_bucket,
        "plant_location": body.plant_location,
        "plant_location_bucket": plant_location_bucket,
        "humidity_level": body.humidity_level,
        "humidity_level_bucket": humidity_bucket,
        "room_temperature": body.room_temperature,
        "room_temperature_bucket": room_temp_bucket,
        "death_reason": body.death_reason.strip() if body.death_reason else None,
        "plant_profile": body.plant_profile,
        "user_profile": body.user_profile,
    }
    death_coll.insert_one(death_doc)

    garden_coll.delete_one({"username": username, "plant_id": body.plant_id})

    return {"message": "Plant marked as dead", "plant_id": body.plant_id}


def get_dead_plant_ids(username: str) -> list[int]:
    """Return list of plant_ids the user has marked as dead. Used for death penalty in recommendations."""
    death_coll = get_death_collection()
    docs = death_coll.find({"username": username}, {"plant_id": 1})
    return list({d["plant_id"] for d in docs})
