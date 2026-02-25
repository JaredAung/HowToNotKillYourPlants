"""
Plant detail API. Fetch a single plant by ID.
"""
from fastapi import APIRouter, Depends, HTTPException

from auth.jwt import get_current_username
from database import get_plant_collection

router = APIRouter(prefix="/plant", tags=["plant"])


def _flatten_plant(p: dict) -> dict:
    """Flatten MongoDB plant doc to API response format."""
    info = p.get("Info", {}) or {}
    care = p.get("Care", {}) or {}
    light_req = care.get("light_req", {}) or {}
    ideal = light_req.get("ideal_light", {}) or {}
    tolerated = light_req.get("tolerated_light", {}) or {}
    temp_req = care.get("temp_req", {}) or {}
    desc = info.get("desc", {}) or {}
    soil = care.get("soil", {}) or {}
    return {
        "plant_id": p["plant_id"],
        "img_url": p.get("img_url"),
        "latin": info.get("latin"),
        "common_name": info.get("common_name"),
        "category": info.get("category"),
        "origin": info.get("origin"),
        "size": info.get("size"),
        "growth_rate": info.get("growth_rate"),
        "physical_desc": desc.get("physical_desc"),
        "symbolism": desc.get("symbolism"),
        "sunlight_type": ideal.get("sunlight_type") or tolerated.get("sunlight_type"),
        "ideal_light": ideal.get("sunlight_type") or ideal.get("sunlight_bucket"),
        "tolerated_light": tolerated.get("sunlight_type") or tolerated.get("sunlight_bucket"),
        "humidity": care.get("humidity_req_bucket") or care.get("humidity_req"),
        "humidity_req": care.get("humidity_req"),
        "care_level": care.get("care_level"),
        "water_req": care.get("water_req_bucket") or care.get("water_req"),
        "water_req_raw": care.get("water_req"),
        "temp_min": temp_req.get("min_temp"),
        "temp_max": temp_req.get("max_temp"),
        "climate": care.get("climate"),
        "soil_type": soil.get("soil_type"),
        "drainage_level": soil.get("drainage_level"),
        "bugs": care.get("bugs") or [],
        "disease": care.get("disease") or [],
    }


@router.get("/{plant_id}")
def get_plant(plant_id: int, _username: str = Depends(get_current_username)):
    """Fetch a single plant by ID. Requires auth."""
    plant_coll = get_plant_collection()
    p = plant_coll.find_one({"plant_id": plant_id})
    if not p:
        raise HTTPException(status_code=404, detail="Plant not found")
    return _flatten_plant(p)
