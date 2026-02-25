"""
Garden API. Add plants to user's garden and list them.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.jwt import get_current_username
from database import get_garden_collection, get_plant_collection

router = APIRouter(prefix="/garden", tags=["garden"])


class AddToGardenRequest(BaseModel):
    plant_id: int
    custom_name: str | None = None


@router.post("/add")
def add_to_garden(
    body: AddToGardenRequest,
    username: str = Depends(get_current_username),
):
    """
    Add a plant to the user's garden.
    plant_id: required.
    custom_name: optional; if empty, uses latin name from plant catalog.
    """
    plant_coll = get_plant_collection()
    garden_coll = get_garden_collection()
    plant_id = body.plant_id

    plant = plant_coll.find_one({"plant_id": plant_id})
    display_name = body.custom_name.strip() if body.custom_name and body.custom_name.strip() else None
    if not display_name:
        if plant:
            info = plant.get("Info", {}) or {}
            display_name = info.get("latin") or info.get("common_name") or f"Plant #{plant_id}"
        else:
            display_name = f"Plant #{plant_id}"

    doc = {
        "username": username,
        "plant_id": plant_id,
        "custom_name": display_name,
        "added_at": datetime.now(timezone.utc),
    }
    garden_coll.insert_one(doc)
    return {"message": "Plant added to garden", "plant_id": plant_id, "custom_name": display_name}


@router.get("/")
def get_my_garden(username: str = Depends(get_current_username)):
    """List all plants in the user's garden."""
    garden_coll = get_garden_collection()
    plant_coll = get_plant_collection()

    items = list(garden_coll.find({"username": username}).sort("added_at", -1))
    if not items:
        return {"plants": []}

    plant_ids = [i["plant_id"] for i in items]
    plants_by_id = {p["plant_id"]: p for p in plant_coll.find({"plant_id": {"$in": plant_ids}})}

    results = []
    for item in items:
        p = plants_by_id.get(item["plant_id"], {})
        info = p.get("Info", {}) or {}
        care = p.get("Care", {}) or {}
        light_req = care.get("light_req", {}) or {}
        ideal = light_req.get("ideal_light", {}) or {}
        tolerated = light_req.get("tolerated_light", {}) or {}
        temp_req = care.get("temp_req", {}) or {}
        results.append({
            "plant_id": item["plant_id"],
            "custom_name": item.get("custom_name", info.get("latin") or f"Plant #{item['plant_id']}"),
            "added_at": item.get("added_at"),
            "img_url": p.get("img_url"),
            "latin": info.get("latin"),
            "common_name": info.get("common_name"),
            "sunlight_type": ideal.get("sunlight_type") or tolerated.get("sunlight_type"),
            "humidity": care.get("humidity_req_bucket") or care.get("humidity_req"),
            "care_level": care.get("care_level"),
            "water_req": care.get("water_req_bucket") or care.get("water_req"),
            "temp_min": temp_req.get("min_temp"),
            "temp_max": temp_req.get("max_temp"),
        })

    return {"plants": results}
