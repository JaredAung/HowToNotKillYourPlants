"""
Garden API. Add plants to user's garden and list them.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from auth.jwt import get_current_username
from database import get_garden_collection, get_plant_collection
from schemas import AddToGardenRequest
from plant.mongo_plant import flatten_catalog_plant_for_api

from garden.death import router as death_router

router = APIRouter(prefix="/garden", tags=["garden"])
router.include_router(death_router)


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
            info = plant.get("info") or {}
            display_name = (
                plant.get("scientific_name")
                or plant.get("name")
                or info.get("latin")
                or info.get("common_name")
                or f"Plant #{plant_id}"
            )
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
        row = flatten_catalog_plant_for_api({**p, "score": 0})
        results.append({
            "plant_id": item["plant_id"],
            "custom_name": item.get(
                "custom_name",
                row.get("latin") or row.get("common_name") or f"Plant #{item['plant_id']}",
            ),
            "added_at": item.get("added_at"),
            "match_percentage": item.get("match_percentage"),
            "img_url": row.get("img_url"),
            "latin": row.get("latin"),
            "common_name": row.get("common_name"),
            "sunlight_type": row.get("sunlight_type"),
            "humidity": row.get("humidity"),
            "care_level": row.get("care_level"),
            "water_req": row.get("water_req"),
            "temp_min": row.get("temp_min"),
            "temp_max": row.get("temp_max"),
        })

    return {"plants": results}
