"""
Plant detail API. Fetch a single plant by ID.
"""
from fastapi import APIRouter, Depends, HTTPException

from auth.jwt import get_current_username
from database import get_plant_collection
from plant.mongo_plant import flatten_catalog_plant_for_api, sanitize_catalog_document

router = APIRouter(prefix="/plant", tags=["plant"])


def _temperature_display(row: dict, ec: dict) -> str | None:
    """Human-readable temperature / zone line for Permapeople ``environment_care``."""
    tmin, tmax = row.get("temp_min"), row.get("temp_max")
    if tmin is not None and tmax is not None:
        try:
            return f"{round(float(tmin))}–{round(float(tmax))}°F"
        except (TypeError, ValueError):
            pass
    note = ec.get("Temperature")
    if isinstance(note, str) and note.strip():
        return note.strip()
    usda = ec.get("USDA Hardiness zone")
    if isinstance(usda, dict):
        lo = usda.get("min")
        hi = usda.get("max")
        if lo is not None or hi is not None:
            return f"USDA zones {lo}–{hi}"
    return None


def _flatten_plant(p: dict) -> dict:
    """Flatten MongoDB catalog doc (``info``, ``environment_care``, ``images``, …) to API shape."""
    row = flatten_catalog_plant_for_api({**p, "score": 0})
    info = p.get("info") or {}
    ec = p.get("environment_care") or {}
    prob = p.get("problems") or {}
    desc = info.get("desc") if isinstance(info.get("desc"), dict) else {}

    soil_type = None
    soil_raw = ec.get("soil")
    if isinstance(soil_raw, dict):
        parts = [soil_raw.get("ideal_soil"), soil_raw.get("tolerated_soil")]
        joined = ", ".join(str(x) for x in parts if x)
        soil_type = joined or None
    st = ec.get("Soil type")
    if isinstance(st, str) and st.strip():
        soil_type = st

    def listish(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i) for i in x]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    w_raw = ec.get("water")
    if isinstance(w_raw, dict):
        water_raw_display = ", ".join(
            f"{k}: {v}" for k, v in w_raw.items() if v is not None
        ) or None
    else:
        water_raw_display = w_raw
    if water_raw_display is not None and not isinstance(water_raw_display, str):
        water_raw_display = str(water_raw_display)

    out = {
        **row,
        "origin": info.get("Native to"),
        "growth_rate": ec.get("Growth") or ec.get("growth_req"),
        "physical_desc": row.get("physical_desc") or desc.get("physical_desc"),
        "symbolism": row.get("symbolism") or desc.get("symbolism"),
        "humidity_req": row.get("humidity"),
        "water_req_raw": water_raw_display,
        "soil_type": soil_type,
        "drainage_level": None,
        "temperature_display": _temperature_display(row, ec),
        "bugs": listish(prob.get("Pests")),
        "disease": listish(prob.get("Diseases")),
    }
    out.pop("score", None)
    return out


@router.get("/{plant_id}")
def get_plant(plant_id: int, _username: str = Depends(get_current_username)):
    """Fetch a single plant by ID. Requires auth.

    Returns flattened UI fields plus ``catalog``: the full Mongo document without ids or embeddings.
    """
    plant_coll = get_plant_collection()
    p = plant_coll.find_one({"plant_id": plant_id})
    if not p:
        raise HTTPException(status_code=404, detail="Plant not found")
    out = _flatten_plant(p)
    out["catalog"] = sanitize_catalog_document(p)
    return out
