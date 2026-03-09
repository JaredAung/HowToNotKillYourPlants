"""
Semantic search: embed NLP query with Voyage AI, cosine similarity against plant desc_embeddings.
"""
import json
import math
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException

from auth.jwt import get_current_username
from pydantic import BaseModel
import voyageai

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

PLANTS_PATH = ROOT / "resources" / "plant_profiles.json"
EMBED_MODEL = "voyage-4-lite"
DEFAULT_K = 20

router = APIRouter(prefix="/semantic", tags=["search"])


def _load_plants() -> list[dict]:
    """Load plants from plant_profiles.json with desc_embeddings."""
    if not PLANTS_PATH.exists():
        raise FileNotFoundError(f"Plant profiles not found: {PLANTS_PATH}")
    with open(PLANTS_PATH) as f:
        plants = json.load(f)
    return [p for p in plants if p.get("desc_embeddings")]


def _embed_query(query: str) -> list[float]:
    """Embed query with Voyage AI (input_type=query for search)."""
    vo = voyageai.Client()
    result = vo.embed([query], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dp = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dp / (na * nb)


def _plant_to_result(p: dict, score: float) -> dict:
    """Convert plant dict to API result format."""
    info = p.get("Info", {}) or {}
    care = p.get("Care", {}) or {}
    light_req = care.get("light_req", {}) or {}
    ideal = light_req.get("ideal_light", {}) or {}
    tolerated = light_req.get("tolerated_light", {}) or {}
    temp_req = care.get("temp_req", {}) or {}
    return {
        "plant_id": p["plant_id"],
        "score": round(score, 4),
        "img_url": p.get("img_url"),
        "latin": info.get("latin"),
        "common_name": info.get("common_name"),
        "sunlight_type": ideal.get("sunlight_type") or tolerated.get("sunlight_type"),
        "humidity": care.get("humidity_req_bucket") or care.get("humidity_req"),
        "care_level": care.get("care_level"),
        "water_req": care.get("water_req_bucket") or care.get("water_req"),
        "temp_min": temp_req.get("min_temp"),
        "temp_max": temp_req.get("max_temp"),
    }


class SemanticSearchBody(BaseModel):
    query: str = ""
    k: int = DEFAULT_K


@router.post("/query")
def semantic_search(
    body: SemanticSearchBody,
    _username: str = Depends(get_current_username),
):
    """
    Embed user NLP query with Voyage AI, compute cosine similarity against
    plant desc_embeddings, return ranked plants.
    """
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    k = max(1, min(body.k or DEFAULT_K, 100))

    plants = _load_plants()
    if not plants:
        raise HTTPException(
            status_code=503,
            detail="No plants with desc_embeddings. Run plant_data_clean to generate embeddings.",
        )

    query_emb = _embed_query(query)

    scored = []
    for p in plants:
        desc_emb = p.get("desc_embeddings") or []
        if not desc_emb:
            continue
        sim = _cosine_similarity(query_emb, desc_emb)
        scored.append((p, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = [_plant_to_result(p, s) for p, s in scored[:k]]

    return {"plants": results, "query": query}
