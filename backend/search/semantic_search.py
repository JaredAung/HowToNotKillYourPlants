"""
Semantic search: embed NLP query with Voyage AI, cosine similarity against plant vectors in MongoDB.

Uses ``profile_embedding`` (from embed/upload ETL) or legacy ``desc_embeddings`` on each catalog doc.
"""
import math
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException

from auth.jwt import get_current_username
from database.mongodb import get_plant_collection
from plant.mongo_plant import flatten_catalog_plant_for_api
from pydantic import BaseModel
import voyageai

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

EMBED_MODEL = "voyage-4-lite"
DEFAULT_K = 20

router = APIRouter(prefix="/semantic", tags=["search"])

# Query only docs that have a non-empty embedding array (Mongo catalog field names).
_EMBEDDING_QUERY = {
    "$or": [
        {"profile_embedding.0": {"$exists": True}},
        {"desc_embeddings.0": {"$exists": True}},
    ]
}


def _semantic_embedding_vector(p: dict) -> list[float]:
    """Prefer profile_embedding (NewPlantCollection / ETL); fall back to desc_embeddings."""
    for key in ("profile_embedding", "desc_embeddings"):
        raw = p.get(key)
        if not isinstance(raw, list) or not raw:
            continue
        try:
            return [float(x) for x in raw]
        except (TypeError, ValueError):
            continue
    return []


def _load_plants() -> list[dict]:
    """Load catalog plants from MongoDB that have a semantic embedding vector."""
    coll = get_plant_collection()
    plants = list(coll.find(_EMBEDDING_QUERY))
    return [p for p in plants if _semantic_embedding_vector(p)]


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
    """Convert plant dict to API result format (catalog schema: ``info``, ``environment_care``, …)."""
    base = dict(p)
    base["score"] = score
    return flatten_catalog_plant_for_api(base)


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
    plant profile_embedding / desc_embeddings in MongoDB, return ranked plants.
    """
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    k = max(1, min(body.k or DEFAULT_K, 100))

    plants = _load_plants()
    if not plants:
        raise HTTPException(
            status_code=503,
            detail=(
                "No plants with profile_embedding or desc_embeddings in MongoDB. "
                "Run resources ETL embed/upload so NEW_PLANT_COLLECTION has vectors."
            ),
        )

    query_emb = _embed_query(query)

    scored = []
    for p in plants:
        plant_emb = _semantic_embedding_vector(p)
        if not plant_emb:
            continue
        sim = _cosine_similarity(query_emb, plant_emb)
        scored.append((p, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = [_plant_to_result(p, s) for p, s in scored[:k]]

    return {"plants": results, "query": query}
