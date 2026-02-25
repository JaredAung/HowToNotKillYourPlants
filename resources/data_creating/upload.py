"""
Upload plant profiles with two-tower embeddings to MongoDB.
Merges plant_profiles.json with plant_embeddings.json (by plant_id) and inserts into PlantCollection.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# Paths (relative to resources/)
ROOT = Path(__file__).resolve().parent.parent
PLANT_PROFILES_PATH = ROOT / "plant_profiles.json"
PLANT_EMBEDDINGS_PATH = ROOT / "two_tower_training" / "output" / "plant_embeddings.json"

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
PLANT_COLLECTION = os.getenv("PLANT_MONGO_COLLECTION", "PlantCollection")


def get_plant_collection():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DATABASE][PLANT_COLLECTION]


def load_and_merge() -> list[dict]:
    """Load plant profiles and embeddings, merge by plant_id."""
    with open(PLANT_PROFILES_PATH) as f:
        profiles = json.load(f)
    with open(PLANT_EMBEDDINGS_PATH) as f:
        embeddings_raw = json.load(f)

    embeddings_by_id = {item["plant_id"]: item["embedding"] for item in embeddings_raw}

    merged = []
    for profile in profiles:
        pid = profile["plant_id"]
        emb = embeddings_by_id.get(pid)
        if emb is None:
            continue  # skip plants without embedding
        doc = {**profile, "plant_tower_embedding": emb}
        merged.append(doc)
    return merged


def upload(dry_run: bool = False) -> int:
    """Upload merged documents to MongoDB. Returns count inserted."""
    docs = load_and_merge()
    if not docs:
        print("No documents to upload.")
        return 0

    if dry_run:
        print(f"DRY RUN: Would insert {len(docs)} documents.")
        return len(docs)

    coll = get_plant_collection()
    coll.delete_many({})  # replace all
    result = coll.insert_many(docs)
    print(f"Inserted {len(result.inserted_ids)} documents into {MONGO_DATABASE}.{PLANT_COLLECTION}")
    return len(result.inserted_ids)


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    upload(dry_run=dry_run)
