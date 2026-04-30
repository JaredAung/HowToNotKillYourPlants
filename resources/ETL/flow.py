"""
Prefect workflow: retrieve plants from MongoDB, load users from synthetic_users.json.
Applies feature engineering to both. If Mongo documents include ``plant_tower_embedding``
(64-d), writes ``plant_tower_features.parquet`` and materializes with Feast in the same
run as plant/user features.

Run from project root:
  python -m resources.ETL.flow
  python -m resources.ETL.flow --collection NewPlantCollection
"""
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCES = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

USERS_PATH = RESOURCES / "data" / "synthetic_users.json"

from prefect import flow, task


@task
def load_users_task() -> list[dict]:
    """Load users from synthetic_users.json."""
    with open(USERS_PATH) as f:
        return json.load(f)


@task
def apply_user_feature_engineering_task(users: list[dict]) -> list[dict]:
    """Apply feature engineering to users (same structure as plants)."""
    from resources.ETL.feature_engineer import apply_user_embeddings
    return apply_user_embeddings(users)


@task
def push_features_to_feast_task(
    plants: list[dict],
    users: list[dict],
    plant_tower_by_id: dict[int, list[float]] | None = None,
) -> None:
    """Write plant and user features to parquet; optionally plant_tower_features (64-d from Mongo)."""
    from resources.ETL.feast_store import push_features_to_feast
    push_features_to_feast(plants, users, plant_tower_by_id=plant_tower_by_id)


@task
def apply_feature_engineering_task(plants: list[dict]) -> list[dict]:
    """Apply feature engineering: ordinal numerics, categorical embeddings, water, USDA zone."""
    from resources.ETL.feature_engineer import apply_categorical_embeddings
    return apply_categorical_embeddings(plants)


@task
def fetch_plants_from_mongo(
    collection_name: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Retrieve plants from MongoDB. Returns list of plant documents."""
    try:
        from pymongo import MongoClient
    except ImportError:
        raise ImportError("pymongo not installed. Run: pip install pymongo")

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    coll_name = collection_name or os.getenv("NEW_PLANT_COLLECTION", "NewPlantCollection")

    if not mongo_uri:
        raise RuntimeError("MONGO_URI not set in .env")

    client = MongoClient(mongo_uri)
    coll = client[mongo_db][coll_name]
    cursor = coll.find({})
    if limit is not None:
        cursor = cursor.limit(limit)
    plants = list(cursor)
    # Convert ObjectId to str for JSON serialization
    for p in plants:
        if "_id" in p:
            p["_id"] = str(p["_id"])
    return plants


@flow(name="etl-retrieve-plants", log_prints=True)
def retrieve_plants_flow(
    collection_name: str | None = None,
    limit: int | None = None,
) -> dict:
    """
    Retrieve plants from MongoDB.

    Args:
        collection_name: Override collection (default: NEW_PLANT_COLLECTION env or NewPlantCollection).
        limit: Max number of plants to fetch (None = all).
    """
    plants = fetch_plants_from_mongo(
        collection_name=collection_name,
        limit=limit,
    )
    count = len(plants)
    print(f"Retrieved {count} plants from MongoDB")

    plants = apply_feature_engineering_task(plants)
    if plants:
        dim = len(plants[0].get("categorical_embedding", []))
        print(f"Applied plant embeddings ({dim} dim)")

        # Sample plant features
        sample = plants[0]
        emb = sample.get("categorical_embedding", [])
        if emb:
            feat_names = [
                "origin_climate_0", "origin_climate_1", "origin_climate_2", "origin_climate_3",
                "origin_climate_4", "origin_climate_5", "origin_climate_6", "origin_climate_7",
                "ideal_light", "tolerated_light", "ideal_soil", "tolerated_soil",
                "size", "growth", "temperature", "care_level",
                "difficulty_score", "ideal_water_norm", "tolerated_water_norm",
                "usda_min_norm", "usda_max_norm",
            ]
            print("Sample plant features:")
            print(f"  id={sample.get('id')}, name={sample.get('name')}")
            for i, name in enumerate(feat_names):
                if i < len(emb):
                    val = emb[i]
                    print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")

    # Load and feature-engineer users
    users = load_users_task()
    user_count = len(users)
    print(f"Loaded {user_count} users from {USERS_PATH.name}")

    users = apply_user_feature_engineering_task(users)
    if users:
        dim = len(users[0].get("categorical_embedding", []))
        print(f"Applied user embeddings ({dim} dim)")

        # Sample user features
        sample = users[0]
        emb = sample.get("categorical_embedding", [])
        if emb:
            feat_names = [
                "origin_climate_0", "origin_climate_1", "origin_climate_2", "origin_climate_3",
                "origin_climate_4", "origin_climate_5", "origin_climate_6", "origin_climate_7",
                "ideal_light", "tolerated_light", "ideal_soil", "tolerated_soil",
                "size", "growth", "temperature", "care_level",
                "difficulty_score", "ideal_water_norm", "tolerated_water_norm",
                "usda_min_norm", "usda_max_norm",
            ]
            print("Sample user features:")
            print(f"  user_id={sample.get('user_id')}, persona={sample.get('persona')}")
            for i, name in enumerate(feat_names):
                if i < len(emb):
                    val = emb[i]
                    print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")

    from resources.ETL.feast_store import plant_tower_embeddings_from_mongo_plants

    plant_tower_by_id = plant_tower_embeddings_from_mongo_plants(plants)
    if plant_tower_by_id:
        print(
            f"Loaded plant_tower_embedding from Mongo for {len(plant_tower_by_id)} plants "
            "(Feast plant_tower_features)"
        )
    else:
        print("No plant_tower_embedding on Mongo plants; Feast plant_tower_features parquet unchanged")

    push_features_to_feast_task(plants, users, plant_tower_by_id=plant_tower_by_id or None)
    print("Pushed plant and user features to Feast")

    return {
        "count": count,
        "plants": plants,
        "users": users,
        "user_count": user_count,
        "plant_tower_count": len(plant_tower_by_id),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=None, help="MongoDB collection name")
    parser.add_argument("--limit", type=int, default=None, help="Max plants to fetch")
    args = parser.parse_args()
    result = retrieve_plants_flow(
        collection_name=args.collection,
        limit=args.limit,
    )
    print(
        f"Result: {result['count']} plants, {result['user_count']} users, "
        f"plant_tower={result['plant_tower_count']}"
    )
