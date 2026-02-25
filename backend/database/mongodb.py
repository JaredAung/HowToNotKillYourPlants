"""
Shared MongoDB connection and collection access.
Uses env vars: MONGO_URI, MONGO_DATABASE, MONGO_USER_PROFILES_COLLECTION, PLANT_MONGO_COLLECTION
"""
import os

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
USER_COLLECTION = os.getenv("MONGO_USER_PROFILES_COLLECTION", "UserCollection")
PLANT_COLLECTION = os.getenv("PLANT_MONGO_COLLECTION", "PlantCollection")
GARDEN_COLLECTION = os.getenv("MONGO_USER_GARDEN_COLLECTION", "User_Garden_Collection")
TOKEN_BLACKLIST_COLLECTION = os.getenv("MONGO_TOKEN_BLACKLIST_COLLECTION", "TokenBlacklist")


def get_db() -> Database:
    """Get MongoDB database. Raises if MONGO_URI not set."""
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI not set. Add MONGO_URI to .env in the project root.")
    client = MongoClient(MONGO_URI)
    return client[MONGO_DATABASE]


def get_user_collection() -> Collection:
    """Get UserCollection with unique index on auth.username."""
    coll = get_db()[USER_COLLECTION]
    coll.create_index("auth.username", unique=True, sparse=True)
    return coll


def get_plant_collection() -> Collection:
    """Get PlantCollection (plants with plant_tower_embedding)."""
    return get_db()[PLANT_COLLECTION]


def get_garden_collection() -> Collection:
    """Get User_Garden_Collection (user's planted plants)."""
    return get_db()[GARDEN_COLLECTION]


def get_token_blacklist_collection() -> Collection:
    """Get TokenBlacklist (revoked JWT tokens)."""
    coll = get_db()[TOKEN_BLACKLIST_COLLECTION]
    coll.create_index("exp", expireAfterSeconds=0)  # TTL: auto-delete when exp is past
    return coll
