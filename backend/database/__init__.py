"""Database connection and collection access."""
from database.mongodb import get_db, get_user_collection, get_plant_collection, get_garden_collection, get_death_collection, get_token_blacklist_collection

__all__ = ["get_db", "get_user_collection", "get_plant_collection", "get_garden_collection", "get_death_collection", "get_token_blacklist_collection"]
