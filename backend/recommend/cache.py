"""
Redis cache for recommendation pipeline results.
Stores recommendations with short TTL and schema version for cache invalidation.
"""
import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Schema version: bump when recommendation output format changes
CACHE_SCHEMA_VERSION = 1

# Default TTL in seconds (5 min)
DEFAULT_TTL = int(os.getenv("RECOMMEND_CACHE_TTL", "300"))

KEY_PREFIX = "rec:"


def _get_redis():
    """Lazy Redis client. Returns None if Redis is not configured."""
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis
        return redis.from_url(url, decode_responses=True)
    except Exception as e:
        logger.debug("Redis not available: %s", e)
        return None


def _profile_hash(profile: dict) -> str:
    """Stable hash of profile dict for cache key."""
    # Normalize: sort keys, exclude volatile fields
    normalized = json.dumps(profile, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _cache_key(username: str, profile: dict, k: int, use_rerank: bool) -> str:
    return f"{KEY_PREFIX}{username}:{_profile_hash(profile)}:{k}:{use_rerank}"


def _serialize(result: dict) -> str:
    """Wrap result with version for schema evolution."""
    payload = {
        "v": CACHE_SCHEMA_VERSION,
        "username": result.get("username"),
        "plants": result.get("plants", []),
    }
    return json.dumps(payload, default=str)


def _deserialize(raw: str) -> dict | None:
    """Parse cached payload. Returns None if version mismatch or invalid."""
    try:
        data = json.loads(raw)
        if data.get("v") != CACHE_SCHEMA_VERSION:
            return None
        return {"username": data["username"], "plants": data["plants"]}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def get_cached(username: str, profile: dict, k: int, use_rerank: bool) -> dict | None:
    """
    Get cached recommendations if available and schema version matches.
    Returns None on cache miss or version mismatch.
    """
    client = _get_redis()
    if not client:
        return None
    key = _cache_key(username, profile, k, use_rerank)
    try:
        raw = client.get(key)
        if not raw:
            return None
        return _deserialize(raw)
    except Exception as e:
        logger.warning("Redis get failed: %s", e)
        return None


def set_cached(
    username: str,
    profile: dict,
    result: dict,
    k: int,
    use_rerank: bool,
    ttl: int = DEFAULT_TTL,
) -> bool:
    """
    Store recommendations in Redis with TTL.
    Returns True on success, False on failure.
    """
    client = _get_redis()
    if not client:
        return False
    key = _cache_key(username, profile, k, use_rerank)
    try:
        client.setex(key, ttl, _serialize(result))
        return True
    except Exception as e:
        logger.warning("Redis set failed: %s", e)
        return False


def inspect_cache() -> dict:
    """
    Inspect Redis cache: list keys, count, TTL, and sample value.
    Returns dict with keys, count, and optional sample. For debugging.
    """
    client = _get_redis()
    if not client:
        return {"status": "redis_unavailable", "keys": [], "count": 0}
    try:
        keys = client.keys(f"{KEY_PREFIX}*")
        count = len(keys)
        sample = None
        if keys:
            first_key = keys[0]
            ttl = client.ttl(first_key)
            raw = client.get(first_key)
            if raw:
                sample = _deserialize(raw)
                if sample:
                    sample = {"key": first_key, "ttl_seconds": ttl, "username": sample.get("username"), "plants_count": len(sample.get("plants", []))}
        return {
            "status": "ok",
            "keys": keys[:20],
            "count": count,
            "sample": sample,
        }
    except Exception as e:
        logger.warning("Redis inspect failed: %s", e)
        return {"status": "error", "error": str(e), "keys": [], "count": 0}
