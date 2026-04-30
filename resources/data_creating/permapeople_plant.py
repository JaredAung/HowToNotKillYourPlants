"""
Permapeople API client for fetching plant data.

Uses PERMAPEOPLE_KEY_ID and PERMAPEOPLE_KEY_SECRET from environment.
API docs: https://permapeople.org/knowledgebase/api-docs.html
"""
import os
from typing import Any

import requests

BASE_URL = "https://permapeople.org/api"


def _headers() -> dict[str, str]:
    key_id = os.environ.get("PERMAPEOPLE_KEY_ID")
    key_secret = os.environ.get("PERMAPEOPLE_KEY_SECRET")
    if not key_id or not key_secret:
        raise ValueError(
            "PERMAPEOPLE_KEY_ID and PERMAPEOPLE_KEY_SECRET must be set in environment"
        )
    return {
        "x-permapeople-key-id": key_id,
        "x-permapeople-key-secret": key_secret,
        "Accept": "application/json",
    }


def list_plants(*, last_id: int | None = None) -> list[dict[str, Any]]:
    """
    Fetch one page of plants (up to 100) from the Permapeople API.

    Args:
        last_id: Cursor for pagination; returns only plants after this ID.

    Returns:
        List of plant objects.
    """
    params = {}
    if last_id is not None:
        params["last_id"] = last_id
    r = requests.get(
        f"{BASE_URL}/plants",
        headers=_headers(),
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("plants", [])


def fetch_all_plants() -> list[dict[str, Any]]:
    """
    Paginate through the full plant list until no more plants are returned.

    Returns:
        List of all plant objects.
    """
    all_plants: list[dict[str, Any]] = []
    last_id: int | None = None
    while True:
        batch = list_plants(last_id=last_id)
        if not batch:
            break
        all_plants.extend(batch)
        last_id = batch[-1].get("id") if batch else None
        if last_id is None:
            break
    return all_plants
