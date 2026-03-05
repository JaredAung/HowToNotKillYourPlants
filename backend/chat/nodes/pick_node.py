"""
Pick phase: add selected plant to user's garden.
"""
from datetime import datetime, timezone

from chat.chat import State
from database import get_garden_collection, get_plant_collection


def _compute_match_percentage(selected: dict, recommended_plants: list[dict]) -> int | None:
    """
    Compute match percentage from recommendation pipeline.
    Uses rerank_score if present, else score normalized by top plant's score.
    """
    if not selected:
        return None
    rerank = selected.get("rerank_score")
    if rerank is not None:
        return round(float(rerank) * 100)
    score = selected.get("score")
    if score is None or not recommended_plants:
        return None
    top_score = max((p.get("score") or 0) for p in recommended_plants)
    if top_score <= 0:
        return None
    return round((float(score) / float(top_score)) * 100)


def pick_agent(state: State) -> dict:
    """
    Add selected plant to user's garden when routed to PICK.
    Includes: username, plant_id, added_at, match_percentage (from rec pipeline).
    """
    messages = list(state.get("messages") or [])
    selected = state.get("selected_plant") or {}
    username = state.get("user_id") or ""
    recommended_plants = state.get("recommended_plants") or []

    plant_id = selected.get("plant_id")
    if not plant_id or not username:
        reply = "Please select a plant first (tap + on a recommendation card), then say \"add to garden\" or \"I'll take it\"."
    else:
        garden_coll = get_garden_collection()
        plant_coll = get_plant_collection()

        plant = plant_coll.find_one({"plant_id": plant_id})
        info = (plant or {}).get("Info", {}) or {}
        display_name = (
            selected.get("common_name")
            or selected.get("latin")
            or info.get("latin")
            or info.get("common_name")
            or f"Plant #{plant_id}"
        )

        match_percentage = _compute_match_percentage(selected, recommended_plants)

        doc = {
            "username": username,
            "plant_id": plant_id,
            "custom_name": display_name,
            "added_at": datetime.now(timezone.utc),
            "match_percentage": match_percentage,
        }
        garden_coll.insert_one(doc)

        match_str = f" (match: {match_percentage}%)" if match_percentage is not None else ""
        reply = f"Added {display_name} to your garden{match_str}. You can view it in your garden."

    new_msg = {"role": "assistant", "content": reply}
    return {
        "messages": messages + [new_msg],
        "phase": "PICK",
    }
