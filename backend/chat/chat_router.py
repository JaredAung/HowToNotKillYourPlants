"""
Chat API. Initializes graph state with user_profile, selected_plant, recommended_plants.
"""
import json

from fastapi import APIRouter, Depends, HTTPException

from auth.jwt import get_current_username
from chat.chat import invoke_chat
from database import get_user_collection
from recommend.recommend import recommend_for_profile

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_user_profile(username: str) -> dict:
    """Fetch user profile from DB (same shape as GET /profile)."""
    user_coll = get_user_collection()
    user = user_coll.find_one({"auth.username": username}) or user_coll.find_one(
        {"auth.email": username.lower()}
    )
    if not user:
        return {}
    auth = user.get("auth", {}) or {}
    return {
        "username": auth.get("username") or username,
        "profile": user.get("profile", {}) or {},
        "location": user.get("location", {}) or {},
        "environment": user.get("environment", {}) or {},
        "climate": user.get("climate"),
        "safety": user.get("safety", {}) or {},
        "constraints": user.get("constraints", {}) or {},
        "preferences": user.get("preferences", {}) or {},
    }


@router.post("/invoke")
def chat_invoke(
    body: dict,
    username: str = Depends(get_current_username),
):
    """
    Run the chat graph. Initializes state with user_profile, selected_plant, top 5 recommendations.
    Body: { "messages": [{"role": "user"|"assistant", "content": "..."}], "selected_plant"?: {...}, "recommended_plants"?: [...] }
    """
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    # Normalize messages to {role, content}
    normalized = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", m.get("text", ""))
            normalized.append({"role": role, "content": str(content)})
        else:
            raise HTTPException(status_code=400, detail="Each message must be {role, content}")

    # Initialize state: user_profile, selected_plant, recommended_plants
    user_profile = _get_user_profile(username)
    selected_plant = body.get("selected_plant")
    recommended_plants = body.get("recommended_plants")

    print("[CHAT_ROUTER] body.selected_plant:", json.dumps(selected_plant, default=str))
    print("[CHAT_ROUTER] body.recommended_plants count:", len(recommended_plants) if recommended_plants else 0)
    print("[CHAT_ROUTER] user_profile keys:", list(user_profile.keys()) if user_profile else [])

    if recommended_plants is None:
        user_coll = get_user_collection()
        user = user_coll.find_one({"auth.username": username}) or user_coll.find_one(
            {"auth.email": username.lower()}
        )
        if user:
            rec = recommend_for_profile(user, username, k=5)
            recommended_plants = rec.get("plants", [])[:5]
        else:
            recommended_plants = []

    result = invoke_chat(
        normalized,
        config={"configurable": {"thread_id": username}},
        user_id=username,
        user_profile=user_profile,
        selected_plant=selected_plant,
        recommended_plants=recommended_plants,
    )

    raw = result.get("messages", [])
    out_messages = []
    for m in raw:
        if isinstance(m, dict):
            out_messages.append({"role": m.get("role", "assistant"), "content": m.get("content", "")})
        elif hasattr(m, "content"):
            role = "assistant" if getattr(m, "type", "").lower() in ("ai", "assistant") else "user"
            out_messages.append({"role": role, "content": str(getattr(m, "content", ""))})
        else:
            out_messages.append({"role": "assistant", "content": str(m)})

    resp = {"messages": out_messages, "phase": result.get("phase")}
    print("[CHAT_ROUTER] response messages count:", len(out_messages))
    print("[CHAT_ROUTER] last assistant content:", (out_messages[-1].get("content", "")[:200] + "...") if out_messages and out_messages[-1].get("content") else "(none)")
    return resp
