"""
Recommendation tool for the chat LLM: wraps the same pipeline as ``GET /recommend/`` in
``recommend.recommend`` (two-tower embedding, MongoDB vector search, Cohere rerank).
"""
from __future__ import annotations

import logging
import os

from langchain_core.tools import tool

from database import get_user_collection
from recommend.recommend import DEFAULT_TOP_K, _format_plant_for_llm, recommend_for_profile

logger = logging.getLogger(__name__)

# Pipeline may retrieve DEFAULT_TOP_K (~20) for rerank/quality; only this many go to the LLM / tool text.
LLM_RECOMMEND_TOP_N = 5


def annotate_recommendation_ranks(plants: list) -> list[dict]:
    """
    Shallow-copy each plant dict and set ``recommend_rank`` to 1..n in **list order**.

    The pipeline already returns plants best-match-first (vector search / rerank).
    **Do not sort** this list—index 0 is rank 1 (top), the last index is the bottom of this batch.
    Enables references like \"third recommendation\", \"top 2\", \"last two\".
    """
    out: list[dict] = []
    for i, p in enumerate(plants or []):
        if not isinstance(p, dict):
            continue
        q = dict(p)
        q["recommend_rank"] = i + 1
        out.append(q)
    return out


def run_recommendation_pipeline(
    username: str,
    *,
    top_k: int | None = None,
    use_rerank: bool | None = None,
) -> dict:
    """
    Same data path as :func:`recommend.recommend.get_recommendations` (full Mongo user doc +
    :func:`recommend_for_profile`). Does not use Redis cache (chat tool path).

    Returns ``{"username", "plants", "message"?}`` from the recommender.
    """
    k = top_k if top_k is not None else DEFAULT_TOP_K
    k = max(1, min(int(k), 50))

    user_coll = get_user_collection()
    user = user_coll.find_one({"auth.username": username}) or user_coll.find_one(
        {"auth.email": (username or "").lower()}
    )
    if not user:
        return {"username": username, "plants": [], "message": f"User '{username}' not found."}

    if use_rerank is None:
        use_rerank = os.getenv("USE_RERANK", "true").lower() in ("true", "1", "yes")

    try:
        out = recommend_for_profile(user, username, k=k, use_rerank=use_rerank)
    except Exception as e:
        logger.exception("recommend_for_profile failed for %s", username)
        return {"username": username, "plants": [], "message": f"Recommendation failed: {e}"}

    if not out.get("plants"):
        out.setdefault(
            "message",
            "No plants with embeddings in database. Run upload / training pipeline first.",
        )
    return out


def format_recommendations_for_llm(result: dict) -> str:
    """Turn pipeline output into a concise string for tool return / LLM context."""
    plants = (result.get("plants") or [])[:LLM_RECOMMEND_TOP_N]
    msg = result.get("message")
    if msg and not plants:
        return msg
    lines: list[str] = []
    if msg:
        lines.append(msg)
    if not plants:
        return "\n".join(lines) if lines else "No recommendations returned."
    lines.append(
        "Order is fixed: rank 1 = best match, rank 2 = second, … last = bottom of this list. "
        "Use these positions when the user says top 2, third, bottom, etc."
    )
    lines.append(f"Top {len(plants)} personalized matches (two-tower + vector search):")
    for i, p in enumerate(plants, 1):
        lines.append(f"Rank {i}: {_format_plant_for_llm(p)}")
        if p.get("score") is not None:
            lines.append(f"   score: {p['score']}")
    return "\n".join(lines)


@tool
def recommend_plants_for_user(username: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Get personalized plant recommendations for a user.

    Uses the production pipeline: user profile from the database, two-tower user embedding,
    MongoDB vector search on plant_tower_embedding, optional Cohere reranking.

    Call when the user asks what plants suit them, wants suggestions for their home, or asks
    for recommendations similar to the home page. Requires the correct username (same as auth).

    Args:
        username: Authenticated user's username (must exist in the user collection).
        top_k: How many candidates the pipeline scores (default 20, max 50). Only the top 5 are returned in the tool text for the assistant.
    """
    k = max(1, min(int(top_k), 50))
    result = run_recommendation_pipeline(username, top_k=k)
    top = (result.get("plants") or [])[:LLM_RECOMMEND_TOP_N]
    ranked = annotate_recommendation_ranks(top)
    return format_recommendations_for_llm({**result, "plants": ranked})


# --- Recommend phase agent (LLM + bound tool) ---

_RECOMMEND_SYSTEM = """You are a friendly plant care assistant in the RECOMMEND phase.
When the user wants personalized plant ideas—what to grow, what fits their home, or suggestions
like on the home page—call the tool `recommend_plants_for_user` with their username (provided below)
and optionally `top_k` (how many candidates the pipeline considers, default 20, max 50). The tool only lists the top 5 matches for you to discuss.
After tool results return, write a short, warm reply: highlight a few standouts, invite follow-up.
If the user is only chatting casually, you may answer without the tool unless they need fresh rankings."""


def _dict_messages_to_lc(messages: list):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    out = []
    for m in messages or []:
        if isinstance(m, dict):
            role = (m.get("role") or "user").lower()
            content = str(m.get("content", m.get("text", "")) or "")
            if role == "system":
                out.append(SystemMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
    return out


def recommend_agent(state: dict) -> dict:
    """
    RECOMMEND phase: LLM with ``recommend_plants_for_user`` bound. Injects ``user_id`` on tool calls.
    Updates ``recommended_plants`` in state when the pipeline runs (top 5 for UI).
    """
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    from llm import gemini_llm

    messages = list(state.get("messages") or [])
    user_id = (state.get("user_id") or "").strip()
    # Normalize ranks by current list order (best-first from last retrieval).
    prior_recs = annotate_recommendation_ranks(list(state.get("recommended_plants") or []))

    if not user_id:
        reply = "Sign in to get personalized plant recommendations matched to your profile."
        return {
            "messages": messages + [{"role": "assistant", "content": reply}],
            "phase": "RECOMMEND",
        }

    lc_msgs = _dict_messages_to_lc(messages)
    system = SystemMessage(
        content=_RECOMMEND_SYSTEM + f"\n\nAuthenticated username (use for recommend_plants_for_user): {user_id}"
    )

    llm_tools = gemini_llm.bind_tools([recommend_plants_for_user])

    try:
        ai = llm_tools.invoke([system, *lc_msgs])
    except Exception as e:
        logger.exception("recommend_agent first LLM call failed")
        return {
            "messages": messages + [{"role": "assistant", "content": f"Something went wrong: {e}"}],
            "phase": "RECOMMEND",
        }

    tool_calls = getattr(ai, "tool_calls", None) or []
    new_recommended: list[dict] = prior_recs
    pipeline_updated = False

    if tool_calls:
        tool_msgs: list[ToolMessage] = []
        for tc in tool_calls:
            tid = tc.get("id") or "tool_call"
            name = tc.get("name") or ""
            args = dict(tc.get("args") or {})
            if name == "recommend_plants_for_user":
                args["username"] = user_id
                k = max(1, min(int(args.get("top_k", DEFAULT_TOP_K)), 50))
                try:
                    rec = run_recommendation_pipeline(user_id, top_k=k)
                    ranked = annotate_recommendation_ranks((rec.get("plants") or [])[:LLM_RECOMMEND_TOP_N])
                    rec_for_display = {**rec, "plants": ranked}
                    text = format_recommendations_for_llm(rec_for_display)
                    new_recommended = ranked
                    pipeline_updated = True
                except Exception as e:
                    logger.exception("recommend pipeline in agent")
                    text = f"Recommendation failed: {e}"
                tool_msgs.append(ToolMessage(content=text, tool_call_id=tid))
            else:
                tool_msgs.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=tid))

        try:
            final = gemini_llm.invoke([system, *lc_msgs, ai, *tool_msgs])
            reply = getattr(final, "content", None) or str(final)
        except Exception as e:
            logger.exception("recommend_agent second LLM call failed")
            reply = tool_msgs[0].content if tool_msgs else str(e)
    else:
        reply = getattr(ai, "content", None) or ""
        if not str(reply).strip():
            reply = (
                "I can pull personalized picks from your profile—say what you’re looking for "
                "(e.g. low light, easy care, pet-safe), or ask me to recommend plants for you."
            )

    out: dict = {
        "messages": messages + [{"role": "assistant", "content": reply}],
        "phase": "RECOMMEND",
    }
    if pipeline_updated:
        out["recommended_plants"] = new_recommended
    elif prior_recs:
        out["recommended_plants"] = prior_recs
    return out
