"""
Chat graph nodes. Each node handles a phase and returns state updates.
"""
import json
import logging

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from chat.agent_tools import _flatten_plant, retrieve_plant_profile, tavily_search
from chat.chat import State
from llm import ollama_llm

logger = logging.getLogger(__name__)

DEBUG_PREFIX = "[EXPAND_DEBUG]"


def _debug(msg: str, data: object = None) -> None:
    """Print debug info to stdout (visible in uvicorn terminal)."""
    print(f"{DEBUG_PREFIX} {msg}")
    if data is not None:
        if isinstance(data, (dict, list)):
            print(f"{DEBUG_PREFIX}   {json.dumps(data, default=str, indent=2)}")
        else:
            print(f"{DEBUG_PREFIX}   {data}")


def _last_user_content(messages: list) -> str:
    """Extract the last user message content."""
    for m in reversed(messages or []):
        if isinstance(m, dict):
            if m.get("role") in ("user", "human"):
                return m.get("content", "") or ""
        elif isinstance(m, HumanMessage):
            return getattr(m, "content", "") or ""
        elif hasattr(m, "type") and getattr(m, "type", "") in ("human", "user"):
            return getattr(m, "content", "") or ""
    return ""


def _extract_explore_query(user_content: str, selected_query: str | None) -> str | None:
    """
    LLM extracts plant name, search query, or selected_plant.
    Returns 'selected_plant' when user wants details on the plant they selected.
    """
    _debug("_extract_explore_query INPUT", {"user_content": user_content, "selected_query": selected_query})
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract the plant name or search query from the user's message.
Examples:
- "I want to know more about the selected plant" -> selected_plant
- "tell me more about this one" -> selected_plant
- "show me snake plants" -> snake plant
- "what about monstera?" -> monstera
- "I want something for low light" -> low light

Reply with ONLY the plant name, search query, or "selected_plant". If unclear, return selected_plant."""),
        ("human", "{message}"),
    ])
    response = (extract_prompt | ollama_llm).invoke({"message": user_content})
    raw = getattr(response, "content", str(response)) or ""
    extracted = raw.strip()
    _debug("_extract_explore_query LLM raw response", raw)
    if not extracted:
        result = selected_query
    elif "selected_plant" in extracted.lower():
        result = selected_query if selected_query else user_content.strip() or None
    else:
        result = extracted
    _debug("_extract_explore_query OUTPUT (result)", result)
    return result


def _resolve_plant_query(state: State, user_content: str) -> tuple[str | None, bool]:
    """
    Determine which plant to use. Returns (query_or_none, use_selected).
    use_selected=True: use state.selected_plant (no DB call). use_selected=False: call retrieve_plant_profile.
    """
    selected = state.get("selected_plant") or {}
    _debug("_resolve_plant_query state.selected_plant", selected)
    if isinstance(selected, dict):
        selected_query = (
            selected.get("common_name")
            or selected.get("latin")
            or (str(selected["plant_id"]) if selected.get("plant_id") is not None else None)
        )
        selected_query = str(selected_query).strip() if selected_query else None
    else:
        selected_query = None
    _debug("_resolve_plant_query selected_query (from selected_plant)", selected_query)

    if not user_content.strip():
        out = (selected_query, bool(selected_query))
        _debug("_resolve_plant_query OUTPUT (no user content)", {"search_query": out[0], "use_selected": out[1]})
        return out

    result = _extract_explore_query(user_content, selected_query)
    use_selected = bool(result == selected_query and selected_query)
    _debug("_resolve_plant_query OUTPUT", {"search_query": result, "use_selected": use_selected})
    return (result, use_selected)


PLANT_EXCLUDE_KEYS = frozenset({"img_url", "rerank_score", "plant_id"})


def _format_user_profile_for_llm(profile: dict | None) -> str:
    """Extract limited user profile: name, environment, climate, time_commit, preferences, experience_level."""
    if not profile:
        return "(No user profile)"
    parts = []
    p = profile.get("profile") or {}
    if p.get("name"):
        parts.append(f"  name: {p['name']}")
    env = profile.get("environment") or {}
    if env:
        env_str = ", ".join(f"{k}: {v}" for k, v in env.items() if v is not None and not isinstance(v, dict))
        if env_str:
            parts.append(f"  environment: {env_str}")
        temp = env.get("temperature_pref") or {}
        if temp and (temp.get("min_f") is not None or temp.get("max_f") is not None):
            parts.append(f"  temp_pref: {temp.get('min_f')}-{temp.get('max_f')}°F")
    if profile.get("climate"):
        parts.append(f"  climate: {profile['climate']}")
    constraints = profile.get("constraints") or {}
    if constraints.get("time_commit"):
        parts.append(f"  time_commit: {constraints['time_commit']}")
    prefs = profile.get("preferences") or {}
    if prefs.get("experience_level"):
        parts.append(f"  experience_level: {prefs['experience_level']}")
    care_prefs = prefs.get("care_preferences") or {}
    if care_prefs:
        care_str = ", ".join(f"{k}: {v}" for k, v in care_prefs.items() if v is not None)
        if care_str:
            parts.append(f"  preferences: {care_str}")
    return "\n".join(parts) if parts else "(No user profile)"


def _eval_missing_info_and_tavily_query(user_query: str, plant_results: str, search_query: str) -> str | None:
    """
    Separate LLM call: evaluate if the user's query asks for info not in the plant profile.
    If missing, return a Tavily search query. Otherwise return None.
    """
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You evaluate whether the user's question can be fully answered from the plant profile alone.

Given:
- User's question
- Plant profile (from our database)

Output a web search query ONLY if the user is asking for something likely missing from the profile, e.g.:
- Specific care tips (repotting, pruning, propagation)
- Troubleshooting (yellow leaves, pests, diseases)
- Recent tips, "how to" guides, or expert advice
- Symbolism, history, or cultural info not in the profile

If the profile already has enough info to answer, reply with exactly: NONE

If a web search would help, reply with a short search query (3-8 words) combining plant name + the missing topic. E.g. "Anthurium superbum repotting guide" or "Monstera yellow leaves fix".

Reply with ONLY "NONE" or the search query, nothing else."""),
        ("human", "User asked: {user_query}\n\nPlant (search: {search_query}):\n{plant_results}"),
    ])
    response = (eval_prompt | ollama_llm).invoke({
        "user_query": user_query,
        "search_query": search_query,
        "plant_results": plant_results[:2000],
    })
    raw = (getattr(response, "content", str(response)) or "").strip()
    if not raw or len(raw) < 5:
        return None
    if raw.upper().startswith("NONE") or raw.upper() == "NONE":
        return None
    return raw if len(raw) > 10 else None


def _format_tavily_results(results: list) -> str:
    """Format Tavily result list for LLM consumption."""
    parts = []
    for i, r in enumerate(results[:5]):
        if isinstance(r, dict):
            title = r.get("title", r.get("name", ""))
            content = r.get("content", r.get("snippet", r.get("raw_content", "")))
            url = r.get("url", "")
            content_str = str(content)[:600] + ("..." if len(str(content)) > 600 else "")
            parts.append(f"[{i + 1}] {title}\n{content_str}\nSource: {url}")
        else:
            parts.append(str(r)[:600])
    return "\n\n".join(parts) if parts else ""


def _fetch_tavily(query: str) -> str:
    """Run Tavily search and return formatted results for the explainer."""
    try:
        result = tavily_search.invoke(query)
        if isinstance(result, str):
            return result[:3000]
        if isinstance(result, list):
            return _format_tavily_results(result)
        if isinstance(result, dict) and "results" in result:
            return _format_tavily_results(result["results"])
        return str(result)[:2000]
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return ""


def _format_plant_for_display(plant: dict) -> str:
    """Format selected_plant for LLM display. Excludes null, img_url, rerank_score, plant_id."""
    if not isinstance(plant, dict):
        return str(plant)
    flat = _flatten_plant(plant)
    # _flatten_plant expects MongoDB structure; PlantRec from frontend is flat
    if not flat or len(flat) <= 2:
        flat = {k: v for k, v in plant.items() if v is not None and not isinstance(v, dict)}
    filtered = {k: v for k, v in flat.items() if v is not None and k not in PLANT_EXCLUDE_KEYS}
    return "\n".join(f"  {k}: {v}" for k, v in filtered.items()) or str(plant)


def expand_agent(state: State) -> dict:
    """
    Handle EXPAND phase: user exploring, browsing, or discovering plants.
    Uses selected_plant from state when user wants that; otherwise fetches from DB via retrieve_plant_profile.
    """
    _debug("=== EXPAND_AGENT START ===")
    _debug("state.user_id", state.get("user_id"))
    _debug("state.user_profile", state.get("user_profile"))
    _debug("state.selected_plant", state.get("selected_plant"))
    _debug("state.recommended_plants (count)", len(state.get("recommended_plants") or []))
    _debug("state.phase", state.get("phase"))

    messages = list(state.get("messages") or [])
    user_content = _last_user_content(messages)
    _debug("user_content (last user message)", user_content)

    if not user_content.strip():
        reply = "What kind of plants are you looking for? For example: easy care, low light, bathroom plants, or something specific like monstera."
    else:
        # 1. Determine which plant: selected_plant or explore query
        search_query, use_selected = _resolve_plant_query(state, user_content)

        if not search_query:
            reply = "Which plant would you like to know more about? You can name one (e.g. monstera, snake plant) or describe what you're looking for."
        else:
            # 2. Get plant data: from state (selected) or DB (explore)
            if use_selected:
                plant_results = _format_plant_for_display(state.get("selected_plant") or {})
            else:
                plant_results = retrieve_plant_profile(
                    search_query=search_query,
                    limit=1,
                    include_plant_id=False,
                    include_info=True,
                    include_care=True,
                    include_img_url=False,
                )

            # 3. Eval: is any requested info missing? If so, run Tavily
            tavily_query = _eval_missing_info_and_tavily_query(user_content, plant_results, search_query)
            tavily_results = ""
            if tavily_query:
                _debug("eval: Tavily query", tavily_query)
                tavily_results = _fetch_tavily(tavily_query)
                _debug("eval: Tavily results length", len(tavily_results))
            else:
                _debug("eval: no Tavily query (profile sufficient)")

            # 4. LLM explainer: plant + user profile + optional Tavily
            user_profile_str = _format_user_profile_for_llm(state.get("user_profile"))
            _debug("plant_results passed to explainer LLM", {"use_selected": use_selected, "plant_results": plant_results[:300]})
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful plant care assistant. First verify: does the plant profile match what the user asked for?

- If YES (plant data matches the query): Write a friendly, concise response (2-4 sentences) that acknowledges what they want, answer their query (describe the plant/answer their question), and invites them to ask for more or compare. Keep it warm. Mention the plant name (common_name), how to care for it and why they might like it. Use the user's profile (environment, experience_level, time_commit) to personalize when relevant. If web search results are provided, use them to enrich your answer when they add useful detail.
- If NO (no results, wrong plant, empty data, mismatch): Write a short message politely asking the user to clarify or try a different plant name/search.

Reply with only the final message to the user."""),
                ("human", "User profile:\n{user_profile}\n\nUser asked: {user_query}\n\nSearch/context: {search_query}\n\nPlant results:\n{plant_results}\n\n{web_search_section}"),
            ])
            web_section = f"Web search results (use to enrich if relevant):\n{tavily_results}" if tavily_results else ""
            response = (prompt | ollama_llm).invoke({
                "user_profile": user_profile_str,
                "user_query": user_content,
                "search_query": search_query,
                "plant_results": plant_results,
                "web_search_section": web_section,
            })
            reply = getattr(response, "content", str(response)) or plant_results

    new_msg = {"role": "assistant", "content": reply}
    out = {
        "messages": messages + [new_msg],
        "phase": "EXPAND",
    }
    _debug("=== EXPAND_AGENT END ===")
    _debug("reply (assistant message)", reply[:500] + "..." if len(reply) > 500 else reply)
    _debug("out.messages count", len(out["messages"]))
    _debug("out.phase", out["phase"])
    return out


def _placeholder_agent(phase: str):
    """Return a placeholder node for unimplemented phases."""

    def node(state: State) -> dict:
        messages = list(state.get("messages") or [])
        reply = f"[{phase} phase - coming soon] How can I help?"
        return {
            "messages": messages + [{"role": "assistant", "content": reply}],
            "phase": phase,
        }

    return node
