"""
Compare phase: compare two or more plants.
"""
import json
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from chat.agent_tools import retrieve_plant_profile, tavily_search
from chat.chat import State
from llm import ollama_llm

logger = logging.getLogger(__name__)

DEBUG_PREFIX = "[COMPARE_DEBUG]"


def _debug(msg: str, data: object = None) -> None:
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


def _format_user_profile_for_llm(profile: dict | None) -> str:
    """Extract limited user profile for personalization."""
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


def _format_plant_for_display(plant: dict) -> str:
    """Format flat plant dict for LLM. Excludes img_url, rerank_score, plant_id."""
    if not isinstance(plant, dict):
        return str(plant)
    excluded = {"img_url", "rerank_score", "plant_id"}
    filtered = {k: v for k, v in plant.items() if v is not None and k not in excluded}
    return "\n".join(f"  {k}: {v}" for k, v in filtered.items()) or str(plant)


def _extract_plant_names_and_focus_for_compare(
    user_content: str,
    selected_plant: dict | None,
) -> tuple[list[str], str | None]:
    """
    LLM extracts 2+ plant names and optional focus of comparison.
    Use "selected" for "this plant" when user has a selected plant. Otherwise names only.
    """
    selected_context = ""
    if selected_plant:
        name = selected_plant.get("common_name") or selected_plant.get("latin") or "selected plant"
        selected_context = f"\nUser has selected: {name}. Use 'selected' for this plant (e.g. 'compare this plant with monstera' -> selected, monstera)."
    else:
        selected_context = "\nNo plant selected. Use explicit names only."

    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract from the user's comparison request:

1. PLANTS: 2+ plant names or references, one per line.
   - "compare monstera and pothos" -> monstera, pothos
   - "snake plant vs zz plant" -> snake plant, zz plant
   - "compare this plant with monstera" -> selected, monstera (when user has a selected plant)
   - "this one vs pothos" -> selected, pothos
   - "fiddle leaf fig versus rubber plant" -> fiddle leaf fig, rubber plant
   {selected_context}

2. FOCUS: What aspect to compare? E.g. ease of care, watering needs, edibility, toxicity.
   - "compare X and Y for ease of care" -> ease of care
   - "in terms of edibility" -> edibility
   - No focus -> None

Output format, exactly:
PLANTS:
plant1
plant2
FOCUS: <focus or None>"""),
        ("human", "User: {message}"),
    ])
    response = (extract_prompt | ollama_llm).invoke({
        "message": user_content,
        "selected_context": selected_context or "",
    })
    raw = (getattr(response, "content", str(response)) or "").strip()
    _debug("_extract_plant_names_and_focus raw", raw)

    plant_refs = []
    focus: str | None = None

    in_plants = True
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("FOCUS:"):
            in_plants = False
            rest = line[6:].strip()
            if rest and rest.upper() != "NONE":
                focus = rest
            continue
        if in_plants and not line.upper().startswith("PLANTS"):
            # Split on comma or " and " in case LLM puts multiple plants on one line
            for part in re.split(r",\s*|\s+and\s+", line):
                part = part.strip()
                if part:
                    key = part.lower()
                    if key not in {ln.lower() for ln in plant_refs}:
                        plant_refs.append(part)

    # Dedupe
    seen = set()
    result = []
    for ln in plant_refs:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            result.append(ln)

    _debug("plant_names", result)
    _debug("focus", focus)
    return (result[:10], focus)


def _db_search_plant(query: str) -> str | None:
    """
    Search DB for plant. Tries full query, then word variants.
    Returns profile string or None if not found.
    """
    def _try(q: str) -> str:
        return retrieve_plant_profile(
            search_query=q,
            limit=1,
            include_plant_id=False,
            include_info=True,
            include_care=True,
            include_img_url=False,
        )

    result = _try(query)
    _debug("DB search", f"query='{query}' -> {'found' if (result and 'No plants found' not in result) else 'not found'}")
    if result and "No plants found" not in result:
        _debug("DB search result (truncated)", result[:500] + "..." if len(result) > 500 else result)
        return result
    # Try word variants for multi-word names (e.g. "century plant" -> "century")
    words = [w for w in query.split() if len(w) > 1 and w.lower() not in ("the", "a", "an")]
    for w in words:
        if w.lower() == query.lower():
            continue
        result = _try(w)
        _debug("DB search variant", f"query='{w}' -> {'found' if (result and 'No plants found' not in result) else 'not found'}")
        if result and "No plants found" not in result:
            _debug("DB search result (truncated)", result[:500] + "..." if len(result) > 500 else result)
            return result
    _debug("DB search", f"query='{query}' -> no match, will use Tavily")
    return None


def _resolve_plants_for_compare(
    plant_refs: list[str],
    selected_plant: dict | None,
) -> list[tuple[str, str]]:
    """
    Resolve each plant ref to (name, formatted_profile).
    "selected" -> selected_plant. Otherwise: DB first, Tavily fallback.
    """
    results = []
    selected = selected_plant or {}

    for ref in plant_refs:
        ref = ref.strip()
        if not ref:
            continue
        ref_lower = ref.lower()
        # "selected" -> selected_plant
        if ref_lower == "selected" and selected:
            name = selected.get("common_name") or selected.get("latin") or f"Plant #{selected.get('plant_id')}"
            results.append((str(name), _format_plant_for_display(selected)))
            continue
        # By name: DB search first (try query, then word variants), Tavily fallback
        profile_str = _db_search_plant(ref)
        if profile_str:
            name_match = re.search(r"latin:\s*(\S+)", profile_str) or re.search(r"common_name:\s*(.+)", profile_str)
            display_name = name_match.group(1).strip() if name_match else ref
            results.append((display_name, profile_str))
        else:
            _debug("plant not in DB, fetching via Tavily", ref)
            tavily_profile = _fetch_plant_info_via_tavily(ref)
            results.append((ref, tavily_profile))

    return results


def _eval_missing_info_and_tavily_query_for_compare(
    user_query: str,
    focus: str | None,
    plants_text: str,
    plant_names: list[str],
) -> str | None:
    """
    LLM eval: is the comparison info (especially focus) likely missing from our plant profiles?
    If yes, return a Tavily search query. Otherwise return None.
    Query MUST include BOTH plant names for comparison.
    """
    names_str = " and ".join(plant_names[:5]) if plant_names else ""
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You evaluate whether the user's comparison question can be sufficiently answered from the given plant information.

Plant profiles typically have: care_level, light, water, humidity, temp_min, temp_max, sunlight_type, physical_desc, symbolism, climate. These are SUFFICIENT for: environment, care level, light needs, watering, humidity, temperature, climate, ease of care. Reply NONE for these.

Output a web search query ONLY if the user asks for something typically NOT in profiles:
- Edibility, toxicity, poisonous, pet-safe
- Propagation, repotting, pruning, pests, diseases
- Symbolism, history, cultural info (only if not in profile)

If the focus is environment, care, light, water, humidity, temp, or climate → reply with exactly: NONE (profiles have this).

If a web search would help, reply with a short query (5-12 words). CRITICAL: Include BOTH plant names.

Reply with ONLY "NONE" or the search query, nothing else."""),
        ("human", "Plants to compare: {plant_names}\n\nUser asked: {user_query}\n\nFocus of comparison: {focus}\n\nPlant profiles:\n{plants_text}"),
    ])
    response = (eval_prompt | ollama_llm).invoke({
        "user_query": user_query,
        "focus": focus or "(general comparison)",
        "plants_text": plants_text[:2500],
        "plant_names": names_str,
    })
    raw = (getattr(response, "content", str(response)) or "").strip()
    if not raw or len(raw) < 5:
        return None
    if raw.upper().startswith("NONE") or raw.upper() == "NONE":
        return None
    return raw if len(raw) > 8 else None


def _format_tavily_results(results: list) -> str:
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


def _fetch_plant_info_via_tavily(plant_name: str) -> str:
    """
    Look up plant info via Tavily when not in our DB.
    Returns formatted profile string for use in comparison.
    """
    query = f"{plant_name} houseplant care light water humidity characteristics description"
    try:
        raw = _fetch_tavily(query)
        if raw:
            return f"(Web-sourced info for {plant_name}):\n{raw}"
    except Exception as e:
        logger.warning("Tavily plant lookup failed for %s: %s", plant_name, e)
    return f"(No data found for '{plant_name}' in database or web search)"


def compare_agent(state: State) -> dict:
    """
    Handle COMPARE phase: compare two or more plants.
    """
    _debug("=== COMPARE_AGENT START ===")
    messages = list(state.get("messages") or [])
    user_content = _last_user_content(messages)
    _debug("user_content", user_content)

    if not user_content.strip():
        reply = "Which plants would you like to compare? For example: \"compare monstera and pothos\" or \"snake plant vs zz plant\"."
    else:
        # 1. Extract plant names/refs and focus (names + "selected" for this plant)
        selected = state.get("selected_plant")
        plant_refs, focus = _extract_plant_names_and_focus_for_compare(user_content, selected)

        if len(plant_refs) < 2:
            reply = "I need at least two plants to compare. Name them explicitly, e.g. \"compare monstera and pothos\" or \"compare this plant with pothos\" (if you have one selected)."
        elif "selected" in [r.strip().lower() for r in plant_refs] and not selected:
            reply = "To compare 'this plant', select one first from your recommendations, or name the plants explicitly (e.g. \"compare monstera and pothos\")."
        else:
            # 2. Resolve each (selected -> selected_plant; names -> DB + Tavily fallback)
            resolved = _resolve_plants_for_compare(plant_refs, selected)
            _debug("resolved count", len(resolved))

            if not resolved:
                reply = "I couldn't find those plants. Please try different names or check your recommendations."
            else:
                # 3. Build plant profiles for prompt
                plant_sections = []
                for i, (name, profile_str) in enumerate(resolved):
                    plant_sections.append(f"Plant {i + 1} ({name}):\n{profile_str}")

                plants_text = "\n\n---\n\n".join(plant_sections)

                # 4. Eval: is comparison info missing? If so, run Tavily
                tavily_results = ""
                if len(resolved) >= 2:
                    plant_names = [n for n, _ in resolved]
                    tavily_query = _eval_missing_info_and_tavily_query_for_compare(
                        user_content, focus, plants_text, plant_names
                    )
                    # Ensure query includes plant names (fallback if eval returns incomplete query)
                    if tavily_query and plant_names:
                        q_lower = tavily_query.lower()
                        missing = [n for n in plant_names[:5] if n and n.lower() not in q_lower]
                        if missing:
                            tavily_query = " ".join(plant_names[:5]) + f" {focus or 'comparison'}"
                    if tavily_query:
                        _debug("eval: Tavily query", tavily_query)
                        try:
                            tavily_results = _fetch_tavily(tavily_query)
                            if tavily_results:
                                tavily_results = f"Web search (use if relevant):\n{tavily_results}"
                        except Exception:
                            pass
                    else:
                        _debug("eval: no Tavily query (profiles sufficient)")

                # 5. LLM comparison
                user_profile_str = _format_user_profile_for_llm(state.get("user_profile"))
                if focus:
                    focus_section = f"Focus ONLY on: {focus}."
                else:
                    focus_section = """Focus on:
- Care difficulty (easy/medium/hard)
- Light, water, humidity needs
- Which might fit the user's environment (use their profile)
- Pros and cons of each"""
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful plant care assistant. Compare the given plants for the user.

{focus_section}

Plant info may come from our database or from web search (marked as "Web-sourced"). Use both when available. If web search results are provided, incorporate them into your comparison.
Only use information that is explicitly mentioned in the plant profiles or web search results. Do not make up information.
Write a friendly, structured comparison (2-5 short paragraphs or bullet points). Be concise but informative."""),
                    ("human", "User profile:\n{user_profile}\n\nUser asked: {user_query}\n\n{plants_text}\n\n{tavily_section}"),
                ])
                response = (prompt | ollama_llm).invoke({
                    "user_profile": user_profile_str,
                    "user_query": user_content,
                    "plants_text": plants_text,
                    "tavily_section": tavily_results or "",
                    "focus_section": focus_section,
                })
                reply = getattr(response, "content", str(response)) or "I couldn't generate a comparison."

    new_msg = {"role": "assistant", "content": reply}
    out = {
        "messages": messages + [new_msg],
        "phase": "COMPARE",
    }
    _debug("=== COMPARE_AGENT END ===")
    return out
