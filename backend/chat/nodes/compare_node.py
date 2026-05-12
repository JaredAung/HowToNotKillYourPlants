"""
Compare phase: compare two or more plants.
"""
import json
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from chat.agent_tools import format_tower_profile_for_llm, retrieve_pfaff_plant_knowledge, retrieve_plant_profile
from chat.chat import State
from chat.rag_trace import log_rag_retrieval
from llm import gemini_llm

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
    response = (extract_prompt | gemini_llm).invoke({
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
    _debug("DB search", f"query='{query}' -> no catalog match")
    return None


def _resolve_plants_for_compare(
    plant_refs: list[str],
    selected_plant: dict | None,
) -> list[tuple[str, str]]:
    """
    Resolve each plant ref to (name, formatted_profile).
    "selected" -> selected_plant. Otherwise: DB catalog only (no web search).
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
        # By name: DB search only (no web)
        profile_str = _db_search_plant(ref)
        if profile_str:
            name_match = re.search(r"latin:\s*(\S+)", profile_str) or re.search(r"common_name:\s*(.+)", profile_str)
            display_name = name_match.group(1).strip() if name_match else ref
            results.append((display_name, profile_str))
        else:
            _debug("plant not in catalog", ref)
            results.append(
                (
                    ref,
                    f"(No catalog entry for '{ref}'. Try another spelling or pick plants from recommendations.)",
                )
            )

    return results


def _pfaff_excerpts_for_compare(
    resolved: list[tuple[str, str]],
    user_query: str,
    focus: str | None,
    *,
    user_id: str | None = None,
) -> str:
    """PFAF RAG snippets per plant when Latin appears in catalog profile text."""
    qbase = f"{user_query} {focus or ''}".strip()[:500] or "plant comparison"
    parts: list[str] = []
    for _name, profile_str in resolved:
        if "No catalog entry" in profile_str:
            continue
        m = re.search(r"latin:\s*(.+)", profile_str, re.I)
        if not m:
            continue
        latin = m.group(1).strip()
        if not latin:
            continue
        try:
            text = retrieve_pfaff_plant_knowledge(qbase, latin, top_k=3)
        except Exception as e:
            logger.warning("PFAF compare snippet failed for %s: %s", latin, e)
            log_rag_retrieval(
                phase="COMPARE",
                user_query=qbase,
                plant_name=latin,
                top_k=3,
                rag_result="",
                user_id=user_id,
                extra={"error": str(e)},
            )
            continue
        if not text or "Plant knowledge search is unavailable" in text:
            log_rag_retrieval(
                phase="COMPARE",
                user_query=qbase,
                plant_name=latin,
                top_k=3,
                rag_result=text or "",
                user_id=user_id,
                extra={"skipped_empty_or_unavailable": True},
            )
            continue
        log_rag_retrieval(
            phase="COMPARE",
            user_query=qbase,
            plant_name=latin,
            top_k=3,
            rag_result=text,
            user_id=user_id,
        )
        parts.append(f"--- {latin} (PFAF / Plants For A Future) ---\n{text}")
    return "\n\n".join(parts)


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
            # 2. Resolve each (selected -> selected_plant; names -> DB catalog)
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

                # 4. PFAF RAG (per Latin name in catalog profiles)
                pfaff_section = ""
                if len(resolved) >= 1:
                    pfaff_section = _pfaff_excerpts_for_compare(
                        resolved,
                        user_content,
                        focus,
                        user_id=(state.get("user_id") or None),
                    )
                    if pfaff_section:
                        _debug("PFAF excerpts length", len(pfaff_section))

                # 5. LLM comparison
                user_profile_str = format_tower_profile_for_llm(state.get("user_profile"))
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

Use catalog plant profiles and any **PFAF / Plants For A Future** excerpts provided (authoritative for cultivation, uses, habitat). Do not invent facts not supported by the given text.
Write a friendly, structured comparison (2-5 short paragraphs or bullet points). Be concise but informative."""),
                    ("human", "User profile:\n{user_profile}\n\nUser asked: {user_query}\n\n{plants_text}\n\n{pfaff_section}"),
                ])
                response = (prompt | gemini_llm).invoke({
                    "user_profile": user_profile_str,
                    "user_query": user_content,
                    "plants_text": plants_text,
                    "pfaff_section": pfaff_section or "",
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
