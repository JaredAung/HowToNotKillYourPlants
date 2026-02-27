"""
LLM-based phase router for the chat graph.
Routes conversation to the appropriate phase: EXPAND, COMPARE, SHOP, GUIDE, PICK.
"""
import re
from typing import Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from chat.chat import State
from llm import ollama_llm

PHASES: tuple[str, ...] = ("EXPAND", "COMPARE", "SHOP", "GUIDE", "PICK")
Phase = Literal["EXPAND", "COMPARE", "SHOP", "GUIDE", "PICK"]

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a router for a plant care app. Classify the user's intent into exactly one phase.

Phases:
- EXPAND: User is exploring or learning about a plant. Includes: "tell me more", "convince me about this plant", "why is this a good fit for me" When they want to understand a plant better → EXPAND.
- COMPARE: User wants to compare two or more plants. E.g. "compare monstera and pothos", "which is easier, snake plant or zz plant"
- SHOP: User wants to buy, add to cart, or get a plant kit. E.g. "add to cart", "I want to buy this", "get me a starter kit"
- PICK: User wants to proceed with adding the selected plant to their garden. E.g. "add to garden", "I'll take it", "add this one", "Select this plant", "yes add it", "let's do it". Proceeding with the pick/add action → PICK.

Reply with ONLY the phase name, nothing else. One word: EXPAND, COMPARE, SHOP, or PICK."""),
    ("human", "{context}"),
])


def _messages_to_context(messages: list) -> str:
    """Convert message list to a string for the router."""
    if not messages:
        return "(No messages yet)"
    parts = []
    for m in messages[-10:]:  # Last 10 messages
        if isinstance(m, BaseMessage):
            role = m.type if hasattr(m, "type") else getattr(m, "type", "user")
            content = getattr(m, "content", str(m)) or ""
        elif isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "") or ""
        else:
            continue
        role_label = "User" if role in ("human", "user") else "Assistant"
        parts.append(f"{role_label}: {content[:500]}")
    return "\n\n".join(parts) if parts else "(Empty)"


def route_to_phase(state: State) -> str:
    """
    Use LLM to classify the conversation and return the phase to route to.
    Returns one of: EXPAND, COMPARE, SHOP, GUIDE, PICK.
    """
    messages = state.get("messages") or []
    context = _messages_to_context(messages)
    selected = state.get("selected_plant")
    if selected:
        plant_name = selected.get("common_name") or selected.get("latin") or "a plant"
        context = f"[User has selected: {plant_name}]\n\n{context}"
    print("[ROUTER] state.selected_plant:", bool(selected))
    print("[ROUTER] context (last msgs):", context[:500] if context else "(empty)")

    chain = ROUTER_PROMPT | ollama_llm
    response = chain.invoke({"context": context})
    content = getattr(response, "content", str(response)).strip().upper()

    # Parse: take first word that matches a phase
    for phase in PHASES:
        if phase in content or content == phase:
            print("[ROUTER] intent/phase:", phase, "| LLM raw:", repr(content[:100]))
            return phase

    # Fallback: try regex for phase name
    match = re.search(r"\b(EXPAND|COMPARE|SHOP|GUIDE|PICK)\b", content, re.I)
    if match:
        phase = match.group(1).upper()
        print("[ROUTER] intent/phase:", phase)
        return phase

    print("[ROUTER] intent/phase: EXPAND (default)")
    return "EXPAND"  # Default


def get_router():
    """Return the route function for LangGraph conditional edges."""
    return route_to_phase


# Usage with LangGraph:
#
# from langgraph.graph import StateGraph, START, END
# from chat.chat import State
# from chat.router import route_to_phase
#
# workflow = StateGraph(State)
# workflow.add_node("expand", expand_node)
# workflow.add_node("compare", compare_node)
# workflow.add_node("shop", shop_node)
# workflow.add_node("guide", guide_node)
# workflow.add_node("pick", pick_node)
#
# workflow.add_conditional_edges(START, route_to_phase, {
#     "EXPAND": "expand",
#     "COMPARE": "compare",
#     "SHOP": "shop",
#     "GUIDE": "guide",
#     "PICK": "pick",
# })
