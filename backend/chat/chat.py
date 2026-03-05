"""
Chat graph: routes user messages to phase-specific agents.
"""
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict, total=False):
    user_id: str  # username for user context
    user_profile: dict
    phase: Literal["EXPAND", "COMPARE", "SHOP", "GUIDE", "PICK"]
    intent: str
    selected_plant: dict
    recommended_plants: list[dict]  # top 5 recommendations, initialized beforehand

    messages: list  # chat history
    ui_actions: list  # button clicks / confirmed actions


from chat.nodes import compare_agent, expand_agent, pick_agent, _placeholder_agent
from chat.router import route_to_phase


# Build the graph
workflow = StateGraph(State)

# Nodes
workflow.add_node("expand", expand_agent)
workflow.add_node("compare", compare_agent)
workflow.add_node("shop", _placeholder_agent("SHOP"))
workflow.add_node("guide", _placeholder_agent("GUIDE"))
workflow.add_node("pick", pick_agent)

# Router: START -> phase-specific node
workflow.add_conditional_edges(START, route_to_phase, {
    "EXPAND": "expand",
    "COMPARE": "compare",
    "SHOP": "shop",
    "GUIDE": "guide",
    "PICK": "pick",
})

# All phase nodes -> END
workflow.add_edge("expand", END)
workflow.add_edge("compare", END)
workflow.add_edge("shop", END)
workflow.add_edge("guide", END)
workflow.add_edge("pick", END)

# Compile with in-memory checkpointer for thread/conversation support
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


def invoke_chat(
    messages: list,
    config: dict | None = None,
    *,
    user_id: str | None = None,
    user_profile: dict | None = None,
    selected_plant: dict | None = None,
    recommended_plants: list[dict] | None = None,
) -> dict:
    """
    Run the chat graph with the given messages and initial state.
    config can include thread_id for conversation memory.
    """
    initial: dict = {"messages": messages}
    if user_id is not None:
        initial["user_id"] = user_id
    if user_profile is not None:
        initial["user_profile"] = user_profile
    if selected_plant is not None:
        initial["selected_plant"] = selected_plant
    if recommended_plants is not None:
        initial["recommended_plants"] = recommended_plants
    cfg = config or {}
    result = graph.invoke(initial, config=cfg)
    return result
