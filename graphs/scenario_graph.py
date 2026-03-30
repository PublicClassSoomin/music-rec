from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph


class ScenarioState(TypedDict, total=False):
    """시나리오 검색 LangGraph 상태."""

    user_query: str
    retrieval_query: str
    mood_energy_bias: float
    candidate_scores: dict[str, float]
    llm_top_ids: list[str]
    top_song_ids: list[str]
    llm_note: str


def compile_scenario_graph(
    expand_fn: Callable[[ScenarioState], ScenarioState],
    retrieve_fn: Callable[[ScenarioState], ScenarioState],
    llm_select_fn: Callable[[ScenarioState], ScenarioState],
    finalize_fn: Callable[[ScenarioState], ScenarioState],
):
    """
    expand → retrieve → llm_select → finalize → END
    - llm_select: 키·설정이 맞을 때만 후보 내 LLM top-3, 아니면 통과
    """
    graph = StateGraph(ScenarioState)
    graph.add_node("expand", expand_fn)
    graph.add_node("retrieve", retrieve_fn)
    graph.add_node("llm_select", llm_select_fn)
    graph.add_node("finalize", finalize_fn)
    graph.set_entry_point("expand")
    graph.add_edge("expand", "retrieve")
    graph.add_edge("retrieve", "llm_select")
    graph.add_edge("llm_select", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_scenario_graph(
    app: Any,
    user_query: str,
) -> ScenarioState:
    return app.invoke(
        {
            "user_query": user_query,
            "retrieval_query": "",
            "mood_energy_bias": 0.0,
            "candidate_scores": {},
            "llm_top_ids": [],
            "top_song_ids": [],
            "llm_note": "",
        }
    )
