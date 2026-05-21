from langgraph.graph import StateGraph, END
from agents.intake_agent import intake_agent
from agents.analysis_agent import analysis_agent
from agents.decision_agent import decision_agent
from typing import TypedDict, Optional

class FraudState(TypedDict):
    customer: str
    merchant: str
    graph: object
    error: Optional[str]
    status: Optional[str]
    risk_score: Optional[float]
    amount: Optional[float]
    category: Optional[str]
    is_fraud: Optional[int]
    llm_explanation: Optional[str]
    decision: Optional[str]
    decision_reason: Optional[str]

def should_continue(state: dict) -> str:
    if state.get("error"):
        return END
    return "analysis"

def build_graph():
    workflow = StateGraph(FraudState)

    workflow.add_node("intake", intake_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("decision", decision_agent)

    workflow.set_entry_point("intake")
    workflow.add_conditional_edges("intake", should_continue, {
        "analysis": "analysis",
        END: END
    })
    workflow.add_edge("analysis", "decision")
    workflow.add_edge("decision", END)

    return workflow.compile()