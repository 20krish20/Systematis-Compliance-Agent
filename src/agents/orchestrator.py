"""
Orchestrator Agent: LangGraph StateGraph coordinating all 7 sub-agents.
Manages global complaint state, conditional edges, failure recovery, and human escalation.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.agents.audit import run_audit
from src.agents.classifier import run_classifier
from src.agents.intake import run_intake
from src.agents.regulatory_review import run_regulatory_review
from src.agents.resolution import run_resolution
from src.agents.root_cause import run_root_cause
from src.agents.routing import run_routing
from src.schemas.models import AgentStatus, ComplaintDisposition, ComplaintSubmitRequest

logger = logging.getLogger(__name__)


# ─── Routing conditions ───────────────────────────────────────────────────────

def route_after_intake(state: dict[str, Any]) -> Literal["classifier", "human_review"]:
    if state.get("pipeline_error") or not state.get("complaint_record"):
        return "human_review"
    return "classifier"


def route_after_classifier(state: dict[str, Any]) -> Literal["root_cause", "human_review"]:
    if state.get("requires_human_review"):
        return "human_review"
    return "root_cause"


def route_after_regulatory_review(state: dict[str, Any]) -> Literal["resolution", "audit"]:
    if state.get("needs_resolution_revision"):
        return "resolution"
    return "audit"


def human_review_node(state: dict[str, Any]) -> dict[str, Any]:
    logger.warning(
        "Complaint %s escalated to human review. Reason: %s",
        state.get("complaint_record", {}).id if state.get("complaint_record") else "unknown",
        state.get("pipeline_error", "Low confidence classification"),
    )
    return {
        **state,
        "requires_human_review": True,
        "last_agent": "human_review",
    }


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(dict)

    graph.add_node("intake", run_intake)
    graph.add_node("classifier", run_classifier)
    graph.add_node("root_cause", run_root_cause)
    graph.add_node("routing", run_routing)
    graph.add_node("resolution", run_resolution)
    graph.add_node("regulatory_review", run_regulatory_review)
    graph.add_node("audit", run_audit)
    graph.add_node("human_review", human_review_node)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", route_after_intake, {"classifier": "classifier", "human_review": "human_review"})
    graph.add_conditional_edges("classifier", route_after_classifier, {"root_cause": "root_cause", "human_review": "human_review"})
    graph.add_edge("root_cause", "routing")
    graph.add_edge("routing", "resolution")
    graph.add_edge("resolution", "regulatory_review")
    graph.add_conditional_edges(
        "regulatory_review",
        route_after_regulatory_review,
        {"resolution": "resolution", "audit": "audit"},
    )
    graph.add_edge("audit", END)
    graph.add_edge("human_review", "audit")

    return graph


_compiled_graph = build_graph().compile()


# ─── Public interface ─────────────────────────────────────────────────────────

def process_complaint(request: ComplaintSubmitRequest) -> ComplaintDisposition:
    pipeline_run_id = str(uuid.uuid4())

    initial_state: dict[str, Any] = {
        "request": request,
        "pipeline_run_id": pipeline_run_id,
        "complaint_record": None,
        "classification": None,
        "root_cause": None,
        "routing": None,
        "resolution": None,
        "regulatory_review": None,
        "audit": None,
        "agent_steps": [],
        "requires_human_review": False,
        "needs_resolution_revision": False,
        "revision_instructions": None,
        "revision_count": 0,
        "pipeline_error": None,
        "last_agent": None,
    }

    logger.info("Starting pipeline for run_id=%s", pipeline_run_id)

    try:
        final_state = _compiled_graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline invocation failed for run_id=%s: %s", pipeline_run_id, exc)
        final_state = {**initial_state, "pipeline_error": str(exc)}

    # Determine final status
    if final_state.get("pipeline_error") and not final_state.get("complaint_record"):
        final_status = AgentStatus.FAILED
    elif final_state.get("requires_human_review"):
        final_status = AgentStatus.ESCALATED
    else:
        final_status = AgentStatus.COMPLETED

    return ComplaintDisposition(
        complaint=final_state.get("complaint_record"),  # type: ignore[arg-type]
        classification=final_state.get("classification"),
        root_cause=final_state.get("root_cause"),
        routing=final_state.get("routing"),
        resolution=final_state.get("resolution"),
        regulatory_review=final_state.get("regulatory_review"),
        audit=final_state.get("audit"),
        final_status=final_status,
    )
