"""
Resolution Agent: RAG-augmented ReAct pattern for regulatory-compliant complaint resolution.
Generates remediation plan, customer response letter, and preventive recommendations.
"""
from __future__ import annotations

import json
from src.utils.json_parser import extract_json
import logging
from datetime import datetime
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts.resolution import RESOLUTION_SYSTEM_PROMPT, RESOLUTION_USER_TEMPLATE
from src.config.settings import get_settings
from src.rag.knowledge_base import RegulatoryKnowledgeBase
from src.schemas.models import AgentStatus, AgentStep, ResolutionPlan

logger = logging.getLogger(__name__)

_rag = RegulatoryKnowledgeBase()

REGULATION_FILTER_MAP = {
    "Credit card": "REG_Z",
    "Checking/Savings": "REG_E",
    "Money transfer": "REG_E",
    "Credit reporting": "FCRA",
    "Debt collection": "FCRA",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_resolution_llm(
    complaint_id: str,
    masked_text: str,
    product: str,
    issue_type: str,
    severity: str,
    risk: str,
    regulations: list[str],
    root_cause_summary: str,
    rag_context: str,
) -> dict:
    cfg = get_settings()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    user_msg = RESOLUTION_USER_TEMPLATE.format(
        complaint_id=complaint_id,
        product=product,
        issue_type=issue_type,
        severity=severity,
        compliance_risk=risk,
        regulations=", ".join(regulations),
        masked_narrative=masked_text[:3000],
        root_cause_summary=root_cause_summary,
        rag_context=rag_context,
    )

    response = client.messages.create(
        model=cfg.primary_llm_model,
        max_tokens=2048,
        system=RESOLUTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    try:
        return extract_json(response.content[0].text)
    except (ValueError, Exception):
        raw = response.content[0].text.strip()
        return {
            "immediate_actions": ["Review complaint", "Assign compliance officer", "Initiate investigation"],
            "owner": "Compliance Resolution Team",
            "statutory_basis": regulations[0] if regulations else "CFPB Complaint Management",
            "customer_response_draft": raw[:1000],
            "preventive_recommendations": ["Review process controls", "Staff training recommended"],
            "regulatory_citations": regulations,
            "estimated_resolution_days": 30,
        }


def run_resolution(state: dict[str, Any]) -> dict[str, Any]:
    step_start = datetime.utcnow()
    record = state.get("complaint_record")
    classification = state.get("classification")
    root_cause = state.get("root_cause")
    revision_instructions = state.get("revision_instructions")

    if not record or not classification:
        return {**state, "last_agent": "resolution"}

    try:
        # RAG retrieval with regulation-specific filter
        reg_filter = REGULATION_FILTER_MAP.get(classification.product)
        rag_results = _rag.retrieve(
            query=f"{classification.issue_type} {classification.product} resolution {' '.join(classification.applicable_regulations)}",
            n_results=3,
            regulation_filter=reg_filter,
        )
        rag_context = _rag.format_context(rag_results)

        root_cause_summary = ""
        if root_cause:
            root_cause_summary = (
                f"Root cause: {root_cause.cause_category} in {root_cause.affected_process}. "
                f"Hypothesis: {root_cause.causal_hypothesis[:300]}"
            )

        # Include revision instructions if this is a re-run after failed regulatory review
        system_prompt = RESOLUTION_SYSTEM_PROMPT
        if revision_instructions:
            system_prompt += f"\n\nREVISION REQUIRED. Previous response failed review:\n{revision_instructions}"

        resolution_data = _call_resolution_llm(
            complaint_id=record.id,
            masked_text=record.masked_text,
            product=classification.product,
            issue_type=classification.issue_type,
            severity=classification.severity.value,
            risk=classification.compliance_risk.value,
            regulations=classification.applicable_regulations,
            root_cause_summary=root_cause_summary,
            rag_context=rag_context,
        )

        resolution = ResolutionPlan(
            immediate_actions=resolution_data.get("immediate_actions", []),
            owner=resolution_data.get("owner", "Compliance Resolution Team"),
            statutory_basis=resolution_data.get("statutory_basis", ""),
            customer_response_draft=resolution_data.get("customer_response_draft", ""),
            preventive_recommendations=resolution_data.get("preventive_recommendations", []),
            regulatory_citations=resolution_data.get("regulatory_citations", classification.applicable_regulations),
            estimated_resolution_days=int(resolution_data.get("estimated_resolution_days", 30)),
            react_tool_calls=[{"tool": "rag_query", "results": len(rag_results), "filter": reg_filter}],
        )

        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="ResolutionAgent",
            status=AgentStatus.COMPLETED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            output_summary=(
                f"Actions={len(resolution.immediate_actions)}, Owner={resolution.owner}, "
                f"EstDays={resolution.estimated_resolution_days}, RAGHits={len(rag_results)}"
            ),
        )

        return {
            **state,
            "resolution": resolution,
            "revision_instructions": None,  # Clear after use
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "resolution",
        }

    except Exception as exc:
        logger.error("ResolutionAgent failed: %s", exc)
        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="ResolutionAgent",
            status=AgentStatus.FAILED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            error=str(exc),
        )
        return {
            **state,
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "resolution",
        }
