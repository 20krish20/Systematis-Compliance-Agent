"""
Regulatory Review Agent: LLM-as-judge scoring of resolution quality.
Returns responses below 80 to Resolution Agent with specific failure flags.
"""
from __future__ import annotations

import json
from src.utils.json_parser import extract_json
import logging
from datetime import datetime
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts.regulatory_review import REGULATORY_REVIEW_SYSTEM_PROMPT, REGULATORY_REVIEW_USER_TEMPLATE
from src.config.settings import get_settings
from src.rag.knowledge_base import RegulatoryKnowledgeBase
from src.schemas.models import AgentStatus, AgentStep, RegulatoryReviewScore

logger = logging.getLogger(__name__)

_rag = RegulatoryKnowledgeBase()
MAX_REVISION_ATTEMPTS = 2


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_review_llm(
    complaint_id: str,
    product: str,
    regulations: list[str],
    risk_level: str,
    masked_narrative: str,
    resolution_plan: str,
    customer_response: str,
    rag_context: str,
) -> dict:
    cfg = get_settings()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    user_msg = REGULATORY_REVIEW_USER_TEMPLATE.format(
        complaint_id=complaint_id,
        product=product,
        regulations=", ".join(regulations),
        risk_level=risk_level,
        masked_narrative=masked_narrative[:2000],
        resolution_plan=resolution_plan[:2000],
        customer_response=customer_response[:2000],
        rag_context=rag_context,
    )

    response = client.messages.create(
        model=cfg.primary_llm_model,
        max_tokens=2048,
        system=REGULATORY_REVIEW_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    return extract_json(response.content[0].text)


def run_regulatory_review(state: dict[str, Any]) -> dict[str, Any]:
    step_start = datetime.utcnow()
    record = state.get("complaint_record")
    classification = state.get("classification")
    resolution = state.get("resolution")
    revision_count = state.get("revision_count", 0)

    if not record or not classification or not resolution:
        return {**state, "last_agent": "regulatory_review"}

    try:
        rag_results = _rag.retrieve(
            query=" ".join(classification.applicable_regulations) + " " + classification.issue_type,
            n_results=2,
        )
        rag_context = _rag.format_context(rag_results)

        resolution_plan_str = json.dumps({
            "immediate_actions": resolution.immediate_actions,
            "statutory_basis": resolution.statutory_basis,
            "preventive_recommendations": resolution.preventive_recommendations,
            "regulatory_citations": resolution.regulatory_citations,
        }, indent=2)

        review_data = _call_review_llm(
            complaint_id=record.id,
            product=classification.product,
            regulations=classification.applicable_regulations,
            risk_level=classification.compliance_risk.value,
            masked_narrative=record.masked_text,
            resolution_plan=resolution_plan_str,
            customer_response=resolution.customer_response_draft,
            rag_context=rag_context,
        )

        cfg = get_settings()
        total_score = float(review_data.get("total_score", 0))
        pass_review = total_score >= cfg.regulatory_review_pass_score

        review = RegulatoryReviewScore(
            total_score=total_score,
            factual_accuracy=float(review_data.get("factual_accuracy", 0)),
            statutory_citations=float(review_data.get("statutory_citations", 0)),
            tone_compliance=float(review_data.get("tone_compliance", 0)),
            completeness=float(review_data.get("completeness", 0)),
            timeliness_representation=float(review_data.get("timeliness_representation", 0)),
            pass_review=pass_review,
            failure_flags=review_data.get("failure_flags", []),
            revision_instructions=review_data.get("revision_instructions"),
        )

        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="RegulatoryReviewAgent",
            status=AgentStatus.COMPLETED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            output_summary=(
                f"Score={total_score:.1f}/100, Pass={pass_review}, "
                f"Flags={len(review.failure_flags)}, RevisionAttempt={revision_count}"
            ),
        )

        needs_revision = not pass_review and revision_count < MAX_REVISION_ATTEMPTS

        return {
            **state,
            "regulatory_review": review,
            "needs_resolution_revision": needs_revision,
            "revision_instructions": review.revision_instructions if needs_revision else None,
            "revision_count": revision_count + (1 if needs_revision else 0),
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "regulatory_review",
        }

    except Exception as exc:
        logger.error("RegulatoryReviewAgent failed: %s", exc)
        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="RegulatoryReviewAgent",
            status=AgentStatus.FAILED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            error=str(exc),
        )
        return {
            **state,
            "needs_resolution_revision": False,
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "regulatory_review",
        }
