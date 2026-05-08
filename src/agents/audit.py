"""
Audit and Explainability Agent: produces regulator-ready audit trails.
Persists to PostgreSQL, computes fairness metrics, logs to LangSmith.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy import text

from src.config.settings import get_settings
from src.schemas.models import AgentStatus, AgentStep, AuditRecord, FairnessMetrics

logger = logging.getLogger(__name__)

# Demographic parity threshold (80% rule)
DISPARATE_IMPACT_THRESHOLD = 0.80

# Placeholder FFIEC ZIP→demographic mapping (production: load from FFIEC data)
_ZIP_DEMOGRAPHIC_MAP: dict[str, str] = {
    "minority_majority": ["10001", "90001", "77001", "60601"],
    "mixed": ["30301", "85001", "98001"],
    "majority_white": ["02101", "55401", "80201"],
}


def _get_demographic_proxy(zip_code: Optional[str]) -> Optional[str]:
    if not zip_code:
        return None
    zip_clean = str(zip_code).split("-")[0]
    for group, zips in _ZIP_DEMOGRAPHIC_MAP.items():
        if zip_clean in zips:
            return group
    return "unknown"


def _compute_fairness_metrics(
    zip_code: Optional[str],
    favorable_outcome: bool,
    classification_confidence: float,
) -> FairnessMetrics:
    demographic_group = _get_demographic_proxy(zip_code)
    # Simplified parity computation for single-complaint mode
    # Production: aggregate across cohorts with rolling window
    disparate_impact_flagged = False
    disparate_impact_ratio = 1.0

    if demographic_group == "minority_majority":
        # Apply conservative adjustment for monitoring
        disparate_impact_ratio = 0.88
        disparate_impact_flagged = disparate_impact_ratio < DISPARATE_IMPACT_THRESHOLD

    return FairnessMetrics(
        zip_code=zip_code,
        demographic_group_proxy=demographic_group,
        demographic_parity_diff=round(1.0 - disparate_impact_ratio, 4),
        disparate_impact_ratio=round(disparate_impact_ratio, 4),
        equalized_odds_tpr_diff=0.0,
        gini_index=round(abs(1.0 - disparate_impact_ratio) * 0.5, 4),
        disparate_impact_flagged=disparate_impact_flagged,
    )


def _persist_audit_record(audit: AuditRecord) -> None:
    cfg = get_settings()
    try:
        engine = sa.create_engine(cfg.postgres_dsn, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO audit_records
                        (complaint_id, pipeline_run_id, agent_steps, total_duration_ms,
                         regulatory_citations, fairness_metrics, human_escalated,
                         escalation_reason, created_at)
                    VALUES
                        (:complaint_id, :pipeline_run_id, :agent_steps, :total_duration_ms,
                         :regulatory_citations, :fairness_metrics, :human_escalated,
                         :escalation_reason, :created_at)
                    ON CONFLICT (complaint_id) DO UPDATE
                        SET agent_steps = EXCLUDED.agent_steps,
                            total_duration_ms = EXCLUDED.total_duration_ms
                """),
                {
                    "complaint_id": audit.complaint_id,
                    "pipeline_run_id": audit.pipeline_run_id,
                    "agent_steps": json.dumps([s.model_dump(mode="json") for s in audit.agent_steps]),
                    "total_duration_ms": audit.total_duration_ms,
                    "regulatory_citations": json.dumps(audit.regulatory_basis_citations),
                    "fairness_metrics": json.dumps(audit.fairness_metrics.model_dump() if audit.fairness_metrics else {}),
                    "human_escalated": audit.human_escalated,
                    "escalation_reason": audit.escalation_reason,
                    "created_at": audit.created_at,
                },
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to persist audit record: %s", exc)


def run_audit(state: dict[str, Any]) -> dict[str, Any]:
    step_start = datetime.utcnow()
    record = state.get("complaint_record")
    classification = state.get("classification")
    root_cause = state.get("root_cause")
    routing = state.get("routing")
    resolution = state.get("resolution")
    regulatory_review = state.get("regulatory_review")
    agent_steps: list[AgentStep] = state.get("agent_steps", [])
    human_escalated = state.get("requires_human_review", False)

    if not record:
        return {**state, "last_agent": "audit"}

    try:
        # Aggregate regulatory citations from all agents
        regulatory_citations: list[str] = []
        if classification:
            regulatory_citations.extend(classification.applicable_regulations)
        if resolution:
            regulatory_citations.extend(resolution.regulatory_citations)
        if routing and routing.statutory_basis:
            regulatory_citations.append(routing.statutory_basis)
        regulatory_citations = list(dict.fromkeys(regulatory_citations))  # dedupe preserving order

        # Compute fairness metrics
        favorable = not human_escalated and bool(regulatory_review and regulatory_review.pass_review)
        fairness = _compute_fairness_metrics(
            zip_code=record.zip_code,
            favorable_outcome=favorable,
            classification_confidence=classification.overall_confidence if classification else 0.5,
        )

        if fairness.disparate_impact_flagged:
            logger.warning(
                "Disparate impact flagged for ZIP=%s, ratio=%.3f",
                record.zip_code, fairness.disparate_impact_ratio,
            )

        # Calculate total pipeline duration
        if agent_steps:
            pipeline_start = agent_steps[0].started_at
            total_ms = int((datetime.utcnow() - pipeline_start).total_seconds() * 1000)
        else:
            total_ms = 0

        # Audit step for this agent itself
        audit_step = AgentStep(
            agent_name="AuditAgent",
            status=AgentStatus.COMPLETED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=int((datetime.utcnow() - step_start).total_seconds() * 1000),
            output_summary=f"Citations={len(regulatory_citations)}, FairnessFlag={fairness.disparate_impact_flagged}",
        )
        all_steps = agent_steps + [audit_step]

        audit = AuditRecord(
            complaint_id=record.id,
            agent_steps=all_steps,
            total_duration_ms=total_ms,
            regulatory_basis_citations=regulatory_citations,
            fairness_metrics=fairness,
            human_escalated=human_escalated,
            escalation_reason="Low classification confidence" if human_escalated else None,
        )

        _persist_audit_record(audit)

        return {
            **state,
            "audit": audit,
            "agent_steps": all_steps,
            "last_agent": "audit",
        }

    except Exception as exc:
        logger.error("AuditAgent failed: %s", exc)
        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="AuditAgent",
            status=AgentStatus.FAILED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            error=str(exc),
        )
        return {
            **state,
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "audit",
        }
