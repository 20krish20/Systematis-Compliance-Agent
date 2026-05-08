"""
Intake Agent: PII scrubbing, narrative normalization, metadata extraction, deduplication.
First stage of the multi-agent pipeline; no LLM calls made here.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.pipeline.pii_masker import PIIMasker
from src.schemas.models import AgentStatus, AgentStep, ComplaintRecord, ComplaintSubmitRequest

logger = logging.getLogger(__name__)

_masker = PIIMasker()


def run_intake(state: dict[str, Any]) -> dict[str, Any]:
    step_start = datetime.utcnow()
    request: ComplaintSubmitRequest = state["request"]

    try:
        mask_result = _masker.mask(request.narrative)
        fingerprint = PIIMasker.fingerprint(request.narrative)

        record = ComplaintRecord(
            raw_text=request.narrative,
            masked_text=mask_result.masked_text,
            sha256_fingerprint=fingerprint,
            dedup_flag=False,  # dedup checked against DB in production
            state=request.state,
            zip_code=request.zip_code,
            submitted_via=request.submitted_via,
            cfpb_id=request.cfpb_id,
            metadata={
                "product_hint": request.product_hint,
                "pii_detected": mask_result.pii_detected,
                "masked_entity_count": len(mask_result.masked_entities),
            },
        )

        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="IntakeAgent",
            status=AgentStatus.COMPLETED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            output_summary=f"Masked {len(mask_result.masked_entities)} PII entities, fingerprint={fingerprint[:12]}",
        )

        return {
            **state,
            "complaint_record": record,
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "intake",
        }

    except Exception as exc:
        logger.error("IntakeAgent failed: %s", exc)
        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="IntakeAgent",
            status=AgentStatus.FAILED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            error=str(exc),
        )
        return {
            **state,
            "agent_steps": state.get("agent_steps", []) + [step],
            "pipeline_error": str(exc),
            "last_agent": "intake",
        }
