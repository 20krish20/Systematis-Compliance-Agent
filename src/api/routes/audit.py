"""
Audit trail retrieval endpoints for regulatory examination support.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{complaint_id}")
async def get_audit_trail(complaint_id: str) -> dict:
    """Retrieve full audit trail for a complaint by ID."""
    cfg = get_settings()
    try:
        engine = sa.create_engine(cfg.postgres_dsn, pool_pre_ping=True)
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM audit_records WHERE complaint_id = :id"),
                {"id": complaint_id},
            ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"No audit record found for complaint {complaint_id}")

        return {
            "complaint_id": row.complaint_id,
            "pipeline_run_id": row.pipeline_run_id,
            "agent_steps": json.loads(row.agent_steps) if row.agent_steps else [],
            "total_duration_ms": row.total_duration_ms,
            "regulatory_citations": json.loads(row.regulatory_citations) if row.regulatory_citations else [],
            "fairness_metrics": json.loads(row.fairness_metrics) if row.fairness_metrics else {},
            "human_escalated": row.human_escalated,
            "escalation_reason": row.escalation_reason,
            "created_at": str(row.created_at),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/")
async def list_audit_records(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    human_escalated: Optional[bool] = Query(default=None),
) -> dict:
    """List audit records with optional filtering."""
    cfg = get_settings()
    try:
        engine = sa.create_engine(cfg.postgres_dsn, pool_pre_ping=True)
        with engine.connect() as conn:
            where_clause = ""
            params: dict = {"limit": limit, "offset": offset}

            if human_escalated is not None:
                where_clause = "WHERE human_escalated = :human_escalated"
                params["human_escalated"] = human_escalated

            rows = conn.execute(
                text(f"""
                    SELECT complaint_id, pipeline_run_id, total_duration_ms,
                           human_escalated, created_at
                    FROM audit_records
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """),
                params,
            ).fetchall()

            total = conn.execute(
                text(f"SELECT COUNT(*) FROM audit_records {where_clause}"),
                params,
            ).scalar()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "records": [
                {
                    "complaint_id": r.complaint_id,
                    "pipeline_run_id": r.pipeline_run_id,
                    "total_duration_ms": r.total_duration_ms,
                    "human_escalated": r.human_escalated,
                    "created_at": str(r.created_at),
                }
                for r in rows
            ],
        }
    except Exception as exc:
        logger.error("Audit list failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
