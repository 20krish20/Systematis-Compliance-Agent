"""
Complaint submission and status endpoints.
Async: submits to Celery queue; sync: runs pipeline inline (for demo/testing).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from src.agents.orchestrator import process_complaint
from src.schemas.models import (
    AgentStatus,
    ComplaintDisposition,
    ComplaintSubmitRequest,
    ComplaintSubmitResponse,
)
from src.tasks.celery_app import process_complaint_task

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/submit", response_model=ComplaintSubmitResponse, status_code=202)
async def submit_complaint(
    request: ComplaintSubmitRequest,
    sync: bool = Query(default=False, description="Run synchronously (demo/testing only)"),
) -> ComplaintSubmitResponse:
    """
    Submit a consumer complaint for autonomous classification, routing, and resolution.
    Default: async via Celery (202 Accepted). Use sync=true for immediate inline processing.
    """
    if sync:
        try:
            disposition = process_complaint(request)
            return ComplaintSubmitResponse(
                complaint_id=disposition.complaint.id if disposition.complaint else "unknown",
                pipeline_run_id=disposition.audit.pipeline_run_id if disposition.audit else "unknown",
                status=disposition.final_status,
                message="Complaint processed synchronously",
            )
        except Exception as exc:
            logger.error("Synchronous processing failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    # Async via Celery
    task = process_complaint_task.delay(request.model_dump())
    return ComplaintSubmitResponse(
        complaint_id="pending",
        pipeline_run_id=task.id,
        status=AgentStatus.PENDING,
        message=f"Complaint queued for processing. Task ID: {task.id}",
    )


@router.post("/process", response_model=ComplaintDisposition)
async def process_complaint_sync(request: ComplaintSubmitRequest) -> ComplaintDisposition:
    """
    Synchronous complaint processing — returns full disposition JSON.
    Intended for demo, testing, and regulator audit trail generation.
    """
    try:
        return process_complaint(request)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")


@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    """Check async task status."""
    from celery.result import AsyncResult
    from src.tasks.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}
    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)
    return response
