"""
Celery task queue for async complaint pipeline execution.
Supports horizontal scaling with 3+ worker replicas.
"""
from __future__ import annotations

import logging

from celery import Celery

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

celery_app = Celery(
    "compliance_agent",
    broker=cfg.celery_broker_url,
    backend=cfg.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=120,
    task_time_limit=180,
    result_expires=3600,
    task_routes={
        "src.tasks.celery_app.process_complaint_task": {"queue": "complaints"},
        "src.tasks.celery_app.run_embedding_batch": {"queue": "embeddings"},
    },
)


@celery_app.task(
    name="src.tasks.celery_app.process_complaint_task",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def process_complaint_task(self, request_data: dict) -> dict:
    from src.agents.orchestrator import process_complaint
    from src.schemas.models import ComplaintSubmitRequest

    try:
        request = ComplaintSubmitRequest(**request_data)
        disposition = process_complaint(request)
        return {
            "complaint_id": disposition.complaint.id if disposition.complaint else None,
            "status": disposition.final_status.value,
            "pipeline_run_id": disposition.audit.pipeline_run_id if disposition.audit else None,
        }
    except Exception as exc:
        logger.error("Task failed for request %s: %s", request_data, exc)
        raise self.retry(exc=exc)


@celery_app.task(name="src.tasks.celery_app.run_embedding_batch")
def run_embedding_batch(parquet_path: str, start_idx: int, end_idx: int) -> int:
    from src.pipeline.embeddings import EmbeddingPipeline
    import pandas as pd

    pipeline = EmbeddingPipeline()
    collection = pipeline.get_or_create_collection()
    df = pd.read_parquet(parquet_path).iloc[start_idx:end_idx]
    df = df[df["masked_text"].notna()]

    texts = df["masked_text"].str[:2000].tolist()
    ids = df["id"].tolist()
    metas = [
        {
            "product": str(row.get("metadata", {}).get("product_raw", "")),
            "issue": str(row.get("metadata", {}).get("issue", "")),
        }
        for _, row in df.iterrows()
    ]

    if texts:
        embeddings = pipeline.embed_batch(texts)
        collection.upsert(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)

    return len(texts)
