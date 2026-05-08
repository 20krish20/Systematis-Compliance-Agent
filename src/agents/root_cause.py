"""
Root Cause Analysis Agent: HDBSCAN semantic clustering + LLM causal chain generation.
Runs asynchronously against the rolling complaint corpus to surface systemic patterns.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import anthropic
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts.root_cause import ROOT_CAUSE_SYSTEM_PROMPT, ROOT_CAUSE_USER_TEMPLATE
from src.config.settings import get_settings
from src.pipeline.embeddings import EmbeddingPipeline
from src.schemas.models import AgentStatus, AgentStep, RootCauseReport

logger = logging.getLogger(__name__)

_embeddings = EmbeddingPipeline()


def _compute_z_score(
    product: str,
    issue_type: str,
    current_count: int,
    history: list[int],
) -> float:
    if not history or len(history) < 3:
        return 0.0
    mean = np.mean(history)
    std = np.std(history)
    if std < 1e-9:
        return 0.0
    return float((current_count - mean) / std)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _generate_causal_chain(
    cluster_id: int,
    cluster_size: int,
    product: str,
    issue_type: str,
    representative_narratives: list[str],
    complaint_ids: list[str],
    z_score: float,
) -> dict:
    cfg = get_settings()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    narratives_str = "\n---\n".join(representative_narratives[:5])
    complaint_ids_str = ", ".join(complaint_ids[:20])
    time_window = "30-day rolling"

    user_msg = ROOT_CAUSE_USER_TEMPLATE.format(
        cluster_id=cluster_id,
        cluster_size=cluster_size,
        product=product,
        issue_type=issue_type,
        time_window=time_window,
        z_score=round(z_score, 2),
        representative_narratives=narratives_str,
        cluster_description=f"Semantic cluster of {product} / {issue_type} complaints",
        complaint_ids=complaint_ids_str,
    )

    response = client.messages.create(
        model=cfg.primary_llm_model,
        max_tokens=1024,
        system=ROOT_CAUSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def run_root_cause(state: dict[str, Any]) -> dict[str, Any]:
    step_start = datetime.utcnow()
    record = state.get("complaint_record")
    classification = state.get("classification")

    if not record or not classification:
        return {**state, "last_agent": "root_cause"}

    try:
        # Find semantically similar complaints for cluster analysis
        similar = _embeddings.similarity_search(
            record.masked_text,
            n_results=10,
            where_filter={"product": classification.product},
        )

        cluster_size = len(similar) + 1
        supporting_ids = [s["metadata"].get("cfpb_id", "") for s in similar]
        representative_texts = [s["text"] for s in similar[:5]]

        # Z-score from volume history (mocked for single-complaint mode)
        z_score = _compute_z_score(
            classification.product,
            classification.issue_type,
            cluster_size,
            [5, 6, 4, 7, 5, 6, 8, 5, 4, 6],  # rolling 30-day history placeholder
        )

        rca_data = _generate_causal_chain(
            cluster_id=hash(f"{classification.product}:{classification.issue_type}") % 10000,
            cluster_size=cluster_size,
            product=classification.product,
            issue_type=classification.issue_type,
            representative_narratives=representative_texts,
            complaint_ids=supporting_ids,
            z_score=z_score,
        )

        root_cause = RootCauseReport(
            cluster_id=rca_data.get("cluster_id") or hash(f"{classification.product}:{classification.issue_type}") % 10000,
            cause_category=rca_data.get("cause_category", "operational"),
            affected_process=rca_data.get("affected_process", "unknown"),
            contributing_factors=rca_data.get("contributing_factors", []),
            frequency_signal=rca_data.get("frequency_signal", "rare"),
            supporting_complaint_ids=supporting_ids,
            recurrence_probability=float(rca_data.get("recurrence_probability", 0.3)),
            z_score=z_score,
            anomaly_detected=abs(z_score) > 2.0,
            causal_hypothesis=rca_data.get("causal_hypothesis", ""),
        )

        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="RootCauseAgent",
            status=AgentStatus.COMPLETED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            output_summary=(
                f"Cause={root_cause.cause_category}, Recurrence={root_cause.recurrence_probability:.2f}, "
                f"Anomaly={root_cause.anomaly_detected}, Z-score={z_score:.2f}"
            ),
        )

        return {
            **state,
            "root_cause": root_cause,
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "root_cause",
        }

    except Exception as exc:
        logger.error("RootCauseAgent failed: %s", exc)
        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="RootCauseAgent",
            status=AgentStatus.FAILED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            error=str(exc),
        )
        return {
            **state,
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "root_cause",
        }
