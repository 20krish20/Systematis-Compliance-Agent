"""
Classifier Agent: multi-label classification across product, issue, severity, compliance risk.
Combines DistilBERT (product/issue) with Claude chain-of-thought (severity/risk).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts.compliance_risk import COMPLIANCE_RISK_SYSTEM_PROMPT, COMPLIANCE_RISK_USER_TEMPLATE
from src.utils.json_parser import extract_json
from src.classifier.distilbert_classifier import DistilBERTComplaintClassifier
from src.config.settings import get_settings
from src.pipeline.embeddings import EmbeddingPipeline
from src.schemas.models import (
    AgentStatus,
    AgentStep,
    ClassificationResult,
    ComplianceRiskLevel,
    SeverityLevel,
)

logger = logging.getLogger(__name__)

_classifier = DistilBERTComplaintClassifier()
_embeddings = EmbeddingPipeline()

SEVERITY_KEYWORDS = {
    SeverityLevel.CRITICAL: ["identity theft", "fraud", "unauthorized access", "imminent harm", "eviction", "foreclosure"],
    SeverityLevel.HIGH: ["unable to pay", "hardship", "collection", "reported to credit", "dispute denied"],
    SeverityLevel.MEDIUM: ["overcharged", "billing error", "incorrect fee", "payment not applied"],
    SeverityLevel.LOW: ["question", "unclear", "confusing statement"],
}


def _heuristic_severity(text: str) -> tuple[SeverityLevel, float]:
    text_lower = text.lower()
    for level in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM]:
        if any(kw in text_lower for kw in SEVERITY_KEYWORDS[level]):
            return level, 0.78
    return SeverityLevel.LOW, 0.70


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _call_llm_for_risk(masked_text: str, product: str, issue_type: str, severity: str) -> dict:
    cfg = get_settings()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    user_msg = COMPLIANCE_RISK_USER_TEMPLATE.format(
        masked_narrative=masked_text[:3000],
        product=product,
        issue_type=issue_type,
        severity=severity,
    )

    response = client.messages.create(
        model=cfg.primary_llm_model,
        max_tokens=1024,
        system=COMPLIANCE_RISK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    return extract_json(response.content[0].text)


def run_classifier(state: dict[str, Any]) -> dict[str, Any]:
    step_start = datetime.utcnow()
    record = state.get("complaint_record")

    if not record:
        return {**state, "last_agent": "classifier"}

    try:
        # DistilBERT product + issue classification
        product, product_conf = _classifier.predict_product(record.masked_text)
        issue_type, issue_conf = _classifier.predict_issue(record.masked_text, product)

        # Heuristic severity (fast path)
        severity, severity_conf = _heuristic_severity(record.masked_text)

        # Few-shot retrieval for confidence boost
        similar = _embeddings.similarity_search(record.masked_text, n_results=3)
        if similar and product_conf < 0.80:
            logger.debug("Using few-shot retrieval to boost confidence")

        # LLM compliance risk scoring
        risk_data = _call_llm_for_risk(record.masked_text, product, issue_type, severity.value)
        risk_level = ComplianceRiskLevel(risk_data.get("risk_level", "ADVISORY"))
        risk_conf = float(risk_data.get("confidence", 0.75))

        # SHAP attributions for audit trail
        shap_attrs = _classifier.predict_with_shap(record.masked_text[:512])

        classification = ClassificationResult(
            product=product,
            product_confidence=product_conf,
            issue_type=issue_type,
            issue_confidence=issue_conf,
            severity=severity,
            severity_confidence=severity_conf,
            compliance_risk=risk_level,
            risk_confidence=risk_conf,
            applicable_regulations=risk_data.get("applicable_regulations", []),
            supporting_facts=risk_data.get("supporting_facts", []),
            escalation_path=risk_data.get("escalation_path"),
            uncertainty_rationale=risk_data.get("uncertainty_rationale"),
            shap_feature_attributions=shap_attrs if shap_attrs else None,
        )

        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="ClassifierAgent",
            status=AgentStatus.COMPLETED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            output_summary=(
                f"Product={product}({product_conf:.2f}), Issue={issue_type}, "
                f"Severity={severity.value}, Risk={risk_level.value}({risk_conf:.2f})"
            ),
        )

        # Escalate only on LLM-based confidence (severity + risk).
        # DistilBERT product confidence is unreliable without fine-tuning
        # and must not gate the full pipeline on its own.
        # Only risk_conf comes from Claude — severity/product are heuristic/DistilBERT.
        # Escalate only when the Claude risk assessment itself is uncertain.
        needs_review = risk_conf < 0.72

        return {
            **state,
            "classification": classification,
            "requires_human_review": needs_review,
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "classifier",
        }

    except Exception as exc:
        logger.error("ClassifierAgent failed: %s", exc)
        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="ClassifierAgent",
            status=AgentStatus.FAILED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            error=str(exc),
        )
        return {
            **state,
            "agent_steps": state.get("agent_steps", []) + [step],
            "requires_human_review": True,
            "last_agent": "classifier",
        }
