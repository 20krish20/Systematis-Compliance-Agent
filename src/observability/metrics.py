"""
Prometheus metrics for pipeline observability.
Tracked: throughput, classification latency, escalation rate, agent errors, SLA adherence.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

# ─── HTTP Metrics ─────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "compliance_agent_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "compliance_agent_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# ─── Pipeline Metrics ─────────────────────────────────────────────────────────

COMPLAINTS_PROCESSED = Counter(
    "compliance_agent_complaints_processed_total",
    "Total complaints processed",
    ["status"],  # completed, escalated, failed
)

COMPLAINTS_PER_HOUR = Gauge(
    "compliance_agent_complaints_per_hour",
    "Rolling throughput estimate (complaints/hour)",
)

PIPELINE_DURATION = Histogram(
    "compliance_agent_pipeline_duration_seconds",
    "End-to-end pipeline duration per complaint",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# ─── Classification Metrics ──────────────────────────────────────────────────

CLASSIFICATION_CONFIDENCE = Histogram(
    "compliance_agent_classification_confidence",
    "Classification confidence score distribution",
    ["dimension"],  # product, issue, severity, risk
    buckets=[0.5, 0.6, 0.7, 0.72, 0.8, 0.85, 0.9, 0.95, 1.0],
)

CLASSIFICATION_LATENCY = Histogram(
    "compliance_agent_classification_latency_seconds",
    "Classifier agent latency",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 15.0],
)

ESCALATION_RATE = Gauge(
    "compliance_agent_escalation_rate",
    "Current human escalation rate (fraction)",
)

# ─── Agent Metrics ────────────────────────────────────────────────────────────

AGENT_ERRORS = Counter(
    "compliance_agent_agent_errors_total",
    "Agent-level errors",
    ["agent_name"],
)

AGENT_DURATION = Histogram(
    "compliance_agent_agent_duration_seconds",
    "Per-agent execution duration",
    ["agent_name"],
    buckets=[0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 30.0],
)

# ─── Regulatory Review Metrics ────────────────────────────────────────────────

REGULATORY_REVIEW_SCORES = Histogram(
    "compliance_agent_regulatory_review_score",
    "Regulatory review score distribution (0-100)",
    buckets=[50, 60, 70, 75, 80, 85, 90, 95, 100],
)

REVISION_COUNT = Counter(
    "compliance_agent_resolution_revisions_total",
    "Number of resolution revision cycles triggered",
)

# ─── Fairness Metrics ─────────────────────────────────────────────────────────

DISPARATE_IMPACT_FLAGS = Counter(
    "compliance_agent_disparate_impact_flags_total",
    "Number of complaints flagged for disparate impact",
)

GINI_INDEX = Gauge(
    "compliance_agent_gini_index",
    "Current Gini coefficient of resolution outcomes",
)

# ─── LLM Cost Metrics ────────────────────────────────────────────────────────

LLM_TOKENS_USED = Counter(
    "compliance_agent_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "direction"],  # direction: input, output
)

LLM_CALL_LATENCY = Histogram(
    "compliance_agent_llm_call_duration_seconds",
    "LLM API call duration",
    ["agent_name", "model"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
