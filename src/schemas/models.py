from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Enumerations ─────────────────────────────────────────────────────────────

class ProductCategory(str, Enum):
    CREDIT_CARD = "Credit card"
    MORTGAGE = "Mortgage"
    STUDENT_LOAN = "Student loan"
    AUTO_LOAN = "Auto loan"
    CHECKING_SAVINGS = "Checking/Savings"
    PERSONAL_LOAN = "Personal loan"
    DEBT_COLLECTION = "Debt collection"
    CREDIT_REPORTING = "Credit reporting"
    MONEY_TRANSFER = "Money transfer"
    PAYDAY_LOAN = "Payday loan"
    PREPAID_CARD = "Prepaid card"
    OTHER = "Other"


class SeverityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ComplianceRiskLevel(str, Enum):
    NONE = "NONE"
    ADVISORY = "ADVISORY"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    IMMINENT = "IMMINENT"


class AgentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"


class RoutingTeam(str, Enum):
    PAYMENTS_COMPLIANCE = "Payments Compliance"
    CREDIT_OPERATIONS = "Credit Operations"
    MORTGAGE_SERVICING = "Mortgage Servicing"
    CONSUMER_LENDING = "Consumer Lending"
    FRAUD_PREVENTION = "Fraud Prevention"
    CREDIT_BUREAU_RELATIONS = "Credit Bureau Relations"
    LEGAL_COMPLIANCE = "Legal & Compliance"
    EXECUTIVE_ESCALATION = "Executive Escalation"
    HUMAN_REVIEW = "Human Review Queue"


# ─── Core Complaint Record ────────────────────────────────────────────────────

class ComplaintRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cfpb_id: Optional[str] = None
    raw_text: str
    masked_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    dedup_flag: bool = False
    sha256_fingerprint: Optional[str] = None
    source: str = "cfpb"
    submitted_via: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    date_received: Optional[datetime] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Classification Output ────────────────────────────────────────────────────

class ClassificationResult(BaseModel):
    product: ProductCategory
    product_confidence: float = Field(ge=0.0, le=1.0)
    issue_type: str
    issue_confidence: float = Field(ge=0.0, le=1.0)
    severity: SeverityLevel
    severity_confidence: float = Field(ge=0.0, le=1.0)
    compliance_risk: ComplianceRiskLevel
    risk_confidence: float = Field(ge=0.0, le=1.0)
    applicable_regulations: list[str] = Field(default_factory=list)
    supporting_facts: list[str] = Field(default_factory=list)
    escalation_path: Optional[str] = None
    uncertainty_rationale: Optional[str] = None
    shap_feature_attributions: Optional[dict[str, float]] = None

    @property
    def overall_confidence(self) -> float:
        return min(
            self.product_confidence,
            self.issue_confidence,
            self.severity_confidence,
            self.risk_confidence,
        )

    @property
    def requires_human_review(self) -> bool:
        return self.overall_confidence < 0.72


# ─── Root Cause Output ───────────────────────────────────────────────────────

class RootCauseReport(BaseModel):
    cluster_id: Optional[int] = None
    cause_category: str
    affected_process: str
    contributing_factors: list[str]
    frequency_signal: str
    supporting_complaint_ids: list[str] = Field(default_factory=list)
    recurrence_probability: float = Field(ge=0.0, le=1.0)
    z_score: Optional[float] = None
    anomaly_detected: bool = False
    causal_hypothesis: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Routing Output ──────────────────────────────────────────────────────────

class RoutingDecision(BaseModel):
    primary_team: RoutingTeam
    secondary_team: Optional[RoutingTeam] = None
    priority_score: float = Field(ge=0.0, le=100.0)
    statutory_basis: Optional[str] = None
    routing_rationale: str
    jira_ticket_id: Optional[str] = None
    sla_deadline: Optional[datetime] = None
    fast_path_used: bool = False


# ─── Resolution Output ───────────────────────────────────────────────────────

class ResolutionPlan(BaseModel):
    immediate_actions: list[str]
    owner: str
    statutory_basis: str
    customer_response_draft: str
    preventive_recommendations: list[str]
    regulatory_citations: list[str]
    estimated_resolution_days: int
    react_tool_calls: list[dict[str, Any]] = Field(default_factory=list)


# ─── Regulatory Review Output ────────────────────────────────────────────────

class RegulatoryReviewScore(BaseModel):
    total_score: float = Field(ge=0.0, le=100.0)
    factual_accuracy: float = Field(ge=0.0, le=20.0)
    statutory_citations: float = Field(ge=0.0, le=20.0)
    tone_compliance: float = Field(ge=0.0, le=20.0)
    completeness: float = Field(ge=0.0, le=20.0)
    timeliness_representation: float = Field(ge=0.0, le=20.0)
    pass_review: bool
    failure_flags: list[str] = Field(default_factory=list)
    revision_instructions: Optional[str] = None
    reviewer_model: str = "claude-sonnet-4-6"
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Audit Record ────────────────────────────────────────────────────────────

class AgentStep(BaseModel):
    agent_name: str
    status: AgentStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    output_summary: Optional[str] = None
    error: Optional[str] = None


class FairnessMetrics(BaseModel):
    zip_code: Optional[str] = None
    demographic_group_proxy: Optional[str] = None
    demographic_parity_diff: Optional[float] = None
    disparate_impact_ratio: Optional[float] = None
    equalized_odds_tpr_diff: Optional[float] = None
    gini_index: Optional[float] = None
    disparate_impact_flagged: bool = False


class AuditRecord(BaseModel):
    complaint_id: str
    pipeline_run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_steps: list[AgentStep] = Field(default_factory=list)
    total_duration_ms: Optional[int] = None
    regulatory_basis_citations: list[str] = Field(default_factory=list)
    fairness_metrics: Optional[FairnessMetrics] = None
    human_escalated: bool = False
    escalation_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Full Complaint Disposition ──────────────────────────────────────────────

class ComplaintDisposition(BaseModel):
    complaint: ComplaintRecord
    classification: Optional[ClassificationResult] = None
    root_cause: Optional[RootCauseReport] = None
    routing: Optional[RoutingDecision] = None
    resolution: Optional[ResolutionPlan] = None
    regulatory_review: Optional[RegulatoryReviewScore] = None
    audit: Optional[AuditRecord] = None
    final_status: AgentStatus = AgentStatus.PENDING
    pipeline_version: str = "1.0.0"


# ─── API Request/Response ────────────────────────────────────────────────────

class ComplaintSubmitRequest(BaseModel):
    narrative: str = Field(min_length=10, max_length=10000)
    product_hint: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    submitted_via: Optional[str] = None
    cfpb_id: Optional[str] = None

    @field_validator("narrative")
    @classmethod
    def narrative_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Narrative cannot be blank")
        return v.strip()


class ComplaintSubmitResponse(BaseModel):
    complaint_id: str
    pipeline_run_id: str
    status: AgentStatus
    message: str = "Complaint received and processing initiated"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
