"""
Routing Agent: Rule-based fast path + LLM fallback for ambiguous cases.
Calculates priority score and creates mock Jira/ServiceNow tickets.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

import anthropic
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings
from src.schemas.models import (
    AgentStatus,
    AgentStep,
    ComplianceRiskLevel,
    ProductCategory,
    RoutingDecision,
    RoutingTeam,
    SeverityLevel,
)

logger = logging.getLogger(__name__)

SEVERITY_SCORE = {SeverityLevel.LOW: 1, SeverityLevel.MEDIUM: 2, SeverityLevel.HIGH: 3, SeverityLevel.CRITICAL: 4}
RISK_SCORE = {
    ComplianceRiskLevel.NONE: 1,
    ComplianceRiskLevel.ADVISORY: 2,
    ComplianceRiskLevel.MODERATE: 3,
    ComplianceRiskLevel.ELEVATED: 4,
    ComplianceRiskLevel.IMMINENT: 5,
}

# Deterministic routing rules (fast path)
ROUTING_RULES: list[dict] = [
    {
        "product": ProductCategory.CREDIT_CARD,
        "issue_keywords": ["billing", "statement", "dispute"],
        "team": RoutingTeam.CREDIT_OPERATIONS,
        "statutory_basis": "Regulation Z (12 CFR 1026.13)",
        "sla_days": 60,
    },
    {
        "product": ProductCategory.CHECKING_SAVINGS,
        "issue_keywords": ["transfer", "wire", "eft", "electronic", "unauthorized"],
        "team": RoutingTeam.PAYMENTS_COMPLIANCE,
        "statutory_basis": "Regulation E (12 CFR 1005.11)",
        "sla_days": 10,
    },
    {
        "product": ProductCategory.CREDIT_REPORTING,
        "issue_keywords": ["report", "score", "dispute", "inaccurate"],
        "team": RoutingTeam.CREDIT_BUREAU_RELATIONS,
        "statutory_basis": "FCRA (15 U.S.C. 1681i)",
        "sla_days": 30,
    },
    {
        "product": ProductCategory.MORTGAGE,
        "issue_keywords": ["foreclosure", "modification", "servicing", "escrow"],
        "team": RoutingTeam.MORTGAGE_SERVICING,
        "statutory_basis": "Regulation X (12 CFR Part 1024)",
        "sla_days": 30,
    },
    {
        "product": None,  # Applies to all products
        "issue_keywords": ["fraud", "identity theft", "unauthorized access"],
        "team": RoutingTeam.FRAUD_PREVENTION,
        "statutory_basis": "FCRA + ECOA",
        "sla_days": 5,
    },
    {
        "product": None,
        "issue_keywords": ["discrimination", "denied", "race", "sex", "national origin"],
        "team": RoutingTeam.LEGAL_COMPLIANCE,
        "statutory_basis": "ECOA / Regulation B",
        "sla_days": 30,
    },
]


def _calculate_priority_score(severity: SeverityLevel, risk: ComplianceRiskLevel, days_open: int = 0) -> float:
    base = SEVERITY_SCORE[severity] * RISK_SCORE[risk]
    age_factor = 1.0 + (days_open / 30.0) * 0.5
    return min(round(base * age_factor * 10, 1), 100.0)


def _fast_path_routing(masked_text: str, product: str, issue_type: str) -> tuple[RoutingTeam, str, int, bool] | None:
    text_lower = masked_text.lower() + " " + issue_type.lower()
    for rule in ROUTING_RULES:
        if rule["product"] and rule["product"] != product:
            continue
        if any(kw in text_lower for kw in rule["issue_keywords"]):
            return rule["team"], rule["statutory_basis"], rule["sla_days"], True
    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _llm_routing(masked_text: str, product: str, issue_type: str, severity: str) -> dict:
    cfg = get_settings()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    prompt = f"""You are a compliance routing specialist. Based on the complaint below, determine:
1. Which team should handle it (one of: {', '.join([t.value for t in RoutingTeam])})
2. The statutory basis for routing
3. The routing rationale

Complaint (masked): {masked_text[:2000]}
Product: {product}, Issue: {issue_type}, Severity: {severity}

Respond in JSON:
{{"team": "<team>", "statutory_basis": "<statute>", "rationale": "<rationale>", "sla_days": <int>}}"""

    response = client.messages.create(
        model=cfg.primary_llm_model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    return json.loads(raw)


def _create_mock_jira_ticket(team: str, priority_score: float, complaint_id: str) -> str:
    cfg = get_settings()
    ticket_key = f"{cfg.jira_project_key}-{str(uuid.uuid4())[:8].upper()}"

    if not cfg.mock_external_apis:
        try:
            httpx.post(
                f"{cfg.jira_base_url}/rest/api/2/issue",
                json={
                    "fields": {
                        "project": {"key": cfg.jira_project_key},
                        "summary": f"Complaint {complaint_id} - {team}",
                        "issuetype": {"name": "Task"},
                        "priority": {"name": "High" if priority_score > 60 else "Medium"},
                    }
                },
                timeout=5.0,
            )
        except Exception:
            pass

    return ticket_key


def run_routing(state: dict[str, Any]) -> dict[str, Any]:
    step_start = datetime.utcnow()
    record = state.get("complaint_record")
    classification = state.get("classification")

    if not record or not classification:
        return {**state, "last_agent": "routing"}

    try:
        priority_score = _calculate_priority_score(
            classification.severity, classification.compliance_risk, days_open=0
        )

        fast_result = _fast_path_routing(record.masked_text, classification.product, classification.issue_type)

        if fast_result:
            team, statutory_basis, sla_days, fast_path = fast_result
            rationale = f"Deterministic routing: {classification.product} + '{classification.issue_type}' matches {team.value} rule"
        else:
            llm_result = _llm_routing(
                record.masked_text, classification.product, classification.issue_type, classification.severity.value
            )
            team_name = llm_result.get("team", RoutingTeam.HUMAN_REVIEW.value)
            try:
                team = RoutingTeam(team_name)
            except ValueError:
                team = RoutingTeam.HUMAN_REVIEW
            statutory_basis = llm_result.get("statutory_basis", "")
            sla_days = int(llm_result.get("sla_days", 30))
            rationale = llm_result.get("rationale", "LLM-assisted routing")
            fast_path = False

        # Override for IMMINENT risk
        if classification.compliance_risk == ComplianceRiskLevel.IMMINENT:
            team = RoutingTeam.EXECUTIVE_ESCALATION
            sla_days = 1

        ticket_id = _create_mock_jira_ticket(team.value, priority_score, record.id)
        sla_deadline = datetime.utcnow() + timedelta(days=sla_days)

        routing = RoutingDecision(
            primary_team=team,
            priority_score=priority_score,
            statutory_basis=statutory_basis,
            routing_rationale=rationale,
            jira_ticket_id=ticket_id,
            sla_deadline=sla_deadline,
            fast_path_used=fast_path,
        )

        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="RoutingAgent",
            status=AgentStatus.COMPLETED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            output_summary=f"Team={team.value}, Priority={priority_score}, Ticket={ticket_id}, FastPath={fast_path}",
        )

        return {
            **state,
            "routing": routing,
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "routing",
        }

    except Exception as exc:
        logger.error("RoutingAgent failed: %s", exc)
        duration_ms = int((datetime.utcnow() - step_start).total_seconds() * 1000)
        step = AgentStep(
            agent_name="RoutingAgent",
            status=AgentStatus.FAILED,
            started_at=step_start,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            error=str(exc),
        )
        return {
            **state,
            "agent_steps": state.get("agent_steps", []) + [step],
            "last_agent": "routing",
        }
