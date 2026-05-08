"""Unit tests for routing agent rule engine."""
from __future__ import annotations

import pytest

from src.agents.routing import _calculate_priority_score, _fast_path_routing
from src.schemas.models import ComplianceRiskLevel, ProductCategory, RoutingTeam, SeverityLevel


def test_priority_score_imminent_critical():
    # base = severity(4) * risk(5) = 20, age_factor=1.0, score = 20 * 1.0 * 10 = 200 → capped at 100
    score = _calculate_priority_score(SeverityLevel.CRITICAL, ComplianceRiskLevel.IMMINENT)
    assert score == 100.0


def test_priority_score_low_none():
    score = _calculate_priority_score(SeverityLevel.LOW, ComplianceRiskLevel.NONE)
    assert score < 20.0


def test_priority_increases_with_age():
    score_fresh = _calculate_priority_score(SeverityLevel.HIGH, ComplianceRiskLevel.MODERATE, days_open=0)
    score_old = _calculate_priority_score(SeverityLevel.HIGH, ComplianceRiskLevel.MODERATE, days_open=60)
    assert score_old > score_fresh


def test_fast_path_reg_e_routing():
    result = _fast_path_routing(
        "I had an unauthorized electronic transfer from my account",
        ProductCategory.CHECKING_SAVINGS,
        "Unauthorized EFT",
    )
    assert result is not None
    team, statutory_basis, sla_days, fast_path = result
    assert team == RoutingTeam.PAYMENTS_COMPLIANCE
    assert "Regulation E" in statutory_basis
    assert fast_path is True


def test_fast_path_fraud_all_products():
    result = _fast_path_routing(
        "Someone committed identity theft and opened accounts in my name",
        ProductCategory.CREDIT_REPORTING,
        "Identity theft",
    )
    assert result is not None
    team, _, _, _ = result
    assert team == RoutingTeam.FRAUD_PREVENTION


def test_fast_path_returns_none_for_ambiguous():
    result = _fast_path_routing(
        "I have a general question about my account",
        ProductCategory.OTHER,
        "Other",
    )
    assert result is None


def test_fast_path_discrimination_routing():
    result = _fast_path_routing(
        "I was denied credit because of my national origin",
        ProductCategory.MORTGAGE,
        "Credit denial",
    )
    assert result is not None
    team, _, _, _ = result
    assert team == RoutingTeam.LEGAL_COMPLIANCE
