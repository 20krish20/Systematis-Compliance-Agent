"""Unit tests for Pydantic schema validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.models import (
    ClassificationResult,
    ComplaintRecord,
    ComplaintSubmitRequest,
    ComplianceRiskLevel,
    ProductCategory,
    SeverityLevel,
)


def test_complaint_submit_request_valid():
    req = ComplaintSubmitRequest(narrative="This is a valid complaint narrative about my account.")
    assert req.narrative == "This is a valid complaint narrative about my account."


def test_complaint_submit_request_strips_whitespace():
    req = ComplaintSubmitRequest(narrative="  complaint text  ")
    assert req.narrative == "complaint text"


def test_complaint_submit_request_too_short():
    with pytest.raises(ValidationError):
        ComplaintSubmitRequest(narrative="short")


def test_complaint_submit_request_blank():
    with pytest.raises(ValidationError):
        ComplaintSubmitRequest(narrative="   ")


def test_classification_result_overall_confidence():
    clf = ClassificationResult(
        product=ProductCategory.CREDIT_CARD,
        product_confidence=0.95,
        issue_type="Billing dispute",
        issue_confidence=0.88,
        severity=SeverityLevel.HIGH,
        severity_confidence=0.80,
        compliance_risk=ComplianceRiskLevel.ELEVATED,
        risk_confidence=0.75,
    )
    assert clf.overall_confidence == 0.75  # min of all


def test_classification_requires_human_review_below_threshold():
    clf = ClassificationResult(
        product=ProductCategory.CREDIT_CARD,
        product_confidence=0.65,
        issue_type="Other",
        issue_confidence=0.60,
        severity=SeverityLevel.LOW,
        severity_confidence=0.70,
        compliance_risk=ComplianceRiskLevel.NONE,
        risk_confidence=0.68,
    )
    assert clf.requires_human_review is True


def test_classification_does_not_require_review_above_threshold():
    clf = ClassificationResult(
        product=ProductCategory.MORTGAGE,
        product_confidence=0.92,
        issue_type="Foreclosure",
        issue_confidence=0.88,
        severity=SeverityLevel.CRITICAL,
        severity_confidence=0.85,
        compliance_risk=ComplianceRiskLevel.IMMINENT,
        risk_confidence=0.90,
    )
    assert clf.requires_human_review is False


def test_complaint_record_auto_id():
    r1 = ComplaintRecord(raw_text="text1", masked_text="text1")
    r2 = ComplaintRecord(raw_text="text2", masked_text="text2")
    assert r1.id != r2.id


def test_complaint_record_confidence_bounds():
    with pytest.raises(ValidationError):
        ClassificationResult(
            product=ProductCategory.OTHER,
            product_confidence=1.5,  # > 1.0
            issue_type="Other",
            issue_confidence=0.8,
            severity=SeverityLevel.LOW,
            severity_confidence=0.8,
            compliance_risk=ComplianceRiskLevel.NONE,
            risk_confidence=0.8,
        )
