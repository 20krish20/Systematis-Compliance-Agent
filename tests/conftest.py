"""
Shared pytest fixtures and test configuration.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.schemas.models import ComplaintSubmitRequest


@pytest.fixture
def sample_complaint_narrative() -> str:
    return (
        "I have been disputing an unauthorized charge on my credit card for over three months. "
        "The bank keeps saying they are investigating but I have not received a provisional credit "
        "or any resolution. They have also reported this as a delinquency to the credit bureaus "
        "while the dispute is pending, which I understand is not allowed under Regulation Z. "
        "This has dropped my credit score by 85 points and I am now being denied for a mortgage."
    )


@pytest.fixture
def sample_request(sample_complaint_narrative) -> ComplaintSubmitRequest:
    return ComplaintSubmitRequest(
        narrative=sample_complaint_narrative,
        product_hint="Credit card",
        state="CA",
        zip_code="90001",
        submitted_via="Web",
    )


@pytest.fixture
def pii_narrative() -> str:
    return (
        "My name is John Smith and my SSN is 123-45-6789. "
        "My account number is 4111111111111111. "
        "Please contact me at john.smith@example.com or 555-867-5309. "
        "The charge of $5,432.00 was unauthorized."
    )


@pytest.fixture
def app_client():
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def golden_complaints() -> list[dict]:
    return [
        {
            "narrative": (
                "Unauthorized EFT transfer of $2,500 from my checking account. "
                "Bank refused to provide provisional credit within 10 business days as required. "
                "I filed a dispute 3 weeks ago and have heard nothing."
            ),
            "expected_product": "Checking/Savings",
            "expected_severity": "High",
            "expected_risk": "ELEVATED",
        },
        {
            "narrative": (
                "My credit card billing dispute from November has not been resolved "
                "and it's been 75 days. They are still charging interest on the disputed amount."
            ),
            "expected_product": "Credit card",
            "expected_severity": "High",
            "expected_risk": "ELEVATED",
        },
        {
            "narrative": (
                "I was denied credit based on my race. The loan officer explicitly stated "
                "that my neighborhood demographics were a factor in the decision."
            ),
            "expected_product": "Other",
            "expected_severity": "Critical",
            "expected_risk": "IMMINENT",
        },
        {
            "narrative": "My monthly statement arrived two days late this month.",
            "expected_product": "Credit card",
            "expected_severity": "Low",
            "expected_risk": "NONE",
        },
        {
            "narrative": (
                "The credit reporting agency is showing a debt I paid off 18 months ago. "
                "I disputed it but they just verified it without investigation."
            ),
            "expected_product": "Credit reporting",
            "expected_severity": "Medium",
            "expected_risk": "MODERATE",
        },
    ]
