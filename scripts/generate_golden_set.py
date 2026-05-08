"""
Generate a 500-complaint golden evaluation set using Faker + real CFPB narratives.
Includes labeled ground truth for product, severity, compliance risk.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path

import click
import pandas as pd
from faker import Faker

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

fake = Faker()

GOLDEN_TEMPLATES = [
    {
        "product": "Credit card",
        "issue": "Billing dispute",
        "severity": "High",
        "risk": "ELEVATED",
        "template": (
            "I have a billing dispute on my {product_type} credit card for a charge of ${amount} "
            "that I did not authorize. I filed a written dispute on {date} and it has been {days} days "
            "with no provisional credit issued as required by Regulation Z. The charge is still showing "
            "as outstanding and I am being charged interest on it."
        ),
    },
    {
        "product": "Checking/Savings",
        "issue": "Unauthorized EFT",
        "severity": "High",
        "risk": "ELEVATED",
        "template": (
            "An unauthorized electronic fund transfer of ${amount} was made from my checking account "
            "on {date}. I notified the bank within {days} days but they have not provided a provisional "
            "credit within 10 business days as required by Regulation E. The investigation has now "
            "been open for {days2} days with no resolution."
        ),
    },
    {
        "product": "Credit reporting",
        "issue": "Incorrect reporting",
        "severity": "Medium",
        "risk": "MODERATE",
        "template": (
            "My credit report shows a delinquent account that I paid in full {months} months ago. "
            "I disputed this with the credit bureau but they verified the inaccurate information "
            "without contacting me. My credit score has dropped {points} points as a result, "
            "affecting my ability to qualify for a mortgage."
        ),
    },
    {
        "product": "Other",
        "issue": "Discrimination",
        "severity": "Critical",
        "risk": "IMMINENT",
        "template": (
            "I was denied credit and the representative stated that my {protected_class} was a factor "
            "in the decision. I have an excellent credit history with a {score} credit score and "
            "sufficient income. This appears to be a clear violation of the Equal Credit Opportunity Act."
        ),
    },
    {
        "product": "Credit card",
        "issue": "Fee dispute",
        "severity": "Low",
        "risk": "NONE",
        "template": (
            "I noticed a ${amount} annual fee on my statement that I was not expecting. "
            "I would like to understand the fee structure and whether I can have it waived "
            "as a loyal customer of {years} years."
        ),
    },
    {
        "product": "Debt collection",
        "issue": "Debt collection practices",
        "severity": "High",
        "risk": "ELEVATED",
        "template": (
            "A debt collector has been calling me at my workplace multiple times a day after "
            "I explicitly told them to stop. They are also threatening to garnish my wages for "
            "a debt that is past the statute of limitations. They misrepresented the amount owed "
            "by adding ${amount} in unauthorized fees."
        ),
    },
    {
        "product": "Mortgage",
        "issue": "Foreclosure",
        "severity": "Critical",
        "risk": "IMMINENT",
        "template": (
            "My mortgage servicer initiated foreclosure proceedings despite my active loss mitigation "
            "application. I submitted a complete loss mitigation application on {date} and they are "
            "required to review it before proceeding with foreclosure. I am at risk of losing my home "
            "of {years} years."
        ),
    },
    {
        "product": "Student loan",
        "issue": "Income-driven repayment",
        "severity": "Medium",
        "risk": "MODERATE",
        "template": (
            "My student loan servicer calculated my income-driven repayment at ${amount}/month but "
            "my recertification documents clearly show I should qualify for a lower payment. "
            "I have been overpaying for {months} months and they refuse to recalculate or refund "
            "the overpayment."
        ),
    },
]


def generate_complaint(template_data: dict) -> dict:
    template = template_data["template"]
    text = template.format(
        product_type=fake.random_element(["Visa", "Mastercard", "Amex", "Discover"]),
        amount=round(fake.random_number(digits=3) + fake.random.random() * 100, 2),
        date=fake.date_between(start_date="-6m", end_date="-1m").strftime("%B %d, %Y"),
        days=fake.random_int(min=12, max=60),
        days2=fake.random_int(min=40, max=90),
        months=fake.random_int(min=3, max=24),
        points=fake.random_int(min=40, max=150),
        protected_class=fake.random_element(["race", "national origin", "sex", "age"]),
        score=fake.random_int(min=720, max=800),
        years=fake.random_int(min=5, max=20),
    )
    return {
        "id": str(uuid.uuid4()),
        "narrative": text,
        "ground_truth": {
            "product": template_data["product"],
            "issue": template_data["issue"],
            "severity": template_data["severity"],
            "compliance_risk": template_data["risk"],
        },
        "source": "synthetic_golden",
    }


@click.command()
@click.option("--n", default=500, help="Number of golden complaints to generate")
@click.option("--output", default="data/golden_set/golden_complaints.json", help="Output JSON path")
def main(n: int, output: str) -> None:
    """Generate the golden evaluation set for consistent benchmarking."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    complaints = []
    for i in range(n):
        template = GOLDEN_TEMPLATES[i % len(GOLDEN_TEMPLATES)]
        complaints.append(generate_complaint(template))

    with open(output_path, "w") as f:
        json.dump(complaints, f, indent=2)

    logger.info("Generated %d golden complaints → %s", n, output_path)

    # Summary
    products = [c["ground_truth"]["product"] for c in complaints]
    risks = [c["ground_truth"]["compliance_risk"] for c in complaints]
    from collections import Counter
    logger.info("Product distribution: %s", dict(Counter(products)))
    logger.info("Risk distribution: %s", dict(Counter(risks)))


if __name__ == "__main__":
    main()
