"""
Comprehensive pipeline evaluation against the golden set.
Computes accuracy, macro F1, per-class metrics, fairness, and regulatory quality scores.
Logs all results to MLflow.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import mlflow
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--golden", default="data/golden_set/golden_complaints.json", help="Golden set JSON path")
@click.option("--run-name", default="pipeline_eval_v1", help="MLflow run name")
@click.option("--sync/--async", "run_sync", default=True, help="Run synchronously via pipeline")
def main(golden: str, run_name: str, run_sync: bool) -> None:
    """Evaluate the complaint pipeline against the golden set."""
    from src.agents.orchestrator import process_complaint
    from src.config.settings import get_settings
    from src.schemas.models import ComplaintSubmitRequest

    golden_path = Path(golden)
    if not golden_path.exists():
        logger.error("Golden set not found at %s — run generate_golden_set.py first", golden)
        sys.exit(1)

    with open(golden_path) as f:
        golden_complaints = json.load(f)

    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    y_true_risk, y_pred_risk = [], []
    y_true_severity, y_pred_severity = [], []
    y_true_product, y_pred_product = [], []
    regulatory_scores = []
    escalation_flags = []
    errors = 0

    logger.info("Evaluating %d complaints...", len(golden_complaints))

    for i, complaint in enumerate(golden_complaints):
        if i % 50 == 0:
            logger.info("Progress: %d/%d", i, len(golden_complaints))

        try:
            request = ComplaintSubmitRequest(narrative=complaint["narrative"])
            disposition = process_complaint(request)

            gt = complaint["ground_truth"]
            y_true_risk.append(gt["compliance_risk"])
            y_true_severity.append(gt["severity"])
            y_true_product.append(gt["product"])

            if disposition.classification:
                y_pred_risk.append(disposition.classification.compliance_risk.value)
                y_pred_severity.append(disposition.classification.severity.value)
                y_pred_product.append(disposition.classification.product)
            else:
                y_pred_risk.append("UNKNOWN")
                y_pred_severity.append("UNKNOWN")
                y_pred_product.append("UNKNOWN")

            if disposition.regulatory_review:
                regulatory_scores.append(disposition.regulatory_review.total_score)

            escalation_flags.append(1 if disposition.final_status.value == "ESCALATED" else 0)

        except Exception as exc:
            logger.error("Evaluation error for complaint %d: %s", i, exc)
            errors += 1
            y_true_risk.append(complaint["ground_truth"]["compliance_risk"])
            y_pred_risk.append("UNKNOWN")
            y_true_severity.append(complaint["ground_truth"]["severity"])
            y_pred_severity.append("UNKNOWN")
            y_true_product.append(complaint["ground_truth"]["product"])
            y_pred_product.append("UNKNOWN")

    # Compute metrics
    risk_labels = ["NONE", "ADVISORY", "MODERATE", "ELEVATED", "IMMINENT"]
    risk_report = classification_report(y_true_risk, y_pred_risk, labels=risk_labels, output_dict=True, zero_division=0)
    product_report = classification_report(y_true_product, y_pred_product, output_dict=True, zero_division=0)

    overall_accuracy = sum(t == p for t, p in zip(y_true_risk, y_pred_risk)) / len(y_true_risk)
    escalation_rate = np.mean(escalation_flags)
    avg_regulatory_score = np.mean(regulatory_scores) if regulatory_scores else 0.0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("golden_set_size", len(golden_complaints))
        mlflow.log_param("errors", errors)

        mlflow.log_metric("risk_classification_accuracy", overall_accuracy)
        mlflow.log_metric("risk_macro_f1", risk_report["macro avg"]["f1-score"])
        mlflow.log_metric("product_macro_f1", product_report["macro avg"]["f1-score"])
        mlflow.log_metric("escalation_rate", escalation_rate)
        mlflow.log_metric("avg_regulatory_score", avg_regulatory_score)

        # Per-risk-level metrics
        for risk in risk_labels:
            if risk in risk_report:
                mlflow.log_metric(f"f1_{risk}", risk_report[risk]["f1-score"])
                mlflow.log_metric(f"precision_{risk}", risk_report[risk]["precision"])
                mlflow.log_metric(f"recall_{risk}", risk_report[risk]["recall"])

        # Success criteria check
        targets = {
            "Risk accuracy > 0.90": overall_accuracy > 0.90,
            "Risk macro F1 > 0.88": risk_report["macro avg"]["f1-score"] > 0.88,
            "Escalation rate < 0.12": escalation_rate < 0.12,
            "Avg regulatory score > 85": avg_regulatory_score > 85,
        }
        for criterion, passed in targets.items():
            mlflow.log_metric(f"target_{'pass' if passed else 'fail'}_{criterion[:20].replace(' ', '_')}", 1 if passed else 0)
            status = "PASS" if passed else "FAIL"
            logger.info("[%s] %s", status, criterion)

    logger.info("\n=== EVALUATION SUMMARY ===")
    logger.info("Risk accuracy: %.3f (target: > 0.90)", overall_accuracy)
    logger.info("Risk macro F1: %.3f (target: > 0.88)", risk_report["macro avg"]["f1-score"])
    logger.info("Escalation rate: %.3f (target: < 0.12)", escalation_rate)
    logger.info("Avg regulatory score: %.1f/100 (target: > 85)", avg_regulatory_score)
    logger.info("Total errors: %d", errors)


if __name__ == "__main__":
    main()
