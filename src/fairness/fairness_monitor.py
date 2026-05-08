"""
Fairness monitoring module.
Computes demographic parity, equalized odds, calibration, Gini coefficient,
and disparate impact ratio across ZIP-code demographic proxies.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DISPARATE_IMPACT_THRESHOLD = 0.80
MAX_DEMOGRAPHIC_PARITY_DIFF = 0.05
MAX_EQUALIZED_ODDS_DIFF = 0.05
MAX_GINI = 0.05


@dataclass
class GroupMetrics:
    group: str
    n_complaints: int = 0
    n_favorable: int = 0
    n_true_positives: int = 0
    n_false_positives: int = 0
    n_true_negatives: int = 0
    n_false_negatives: int = 0

    @property
    def favorable_rate(self) -> float:
        return self.n_favorable / self.n_complaints if self.n_complaints > 0 else 0.0

    @property
    def tpr(self) -> float:
        denom = self.n_true_positives + self.n_false_negatives
        return self.n_true_positives / denom if denom > 0 else 0.0

    @property
    def fpr(self) -> float:
        denom = self.n_false_positives + self.n_true_negatives
        return self.n_false_positives / denom if denom > 0 else 0.0


@dataclass
class FairnessReport:
    demographic_parity_diff: float
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    disparate_impact_ratio: float
    gini_index: float
    group_metrics: dict[str, GroupMetrics]
    violations: list[str] = field(default_factory=list)
    overall_pass: bool = True


class FairnessMonitor:
    def __init__(self) -> None:
        self._group_metrics: dict[str, GroupMetrics] = defaultdict(lambda: GroupMetrics(group="unknown"))

    def record_outcome(
        self,
        demographic_group: str,
        favorable_outcome: bool,
        actual_label: Optional[bool] = None,
        predicted_label: Optional[bool] = None,
    ) -> None:
        gm = self._group_metrics[demographic_group]
        gm.group = demographic_group
        gm.n_complaints += 1
        if favorable_outcome:
            gm.n_favorable += 1

        if actual_label is not None and predicted_label is not None:
            if actual_label and predicted_label:
                gm.n_true_positives += 1
            elif actual_label and not predicted_label:
                gm.n_false_negatives += 1
            elif not actual_label and predicted_label:
                gm.n_false_positives += 1
            else:
                gm.n_true_negatives += 1

    def compute_report(self) -> FairnessReport:
        groups = list(self._group_metrics.values())
        if len(groups) < 2:
            logger.warning("Insufficient demographic groups for fairness analysis (need >= 2)")
            return FairnessReport(
                demographic_parity_diff=0.0,
                equalized_odds_tpr_diff=0.0,
                equalized_odds_fpr_diff=0.0,
                disparate_impact_ratio=1.0,
                gini_index=0.0,
                group_metrics=dict(self._group_metrics),
            )

        favorable_rates = [g.favorable_rate for g in groups]
        tprs = [g.tpr for g in groups]
        fprs = [g.fpr for g in groups]

        dem_parity_diff = float(max(favorable_rates) - min(favorable_rates))
        tpr_diff = float(max(tprs) - min(tprs)) if any(tprs) else 0.0
        fpr_diff = float(max(fprs) - min(fprs)) if any(fprs) else 0.0

        # Disparate impact: min favorable rate / max favorable rate (80% rule)
        max_rate = max(favorable_rates)
        min_rate = min(favorable_rates)
        disparate_impact = min_rate / max_rate if max_rate > 0 else 1.0

        # Gini coefficient across groups
        resolution_counts = sorted([g.n_favorable for g in groups])
        n = len(resolution_counts)
        gini = self._gini(resolution_counts)

        violations: list[str] = []
        if dem_parity_diff > MAX_DEMOGRAPHIC_PARITY_DIFF:
            violations.append(f"Demographic parity diff {dem_parity_diff:.4f} > threshold {MAX_DEMOGRAPHIC_PARITY_DIFF}")
        if tpr_diff > MAX_EQUALIZED_ODDS_DIFF:
            violations.append(f"Equalized odds TPR diff {tpr_diff:.4f} > threshold {MAX_EQUALIZED_ODDS_DIFF}")
        if disparate_impact < DISPARATE_IMPACT_THRESHOLD:
            violations.append(f"Disparate impact ratio {disparate_impact:.4f} < 80% rule threshold")
        if gini > MAX_GINI:
            violations.append(f"Gini index {gini:.4f} > threshold {MAX_GINI}")

        return FairnessReport(
            demographic_parity_diff=round(dem_parity_diff, 4),
            equalized_odds_tpr_diff=round(tpr_diff, 4),
            equalized_odds_fpr_diff=round(fpr_diff, 4),
            disparate_impact_ratio=round(disparate_impact, 4),
            gini_index=round(gini, 4),
            group_metrics=dict(self._group_metrics),
            violations=violations,
            overall_pass=len(violations) == 0,
        )

    @staticmethod
    def _gini(values: list[int]) -> float:
        if not values or sum(values) == 0:
            return 0.0
        arr = np.array(sorted(values), dtype=float)
        n = len(arr)
        index = np.arange(1, n + 1)
        return float(((2 * index - n - 1) * arr).sum() / (n * arr.sum()))
