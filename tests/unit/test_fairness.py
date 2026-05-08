"""Unit tests for fairness monitoring module."""
from __future__ import annotations

import pytest

from src.fairness.fairness_monitor import FairnessMonitor


def test_gini_equal_distribution():
    monitor = FairnessMonitor()
    assert monitor._gini([10, 10, 10]) == pytest.approx(0.0, abs=0.01)


def test_gini_perfect_inequality():
    monitor = FairnessMonitor()
    gini = monitor._gini([0, 0, 100])
    assert gini > 0.5


def test_disparate_impact_detected():
    monitor = FairnessMonitor()
    for _ in range(100):
        monitor.record_outcome("majority", favorable_outcome=True)
    for _ in range(50):
        monitor.record_outcome("majority", favorable_outcome=False)
    for _ in range(100):
        monitor.record_outcome("minority", favorable_outcome=True)
    for _ in range(200):
        monitor.record_outcome("minority", favorable_outcome=False)

    report = monitor.compute_report()
    assert report.disparate_impact_ratio < 0.80
    assert not report.overall_pass
    assert any("disparate impact" in v.lower() for v in report.violations)


def test_fair_distribution_passes():
    monitor = FairnessMonitor()
    for _ in range(80):
        monitor.record_outcome("group_a", favorable_outcome=True)
    for _ in range(20):
        monitor.record_outcome("group_a", favorable_outcome=False)
    for _ in range(78):
        monitor.record_outcome("group_b", favorable_outcome=True)
    for _ in range(22):
        monitor.record_outcome("group_b", favorable_outcome=False)

    report = monitor.compute_report()
    assert report.disparate_impact_ratio >= 0.80


def test_single_group_insufficient():
    monitor = FairnessMonitor()
    for _ in range(10):
        monitor.record_outcome("only_group", favorable_outcome=True)

    report = monitor.compute_report()
    assert report.demographic_parity_diff == 0.0
