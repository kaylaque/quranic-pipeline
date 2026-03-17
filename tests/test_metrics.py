"""
test_metrics.py - Tests for eval/metrics.py and eval/evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from eval.metrics import (
    precision_recall_f1,
    compute_all_metrics,
    compute_gop_metrics,
    collect_false_positives,
    collect_false_negatives,
    _compute_gop_auc,
)
from eval.evaluate import format_eval_report
from src.error_classifier import ErrorRecord, ErrorType, Severity


# ---------------------------------------------------------------------------
# Helper: build an ErrorRecord
# ---------------------------------------------------------------------------
def _make_record(
    position: str,
    error_type: str = "SUBSTITUTION",
    severity: str = "CONFIRMED",
    gop: float = 0.5,
) -> ErrorRecord:
    return ErrorRecord(
        position=position,
        reference_token="ref",
        hypothesis_token="hyp",
        error_type=ErrorType(error_type),
        severity=Severity(severity),
        confidence_score=1.0 - gop,
        gop_score=gop,
        alignment_confidence="HIGH",
        notes="test",
    )


# ---------------------------------------------------------------------------
# Test 1: precision_recall_f1(5, 2, 3) → P≈0.714, R≈0.625, F1≈0.667
# ---------------------------------------------------------------------------
def test_precision_recall_f1_values():
    result = precision_recall_f1(5, 2, 3)
    assert abs(result["precision"] - 5 / 7) < 1e-3
    assert abs(result["recall"] - 5 / 8) < 1e-3
    expected_f1 = 2 * (5 / 7) * (5 / 8) / ((5 / 7) + (5 / 8))
    assert abs(result["f1"] - expected_f1) < 1e-3


# ---------------------------------------------------------------------------
# Test 2: precision_recall_f1(0, 0, 5) → all 0.0
# ---------------------------------------------------------------------------
def test_precision_recall_f1_zero_division():
    result = precision_recall_f1(0, 0, 5)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


# ---------------------------------------------------------------------------
# Test 3: All-correct predictions → micro-F1 = 1.0
# ---------------------------------------------------------------------------
def test_compute_all_metrics_perfect():
    """When predictions exactly match ground truth, micro-F1 should be 1.0."""
    positions = ["1:1:0", "1:1:1", "1:1:2"]
    predictions = [_make_record(p) for p in positions]
    ground_truth = [
        {"position": p, "error_type": "SUBSTITUTION", "reference_token": "ref", "hypothesis_token": "hyp"}
        for p in positions
    ]
    metrics = compute_all_metrics(predictions, ground_truth)
    micro = metrics["micro"]
    assert abs(micro["f1"] - 1.0) < 1e-6
    assert abs(micro["precision"] - 1.0) < 1e-6
    assert abs(micro["recall"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: collect_false_positives returns at most max_examples
# ---------------------------------------------------------------------------
def test_collect_false_positives_max_examples():
    """collect_false_positives should never return more than max_examples items."""
    # 10 predictions, 0 in ground truth → all are FPs
    predictions = [_make_record(f"1:1:{i}") for i in range(10)]
    ground_truth: list[dict] = []

    fp = collect_false_positives(predictions, ground_truth, max_examples=4)
    assert len(fp) <= 4


# ---------------------------------------------------------------------------
# Test 5: format_eval_report contains "## Ablation"
# ---------------------------------------------------------------------------
def test_format_eval_report_contains_ablation():
    """The formatted eval report must contain the '## Ablation' section header."""
    metrics = {
        "micro": {"precision": 0.8, "recall": 0.75, "f1": 0.77},
        "per_type": {},
    }
    ablation = {
        "A": {"label": "text_only", "metrics": {"micro": {"precision": 0.7, "recall": 0.6, "f1": 0.65}}},
        "B": {"label": "with_phonemizer", "metrics": {"micro": {"precision": 0.75, "recall": 0.7, "f1": 0.72}}},
        "C": {"label": "full", "metrics": {"micro": {"precision": 0.8, "recall": 0.75, "f1": 0.77}}},
    }
    report_md = format_eval_report(metrics, ablation, [], [])
    assert "## Ablation" in report_md


# ---------------------------------------------------------------------------
# Test 6: compute_gop_metrics with no GOP scores returns safe defaults
# ---------------------------------------------------------------------------
def test_compute_gop_metrics_no_gop_scores_returns_safe_defaults():
    """
    When all predictions have gop_score=None, compute_gop_metrics must
    return zeros (mean_gop_tp=0.0, mean_gop_tn=0.0, gop_separation=0.0)
    and a default AUC of 0.5 with n_tp_scored=0 and n_tn_scored=0.
    """
    preds = [
        _make_record("1:1:0", error_type="SUBSTITUTION", gop=0.5),
    ]
    # Override gop_score to None manually since _make_record sets it
    preds[0] = ErrorRecord(
        position="1:1:0",
        reference_token="ref",
        hypothesis_token="hyp",
        error_type=ErrorType.SUBSTITUTION,
        severity=Severity.CONFIRMED,
        confidence_score=0.5,
        gop_score=None,
        alignment_confidence="HIGH",
        notes="no gop",
    )
    ground_truth = [{"position": "1:1:0", "error_type": "SUBSTITUTION"}]

    result = compute_gop_metrics(preds, ground_truth)

    assert result["mean_gop_tp"] == 0.0
    assert result["mean_gop_tn"] == 0.0
    assert result["gop_separation"] == 0.0
    assert result["n_tp_scored"] == 0
    assert result["n_tn_scored"] == 0
    # AUC defaults to 0.5 when there are no scored predictions
    assert result["gop_auc"] == 0.5


# ---------------------------------------------------------------------------
# Test 7: compute_gop_metrics – TP with low GOP, TN with high GOP → positive separation
# ---------------------------------------------------------------------------
def test_compute_gop_metrics_low_tp_gop_high_tn_gop_gives_positive_separation():
    """
    True-positive errors should have a lower GOP than true-negative matches.
    gop_separation = mean_gop_tn - mean_gop_tp must be positive (> 0).
    """
    preds = [
        # True positive: predicted as error AND is in ground truth
        ErrorRecord(
            position="1:1:0",
            reference_token="ref",
            hypothesis_token="hyp",
            error_type=ErrorType.SUBSTITUTION,
            severity=Severity.CONFIRMED,
            confidence_score=0.8,
            gop_score=0.2,         # low GOP — poor pronunciation
            alignment_confidence="HIGH",
            notes="",
        ),
        # True negative: predicted as MATCH AND not in ground truth
        ErrorRecord(
            position="1:1:1",
            reference_token="ref",
            hypothesis_token="hyp",
            error_type=ErrorType.MATCH,
            severity=Severity.CONFIRMED,
            confidence_score=0.1,
            gop_score=0.9,         # high GOP — good pronunciation
            alignment_confidence="HIGH",
            notes="",
        ),
    ]
    ground_truth = [{"position": "1:1:0", "error_type": "SUBSTITUTION"}]

    result = compute_gop_metrics(preds, ground_truth)

    assert result["gop_separation"] > 0.0
    assert result["mean_gop_tp"] < result["mean_gop_tn"]


# ---------------------------------------------------------------------------
# Test 8: format_eval_report with gop_metrics dict includes "## GOP Metrics" section
# ---------------------------------------------------------------------------
def test_format_eval_report_with_gop_metrics_includes_gop_section():
    """
    When a non-None gop_metrics dict is provided, the formatted report
    must include the '## GOP Metrics' header and display AUC data.
    """
    metrics = {
        "micro": {"precision": 0.8, "recall": 0.75, "f1": 0.77},
        "per_type": {},
    }
    ablation = {
        "A": {"label": "text_only", "metrics": {"micro": {"precision": 0.7, "recall": 0.6, "f1": 0.65}}},
    }
    gop_metrics = {
        "mean_gop_tp": 0.2,
        "mean_gop_tn": 0.9,
        "gop_separation": 0.7,
        "gop_auc": 1.0,
        "optimal_threshold": 0.5,
        "n_tp_scored": 3,
        "n_tn_scored": 3,
    }

    report_md = format_eval_report(metrics, ablation, [], [], gop_metrics=gop_metrics)

    assert "## GOP Metrics" in report_md
    # Check that key GOP values appear in the rendered table
    assert "GOP AUC" in report_md or "gop_auc" in report_md.lower() or "1.0000" in report_md


# ---------------------------------------------------------------------------
# Test 9: _compute_gop_auc with perfect separation returns AUC near 1.0
# ---------------------------------------------------------------------------
def test_compute_gop_auc_perfect_separation_returns_auc_near_one():
    """
    When error GOP scores are all low and correct GOP scores are all high
    with no overlap, _compute_gop_auc must return an AUC >= 0.95.
    """
    error_gop = [0.1, 0.15, 0.2]    # low GOP — clear errors
    correct_gop = [0.8, 0.85, 0.9]  # high GOP — good pronunciation

    auc, optimal_threshold = _compute_gop_auc(error_gop, correct_gop)

    assert auc >= 0.95, f"Expected AUC near 1.0, got {auc}"
    # Optimal threshold should fall between the two clusters
    assert 0.2 <= optimal_threshold <= 0.9
