"""
metrics.py - Evaluation metrics for Quranic recitation error detection.

Includes precision/recall/F1, GOP-based pronunciation metrics, and
false positive/negative analysis.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from src.error_classifier import ErrorRecord


# ---------------------------------------------------------------------------
# precision_recall_f1
# ---------------------------------------------------------------------------

def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    """
    Compute precision, recall, and F1 from raw counts.

    Zero-division is handled by returning 0.0 for all metrics.

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: List[ErrorRecord],
    ground_truth: List[dict],
) -> dict:
    """
    Compute per-type and micro-average precision, recall, and F1.

    Ground-truth items are dicts with at least:
        position : str   – matches ErrorRecord.position
        error_type : str – matches ErrorType.value

    Returns
    -------
    dict with keys:
        per_type : { error_type_str : {precision, recall, f1, tp, fp, fn} }
        micro    : {precision, recall, f1}
    """
    # Build GT index by position
    gt_by_position: Dict[str, str] = {
        item["position"]: item["error_type"]
        for item in ground_truth
        if item.get("error_type", "MATCH") != "MATCH"
    }

    # Collect all error types
    all_types: set = set()
    for p in predictions:
        if p.error_type.value != "MATCH":
            all_types.add(p.error_type.value)
    for item in ground_truth:
        if item.get("error_type", "MATCH") != "MATCH":
            all_types.add(item["error_type"])

    per_type: Dict[str, dict] = {}
    micro_tp = micro_fp = micro_fn = 0

    for etype in sorted(all_types):
        pred_positions = {
            p.position
            for p in predictions
            if p.error_type.value == etype and p.error_type.value != "MATCH"
        }
        gt_positions = {
            pos for pos, et in gt_by_position.items() if et == etype
        }

        tp = len(pred_positions & gt_positions)
        fp = len(pred_positions - gt_positions)
        fn = len(gt_positions - pred_positions)

        per_type[etype] = {**precision_recall_f1(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro = precision_recall_f1(micro_tp, micro_fp, micro_fn)

    return {"per_type": per_type, "micro": micro}


# ---------------------------------------------------------------------------
# GOP evaluation metrics
# ---------------------------------------------------------------------------

def compute_gop_metrics(
    predictions: List[ErrorRecord],
    ground_truth: List[dict],
) -> dict:
    """
    Compute GOP-based evaluation metrics.

    Uses GOP scores from predictions to evaluate how well GOP discriminates
    between true errors and correct pronunciations.

    Returns
    -------
    dict with keys:
        mean_gop_tp       : mean GOP for true positive errors (should be low)
        mean_gop_tn       : mean GOP for true negatives / matches (should be high)
        gop_separation    : mean_gop_tn - mean_gop_tp (higher = better)
        gop_auc           : area under ROC curve (GOP as binary classifier)
        optimal_threshold : GOP threshold maximising Youden's J statistic
    """
    gt_error_positions = {
        item["position"]
        for item in ground_truth
        if item.get("error_type", "MATCH") != "MATCH"
    }

    # Collect GOP scores for true positives (real errors) and true negatives
    tp_gop: List[float] = []
    tn_gop: List[float] = []

    for pred in predictions:
        if pred.gop_score is None:
            continue
        if pred.error_type.value == "MATCH":
            if pred.position not in gt_error_positions:
                tn_gop.append(pred.gop_score)
        else:
            if pred.position in gt_error_positions:
                tp_gop.append(pred.gop_score)
            else:
                tn_gop.append(pred.gop_score)

    mean_gop_tp = sum(tp_gop) / len(tp_gop) if tp_gop else 0.0
    mean_gop_tn = sum(tn_gop) / len(tn_gop) if tn_gop else 0.0
    gop_separation = mean_gop_tn - mean_gop_tp

    # Compute AUC and optimal threshold using simple threshold sweep
    gop_auc, optimal_threshold = _compute_gop_auc(tp_gop, tn_gop)

    return {
        "mean_gop_tp": round(mean_gop_tp, 4),
        "mean_gop_tn": round(mean_gop_tn, 4),
        "gop_separation": round(gop_separation, 4),
        "gop_auc": round(gop_auc, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "n_tp_scored": len(tp_gop),
        "n_tn_scored": len(tn_gop),
    }


def _compute_gop_auc(
    error_gop: List[float],
    correct_gop: List[float],
) -> tuple[float, float]:
    """
    Compute AUC and optimal threshold via threshold sweep.

    Errors should have lower GOP; correct should have higher GOP.
    A threshold classifies: GOP < threshold => error, GOP >= threshold => correct.

    Returns (auc, optimal_threshold).
    """
    if not error_gop or not correct_gop:
        return 0.5, 0.7  # default

    # Combine all scores and sort for threshold candidates
    all_scores = sorted(set(error_gop + correct_gop))
    if len(all_scores) < 2:
        return 0.5, 0.7

    n_errors = len(error_gop)
    n_correct = len(correct_gop)

    best_j = -1.0
    best_threshold = 0.7
    tpr_list: List[float] = []
    fpr_list: List[float] = []

    # Add boundary thresholds
    thresholds = [all_scores[0] - 0.01] + all_scores + [all_scores[-1] + 0.01]

    for thresh in thresholds:
        # TP: errors correctly identified (GOP < threshold)
        tp = sum(1 for g in error_gop if g < thresh)
        # FP: correct words incorrectly flagged (GOP < threshold)
        fp = sum(1 for g in correct_gop if g < thresh)

        tpr = tp / n_errors if n_errors > 0 else 0.0
        fpr = fp / n_correct if n_correct > 0 else 0.0
        specificity = 1.0 - fpr

        tpr_list.append(tpr)
        fpr_list.append(fpr)

        # Youden's J = sensitivity + specificity - 1
        j_stat = tpr + specificity - 1.0
        if j_stat > best_j:
            best_j = j_stat
            best_threshold = thresh

    # Compute AUC using trapezoidal rule
    # Sort by FPR
    points = sorted(zip(fpr_list, tpr_list))
    auc = 0.0
    for k in range(1, len(points)):
        dx = points[k][0] - points[k - 1][0]
        dy = (points[k][1] + points[k - 1][1]) / 2.0
        auc += dx * dy

    return auc, best_threshold


# ---------------------------------------------------------------------------
# collect_false_positives / collect_false_negatives
# ---------------------------------------------------------------------------

def collect_false_positives(
    predictions: List[ErrorRecord],
    ground_truth: List[dict],
    max_examples: int = 5,
) -> List[dict]:
    """
    Return up to *max_examples* false-positive error records with a heuristic label.

    Heuristics
    ----------
    - GOP > 0.8                  → "ASR_HALLUCINATION"
    - error_type == HARAKAT_ERROR → "NORMALISATION_MISMATCH"
    - otherwise                  → "ALIGNMENT_ERROR"
    """
    gt_positions = {item["position"] for item in ground_truth}

    fps: List[dict] = []
    for pred in predictions:
        if pred.error_type.value == "MATCH":
            continue
        if pred.position in gt_positions:
            continue  # true positive

        if pred.gop_score is not None and pred.gop_score > 0.8:
            label = "ASR_HALLUCINATION"
        elif pred.error_type.value == "HARAKAT_ERROR":
            label = "NORMALISATION_MISMATCH"
        else:
            label = "ALIGNMENT_ERROR"

        fps.append(
            {
                "position": pred.position,
                "error_type": pred.error_type.value,
                "reference_token": pred.reference_token,
                "hypothesis_token": pred.hypothesis_token,
                "gop_score": pred.gop_score,
                "heuristic_label": label,
            }
        )
        if len(fps) >= max_examples:
            break

    return fps


def collect_false_negatives(
    predictions: List[ErrorRecord],
    ground_truth: List[dict],
    max_examples: int = 5,
) -> List[dict]:
    """
    Return up to *max_examples* false-negative ground-truth items (missed errors).
    """
    pred_positions = {p.position for p in predictions if p.error_type.value != "MATCH"}

    fns: List[dict] = []
    for item in ground_truth:
        if item.get("error_type", "MATCH") == "MATCH":
            continue
        if item["position"] in pred_positions:
            continue  # detected

        fns.append(
            {
                "position": item["position"],
                "error_type": item.get("error_type", "UNKNOWN"),
                "reference_token": item.get("reference_token", ""),
                "hypothesis_token": item.get("hypothesis_token", ""),
                "heuristic_label": "MISSED_ERROR",
            }
        )
        if len(fns) >= max_examples:
            break

    return fns
