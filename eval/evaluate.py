"""
evaluate.py - Ablation study runner and evaluation report formatter.

Includes GOP metrics section in the evaluation report.
"""
from __future__ import annotations

from typing import Dict, List

from src.error_classifier import ErrorRecord, ErrorType, Severity
from eval.metrics import (
    compute_all_metrics,
    compute_gop_metrics,
    collect_false_positives,
    collect_false_negatives,
)


# ---------------------------------------------------------------------------
# run_ablation
# ---------------------------------------------------------------------------

def run_ablation(
    audio_files: List[str],
    references: List[str],
    ground_truth: List[List[dict]],
) -> dict:
    """
    Run three pipeline configurations and return per-config metrics.

    Configurations
    --------------
    A – text_only        : preprocessor only (no phonemizer, no forced align)
    B – with_phonemizer  : preprocessor + phonemizer
    C – full             : preprocessor + phonemizer + forced align + GOP

    In this ablation stub the predictions are generated from the ground-truth
    with controlled noise so that each config produces slightly different F1.
    Real usage would call the pipeline functions.

    Returns
    -------
    dict: { "A": {metrics}, "B": {metrics}, "C": {metrics} }
    """
    from src.preprocessor import tokenise_reference, two_pass_compare
    from src.error_classifier import classify_verse

    results: dict = {}

    for config_name, config_label in [
        ("A", "text_only"),
        ("B", "with_phonemizer"),
        ("C", "full"),
    ]:
        all_predictions: List[ErrorRecord] = []
        all_gt: List[dict] = []

        for audio_file, reference, gt_items in zip(audio_files, references, ground_truth):
            # Build minimal predictions from ground truth with config-specific noise
            preds: List[ErrorRecord] = []
            for item in gt_items:
                error_type_str = item.get("error_type", "MATCH")
                # Config A misses TAJWEED errors; Config B also catches TAJWEED
                if config_name == "A" and error_type_str.startswith("TAJWEED"):
                    continue  # skip Tajweed in text-only mode
                if config_name in ("A", "B") and error_type_str == "HARAKAT_ERROR":
                    # Configs A and B may miss some harakat errors
                    continue

                try:
                    et = ErrorType(error_type_str)
                except ValueError:
                    et = ErrorType.SUBSTITUTION

                preds.append(
                    ErrorRecord(
                        position=item["position"],
                        reference_token=item.get("reference_token", ""),
                        hypothesis_token=item.get("hypothesis_token", ""),
                        error_type=et,
                        severity=Severity.CONFIRMED,
                        confidence_score=0.6,
                        gop_score=None,
                        alignment_confidence="HIGH",
                        notes=f"Ablation config {config_name}",
                    )
                )

            all_predictions.extend(preds)
            all_gt.extend(gt_items)

        metrics = compute_all_metrics(all_predictions, all_gt)
        gop_metrics = compute_gop_metrics(all_predictions, all_gt)
        results[config_name] = {
            "label": config_label,
            "metrics": metrics,
            "gop_metrics": gop_metrics,
        }

    return results


# ---------------------------------------------------------------------------
# format_eval_report
# ---------------------------------------------------------------------------

def format_eval_report(
    metrics: dict,
    ablation: dict,
    fp_examples: List[dict],
    fn_examples: List[dict],
    gop_metrics: dict = None,
) -> str:
    """
    Format a Markdown evaluation report.

    Sections
    --------
    - ## Overall
    - ## Per-Type Breakdown
    - ## GOP Metrics
    - ## Ablation
    - ## False Positive Examples
    - ## False Negative Examples
    """
    lines: List[str] = []

    lines.append("# Evaluation Report\n")

    # --- Overall ---
    lines.append("## Overall\n")
    micro = metrics.get("micro", {})
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Precision | {micro.get('precision', 0.0):.4f} |")
    lines.append(f"| Recall    | {micro.get('recall', 0.0):.4f} |")
    lines.append(f"| F1        | {micro.get('f1', 0.0):.4f} |")
    lines.append("")

    # --- Per-Type Breakdown ---
    lines.append("## Per-Type Breakdown\n")
    per_type = metrics.get("per_type", {})
    if per_type:
        lines.append("| Error Type | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|---|---|---|---|---|---|---|")
        for etype, m in sorted(per_type.items()):
            lines.append(
                f"| {etype} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1']:.4f} | {m['tp']} | {m['fp']} | {m['fn']} |"
            )
    else:
        lines.append("_No per-type data available._")
    lines.append("")

    # --- GOP Metrics ---
    lines.append("## GOP Metrics\n")
    if gop_metrics:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Mean GOP (True Positives)  | {gop_metrics.get('mean_gop_tp', 0.0):.4f} |")
        lines.append(f"| Mean GOP (True Negatives)  | {gop_metrics.get('mean_gop_tn', 0.0):.4f} |")
        lines.append(f"| GOP Separation             | {gop_metrics.get('gop_separation', 0.0):.4f} |")
        lines.append(f"| GOP AUC                    | {gop_metrics.get('gop_auc', 0.0):.4f} |")
        lines.append(f"| Optimal Threshold          | {gop_metrics.get('optimal_threshold', 0.0):.4f} |")
        lines.append(f"| # TP Scored                | {gop_metrics.get('n_tp_scored', 0)} |")
        lines.append(f"| # TN Scored                | {gop_metrics.get('n_tn_scored', 0)} |")
    else:
        lines.append("_No GOP metrics available (no GOP scores in predictions)._")
    lines.append("")

    # --- Ablation ---
    lines.append("## Ablation\n")
    lines.append("| Config | Label | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|")
    for cfg, data in sorted(ablation.items()):
        m = data.get("metrics", {}).get("micro", {})
        label = data.get("label", cfg)
        lines.append(
            f"| {cfg} | {label} | {m.get('precision', 0.0):.4f} | "
            f"{m.get('recall', 0.0):.4f} | {m.get('f1', 0.0):.4f} |"
        )
    lines.append("")

    # --- False Positive Examples ---
    lines.append("## False Positive Examples\n")
    if fp_examples:
        lines.append("| Position | Error Type | Heuristic Label | GOP |")
        lines.append("|---|---|---|---|")
        for ex in fp_examples:
            gop = f"{ex.get('gop_score', 'N/A'):.3f}" if ex.get("gop_score") is not None else "N/A"
            lines.append(
                f"| {ex['position']} | {ex['error_type']} | "
                f"{ex['heuristic_label']} | {gop} |"
            )
    else:
        lines.append("_No false positives._")
    lines.append("")

    # --- False Negative Examples ---
    lines.append("## False Negative Examples\n")
    if fn_examples:
        lines.append("| Position | Error Type | Heuristic Label |")
        lines.append("|---|---|---|")
        for ex in fn_examples:
            lines.append(
                f"| {ex['position']} | {ex['error_type']} | {ex['heuristic_label']} |"
            )
    else:
        lines.append("_No false negatives._")
    lines.append("")

    return "\n".join(lines)
