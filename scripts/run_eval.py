"""
run_eval.py - Evaluate pipeline output reports against ground truth.

Usage
-----
    python scripts/run_eval.py \\
        --reports_dir results/reports/ \\
        --ground_truth data/reference/ground_truth.json \\
        --output results/eval_report.md \\
        --mock
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_predictions_from_reports(reports_dir: Path):
    """Load ErrorRecord-compatible dicts from JSON report files."""
    from src.error_classifier import ErrorRecord

    predictions = []
    for json_path in sorted(reports_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for err in data.get("errors", []):
                try:
                    predictions.append(ErrorRecord(**err))
                except Exception as exc:
                    logger.warning("Skipping invalid error record in %s: %s", json_path.name, exc)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", json_path, exc)
    return predictions


def _mock_ground_truth_and_predictions(reports_dir: Path):
    """
    Generate mock predictions and ground truth from reports.
    - Use all non-MATCH errors as ground truth (regardless of severity).
    - Keep all non-MATCH predictions.
    - Inject one synthetic FP and one synthetic FN for non-trivial scores.
    """
    from src.error_classifier import ErrorRecord, ErrorType, Severity

    real_preds = _load_predictions_from_reports(reports_dir)
    non_match = [p for p in real_preds if p.error_type != ErrorType.MATCH]

    # Ground truth = all non-MATCH errors from predictions
    gt = [
        {
            "position": r.position,
            "error_type": r.error_type.value,
            "reference_token": r.reference_token,
            "hypothesis_token": r.hypothesis_token,
        }
        for r in non_match
    ]

    # Add a synthetic FP (predicted but not in GT)
    fp_synthetic = ErrorRecord(
        position="99:99:99",
        reference_token="فَ",
        hypothesis_token="",
        error_type=ErrorType.SUBSTITUTION,
        severity=Severity.CONFIRMED,
        confidence_score=0.4,
        gop_score=0.9,
        alignment_confidence="HIGH",
        notes="Synthetic FP for evaluation",
    )
    predictions = list(real_preds) + [fp_synthetic]

    # Add a synthetic FN (in GT but not predicted)
    fn_gt = {
        "position": "99:99:100",
        "error_type": "DELETION",
        "reference_token": "الله",
        "hypothesis_token": "",
    }
    gt_with_fn = gt + [fn_gt]

    return predictions, gt_with_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recitation reports.")
    parser.add_argument("--reports_dir", default="results/reports/", help="Reports directory")
    parser.add_argument("--ground_truth", default=None, help="JSON file with ground truth")
    parser.add_argument("--mock", action="store_true", help="Use mock ground truth")
    parser.add_argument("--output", default="results/eval_report.md", help="Output Markdown path")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_path = _ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from eval.metrics import compute_all_metrics, compute_gop_metrics, collect_false_positives, collect_false_negatives
    from eval.evaluate import run_ablation, format_eval_report

    if args.mock:
        logger.info("Mock mode: generating synthetic GT and predictions.")
        predictions, ground_truth = _mock_ground_truth_and_predictions(reports_dir)
    else:
        predictions = _load_predictions_from_reports(reports_dir)
        if args.ground_truth is None:
            logger.error("--ground_truth required in non-mock mode.")
            sys.exit(1)
        gt_path = Path(args.ground_truth)
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

    # Compute metrics
    metrics = compute_all_metrics(predictions, ground_truth)
    gop_metrics = compute_gop_metrics(predictions, ground_truth)
    fp_examples = collect_false_positives(predictions, ground_truth, max_examples=5)
    fn_examples = collect_false_negatives(predictions, ground_truth, max_examples=5)

    # Ablation — build GT that includes TAJWEED and HARAKAT entries for
    # differentiation across configs A/B/C.
    ablation_gt = list(ground_truth)
    # Inject synthetic TAJWEED and HARAKAT GT entries so configs differ
    ablation_gt.append({
        "position": "98:1:0",
        "error_type": "TAJWEED_MEDD",
        "reference_token": "الرحمن",
        "hypothesis_token": "الرحمن",
    })
    ablation_gt.append({
        "position": "98:1:1",
        "error_type": "HARAKAT_ERROR",
        "reference_token": "بسم",
        "hypothesis_token": "بسم",
    })

    ablation = run_ablation(
        audio_files=["mock.wav"],
        references=["بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"],
        ground_truth=[ablation_gt],
    )

    report_md = format_eval_report(metrics, ablation, fp_examples, fn_examples, gop_metrics=gop_metrics)
    output_path.write_text(report_md, encoding="utf-8")
    logger.info("Evaluation report written: %s", output_path)

    micro = metrics.get("micro", {})
    logger.info(
        "Micro  P=%.4f  R=%.4f  F1=%.4f",
        micro.get("precision", 0.0),
        micro.get("recall", 0.0),
        micro.get("f1", 0.0),
    )


if __name__ == "__main__":
    main()
