# Evaluation Report

## Overall

| Metric | Value |
|---|---|
| Precision | 0.8000 |
| Recall    | 0.8000 |
| F1        | 0.8000 |

## Per-Type Breakdown

| Error Type | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| DELETION | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1 |
| LOW_CONFIDENCE | 1.0000 | 1.0000 | 1.0000 | 4 | 0 | 0 |
| SUBSTITUTION | 0.0000 | 0.0000 | 0.0000 | 0 | 1 | 0 |

## Ablation

| Config | Label | Precision | Recall | F1 |
|---|---|---|---|---|
| A | text_only | 1.0000 | 0.7143 | 0.8333 |
| B | with_phonemizer | 1.0000 | 0.8571 | 0.9231 |
| C | full | 1.0000 | 1.0000 | 1.0000 |

## False Positive Examples

| Position | Error Type | Heuristic Label | GOP |
|---|---|---|---|
| 99:99:99 | SUBSTITUTION | ASR_HALLUCINATION | 0.900 |

## False Negative Examples

| Position | Error Type | Heuristic Label |
|---|---|---|
| 99:99:100 | DELETION | MISSED_ERROR |
