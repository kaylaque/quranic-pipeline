# Evaluation -- Test Evaluation & Metrics

## Overview

The evaluation modules (`eval/metrics.py` and `eval/evaluate.py`) measure pipeline performance using precision/recall/F1, GOP-based pronunciation metrics, and ablation studies across pipeline configurations.

---

## Metrics

### Precision / Recall / F1

Standard information retrieval metrics applied to error detection:

```
Precision = TP / (TP + FP)     -- of detected errors, how many are real?
Recall    = TP / (TP + FN)     -- of real errors, how many were detected?
F1        = 2 * P * R / (P + R) -- harmonic mean
```

**Granularity:**
- **Per-type**: Computed separately for each ErrorType (SUBSTITUTION, DELETION, TAJWEED_MEDD, etc.)
- **Micro-average**: Aggregated TP/FP/FN across all types, then compute P/R/F1

**Zero-division:** Returns 0.0 when denominator is zero.

### Ground Truth Format

```python
ground_truth = [
    {"position": "1:1:1", "error_type": "SUBSTITUTION", "reference_token": "...", "hypothesis_token": "..."},
    {"position": "1:1:3", "error_type": "TAJWEED_MEDD"},
    ...
]
```

Entries with `error_type == "MATCH"` are excluded from evaluation.

---

## GOP (Goodness of Pronunciation) Score

### What Is GOP?

GOP is a phoneme-level metric for automated pronunciation assessment, originally introduced by Witt & Young (2000). It quantifies how well a speaker's realisation of a phoneme matches the expected acoustic model.

### Mathematical Formulas

**DNN-Based GOP (primary formula used):**

```
GOP(p) = -(1/D) * SUM(t=T to T+D-1) log P_t(p | O)
```

Where:
- `p` = target phoneme
- `D` = number of aligned frames for phoneme `p`
- `T` = start frame of phoneme alignment
- `P_t(p|O)` = posterior probability of phoneme `p` at frame `t`
- `O` = full acoustic observation sequence

**Interpretation:** Lower GOP values (closer to 0) indicate better pronunciation. Higher negative values indicate mispronunciation.

**CTC-Based Alignment-Free GOP (alternative):**

```
GOP_AF = log[ P(L_canonical | O) / P(L_perturbed | O) ]
```

Where `L_canonical` is the expected phoneme sequence and `L_perturbed` has the target phoneme replaced with the most confusable alternative.

### GOP as an Evaluation Metric

In `eval/metrics.py`, GOP is used as a continuous evaluation metric:

#### `compute_gop_metrics(predictions, ground_truth) -> dict`

| Metric              | Description                                          |
|---------------------|------------------------------------------------------|
| `mean_gop_tp`       | Mean GOP for true positive errors (should be low)    |
| `mean_gop_tn`       | Mean GOP for true negatives (should be high)         |
| `gop_separation`    | `mean_gop_tn - mean_gop_tp` (higher = better discrimination) |
| `gop_auc`           | Area under ROC curve treating GOP as a binary classifier |
| `optimal_threshold` | GOP threshold maximising Youden's J = sensitivity + specificity - 1 |

#### Typical GOP Threshold Values

Thresholds are dataset-specific and should be optimised per corpus. The default thresholds in this pipeline:

| Threshold            | Value | Purpose                              |
|----------------------|-------|--------------------------------------|
| `gop_confirmed`     | 0.7   | Below: error is CONFIRMED            |
| `gop_possible`      | 0.85  | Above: minor errors are SUPPRESSED   |

Optimal thresholds can be computed using `compute_gop_metrics()` via the Youden's J statistic.

---

## Ablation Study

### Configurations

| Config | Label             | Components                               |
|--------|-------------------|------------------------------------------|
| A      | text_only         | Preprocessor only (no phonemizer, no alignment) |
| B      | with_phonemizer   | Preprocessor + Phonemizer                |
| C      | full              | Preprocessor + Phonemizer + Alignment + GOP |

### Expected Behaviour

- Config A misses TAJWEED_* errors entirely (no phoneme analysis)
- Config A and B miss some HARAKAT_ERROR cases (no audio confirmation)
- Config C achieves highest recall through phoneme-level and GOP analysis
- Config C may have slightly lower precision due to GOP-based false positives

### `run_ablation(audio_files, references, ground_truth) -> dict`

Returns per-config metrics allowing comparison of pipeline stages:

```python
{
    "A": {"label": "text_only",       "metrics": {"micro": {"precision": ..., "recall": ..., "f1": ...}, ...}},
    "B": {"label": "with_phonemizer", "metrics": {...}},
    "C": {"label": "full",            "metrics": {...}},
}
```

---

## False Positive / False Negative Analysis

### False Positive Heuristics

| Heuristic Label       | Condition                          | Likely Cause            |
|-----------------------|------------------------------------|-------------------------|
| ASR_HALLUCINATION     | GOP > 0.8 for a flagged error     | ASR model hallucination |
| NORMALISATION_MISMATCH| Error type is HARAKAT_ERROR        | Unicode normalisation   |
| ALIGNMENT_ERROR       | Other false positives              | CTC alignment failure   |

### False Negative Collection

Missed errors from ground truth that were not detected by the pipeline. These indicate recall gaps, typically in:
- Subtle tajweed violations
- Harakat errors with high acoustic similarity
- Short-duration words where alignment is imprecise

---

## Evaluation Report Format

Markdown report with sections:

```markdown
# Evaluation Report

## Overall
| Metric    | Value  |
|-----------|--------|
| Precision | 0.8500 |
| Recall    | 0.7200 |
| F1        | 0.7797 |

## Per-Type Breakdown
| Error Type   | Precision | Recall | F1     | TP | FP | FN |
|-------------|-----------|--------|--------|----|----|-----|
| SUBSTITUTION | 0.9000   | 0.8000 | 0.8471 | 9  | 1  | 2  |
| TAJWEED_MEDD | 0.7500   | 0.6000 | 0.6667 | 3  | 1  | 2  |

## GOP Metrics
| Metric              | Value  |
|---------------------|--------|
| Mean GOP (TP)       | 0.3200 |
| Mean GOP (TN)       | 0.8700 |
| GOP Separation      | 0.5500 |
| GOP AUC             | 0.8900 |
| Optimal Threshold   | 0.6500 |

## Ablation
| Config | Label           | Precision | Recall | F1     |
|--------|-----------------|-----------|--------|--------|
| A      | text_only       | 0.9000    | 0.5000 | 0.6429 |
| B      | with_phonemizer | 0.8500    | 0.7000 | 0.7692 |
| C      | full            | 0.8500    | 0.7200 | 0.7797 |

## False Positive Examples
...

## False Negative Examples
...
```
