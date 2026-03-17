# Error Classifier -- Mismatch Detection & Error Classification

## Overview

The error classifier module (`src/error_classifier.py`) implements a decision-tree classification engine that combines text-level diffs, phoneme-level diffs, and GOP scores to produce structured `ErrorRecord` objects with error types and severity levels.

---

## Error Types

| ErrorType           | Description                                      | Source             |
|---------------------|--------------------------------------------------|--------------------|
| MATCH               | No error detected                                 | Text comparison    |
| SUBSTITUTION        | Wrong word spoken                                 | Text comparison    |
| DELETION            | Word omitted                                      | Text comparison    |
| INSERTION           | Extra word added                                  | Text comparison    |
| HARAKAT_ERROR       | Diacritic mismatch (rasm matches, harakat differs)| Text comparison    |
| TAJWEED_MEDD        | Madd (elongation) rule violation                  | Phoneme diff       |
| TAJWEED_GHUNNA      | Ghunna (nasal) rule violation                     | Phoneme diff       |
| TAJWEED_IDGHAM      | Idgham (assimilation) rule violation              | Phoneme diff       |
| TAJWEED_IKHFA       | Ikhfa (hiding) rule violation                     | Phoneme diff       |
| TAJWEED_QALQALA     | Qalqala (echoing) rule violation                  | Phoneme diff       |
| TAJWEED_IQLAB       | Iqlab (conversion) rule violation                 | Phoneme diff       |
| TAJWEED_IZHAR       | Izhar (clear pronunciation) rule violation        | Phoneme diff       |
| TAJWEED_TAFKHEEM    | Tafkheem (emphasis) rule violation                | Phoneme diff       |
| LOW_CONFIDENCE      | Error detected but alignment unreliable           | Fallback alignment |

---

## Severity Levels

| Severity    | Meaning                                           | When Applied                       |
|-------------|---------------------------------------------------|------------------------------------|
| CONFIRMED   | High-confidence error                              | GOP < 0.7 or no GOP available      |
| POSSIBLE    | Moderate-confidence error                          | GOP 0.7-0.85 or FALLBACK alignment |
| SUPPRESSED  | Likely false positive (ASR artefact)               | GOP > 0.85 on substitution/harakat |

---

## Decision Tree

The `classify_error()` function applies a 6-step priority-based decision tree:

```
Step 1: MATCH?
    YES --> ErrorType.MATCH, Severity.CONFIRMED, confidence=0.0
    NO  --> continue

Step 2: FALLBACK alignment?
    YES --> ErrorType.LOW_CONFIDENCE, Severity.POSSIBLE
            (underlying error noted)
    NO  --> continue

Step 3: HARAKAT_ERROR?
    YES --> ErrorType.HARAKAT_ERROR
            GOP > 0.85  --> SUPPRESSED ("normalisation mismatch")
            GOP > 0.70  --> POSSIBLE
            else        --> CONFIRMED
    NO  --> continue

Step 4: TAJWEED_* from phone_diff?
    YES --> Map diff_type to ErrorType via tajweed_map:
            TAJWEED_MEDD    --> ErrorType.TAJWEED_MEDD
            TAJWEED_GHUNNA  --> ErrorType.TAJWEED_GHUNNA
            TAJWEED_IDGHAM  --> ErrorType.TAJWEED_IDGHAM
            TAJWEED_IKHFA   --> ErrorType.TAJWEED_IKHFA
            TAJWEED_QALQALA --> ErrorType.TAJWEED_QALQALA
            TAJWEED_IQLAB   --> ErrorType.TAJWEED_IQLAB
            TAJWEED_IZHAR   --> ErrorType.TAJWEED_IZHAR
            TAJWEED_TAFKHEEM--> ErrorType.TAJWEED_TAFKHEEM
            Severity: CONFIRMED
    NO  --> continue

Step 5: SUBSTITUTION / DELETION / INSERTION
    Map error_type string to ErrorType enum

Step 6: Suppression check
    SUBSTITUTION + GOP > 0.85 --> SUPPRESSED ("ASR artefact")
    else                      --> CONFIRMED
```

### Confidence Score

```
confidence_score = 1.0 - gop_score    (if GOP available)
confidence_score = 0.6                (if no GOP)
```

---

## ErrorRecord Model

Pydantic model containing all classification results:

| Field                | Type           | Description                     |
|----------------------|----------------|---------------------------------|
| position             | str            | `"surah:ayah:word_index"`       |
| reference_token      | str            | Expected word                    |
| hypothesis_token     | str            | Recognised word                  |
| error_type           | ErrorType      | Classification result            |
| severity             | Severity       | CONFIRMED / POSSIBLE / SUPPRESSED|
| confidence_score     | float          | Error confidence [0, 1]          |
| gop_score            | Optional[float]| GOP pronunciation score          |
| alignment_confidence | str            | HIGH / LOW / FALLBACK            |
| start_sec            | Optional[float]| Start time in audio              |
| end_sec              | Optional[float]| End time in audio                |
| notes                | str            | Human-readable explanation       |

---

## Verse-Level Classification

### `classify_verse(diffs, phone_diffs=None, aligned_words=None) -> List[ErrorRecord]`

Iterates through all diffs for a verse, matching by index with `phone_diffs` and `aligned_words` when available:

```python
for idx, diff in enumerate(diffs):
    phone_diff = phone_diffs[idx] if available
    aligned_word = aligned_words[idx] if available
    record = classify_error(diff, phone_diff, aligned_word)
```

---

## Data Flow

```
DiffResult (from preprocessor)
    +
PhoneDiff (from phonemizer)
    +
AlignedWord (from aligner, with GOP)
    |
    v
classify_error() --> ErrorRecord
    |
    v
List[ErrorRecord] --> generate_report()
```

---

## GOP Threshold Configuration

| Parameter                | Default | Description                        |
|--------------------------|---------|------------------------------------|
| gop_threshold_confirmed  | 0.7     | Below this: CONFIRMED severity     |
| gop_threshold_possible   | 0.85    | Above this: SUPPRESSED for minor errors |

These thresholds are empirically determined and should be tuned per-dataset using the evaluation metrics (GOP-AUC analysis).
