# Report -- Report Generation

## Overview

The report module (`src/report.py`) generates structured output from classified errors in two formats: JSON (machine-readable) and bordered text (human-readable).

---

## RecitationReport Model

Pydantic model containing the complete analysis result:

| Field           | Type                | Description                          |
|-----------------|---------------------|--------------------------------------|
| audio_id        | str                 | Identifier for the audio file        |
| surah           | int                 | Surah number                         |
| ayah            | int                 | Ayah number                          |
| reference_text  | str                 | Ground-truth Arabic text             |
| hypothesis_text | str                 | ASR-transcribed text                 |
| errors          | List[ErrorRecord]   | All classified errors                |
| summary         | dict                | Aggregated counts                    |
| asr_wer         | Optional[float]     | Word Error Rate from ASR             |
| pipeline_mode   | str                 | Pipeline execution mode              |

---

## Pipeline Modes

| Mode             | Description                                          |
|------------------|------------------------------------------------------|
| FULL             | All stages executed (ASR + alignment + phonemizer + GOP) |
| NO_GOP           | Alignment succeeded but GOP scoring failed           |
| TEXT_ONLY         | No alignment available; text comparison only         |
| TEXT_ONLY_CLEAN  | Early exit -- all words matched (no errors detected) |
| MOCK             | Mock mode -- simulated ASR and fallback alignment    |
| ASR_ONLY         | Audio-only mode -- no reference provided, surah/ayah auto-detected or unknown |

---

## Summary Computation

```python
summary = {
    "total_errors": count of non-MATCH errors,
    "by_type": { ErrorType.value: count, ... },
    "by_severity": { Severity.value: count, ... },
}
```

---

## Output Formats

### JSON Output (`report_to_json`)

Full Pydantic model serialisation via `model_dump_json(indent=2)`. Includes all ErrorRecord fields, timestamps, GOP scores, and notes.

Output path: `{output_dir}/{audio_id}.json`

### Text Output (`report_to_text`)

Human-readable bordered report with sections:

```
============================================================
  Recitation Report  --  {audio_id}
============================================================
Surah : 1   Ayah : 1
Mode  : FULL
ASR WER : 0.1429

============================================================
  Reference Text
============================================================
{reference_text}

============================================================
  Hypothesis Text
============================================================
{hypothesis_text}

============================================================
  Summary
============================================================
  Total errors  : 2
  By type:
    SUBSTITUTION              1
    TAJWEED_MEDD              1
  By severity:
    CONFIRMED                 2

============================================================
  Errors  (2 shown)
============================================================
  [CONFIRMED] SUBSTITUTION  @ 1:1:1
    Reference  : اللَّهِ
    Hypothesis : خطأ
    Time       : 0.50s -- 1.00s
    GOP        : 0.3200
    Notes      : Substitution error detected.
```

**Filtering:** Only `CONFIRMED` and `POSSIBLE` errors are shown. `MATCH` and `SUPPRESSED` entries are hidden.

---

## Data Flow

```
List[ErrorRecord] (from classifier)
      |
      v
generate_report() --> RecitationReport
      |
      +---> report_to_json() --> {audio_id}.json
      +---> report_to_text() --> {audio_id}.txt (printed to console if --verbose)
```
