# Aligner -- Arabic Alignment & Forced Alignment

## Overview

The aligner module (`src/aligner.py`) provides two alignment stages:

1. **Text-level alignment** -- Wagner-Fischer DP on word rasm sequences (handled by `src/preprocessor.py:two_pass_compare()`)
2. **Audio-level alignment** -- Whisper word-timestamp alignment mapping words to audio timestamps, with GOP scoring

---

## Two-Pass Text Comparison (in preprocessor.py)

### Algorithm: Wagner-Fischer Dynamic Programming

**Pass 1 -- Rasm alignment:**

Operates on stripped/normalised (rasm) word forms. Computes a word-level Levenshtein distance matrix:

```
dp[i][j] = min(
    dp[i-1][j]   + 1,    # deletion  (ref word has no hyp counterpart)
    dp[i][j-1]   + 1,    # insertion (hyp word has no ref counterpart)
    dp[i-1][j-1] + cost, # substitution (cost=0 if rasm match, 1 otherwise)
)
```

**Pass 2 -- Harakat comparison:**

For aligned MATCH pairs from Pass 1, compares full forms (including diacritics):
- Full forms match -> `MATCH`
- Full forms differ -> `HARAKAT_ERROR`

### DiffResult Output

| Field            | Type | Values                                           |
|------------------|------|--------------------------------------------------|
| error_type       | str  | MATCH, SUBSTITUTION, DELETION, INSERTION, HARAKAT_ERROR |
| rasm_match       | bool | Whether rasm forms are identical                 |
| harakat_match    | bool | Whether full diacritised forms are identical     |
| reference_full   | str  | Original reference word (with harakat)           |
| hypothesis_full  | str  | ASR output word                                  |

---

## Whisper Word-Timestamp Alignment

### `align_audio_to_reference(audio_array, sr, reference_text, ...) -> List[AlignedWord]`

Uses Whisper's word-level timestamp output to align reference text to audio:

1. Guard checks: minimum audio length (0.5 s) and RMS silence threshold
2. Build `AutomaticSpeechRecognitionPipeline` from the loaded model + processor
3. Attempt **word-level timestamps** via `pipe(audio_array, return_timestamps="word")`
4. If that fails, fall back to **chunk-level timestamps** with proportional word interpolation
5. Map resulting chunks to reference words by index; distribute leftover reference words evenly over remaining audio span

**Minimum audio length:** 0.5 seconds (shorter audio returns `None`)

**Silence guard:** RMS < 1e-4 returns `None` immediately (Whisper produces empty output on silent audio)

**Requires** a loaded Whisper model and processor — returns `None` when either is absent.

### Fallback Cascade

```
Strategy 1: Whisper word-timestamps → reference text   --> HIGH confidence
     |
     +--> FAIL (no chunks returned, model/processor missing)
     |
Strategy 2: Whisper word-timestamps → ASR transcript   --> LOW confidence
     |
     +--> FAIL
     |
Strategy 3: Equal-duration distribution                --> FALLBACK confidence
```

When `model`/`processor` are `None`, Strategies 1 and 2 are skipped and the result falls directly to equal-duration distribution.

### AlignedWord Fields

| Field                | Type           | Description                     |
|----------------------|----------------|---------------------------------|
| word_position        | str            | `"surah:ayah:word_index"`       |
| text                 | str            | Word text                        |
| start_sec            | float          | Start time in audio              |
| end_sec              | float          | End time in audio                |
| duration_sec         | float          | Duration (end - start)           |
| gop_score            | Optional[float]| GOP score (None until scored)   |
| alignment_confidence | str            | HIGH, LOW, or FALLBACK          |

---

## GOP (Goodness of Pronunciation) Scoring

### What Is GOP?

GOP is a phoneme-level metric for pronunciation quality assessment. It measures how well the speaker's realisation matches the expected acoustic model for each phoneme.

### Mathematical Formula

**DNN-Based GOP (used in this pipeline):**

```
GOP(p) = -(1/D) * SUM(t=T to T+D-1) log P_t(p | O)
```

Where:
- `p` = target phoneme
- `D` = number of aligned frames
- `T` = start frame
- `P_t(p|O)` = posterior probability of phoneme `p` at frame `t` given observation `O`

### Implementation: `compute_gop_score()`

1. Extract audio segment for the target word using aligned timestamps
2. Compute encoder log-posteriors from the Whisper encoder (via `model(input_features)` or `model.generate()` fallback)
3. Apply `log_softmax` across vocabulary dimension to get log-posteriors per frame
4. Average the maximum log-probability across all frames as proxy for GOP
5. Normalise to [0, 1] via sigmoid scaling (`threshold = 2.0`)

**Score interpretation:**

| GOP Range | Interpretation                   |
|-----------|----------------------------------|
| 0.0 - 0.3 | Poor pronunciation (confirmed error) |
| 0.3 - 0.7 | Moderate (possible error)        |
| 0.7 - 0.85| Acceptable (possible ASR artefact) |
| 0.85 - 1.0| Good pronunciation (likely correct) |

### Selective GOP

GOP is only computed for words at `error_positions` (those flagged by text comparison), not all words. This reduces GPU/CPU time significantly.

---

## Medd Duration Check

### `check_medd_duration(aligned_word, ref_token, min_morae_sec=0.15) -> bool`

Checks whether a Madd (elongation) tajweed rule is violated based on duration:

- Violation: `aligned_word.duration_sec < min_morae_sec * 2`
- `min_morae_sec = 0.15` (150ms per mora, natural Madd = 2 morae = 300ms)
- Only triggers when `ref_token.tajweed_rules` contains a MEDD rule

---

## Mock / Fallback Alignment

### `create_mock_alignment(reference_text, audio_duration, ...) -> List[AlignedWord]`

Creates equal-duration alignment when Whisper alignment fails:

```
audio_duration / num_words = duration_per_word
word_0: [0.0, duration_per_word)
word_1: [duration_per_word, 2*duration_per_word)
...
```

- `gop_score` set to 0.85 (neutral)
- `alignment_confidence` set to `"FALLBACK"`

---

## Data Flow

```
audio_array + reference_text
      |
      v
align_audio_to_reference() --> List[AlignedWord] (timestamps)
      |
      v
score_aligned_words() --> List[AlignedWord] (with GOP scores)
      |                     (only error_positions scored)
      v
classify_verse() --> List[ErrorRecord]
```

The aligned timestamps and GOP scores feed into the error classifier, which uses GOP for severity modulation (CONFIRMED vs POSSIBLE vs SUPPRESSED).
