# Preprocessor -- Reference Resolution, Text Normalisation & Two-Pass Comparison

## Overview

The preprocessing pipeline after ASR spans two files:

- **`src/preprocessor.py`** (295 lines) -- text normalisation, tokenisation, Wagner-Fischer two-pass comparison
- **`scripts/run_pipeline.py`** (lines 47-173, 360-416) -- reference resolution, auto-detection, mock transcription, WER computation, pipeline orchestration

This document covers the complete post-ASR preprocessing flow: from receiving an ASR transcript to producing a list of per-word diff results.

---

## Complete Post-ASR Data Flow

```
ASR transcript (hypothesis)
      |
      v
[Stage 3] Reference Resolution  (run_pipeline.py)
      |
      +--1. --reference flag provided?  → use it directly
      |
      +--2. --surah/--ayah provided?
      |       → _load_reference(surah, ayah)
      |           → file lookup: data/reference/{surah}_{ayah}.txt
      |           → built-in: _DEFAULT_REFERENCES dict
      |
      +--3. Auto-detect from transcript?
      |       → _detect_ayah_from_transcript(transcript)
      |           → rasm fuzzy match against _DEFAULT_REFERENCES
      |           → accept if ≥40% word overlap
      |
      +--4. None of the above?
              → ASR_ONLY mode (transcript = reference)
      |
      v
[WER] compute_wer(strip_harakat(hyp), strip_harakat(ref))
      |
      v
[Stage 4] two_pass_compare(hypothesis, reference)  (preprocessor.py)
      |
      +--→ Pass 1: Wagner-Fischer on rasm tokens
      +--→ Pass 2: harakat comparison on MATCH positions
      |
      v
List[DiffResult]
      |
      v
all_match(diffs)?
      |
      +-- yes → TEXT_ONLY_CLEAN (early exit, skip stages 5-8)
      +-- no  → continue to phonemization (Stage 5)
```

---

## Stage 3: Reference Resolution

Implemented in `scripts/run_pipeline.py`. Determines which ayah the reciter was trying to say, so the pipeline has a ground-truth reference to compare against.

### Strategy 1: User-Provided Reference (`--reference` flag)

```bash
python scripts/run_pipeline.py --audio rec.wav --reference "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
```

Used directly with no lookup. Highest priority.

### Strategy 2: File or Built-In Lookup (`--surah` / `--ayah`)

**`_load_reference(surah, ayah) -> str | None`** (`run_pipeline.py:114-123`)

Two sources, tried in order:

1. **File-based:** looks for `data/reference/{surah}_{ayah}.txt`
   ```python
   ref_file = ref_dir / f"{surah}_{ayah}.txt"
   if ref_file.exists():
       return ref_file.read_text(encoding="utf-8").strip()
   ```

2. **Built-in defaults:** `_DEFAULT_REFERENCES` dict (`run_pipeline.py:49-58`)

   Contains 9 pre-loaded ayahs (Al-Fatiha 1:1-1:7 + Al-Baqarah 2:4 + Ayat al-Kursi 2:255):
   ```python
   _DEFAULT_REFERENCES = {
       (1, 1): "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
       (1, 2): "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ",
       (1, 3): "ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
       (1, 4): "مَـٰلِكِ يَوْمِ ٱلدِّينِ",
       (1, 5): "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
       (1, 6): "ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ",
       (1, 7): "صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ",
       (2, 4): "وَٱلَّذِينَ يُؤْمِنُونَ بِمَآ أُنزِلَ إِلَيْكَ وَمَآ أُنزِلَ مِن قَبْلِكَ وَبِٱلْـَٔاخِرَةِ هُمْ يُوقِنُونَ",
       (2, 255): "ٱللَّهُ لَآ إِلَـٰهَ إِلَّا هُوَ ٱلْحَىُّ ٱلْقَيُّومُ",
   }
   ```

### Strategy 3: Auto-Detect from ASR Transcript

**`_detect_ayah_from_transcript(transcript) -> (surah, ayah, reference_text) | None`** (`run_pipeline.py:126-173`)

When no `--surah`/`--ayah` is provided, the pipeline tries to identify the ayah from the ASR transcript using rasm-level fuzzy matching:

1. Strip diacritics and normalise hamza on the transcript
2. Compare against every entry in `_DEFAULT_REFERENCES` using **order-sensitive word overlap**:
   ```python
   for i in range(min_len):
       if ref_rasm[i] == hyp_rasm[i]:
           matches += 1
   score = matches / max_len
   ```
3. Accept the best match if `score >= 0.4` (40% threshold)

**Why 40%?** A beginner who mispronounces 3 out of 7 words still matches. Below 40%, the match is too ambiguous to trust.

**Limitation:** Only matches against the 8 built-in ayahs in `_DEFAULT_REFERENCES`. For ayahs outside this set, the user must provide `--surah`/`--ayah` or `--reference`.

### Strategy 4: ASR_ONLY Fallback

If no reference can be found by any strategy:
```python
reference_text = hypothesis       # transcript becomes its own reference
pipeline_mode = "ASR_ONLY"        # no error detection possible
```

The pipeline produces a transcript-only report with no errors. This is the graceful degradation path.

### Mock Transcription

**`_mock_transcribe(reference, surah, ayah) -> str`** (`run_pipeline.py:104-111`)

For testing (`--mock` flag). Strips diacritics from the reference, then substitutes word[1] with `"خطأ"` to simulate a word-level error:

```python
words = strip_harakat(reference).split()
if len(words) >= 2:
    words[1] = "خطأ"       # inject substitution at position 1
return " ".join(words)
```

### WER Computation (between reference resolution and text diff)

After both hypothesis and reference are available, the pipeline computes WER on **harakat-stripped** forms:

```python
asr_wer_dict = compute_wer(strip_harakat(hypothesis), strip_harakat(reference_text))
```

This measures rasm-level transcription accuracy before the two-pass diff runs. WER is computed on stripped text so that diacritic differences don't inflate the error rate — diacritic errors are handled separately by Pass 2.

---

## Text Normalisation

### `strip_harakat(text) -> str`

Removes all Arabic diacritical marks and tatweel (kashida) from input text.

**Unicode ranges removed:**

| Range           | Description                                    |
|-----------------|------------------------------------------------|
| U+064B -- U+065F | Fathatan, dammatan, kasratan, fatha, damma, kasra, shadda, sukun, extended combining marks |
| U+0670          | Superscript alef (alef above)                  |
| U+0610 -- U+061A | Quranic annotation signs                       |
| U+06D6 -- U+06ED | Quranic marks (sajda, rub-el-hizb, etc.)       |
| U+0640          | Tatweel / kashida (letter extension)           |

**Algorithm:** Two-pass regex substitution:
1. `_HARAKAT_RE.sub("", text)` -- remove diacritics
2. `_TATWEEL_RE.sub("", text)` -- remove tatweel

### `normalise_hamza(text) -> str`

Normalises hamza variants and decomposes lam-alef ligatures.

**Hamza normalisation table:**

| Input | Output | Description              |
|-------|--------|--------------------------|
| أ (U+0623) | ا (U+0627) | Alef with hamza above |
| إ (U+0625) | ا (U+0627) | Alef with hamza below |
| آ (U+0622) | ا (U+0627) | Alef with madda        |
| ٱ (U+0671) | ا (U+0627) | Alef wasla             |
| ئ (U+0626) | ي (U+064A) | Ya with hamza          |
| ؤ (U+0624) | و (U+0648) | Waw with hamza         |

**Lam-alef ligatures** (U+FEF5 -- U+FEFC) are decomposed to `ل + ا`. This covers 8 presentation-form variants (isolated and final forms for each of 4 ligature types).

**Algorithm:** `str.translate()` for the hamza table (O(n) single pass), then regex substitution for ligatures.

---

## Tokenisation

### `tokenise_reference(text, surah, ayah) -> List[WordToken]`

Splits text by whitespace and produces one `WordToken` per word.

**WordToken fields:**

| Field      | Type | Description                              |
|-----------|------|------------------------------------------|
| surah     | int  | Surah number                             |
| ayah      | int  | Ayah number                              |
| position  | int  | 0-based word index within the ayah       |
| rasm      | str  | Skeleton form: `normalise_hamza(strip_harakat(word))` |
| full_text | str  | Original word with all diacritics        |

---

## Stage 4: Two-Pass Comparison

### `two_pass_compare(hypothesis, reference, surah, ayah) -> List[DiffResult]`

The core comparison function. Takes an ASR hypothesis and a diacritised reference, produces a list of per-word diff results.

### Pass 1 -- Wagner-Fischer DP on Rasm Tokens

Both reference and hypothesis are stripped of diacritics and hamza-normalised, then compared using word-level Levenshtein distance.

**`_levenshtein_matrix(ref_words, hyp_words)`** builds an `(m+1) × (n+1)` cost matrix:

```python
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if ref_words[i - 1] == hyp_words[j - 1]:
            dp[i][j] = dp[i - 1][j - 1]           # match (cost 0)
        else:
            dp[i][j] = 1 + min(
                dp[i - 1][j],      # deletion
                dp[i][j - 1],      # insertion
                dp[i - 1][j - 1],  # substitution
            )
```

Time complexity: O(m × n) where m = reference word count, n = hypothesis word count.

**`_traceback(dp, ref_words, hyp_words)`** walks backwards from `dp[m][n]` to recover the optimal alignment. Priority order on ties: **match → substitution → deletion → insertion**.

Returns a list of `(ref_idx, hyp_idx)` pairs:
- `(i, j)` -- ref word `i` aligned with hyp word `j` (MATCH or SUBSTITUTION)
- `(i, None)` -- ref word `i` has no counterpart (DELETION)
- `(None, j)` -- hyp word `j` has no counterpart (INSERTION)

### Pass 2 -- Harakat Comparison on Matched Positions

For each position where `rasm_match == True` (Pass 1 found identical skeletons), the full diacritised forms are compared:

```python
if rasm_match:
    harakat_match = ref_tok.full_text == hyp_word
    error_type = "MATCH" if harakat_match else "HARAKAT_ERROR"
else:
    error_type = "SUBSTITUTION"
```

No second DP computation -- Pass 2 is a simple string equality check.

### DiffResult Fields

| Field            | Type | Description                                        |
|-----------------|------|----------------------------------------------------|
| position        | str  | `"surah:ayah:word_index"` (insertions use `"surah:ayah:INS{hyp_idx}"`) |
| reference_rasm  | str  | Stripped/normalised reference word (empty for INSERTION) |
| hypothesis_rasm | str  | Stripped/normalised hypothesis word (empty for DELETION) |
| reference_full  | str  | Original reference word with diacritics            |
| hypothesis_full | str  | Original hypothesis word                           |
| rasm_match      | bool | Whether skeletal forms are identical               |
| harakat_match   | bool | Whether full diacritised forms are identical       |
| error_type      | str  | MATCH, SUBSTITUTION, DELETION, INSERTION, or HARAKAT_ERROR |

### Position Indexing

`word_index` advances only for reference-side words (MATCH, SUBSTITUTION, DELETION). Insertions use a separate `INS{hyp_idx}` key to avoid aliasing reference positions:

```
ref:  بِسْمِ  اللَّهِ  الرَّحِيمِ
hyp:  بسم    الله    خطأ    الرحيم

Positions:
  1:1:0   بسم / بسم       MATCH → check harakat
  1:1:1   الله / الله      MATCH → check harakat
  1:1:INS2  — / خطأ       INSERTION (word_index does NOT advance)
  1:1:2   الرحيم / الرحيم   MATCH → check harakat
```

### Concrete Example

```
Input:
  Reference:  بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ
  Hypothesis: بسم خطأ الرحمن الرحيم

Pass 1 (Wagner-Fischer on rasm):
  ref_rasms: ["بسم", "الله", "الرحمن", "الرحيم"]
  hyp_rasms: ["بسم", "خطا",  "الرحمن", "الرحيم"]

  DP matrix:
          ""    بسم   خطا   الرحمن  الرحيم
    ""  [  0     1     2      3      4  ]
    بسم [  1     0     1      2      3  ]
    الله[  2     1     1      2      3  ]
    الرحمن[ 3     2     2      1      2  ]
    الرحيم[ 4     3     3      2      1  ]

  Traceback: [(0,0), (1,1), (2,2), (3,3)]
  Results:
    pos 0: بسم == بسم       → rasm_match = True
    pos 1: الله ≠ خطا       → SUBSTITUTION
    pos 2: الرحمن == الرحمن  → rasm_match = True
    pos 3: الرحيم == الرحيم  → rasm_match = True

Pass 2 (harakat on rasm_match positions):
    pos 0: بِسْمِ ≠ بسم       → HARAKAT_ERROR
    pos 2: الرَّحْمَٰنِ ≠ الرحمن → HARAKAT_ERROR
    pos 3: الرَّحِيمِ ≠ الرحيم  → HARAKAT_ERROR

Final: [HARAKAT_ERROR, SUBSTITUTION, HARAKAT_ERROR, HARAKAT_ERROR]
```

---

## Early Exit

### `all_match(diffs) -> bool`

Returns `True` if every `DiffResult` in the list has `error_type == "MATCH"`. Used by the pipeline to trigger early exit (TEXT_ONLY_CLEAN mode, skipping stages 5--8):

```python
if all_match(diffs):
    # Skip phonemization, alignment, GOP, classification
    # Generate report directly with zero errors
    pipeline_mode = "TEXT_ONLY_CLEAN"
```

---

## Input/Output Contract

### Stage 3 (Reference Resolution)

**Input:** ASR transcript (str), optional `--surah`/`--ayah`/`--reference` CLI args

**Output:** `reference_text` (str) -- the fully diacritised ground-truth ayah text, or ASR_ONLY mode if no reference found

### Stage 4 (Two-Pass Comparison)

**Input:** hypothesis (str from ASR), reference (str from Stage 3), surah (int), ayah (int)

**Output:** `List[DiffResult]` where each result classifies one word position as MATCH, SUBSTITUTION, DELETION, INSERTION, or HARAKAT_ERROR

The `WordToken.full_text` field preserves original diacritics to feed into the phonemizer (Stage 5), which requires harakat for accurate G2P conversion and tajweed rule detection.

---

## Source Locations

| Function | File | Lines |
|----------|------|-------|
| `_DEFAULT_REFERENCES` | `scripts/run_pipeline.py` | 49-58 |
| `_mock_transcribe()` | `scripts/run_pipeline.py` | 104-111 |
| `_load_reference()` | `scripts/run_pipeline.py` | 114-123 |
| `_detect_ayah_from_transcript()` | `scripts/run_pipeline.py` | 126-173 |
| Reference resolution orchestration | `scripts/run_pipeline.py` | 360-406 |
| `strip_harakat()` | `src/preprocessor.py` | 27-31 |
| `normalise_hamza()` | `src/preprocessor.py` | 62-66 |
| `tokenise_reference()` | `src/preprocessor.py` | 86-105 |
| `_levenshtein_matrix()` | `src/preprocessor.py` | 128-149 |
| `_traceback()` | `src/preprocessor.py` | 152-187 |
| `two_pass_compare()` | `src/preprocessor.py` | 190-286 |
| `all_match()` | `src/preprocessor.py` | 293-295 |
