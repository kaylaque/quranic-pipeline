# Preprocessor -- Audio Ingestion & Text Preprocessing

## Overview

The preprocessor module handles two responsibilities:

1. **Text normalisation** -- preparing Arabic reference and hypothesis texts for comparison
2. **Text tokenisation** -- splitting text into aligned `WordToken` objects with rasm (skeleton) forms

This is the entry point of the pipeline. Its outputs feed into both the two-pass alignment (Step 4) and the phonemizer (Step 6).

---

## Audio Ingestion

Audio loading is handled by `src/asr.py:load_audio()`, not the preprocessor itself. The preprocessor operates purely on text.

| Parameter    | Value                          |
|-------------|-------------------------------|
| Library      | soundfile (primary), librosa (fallback/resample) |
| Target SR    | 16,000 Hz (Whisper standard) |
| Channels     | Mono (averaged from multi-channel) |
| Dtype        | float32, normalised to [-1, 1] |
| Chunking     | Window-based parallel (configurable, default 10s with 1s overlap) |

### Supported Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV    | `.wav`    | Recommended |
| FLAC   | `.flac`   | Lossless |
| OGG    | `.ogg`    | Lossy |
| AIFF   | `.aiff`, `.aif` | macOS common |

### Window-Based Parallel Processing

For latency reduction, audio is segmented into overlapping windows before ASR:

```
|--- chunk 0 (0s-10s) ---|
                    |--- chunk 1 (9s-19s) ---|
                                       |--- chunk 2 (18s-28s) ---|
```

- **chunk_seconds**: Window size (default: 10.0s)
- **overlap_seconds**: Overlap between consecutive windows (default: 1.0s)
- Overlap regions use the middle-chunk preference for merging transcriptions
- CPU: parallel via `concurrent.futures.ProcessPoolExecutor`
- GPU/MPS: sequential chunked (GPU parallelism is internal to the model)

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

**Lam-alef ligatures** (U+FEF5 -- U+FEFC) are decomposed to `ل + ا`.

**Algorithm:** `str.translate()` for hamza table, then regex substitution for ligatures.

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

## Data Flow

```
Raw Arabic text (with harakat)
      |
      v
strip_harakat() --> rasm (skeleton)
      |
      v
normalise_hamza() --> normalised rasm
      |
      v
tokenise_reference() --> List[WordToken]
      |
      +---> two_pass_compare() [Step 4]
      +---> phonemize_verse()  [Step 6 -- uses full_text for phonemization]
```

---

## Input/Output Contract

**Input:** Raw Arabic text string (UTF-8), surah number, ayah number

**Output:** `List[WordToken]` where each token contains both the original diacritised form (`full_text`) and the stripped/normalised form (`rasm`) for alignment.

The `full_text` field is preserved to feed into the phonemizer, which requires diacritics for accurate grapheme-to-phoneme conversion and tajweed rule detection.
