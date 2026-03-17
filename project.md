# Quranic Recitation Error Detection Pipeline — Project Summary

**Author:** Kayla Queenazima Santoso
**Last Updated:** 2026-03-17
**Status:** Production-ready (research prototype)

---

## Architecture

A 9-stage pipeline that takes Quranic recitation audio + reference text and produces a structured JSON error report.

```
Audio + Reference → [1] Load Audio → [2] ASR → [3] Reference Resolution
→ [4] Two-Pass Text Diff → [5] Phonemization → [6] Forced Alignment
→ [7] GOP Scoring → [8] Error Classification → [9] Report
```

### Modules

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Preprocessor | `src/preprocessor.py` | 295 | Unicode normalization, rasm tokenization, Wagner-Fischer two-pass diff |
| ASR | `src/asr.py` | 665 | Dual-backend Whisper (faster-whisper int8 + HuggingFace PyTorch), audio I/O |
| Phonemizer | `src/phonemizer.py` | 587 | Arabic G2P (29 IPA consonants), 15+ Tajweed rules, cross-word context |
| Aligner | `src/aligner.py` | 485 | Whisper word-timestamp alignment, 3-strategy fallback, GOP scoring |
| Classifier | `src/error_classifier.py` | 273 | 6-step decision tree, 14 error types, GOP-based severity |
| Report | `src/report.py` | 161 | JSON + bordered text output, Pydantic models |
| Metrics | `eval/metrics.py` | 306 | P/R/F1, GOP-AUC, Youden's J threshold |
| Evaluator | `eval/evaluate.py` | 212 | Ablation runner, markdown report formatter |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` (582 lines) | End-to-end pipeline runner (single file) |
| `scripts/run_benchmark.py` (531 lines) | Batch WAV folder processing |
| `scripts/run_hf_benchmark.py` (503 lines) | Large-scale HuggingFace parquet benchmarking |
| `scripts/run_eval.py` (164 lines) | Evaluation against ground truth |
| `scripts/run_asr_baseline.py` (172 lines) | ASR-only WER/CER baseline |
| `scripts/check_env.py` (45 lines) | Dependency verification |
| `scripts/hf_parquet_utils.py` (159 lines) | Parquet reading + MP3 decoding utilities |

**Total:** ~4,600 lines production code, ~1,450 lines tests, ~6,050 lines total.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| ASR (default) | faster-whisper (CTranslate2, int8) — ~1-2s/ayah on CPU |
| ASR (fallback) | HuggingFace transformers (PyTorch float32) — ~5-8s/ayah |
| Audio I/O | soundfile (WAV/FLAC/OGG/AIFF), miniaudio (MP3 for HF dataset) |
| Text alignment | Custom Wagner-Fischer DP (word-level Levenshtein) |
| Phonemization | Rule-based Arabic G2P with Tajweed context |
| Forced alignment | Whisper word timestamps + 3-strategy fallback cascade |
| GOP scoring | DNN-based from Whisper encoder posteriors, sigmoid-normalized |
| Data models | Pydantic v2 |
| Metrics | jiwer (WER/CER), custom P/R/F1 |
| Testing | pytest (6 test modules, all passing) |

---

## Pipeline Modes

| Mode | When | What works |
|------|------|------------|
| FULL | Default | All 9 stages, Tajweed + GOP |
| NO_GOP | Alignment OK, GOP fails | All types, binary confidence |
| TEXT_ONLY | No alignment possible | SUB/DEL/INS/HARAKAT only |
| TEXT_ONLY_CLEAN | All words match (early exit) | No errors, skips stages 5-8 |
| ASR_ONLY | No reference found | Transcript only, no error detection |
| MOCK | `--mock` flag | Simulated ASR, integration testing |

---

## Error Detection

### Error Types (14)

- **Word-level:** SUBSTITUTION, DELETION, INSERTION
- **Diacritic:** HARAKAT_ERROR
- **Tajweed:** MEDD, GHUNNA, IDGHAM, IKHFA, QALQALA, IQLAB, IZHAR, TAFKHEEM
- **Meta:** MATCH, LOW_CONFIDENCE

### Severity (GOP-based)

| Severity | GOP Range | Action |
|----------|-----------|--------|
| CONFIRMED | < 0.70 | Show to learner |
| POSSIBLE | 0.70 - 0.85 | Show to teacher for review |
| SUPPRESSED | > 0.85 | Hide (likely false positive) |

---

## Deployment Conditions (as of 2026-03-17)

### Environment

- **OS:** macOS (Darwin 21.6.0, Apple Silicon)
- **Python:** 3.9
- **GPU:** Not available (CPU-only deployment)
- **Backend:** faster-whisper int8 (default) — no MPS/CUDA required

### Models

| Model | Location | Size | Status |
|-------|----------|------|--------|
| tarteel-ai/whisper-base-ar-quran (CT2) | `models/whisper-quran-ct2/` | ~73 MB | Converted, ready |
| wasimlhr/whisper-quran-v1 (CT2) | `models/whisper-quran-v1-ct2/` | ~1.5 GB | Converted, available |

### Data

| Dataset | Location | Description |
|---------|----------|-------------|
| Mock audio | `data/samples/mock.wav`, `mock-2.wav` | 1-second silence for testing |
| HF parquet | `data/benchmark_hf/0044.parquet` | Surah 44, 30 reciters (Buraaq dataset) |
| Reference CSV | `data/reference/benchmark_mapping.csv` | Benchmark file mappings |

### Test Status

All 6 test modules passing:
- `test_preprocessor.py` (141 lines) — text normalization, tokenization, two-pass comparison
- `test_asr.py` (271 lines) — device selection, audio loading, format validation
- `test_phonemizer.py` (202 lines) — phoneme mapping, Tajweed rules, diffs
- `test_aligner.py` (321 lines) — mock alignment, GOP, fallback, Medd duration
- `test_classifier.py` (275 lines) — 6-step decision tree, severity assignment
- `test_metrics.py` (245 lines) — P/R/F1, GOP metrics, AUC

### Recent Fixes (2026-03-17)

1. CRITICAL: Temp file leak in MP3 decoding
2. MAJOR: Missing pyarrow/miniaudio in requirements.txt
3. MAJOR: miniaudio import guard for graceful fallback
4. MAJOR: docs/hf_benchmark.md rewritten to match implementation
5. MINOR: Module-level imports (removed per-row import in hot loop)
6. MINOR: Case-insensitive reciter filtering
7. MINOR: Deduplicated aggregation logic

---

## ASR Benchmark Results

Benchmarked on Buraaq/quran-md-ayahs (Surah 37, ayahs 78-87, Alafasy reciter, 10 samples):

| Model | Backend | Mean WER | Mean F1 | Speed (CPU) |
|-------|---------|----------|---------|-------------|
| tarteel-ai/whisper-base-ar-quran | faster-whisper int8 | 0.613 | 0.786 | ~5.3s/ayah |
| wasimlhr/whisper-quran-v1 | HuggingFace float32 | 0.020 | 0.977 | ~18.6s/ayah |

**Key finding:** tarteel-ai hallucinates tail phrases on short ayahs (24 FP insertions). wasimlhr achieves near-perfect fidelity with 1 normalisation-level FP.

---

## Known Limitations

- GOP scoring requires HuggingFace PyTorch model (CTranslate2 doesn't expose encoder logits)
- Tajweed rule detection is rule-based, not learned — limited to Hafs riwaya
- Forced alignment degrades on heavily erroneous recitations
- Harakat detection limited by ASR diacritic output quality
- No speaker-independent evaluation across demographics
- HF benchmark supports ASR_ONLY mode only (no full 9-stage pipeline)

---

## Documentation Index

| Document | Content |
|----------|---------|
| `README.md` | Installation, quick start, usage examples |
| `project.md` | This file — technical summary |
| `quranic_pipeline_report.md` | Research report with methodology and results |
| `FIXES_APPLIED.md` | Code review fixes log (2026-03-17) |
| `docs/preprocessor.md` | Text normalization and tokenization process |
| `docs/asr.md` | ASR backends, models, audio loading |
| `docs/phonemizer.md` | G2P mapping, Tajweed rules (15+) |
| `docs/aligner.md` | Forced alignment, GOP scoring, fallback strategies |
| `docs/error_classifier.md` | Decision tree, error types, severity |
| `docs/report.md` | Report generation and output formats |
| `docs/evaluation.md` | Metrics, GOP evaluation, ablation design |
| `docs/benchmark.md` | Batch WAV processing |
| `docs/hf_benchmark.md` | HuggingFace parquet large-scale benchmarking |
| `docs/faster_whisper_integration.md` | CTranslate2 setup and model conversion |
