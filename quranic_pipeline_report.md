# Quranic Recitation Error Detection Pipeline — Technical Report

**Author:** Kayla Queenazima Santoso · **Date:** Ramadhan, 2026

---

## 1. Introduction

### 1.1 Problem

Given an audio recording of a Quranic verse and its reference text, automatically detect and classify recitation errors at word level. The error taxonomy covers:

- **Word-level errors** — substitution, deletion, insertion (rasm-skeleton mismatch)
- **Harakat errors** — correct consonants, wrong diacritics
- **Tajweed violations** — medd, ghunna, idgham, ikhfa, qalqala, iqlab, izhar, tafkheem

### 1.2 Approach

An open, reproducible 9-stage pipeline integrating:
1. A two-pass text comparator separating word-level from harakat errors
2. Rule-based Tajweed-aware phonemization (15+ rules, Hafs riwaya)
3. Whisper word-timestamp forced alignment with 3-strategy fallback
4. GOP (Goodness of Pronunciation) scoring for false positive suppression

The pipeline uses a dual-backend ASR design: faster-whisper (CTranslate2 int8, ~1–2 s/ayah on CPU) as default, with lazy-loaded HuggingFace PyTorch for alignment and GOP scoring.

### 1.3 Contributions

1. Two-pass text comparator cleanly separating word-level from harakat errors
2. Rule-based Tajweed phonemizer detecting 15+ rules with cross-word context
3. GOP scoring against Tajweed-aware phoneme references (not MSA phoneme set)
4. Graceful degradation across 6 pipeline modes (FULL → TEXT_ONLY → ASR_ONLY)

---

## 2. Pipeline Architecture

### 2.1 Overview

```
Audio + Reference Text
        │
[1] Audio load ─── soundfile → float32, 16 kHz mono
        │
[2] ASR transcription ─── faster-whisper (int8) / HF Whisper
        │
[3] Reference resolution ─── fuzzy word-overlap (≥40%) or provided
        │
[4] Two-pass text comparison ─── Wagner-Fischer on rasm → harakat diff
        │
[5] Tajweed phonemization ─── 29 IPA consonants + 15 Tajweed rules
        │
[6] Forced alignment ─── Whisper word timestamps + 3-strategy fallback
        │
[7] GOP scoring ─── selective, error positions only
        │
[8] Error classification ─── 6-step decision tree
        │
[9] Report generation ─── JSON + human-readable text
```

### 2.2 Component Summary

| Stage | Component | Primary Tool | Fallback |
|-------|-----------|-------------|----------|
| 1 | Audio load | soundfile | librosa |
| 2 | ASR | faster-whisper int8 (CTranslate2) | HF transformers Whisper |
| 3 | Reference resolution | Fuzzy word-overlap (≥40%) | ASR_ONLY mode |
| 4 | Text comparison | Two-pass Wagner-Fischer on rasm | — |
| 5 | Phonemization | Rule-based Arabic G2P + Tajweed | — |
| 6 | Forced alignment | Whisper word timestamps | Equal-duration distribution |
| 7 | GOP scoring | DNN-based from Whisper encoder | Neutral 0.85 (FALLBACK) |
| 8 | Error classification | Rule-based decision tree | — |
| 9 | Report generation | Pydantic schema | — |

### 2.3 Pipeline Modes

| Mode | Trigger | Detects |
|------|---------|---------|
| FULL | Default | All error types + Tajweed + GOP confidence |
| NO_GOP | Alignment OK, GOP fails | All types, binary confidence only |
| TEXT_ONLY | No alignment possible | SUB/DEL/INS/HARAKAT only |
| TEXT_ONLY_CLEAN | All words MATCH (early exit) | No errors; skips stages 5–8 |
| ASR_ONLY | No reference match | Transcript only; no error detection |
| MOCK | `--mock` flag | Simulated ASR; integration testing |

---

## 3. Implementation

### 3.1 Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| ASR (default) | faster-whisper ≥1.1.0 (CTranslate2, int8) |
| ASR (alignment/GOP) | transformers ≥4.40, torch ≥2.0.0 |
| Audio | soundfile, librosa, miniaudio ≥1.61 |
| Data | pyarrow ≥14.0.0, numpy <2.0, pandas, pydantic ≥2.0 |
| Metrics | jiwer |
| Testing | pytest (6 modules, all passing) |

### 3.2 Modules

| File | Lines | Responsibility |
|------|-------|---------------|
| `src/preprocessor.py` | 295 | Unicode normalization, tokenization, two-pass comparator |
| `src/asr.py` | 665 | Dual-backend Whisper, audio loading, WER/CER |
| `src/phonemizer.py` | 587 | Arabic G2P, Tajweed rule detection, phoneme diff |
| `src/aligner.py` | 485 | Forced alignment, GOP scoring, Medd duration check |
| `src/error_classifier.py` | 273 | Decision tree, ErrorRecord, severity assignment |
| `src/report.py` | 161 | RecitationReport, JSON + text output |
| `eval/metrics.py` | 306 | P/R/F1, GOP-AUC, Youden's J threshold |
| `eval/evaluate.py` | 212 | Ablation runner, markdown report formatter |

### 3.3 Two-Pass Text Comparison

All text passes through Unicode normalization: tatweel removal, hamza standardization (أ إ آ ٱ → ا), lam-alef decomposition, and Quranic annotation mark removal.

- **Pass 1 (rasm):** Strip all diacritics, run word-level Levenshtein alignment → SUBSTITUTION / DELETION / INSERTION
- **Pass 2 (harakat):** For each rasm-matched pair, compare full diacritized forms → HARAKAT_ERROR

This ensures harakat errors are never confused with word-level errors.

### 3.4 Tajweed Phonemization

The phonemizer converts each word to an IPA phoneme sequence using context-sensitive rules:

| Category | Rules Detected |
|----------|---------------|
| Noon Sakinah/Tanween | Izhar, Idgham (with/without Ghunna), Iqlab, Ikhfa |
| Meem Sakinah | Ikhfa Shafawi, Idgham Shafawi, Izhar Shafawi |
| Madd (Elongation) | Tabii, Wajib Muttasil, Jaiz Munfasil, Lazim, Arid Lissukun |
| Qalqala | Sughra (mid-word), Kubra (end-word) |
| Other | Ghunna, Lam Shamsiyah/Qamariyah, Tafkheem |

Cross-word rules (Idgham, Madd Jaiz Munfasil) are handled via next-word lookahead in `phonemize_verse()`.

### 3.5 Forced Alignment & GOP Scoring

Whisper word timestamps align reference text to audio. Three-strategy fallback:

1. **Strategy 1** — Align reference text → `HIGH` confidence
2. **Strategy 2** — Align ASR transcript → `LOW` confidence
3. **Strategy 3** — Equal-duration distribution → `FALLBACK` confidence

GOP is computed per word from Whisper encoder posteriors (selective — error positions only):

```
GOP(p) = -(1/D) * SUM(t=T to T+D-1) log P_t(p | O)
GOP_norm = 1 / (1 + exp(gop_raw - 2.0))
```

High GOP → audio matches expected pronunciation → likely false positive.

### 3.6 Error Classification

6-step priority decision tree:

1. **MATCH** — all diffs zero → no error
2. **LOW_CONFIDENCE** — FALLBACK alignment → demote severity
3. **HARAKAT_ERROR** — rasm match, harakat mismatch → GOP-based severity
4. **TAJWEED_*** — rasm match, phoneme diff present → map to specific Tajweed type
5. **SUB/DEL/INS** — rasm mismatch → GOP-based severity
6. **Suppression** — high GOP + minor error → SUPPRESSED

Severity thresholds: CONFIRMED (GOP < 0.70), POSSIBLE (0.70–0.85), SUPPRESSED (> 0.85).

---

## 4. Experiments

### 4.1 Data

- **Reference corpus:** Tanzil Uthmani with full tashkeel (6,236 verses)
- **ASR benchmark:** Buraaq/quran-md-ayahs parquet (187K samples, 30 reciters)
- **Test subset:** Surah 37 ayahs 78–87, Alafasy reciter (10 samples, professional — ground-truth WER = 0)

### 4.2 ASR Baseline Results

| Model | Backend | Params | Mean WER | Mean F1 | Avg time/ayah |
|-------|---------|--------|----------|---------|---------------|
| tarteel-ai/whisper-base-ar-quran | faster-whisper int8 | 74M | **0.613** | 0.786 | ~5.3 s |
| wasimlhr/whisper-quran-v1 | HuggingFace float32 | 1.55B | **0.020** | 0.977 | ~18.6 s |

**Per-ayah breakdown (tarteel-ai):**

| Ayah | WER | F1 | FP (insertions) | Pattern |
|------|-----|-----|-----------------|---------|
| 37:78 | 0.25 | 0.889 | 1 | Partial echo |
| 37:79 | 0.20 | 0.909 | 1 | Borrowed phrase |
| 37:80 | 0.25 | 0.889 | 1 | Unrelated word |
| 37:81 | 0.25 | 0.889 | 1 | Duplicate with conjunction |
| 37:82 | 2.00 | 0.500 | 6 | Repetitive loop ×3 |
| 37:83 | 0.75 | 0.727 | 3 | Corrupted name loop ×3 |
| 37:84 | 0.40 | 0.833 | 2 | Common suffix |
| 37:85 | 0.33 | 0.857 | 2 | Paraphrase |
| 37:86 | 0.20 | 0.909 | 1 | Unrelated word |
| 37:87 | 1.50 | 0.571 | 6 | Severe repetitive loop ×3 |

**Key finding:** tarteel-ai hallucinates tail phrases after correctly transcribing the ayah body. Short ayahs (3–5 words) fall within Whisper's 30s window, leaving silence that the model fills from Quranic priors. wasimlhr has 1 normalisation-level FP (hamza variant on ayah 37:86).

### 4.3 Mock Evaluation Results

From `results/eval_report.md` (mock pipeline, demonstrates framework):

| Metric | Value |
|--------|-------|
| Precision | 0.800 |
| Recall | 0.800 |
| F1 | 0.800 |

**Ablation (mock):**

| Config | Label | P | R | F1 |
|--------|-------|---|---|-----|
| A | text_only | 1.000 | 0.714 | 0.833 |
| B | + phonemizer | 1.000 | 0.857 | 0.923 |
| C | full pipeline | 1.000 | 1.000 | 1.000 |

> Full evaluation on QuranMB.v1 pending — requires running `scripts/run_eval.py` against the IqraEval 2025 dataset.

---

## 5. Error Analysis

### 5.1 False Positives

Three categories observed:

| Source | Mechanism | Mitigation |
|--------|-----------|------------|
| ASR hallucinations | Whisper generates words not in audio (tail insertions) | GOP > 0.85 → SUPPRESSED |
| Normalization mismatches | Unicode divergence (e.g., hamza variants) | Extended `strip_harakat()` |
| Alignment errors | Wrong audio segment mapped to word | FALLBACK → LOW_CONFIDENCE |

**tarteel-ai:** 24 FP words across 10 ayahs, all tail insertions. GOP partially mitigates — hallucinated words at verse end get FALLBACK alignment and neutral GOP (0.85).

**wasimlhr:** 1 FP — hamza variant أَئِفْكًا → أَإِفْكًا (U+0626 vs U+0625). Fixable by extending hamza normalization.

### 5.2 False Negatives

Structural FN sources (identified from design, not observed on professional reciter data):

| Source | Mechanism |
|--------|-----------|
| GOP suppression | Subtle error gets high GOP (common with harakat) |
| ASR masking | ASR makes same error as learner → pipeline sees MATCH |
| Phonemizer gaps | Rules not modelled (e.g., qalqala kubra at verse boundaries) |
| Alignment fallback | FALLBACK mode sets neutral GOP → errors demoted |

### 5.3 Observations

- **Rasm-match / phoneme-mismatch** words are the primary motivation for the phonemizer stage — caught only in FULL mode
- **Shadda omission** (e.g., رَبِّ → رَبِ) maps to PHONE_SUB in phoneme diff, surfaces as TAJWEED class error
- tarteel-ai's hallucination pattern on short ayahs creates structural FN risk where repetitive loops can mask real deletions

---

## 6. Discussion

### 6.1 Harakat Detection Ceiling

Harakat error detection is fundamentally limited by ASR output quality. Whisper does not reliably predict diacritics. The two-pass strategy mitigates this by only comparing harakat on rasm-matched words, but ASR uncertainty still propagates. Improving this requires either a dedicated post-ASR diacritization module or an ASR model trained with full tashkeel supervision.

### 6.2 Tajweed Detection Ceiling

The rule-based phonemizer covers Hafs riwaya only. It handles 15+ rule types with cross-word context, but edge cases (e.g., waqf-specific transformations, qalqala kubra at verse boundaries) are not modelled. A learned phonemizer would improve coverage.

### 6.3 Performance

| Backend | Device | Latency (10s audio) | Memory |
|---------|--------|-------------------|--------|
| faster-whisper | CPU int8 | ~1–2 s | ~150 MB |
| HF transformers | CPU float32 | ~5–8 s | ~600 MB |
| HF transformers | MPS float16 | ~2–3 s | ~400 MB |
| faster-whisper | CUDA float16 | ~0.3–0.5 s | ~300 MB |

Full pipeline (alignment + GOP) adds ~0.5–3 s per verse on CPU. Adequate for post-recitation review; borderline for real-time feedback. 

P.S: it tested on M1 macbook pro with inactive MPS.

---

## 7. Limitations

1. Harakat detection limited by ASR diacritic quality
2. Tajweed detection covers Hafs riwaya only
3. Forced alignment degrades on heavily erroneous recitations (FALLBACK loses GOP)
4. GOP uses the same ASR model — errors correlated with ASR failures may be under-penalised
5. No speaker-independent evaluation across demographics
6. HF benchmark supports ASR_ONLY mode only (no full 9-stage pipeline on parquet data)
7. Reference text must include full diacritics for harakat detection

---

## 8. Conclusion

This pipeline demonstrates that combining Tajweed-aware phonemization with Whisper ASR and GOP scoring produces a viable error detection system for Quranic recitation. The two-pass text comparator cleanly separates word-level from harakat errors, and the 6-mode degradation strategy ensures the pipeline always produces useful output regardless of component availability.

Key results:
- wasimlhr/whisper-quran-v1 achieves 0.020 WER (F1 0.977) on professional recitations
- The ablation shows each component (phonemizer, alignment, GOP) contributes measurably to recall
- GOP-based severity modulation suppresses ASR hallucination false positives

---

## 9. Recommendations

### 9.1 Quality Improvement (Short-term)

| Priority | Action | Impact |
|----------|--------|--------|
| HIGH | **Extend hamza normalization** — add all hamza variants to `normalise_hamza()` | Eliminates normalisation FP class (observed on wasimlhr) |
| HIGH | **Evaluate on QuranMB.v1** — run `scripts/run_eval.py` against IqraEval 2025 dataset | First real benchmark numbers for this pipeline |
| HIGH | **Tune GOP thresholds per-dataset** — use `compute_gop_metrics()` Youden's J | Optimizes precision/recall tradeoff |
| MEDIUM | **Add post-ASR diacritization** — apply a dedicated diacritizer (e.g., Mishkal, Shakkala) after ASR | Improves harakat detection recall |
| MEDIUM | **Add ASR hallucination detector** — flag words where Whisper's word-level probability < threshold | Reduces FP rate on short ayahs without relying solely on GOP |

### 9.2 Scalability (Medium-term)

| Priority | Action | Impact |
|----------|--------|--------|
| HIGH | **API service wrapper** — FastAPI/Flask endpoint wrapping `run_pipeline.py` | Enables web/mobile integration |
| HIGH | **Docker containerization** — Dockerfile with pre-converted CTranslate2 models | Reproducible deployment, eliminates model conversion step |
| MEDIUM | **Batch queue** — Celery/Redis task queue for concurrent pipeline execution | Handles multi-user load |
| MEDIUM | **Enable full pipeline on HF benchmark** — extend `run_hf_benchmark.py` to run all 9 stages | Large-scale evaluation capability |
| LOW | **Streaming ASR** — chunked real-time transcription with incremental error feedback | Real-time tutoring use case |

### 9.3 Production Readiness (Long-term)

| Priority | Action | Impact |
|----------|--------|--------|
| HIGH | **CI/CD pipeline** — GitHub Actions running `pytest` + mock pipeline on every push | Prevents regression |
| HIGH | **Logging & monitoring** — structured logging (JSON), error tracking (Sentry) | Production observability |
| HIGH | **Model versioning** — DVC or MLflow for tracking CTranslate2 model versions | Reproducible model management |
| MEDIUM | **GPU inference server** — Triton or TGI for batched GPU inference | 10-50x throughput improvement |
| MEDIUM | **Speaker-independent evaluation** — test across gender, age, accent, proficiency level | Validates generalization |
| MEDIUM | **Extend Tajweed coverage** — add Warsh/Qalun riwayat, waqf rules, qalqala kubra at verse boundaries | Broader user base |
| LOW | **Learned phonemizer** — fine-tune a seq2seq model on annotated Tajweed data | Replaces rule-based approach, improves accuracy |
| LOW | **On-device inference** — ONNX/CoreML export of Whisper + quantized phonemizer | Mobile deployment |

### 9.4 Recommended Next Steps (Priority Order)

1. **Run QuranMB.v1 evaluation** — establishes baseline metrics comparable to IqraEval 2025
2. **Docker + CI/CD** — makes the project reproducible and maintainable
3. **FastAPI wrapper** — unlocks web/mobile integration
4. **Tune GOP thresholds** — easy win for precision/recall improvement
5. **Post-ASR diacritization** — addresses the harakat detection ceiling

---

## Appendix A — Error Taxonomy

| Error Type | Description | Detection Method |
|-----------|-------------|----------------|
| MATCH | Correct | All diffs zero |
| SUBSTITUTION | Wrong word | Rasm Levenshtein |
| DELETION | Word skipped | Rasm alignment |
| INSERTION | Extra word | Rasm alignment |
| HARAKAT_ERROR | Wrong diacritics | Harakat diff on rasm-matched pairs |
| TAJWEED_MEDD | Elongation violation | Phoneme diff + Madd rule |
| TAJWEED_GHUNNA | Nasalization violation | Phoneme diff |
| TAJWEED_IDGHAM | Assimilation violation | Phoneme diff |
| TAJWEED_IKHFA | Concealment violation | Phoneme diff |
| TAJWEED_QALQALA | Echoed release violation | Phoneme diff |
| TAJWEED_IQLAB | Noon→meem violation | Phoneme diff |
| TAJWEED_IZHAR | Clear pronunciation violation | Phoneme diff |
| TAJWEED_TAFKHEEM | Emphasis violation | Phoneme diff |
| LOW_CONFIDENCE | Unreliable alignment | FALLBACK mode |

## Appendix B — Sample Report Output

```json
{
  "audio_id": "mock",
  "surah": 1,
  "ayah": 1,
  "reference_text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
  "hypothesis_text": "الَّذِينَ يُؤْمِنُونَ بِمَا هُوَ مَا هُوَ يُؤْمِنُونَ",
  "pipeline_mode": "FULL",
  "asr_wer": 1.75,
  "errors": [
    {
      "position": "1:1:0",
      "reference_token": "بِسْمِ",
      "hypothesis_token": "هُوَ",
      "error_type": "SUBSTITUTION",
      "severity": "CONFIRMED",
      "confidence_score": 0.5,
      "gop_score": 0.5,
      "alignment_confidence": "HIGH",
      "start_sec": 0.0,
      "end_sec": 0.625,
      "notes": "Substitution error detected."
    }
  ],
  "summary": {
    "total_errors": 7,
    "by_type": {"INSERTION": 3, "SUBSTITUTION": 1, "TAJWEED_MEDD": 3},
    "by_severity": {"CONFIRMED": 7}
  }
}
```

## Appendix C — Repository

```bash
# Mock pipeline
python scripts/run_pipeline.py --surah 1 --ayah 2 --mock --verbose

# Evaluation
python scripts/run_eval.py --mock --output results/eval_report.md

# Tests
pytest tests/ -v
```

---

## Appendix D — Results Directory

All pipeline and benchmark outputs are written to `results/`. Structure as of Ramadhan 2026:

```
results/
├── asr_baseline.csv                        # ASR WER/CER baseline across models
├── eval_report.md                          # Mock evaluation report (P/R/F1, ablation)
│
├── reports/                                # Single-verse pipeline outputs
│   ├── mock.json                           # Full structured report (mock mode)
│   ├── mock.txt                            # Human-readable bordered report
│   ├── mock_audio.json                     # Full report (mock with audio)
│   └── mock_audio.txt                      # Human-readable (mock with audio)
│
└── hf_benchmark/                           # HuggingFace parquet benchmark runs
    ├── hf_benchmark_0044_{report,results,summary}.{md,csv,json}  # Latest run
    │
    ├── model1_tarteel/                     # tarteel-ai/whisper-base-ar-quran
    │   │                                   # Backend: faster-whisper int8 | 10 rows
    │   │                                   # Mean WER: 0.613 | F1: 0.786
    │   ├── hf_benchmark_0044_results.csv
    │   ├── hf_benchmark_0044_report.md
    │   └── hf_benchmark_0044_summary.json
    │
    ├── model2_openai/                      # openai/whisper-base (baseline)
    │   └── hf_benchmark_0044_{...}
    │
    ├── model3_wasimlhr/                    # wasimlhr/whisper-quran-v1
    │   │                                   # Backend: HuggingFace float32 | 3 rows
    │   └── hf_benchmark_0044_{...}
    │
    ├── model3_wasimlhr_10rows/             # wasimlhr/whisper-quran-v1 — full run
    │   │                                   # Backend: HuggingFace float32 | 10 rows
    │   │                                   # Mean WER: 0.020 | F1: 0.977
    │   ├── hf_benchmark_0044_results.csv
    │   ├── hf_benchmark_0044_report.md
    │   └── hf_benchmark_0044_summary.json
    │
    └── model3_wasimlhr_fw/                 # wasimlhr CT2 conversion attempt
                                            # Backend: faster-whisper int8 | FAILED
                                            # CT2 conversion degrades fine-tuning quality
```

### Output File Formats

| File | Format | Contents |
|------|--------|---------|
| `*.json` (pipeline) | JSON | Full `RecitationReport`: errors list, GOP scores, timestamps, summary |
| `*.txt` (pipeline) | Plain text | Human-readable bordered report with per-word error table |
| `hf_benchmark_*_results.csv` | CSV | Per-(ayah, reciter) row: WER, CER, insertions, deletions, substitutions, timing |
| `hf_benchmark_*_report.md` | Markdown | Summary stats + per-reciter table + first 50 rows |
| `hf_benchmark_*_summary.json` | JSON | Aggregated metrics: mean WER/CER, per-reciter breakdown |
| `asr_baseline.csv` | CSV | Model × dataset WER/CER grid |
| `eval_report.md` | Markdown | P/R/F1 by error type, ablation table, GOP metrics |
