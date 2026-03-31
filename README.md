# Quranic Recitation Error Detection Pipeline

Automated detection of Quranic recitation errors using Whisper ASR, Tajweed-aware phonemization, forced alignment, and GOP scoring. Given audio + reference verse, produces a structured JSON report with error types, timing, and confidence scores.

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/kaylaque/quranic-pipeline.git
cd quranic-pipeline
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Convert Whisper model to CTranslate2 (one-time, ~300 MB download)
ct2-transformers-converter \
    --model tarteel-ai/whisper-base-ar-quran \
    --output_dir models/whisper-quran-ct2 \
    --quantization int8

# 4. Verify
python scripts/check_env.py
```

> If `ct2-transformers-converter` fails with a `dtype` kwarg error, see `docs/faster_whisper_integration.md` for a Python workaround.
> 
> You can also pull models from Huggingface: [kaylazima/quranic-model](https://huggingface.co/kaylazima/quranic-model/tree/main)
---

## Usage

### Run Pipeline (single file)

```bash
# With audio file (auto-detect surah/ayah)
python scripts/run_pipeline.py --audio data/samples/recitation.wav --verbose

# With known surah/ayah
python scripts/run_pipeline.py --audio data/samples/recitation.wav --surah 1 --ayah 1 --verbose

# Mock mode (no audio/model needed)
python scripts/run_pipeline.py --surah 1 --ayah 1 --mock --verbose
```

### Run Benchmark (batch WAV folder)

```bash
python scripts/run_benchmark.py \
    --audio_dir data/samples/ \
    --output_dir results/benchmark/
```

### Run HuggingFace Benchmark (large-scale)

```bash
python scripts/run_hf_benchmark.py \
    --parquet data/benchmark_hf/0044.parquet \
    --reciters alafasy \
    --max_rows 10
```

### Run Evaluation

```bash
python scripts/run_eval.py --mock --output results/eval_report.md
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Docker

Build and run with Docker Compose (no local Python environment needed):

```bash
# Build image
docker compose build

# Single-verse pipeline
docker compose run pipeline --surah 1 --ayah 1 --audio data/samples/mock.wav --verbose

# Mock mode (no audio/model required)
docker compose run pipeline --surah 1 --ayah 1 --mock --verbose

# Batch WAV benchmark
docker compose run benchmark --audio_dir data/samples/ --output_dir results/benchmark/

# HuggingFace parquet benchmark
docker compose run hf-benchmark --parquet data/benchmark_hf/0044.parquet --reciters alafasy --max_rows 10

# Test suite
docker compose run tests
```

The pre-converted CTranslate2 model (`models/whisper-quran-ct2/`, 74 MB) is baked into the image. `data/`, `results/`, and the HuggingFace model cache are mounted as volumes so outputs persist on the host.

For GPU server deployment, swap the `FROM` line in `Dockerfile` to a CUDA base image and pass `--device cuda --compute_type float16`.

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio` | None | Audio file (WAV/FLAC/OGG/AIFF) |
| `--surah` | 0 | Surah number (0 = auto-detect) |
| `--ayah` | 0 | Ayah number (0 = auto-detect) |
| `--reference` | None | Reference Arabic text (optional) |
| `--output_dir` | `results/reports/` | Output directory |
| `--mock` | False | Mock mode (no audio/models) |
| `--verbose` | False | Debug logging |
| `--backend` | `faster-whisper` | ASR backend: `faster-whisper` or `huggingface` |
| `--model` | `tarteel-ai/whisper-base-ar-quran` | HuggingFace model ID |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |
| `--model_dir` | `models/whisper-quran-ct2` | CTranslate2 model path |

---

## Pipeline Stages

```
Audio + Reference Text
        |
        v
[1] Load Audio ─── soundfile → float32, 16 kHz mono
        |
[2] ASR ─── faster-whisper (int8, default) or HuggingFace (PyTorch)
        |
[3] Reference Resolution ─── fuzzy match (≥40% word overlap) or provided
        |
[4] Two-Pass Text Diff ─── Wagner-Fischer on rasm → harakat comparison
        |
[5] Phonemization ─── 29 IPA consonants + 15 Tajweed rules
        |
[6] Forced Alignment ─── Whisper word timestamps, 3-strategy fallback
        |
[7] GOP Scoring ─── selective, error positions only
        |
[8] Error Classification ─── 6-step decision tree
        |
[9] Report ─── JSON + text output
```

**Pipeline modes:** FULL, NO_GOP, TEXT_ONLY, TEXT_ONLY_CLEAN, ASR_ONLY, MOCK

---

## Error Types

| Type | Source |
|------|--------|
| SUBSTITUTION, DELETION, INSERTION | Rasm text diff |
| HARAKAT_ERROR | Harakat diff (rasm matches) |
| TAJWEED_MEDD, GHUNNA, IDGHAM, IKHFA, QALQALA, IQLAB, IZHAR, TAFKHEEM | Phoneme diff |
| LOW_CONFIDENCE | Fallback alignment |

**Severity:** CONFIRMED (GOP < 0.7), POSSIBLE (0.7-0.85), SUPPRESSED (> 0.85)

---

## Project Structure

```
quranic-pipeline/
├── src/                    # Core modules (2,466 lines)
│   ├── preprocessor.py     # Text normalization, tokenization, two-pass diff
│   ├── asr.py              # Dual-backend Whisper ASR, audio I/O
│   ├── phonemizer.py       # Arabic G2P, Tajweed rules
│   ├── aligner.py          # Forced alignment, GOP scoring
│   ├── error_classifier.py # Error classification, severity
│   └── report.py           # JSON/text report generation
├── eval/                   # Evaluation (518 lines)
│   ├── metrics.py          # P/R/F1, GOP-AUC, threshold optimization
│   └── evaluate.py         # Ablation runner, report formatter
├── scripts/                # Entry points (2,156 lines)
│   ├── run_pipeline.py     # Single-file pipeline
│   ├── run_benchmark.py    # Batch WAV processing
│   ├── run_hf_benchmark.py # HuggingFace parquet benchmarking
│   ├── run_eval.py         # Evaluation runner
│   ├── run_asr_baseline.py # ASR baseline
│   ├── hf_parquet_utils.py # Parquet/MP3 utilities
│   └── check_env.py        # Dependency check
├── tests/                  # Tests (1,455 lines)
├── data/                   # Audio samples, parquet, reference CSVs
├── models/                 # CTranslate2 converted models
├── results/                # Pipeline outputs, benchmark reports
├── docs/                   # Per-module documentation (10 files)
├── requirements.txt
├── project.md              # Technical summary
└── quranic_pipeline_report.md  # Research report
```

---

## ASR Backends

| Backend | Engine | Speed (CPU, 10s audio) | Memory | MPS |
|---------|--------|----------------------|--------|-----|
| **faster-whisper** (default) | CTranslate2 int8 | ~1-2s | ~150 MB | No (CPU fallback) |
| huggingface | PyTorch float32 | ~5-8s | ~600 MB | Yes |

The faster-whisper backend is recommended for CPU-only and production use. HuggingFace backend is lazy-loaded for alignment/GOP only when errors are detected.

---

## Pre-Converted Models

Two CTranslate2 models are included for faster-whisper ASR:

| Model | Size | Source | Architecture | Notes |
|-------|------|--------|--------------|-------|
| `models/whisper-quran-ct2/` | 73 MB | `tarteel-ai/whisper-base-ar-quran` | Whisper Base (74M) | **Recommended** — int8 quantization, production-ready |
| `models/whisper-quran-v1-ct2/` | 2.9 GB | `wasimlhr/whisper-quran-v1` | Whisper Large-v3 (1.55B) | Larger model; int8 conversion degrades fine-tuning quality — **use HuggingFace backend instead** |

To use a different model, pass `--model_dir path/to/model` to `run_pipeline.py` or `run_hf_benchmark.py`.

---

## Optimizations

- **Early exit:** All-MATCH recitations skip stages 5-8 (~80% compute saved)
- **Selective GOP:** Only error-flagged words scored
- **Lazy model loading:** HF PyTorch model loaded only when needed
- **Single audio load:** Numpy array reused across ASR, alignment, GOP
- **Module-level caching:** Models loaded once per process
- **No ffmpeg:** Audio via soundfile numpy arrays directly

---

## Requirements

```
faster-whisper>=1.1.0    # CTranslate2 ASR backend
transformers>=4.40.0     # HuggingFace backend (alignment + GOP)
torch>=2.0.0             # PyTorch
soundfile                # Audio I/O (WAV/FLAC/OGG/AIFF)
librosa                  # Resampling fallback
miniaudio>=1.61          # MP3 decoding (HF benchmark)
pyarrow>=14.0.0          # Parquet reading (HF benchmark)
jiwer                    # WER/CER computation
numpy<2.0                # PyTorch 2.1 compatibility
pandas
pydantic>=2.0
pytest
```

---

## Known Limitations

- CTranslate2 doesn't expose encoder logits — GOP requires HuggingFace model
- Tajweed detection is rule-based (Hafs riwaya only)
- Harakat detection limited by ASR diacritic quality
- Auto-detect needs ≥40% word overlap; non-standard pronunciations may not match
- Short audio (<0.5s) falls back to equal-duration alignment
- Reference text must include full diacritics for harakat detection

---

## Documentation

Detailed per-module docs in `docs/`:

| Doc | Content |
|-----|---------|
| [preprocessor.md](docs/preprocessor.md) | Text normalization, tokenization |
| [asr.md](docs/asr.md) | ASR backends, models, audio loading |
| [phonemizer.md](docs/phonemizer.md) | G2P, Tajweed rules |
| [aligner.md](docs/aligner.md) | Forced alignment, GOP |
| [error_classifier.md](docs/error_classifier.md) | Classification decision tree |
| [report.md](docs/report.md) | Output formats |
| [evaluation.md](docs/evaluation.md) | Metrics, ablation |
| [benchmark.md](docs/benchmark.md) | Batch WAV processing |
| [hf_benchmark.md](docs/hf_benchmark.md) | HuggingFace parquet benchmarking |
| [faster_whisper_integration.md](docs/faster_whisper_integration.md) | CTranslate2 setup |
