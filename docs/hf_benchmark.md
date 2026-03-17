# HuggingFace Dataset Benchmarking -- `Buraaq/quran-md-ayahs`

## Overview

Large-scale benchmarking of the Quranic recitation error detection pipeline using the **Buraaq/quran-md-ayahs** HuggingFace dataset. This dataset provides 187,080 ayah-level audio recordings across 30 reciters and 114 surahs, with ground truth Arabic text, enabling systematic WER/CER evaluation at scale.

This extends the existing `scripts/run_benchmark.py` (WAV folder benchmarking) with a new `scripts/run_hf_benchmark.py` that reads parquet files, decodes embedded MP3 audio directly to numpy arrays (no ffmpeg), and produces stratified reports per reciter, per surah, and overall.

---

## Dataset Description

| Property | Value |
|----------|-------|
| **Source** | [Buraaq/quran-md-ayahs](https://huggingface.co/datasets/Buraaq/quran-md-ayahs) |
| **Total samples** | 187,080 (6,236 ayahs x 30 reciters) |
| **Audio duration** | 450+ hours |
| **Format** | 50 Parquet files (0000-0049), 200MB-970MB each, ~35GB total |
| **Audio encoding** | MP3 (32-192kbps), embedded as binary in parquet |
| **Published** | NeurIPS 2025 (Muslims in ML Workshop) |

### Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `surah_id` | int32 | Chapter number (1-114) |
| `ayah_id` | int32 | Verse number within surah |
| `surah_name_ar` | string | Chapter name in Arabic |
| `surah_name_en` | string | Chapter name in English |
| `surah_name_tr` | string | Chapter name transliterated |
| `ayah_count` | int32 | Total verses in chapter |
| `ayah_ar` | string | Complete verse in Arabic (Uthmani script, with tashkeel) |
| `ayah_en` | string | English translation |
| `ayah_tr` | string | Romanized transliteration |
| `reciter_id` | string | Unique reciter identifier (e.g., `alafasy`, `husary`) |
| `reciter_name` | string | Reciter display name with bitrate (e.g., `Alafasy_64kbps`) |
| `audio` | struct | `{bytes: binary, path: string}` — MP3 audio data and filename |

### Reciters (30 total)

30 professional Quranic reciters with audio bitrates ranging from 32kbps to 192kbps, covering different recitation styles (Murattal, Mujawwad, Teacher, Warsh variant).

---

## Dependencies

### Python Packages (add to requirements.txt)

```
pyarrow>=14.0.0      # Parquet file reading
miniaudio>=1.61      # MP3 decoding (in-memory, no ffmpeg)
```

Both are already in the current `requirements.txt`.

### System Dependencies

**None.** MP3 decoding uses `miniaudio` (pure Python + C extension) with optional `librosa` fallback. No ffmpeg required.

### Already Available

These are already in `requirements.txt` and used by the benchmark:
- `pandas`, `jiwer`, `librosa`, `soundfile`, `numpy`

---

## Process Stages

### Stage 1: Parquet Acquisition

Download one or more parquet files from the HuggingFace Hub. Files are split by surah range (~89 ayahs × 30 reciters = ~2,600 rows per file).

```bash
# Download a single parquet file (manual)
wget -O data/benchmark_hf/0044.parquet \
  'https://huggingface.co/datasets/Buraaq/quran-md-ayahs/resolve/refs%2Fconvert%2Fparquet/default/train/0044.parquet'
```

Or pre-downloaded at `data/benchmark_hf/0044.parquet` (Surah 37, 190MB).

### Stage 2: Audio Decoding

For each row in the parquet:

1. Extract `audio.bytes` (MP3 binary data)
2. Decode to float32 numpy array using `miniaudio.decode()` **entirely in-memory**
3. Normalize to 16kHz mono (matching pipeline expectation)

**No disk writes, no ffmpeg required.** If `miniaudio` is not available, falls back to `librosa` with a temp file.

### Stage 3: Transcription

For each decoded audio:

1. Call the selected ASR backend from `src/asr.py`:
   - **faster-whisper** (default): `transcribe_faster_whisper(fw_model, audio_array, sr)` — CTranslate2 int8, ~4x faster on CPU
   - **huggingface**: `transcribe_whisper(model, processor, audio_array, sr, device)` — PyTorch backend
2. Get hypothesis text
3. Compute WER/CER against ground truth `ayah_ar` after normalizing both with `strip_harakat()`

### Stage 4: Metrics Aggregation

Aggregate results at multiple levels:
- **Per-ayah**: WER, CER, insertions, deletions, substitutions
- **Per-reciter**: Mean/median WER across ayahs
- **Overall**: Global mean WER/CER

---

## Output Format

### Results CSV

**Path**: `results/hf_benchmark/hf_benchmark_{parquet_index}_results.csv`

```csv
parquet_index,surah_id,ayah_id,surah_name_en,reciter_id,reciter_name,wer,cer,insertions,deletions,substitutions,reference_text,hypothesis_text,processing_time_sec,success,error_message
0044,37,78,Those drawn up in Ranks,alafasy,Alafasy_64kbps,0.0,0.0,0,0,0,وَتَرَكْنَا...,وَتَرَكْنَا...,2.38,True,
```

One row per (ayah, reciter) pair processed.

### Markdown Report

**Path**: `results/hf_benchmark/hf_benchmark_{parquet_index}_report.md`

Contains:
- **Summary**: Total rows, success rate, mean WER/CER, avg processing time
- **Per-Reciter Table**: Mean WER/CER per reciter
- **Per-Row Results**: First 50 rows in tabular format
- **Failed Rows**: List of failures with error messages

### JSON Summary

**Path**: `results/hf_benchmark/hf_benchmark_{parquet_index}_summary.json`

Machine-readable aggregated metrics for CI/CD integration or dashboards.

---

## Command-Line Interface

```bash
python3 scripts/run_hf_benchmark.py \
    --parquet <path>           # Path to a parquet file (required)
    --output_dir <path>        # Output directory (default: results/hf_benchmark/)
    --mode <mode>              # "asr_only" (default: only ASR WER/CER computation)
    --model <model_id>         # HuggingFace Whisper model (default: tarteel-ai/whisper-base-ar-quran)
    --device <device>          # "auto", "cpu", "cuda", "mps" (default: auto)
    --backend <backend>        # ASR backend: "faster-whisper" (default) or "huggingface"
    --compute_type <type>      # CTranslate2 compute type: "auto", "int8", "float16", "float32" (default: auto)
    --model_dir <path>         # CTranslate2 model directory (default: models/whisper-quran-ct2)
    --reciters <ids>           # Comma-separated reciter IDs to include (e.g. alafasy,husary)
    --surahs <ids>             # Comma-separated surah IDs to include (e.g. 37)
    --max_rows <int>           # Max rows to process (default: 0 = all)
```

### Examples

**Quick validation — 5 rows, one reciter, faster-whisper (default):**

```bash
python3 scripts/run_hf_benchmark.py \
    --parquet data/benchmark_hf/0044.parquet \
    --reciters alafasy \
    --max_rows 5 \
    --device cpu
```

**Single parquet, all reciters, faster-whisper — ~25 min on CPU (int8):**

```bash
python3 scripts/run_hf_benchmark.py \
    --parquet data/benchmark_hf/0044.parquet \
    --device cpu
```

**Use HuggingFace backend explicitly — ~1.5 hours on CPU:**

```bash
python3 scripts/run_hf_benchmark.py \
    --parquet data/benchmark_hf/0044.parquet \
    --backend huggingface \
    --device cpu
```

**Multiple reciters, filtered to 3 surahs, 30 rows total:**

```bash
python3 scripts/run_hf_benchmark.py \
    --parquet data/benchmark_hf/0044.parquet \
    --reciters alafasy,husary,minshawy_murattal \
    --surahs 37 \
    --max_rows 30 \
    --device cpu
```

---

## Performance Characteristics

### Computation Time

| Operation | Per ayah | Per parquet (2,635 rows) |
|-----------|----------|-------------------------|
| MP3 decode (miniaudio) | ~50ms | ~2 min |
| ASR transcription (CPU, HF) | ~2.4s | ~1.7 hours |
| ASR transcription (CPU, faster-whisper int8) | ~0.6s | ~25 min |
| WER computation | ~1ms | ~3s |

**Backend + Device comparison** (per ayah, ~10s audio):

| Backend | Device | Est. per ayah | Est. per parquet (2,635 rows) |
|---------|--------|---------------|-------------------------------|
| faster-whisper | CPU (int8) | ~0.6s | ~25 min |
| faster-whisper | CUDA (float16) | ~0.2s | ~9 min |
| huggingface | CPU (float32) | ~2.4s | ~1.7 hours |
| huggingface | CUDA (float16) | ~0.5s | ~22 min |

**MPS (Apple M1/M2)**: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically in the script. The faster-whisper backend maps MPS to CPU with int8 (still fast via Apple Accelerate).

### Storage

| Item | Size |
|------|------|
| Single parquet | 200MB-970MB |
| Working memory per row | ~1-2MB (MP3 bytes + decoded audio) |
| Results CSV | ~1MB per parquet |
| **Total disk for 1 parquet benchmark** | Input parquet + output CSV (~1GB total) |

---

## Implementation Details

### Audio Decoding Strategy (ffmpeg-free)

1. **Primary**: `miniaudio.decode()` — pure Python + C extension, decodes MP3 bytes directly to float32 numpy array
2. **Fallback**: If miniaudio not installed, `librosa.load()` writes MP3 bytes to a temp file, decodes with audioread/CoreAudio, cleans up temp file

### ASR Backend Selection

Both `run_hf_benchmark.py` and `run_benchmark.py` default to the **faster-whisper** backend (`--backend faster-whisper`), which uses CTranslate2 int8 quantization for ~4x faster CPU inference.

To use the HuggingFace PyTorch backend instead, pass `--backend huggingface`. Both backends produce the same output contract (hypothesis text + WER/CER), so benchmark results are directly comparable.

| Backend | Model loading | Transcription function |
|---------|--------------|----------------------|
| `faster-whisper` | `load_faster_whisper_model()` | `transcribe_faster_whisper()` |
| `huggingface` | `load_whisper_model()` | `transcribe_whisper()` |

### Text Normalization

Before WER computation, both the hypothesis and reference are normalized via `src/preprocessor.py::strip_harakat()`, which removes:
- Arabic diacritical marks (fatḥah, ḍammah, kasrah, etc.)
- Quranic annotation signs
- Tatweel (letter extension)

This ensures WER reflects actual word errors, not diacritic differences. The stored `reference_text` in the CSV is the **un-normalized** Uthmani script with full tashkeel; the WER was computed on the normalized form.

---

## Edge Cases and Notes

### 1. Empty or Corrupt Audio

Rows with zero-length or corrupt MP3 data are caught and recorded as failures (success=False, error_message set). Benchmarking continues.

### 2. Very Short Ayahs

Single-word ayahs (e.g., "الم", "طه") may produce empty ASR output. These are still processed and WER is computed (likely WER > 1.0 if hypothesis is empty). See note on WER bounds below.

### 3. WER > 1.0

jiwer can return WER > 1.0 when insertions exceed reference word count. This is mathematically valid (100%+ error rate) and is not clamped. Results CSVs will faithfully report these values.

### 4. Reciter Filter Case-Insensitivity

The `--reciters` argument is case-insensitive: `--reciters Alafasy`, `--reciters alafasy`, and `--reciters ALAFASY` all match the `alafasy` reciter in the parquet.

### 5. Max Rows Applied After Filtering

If you specify `--reciters alafasy --max_rows 10`, the benchmark processes 10 rows of alafasy audio (not 10 total rows across all reciters).

---

## Data Flow Diagram

```
HuggingFace Parquet (0044.parquet)
    ↓ read rows
for each (surah, ayah, reciter):
    ↓ extract audio.bytes (MP3)
    miniaudio.decode() → float32 array @ 16kHz mono
    ↓ (in-memory, no disk write)
    --backend?
        ├─ faster-whisper → transcribe_faster_whisper(fw_model, audio, sr)
        └─ huggingface    → transcribe_whisper(model, processor, audio, sr, device)
    ↓ get hypothesis text
    compute_wer(hypothesis, ayah_ar, normalise_fn=strip_harakat)
    ↓ get WER, CER, insertions, deletions, substitutions
    save result
    ↓
aggregate per-reciter, per-surah, overall
    ↓
write results.csv + report.md + summary.json
```

---

## Troubleshooting

### MPS Device on Apple Silicon

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically in the script for the HuggingFace backend. For the faster-whisper backend, MPS is mapped to CPU with int8 automatically (CTranslate2 does not support MPS, but uses Apple Accelerate for fast CPU inference).

If you encounter MPS-related errors with the HuggingFace backend, use `--device cpu` explicitly:

```bash
python3 scripts/run_hf_benchmark.py \
    --parquet data/benchmark_hf/0044.parquet \
    --backend huggingface \
    --device cpu
```

### Out of Memory

If running on a constrained device:
- Use `--device cpu` instead of CUDA
- Reduce parquet file size by filtering: `--reciters alafasy --max_rows 50`
- Process parquets sequentially, not all at once

### ImportError: No module named 'pyarrow'

Install missing dependencies:

```bash
pip install -r requirements.txt
```

---

## Integration with Existing Benchmark

The HF benchmark is **independent** of `scripts/run_benchmark.py` and does not share output directories by default. Both can coexist:

- `scripts/run_benchmark.py` → `results/benchmark/` (WAV files)
- `scripts/run_hf_benchmark.py` → `results/hf_benchmark/` (Parquet files)

---

*Documentation for `scripts/run_hf_benchmark.py`*
