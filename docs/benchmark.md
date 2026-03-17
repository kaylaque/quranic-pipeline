# Benchmarking -- Batch Processing Multiple Audio Files

## Overview

The benchmarking script (`scripts/run_benchmark.py`) enables batch processing of multiple WAV files through the pipeline with optional reference mapping and aggregate metrics collection.

---

## Quick Start

### Mode 1: Auto-Detect Surah/Ayah

Run the pipeline on all WAV files in a directory with automatic surah/ayah detection:

```bash
python scripts/run_benchmark.py \
    --audio_dir data/samples/ \
    --output_dir results/benchmark/
```

**Result**: Each file runs in auto-detect mode; unknown files fallback to `ASR_ONLY` mode.

### Mode 2: With CSV Reference Mapping

Provide explicit surah/ayah or reference text for each file via CSV:

```bash
python scripts/run_benchmark.py \
    --audio_dir data/samples/ \
    --reference_csv data/reference/mapping.csv \
    --output_dir results/benchmark/
```

**CSV Format** (`data/reference/mapping.csv`):
```csv
filename,surah,ayah,reference_text
mock.wav,1,1,بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
sample.wav,2,5,الحمد لله رب العالمين
```

**Fields**:
- `filename` (required): Basename of audio file (e.g., `mock.wav`)
- `surah` (optional): Surah number (1-114)
- `ayah` (optional): Ayah number
- `reference_text` (optional): Full Arabic reference with diacritics

If both surah/ayah and reference_text are provided, reference_text takes precedence.

### Mode 3: Explicit Surah/Ayah for All Files

Apply the same surah/ayah to all files:

```bash
python scripts/run_benchmark.py \
    --audio_dir data/samples/ \
    --surah 1 --ayah 1 \
    --output_dir results/benchmark/
```

---

## Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--audio_dir` | path | Yes | Directory containing `.wav` files to process |
| `--output_dir` | path | No | Output directory for reports and pipeline results (default: `results/benchmark/`) |
| `--reference_csv` | path | No | CSV file mapping filenames to surah/ayah/reference_text |
| `--surah` | int | No | Surah number (applies to all files if provided; 0 = auto-detect) |
| `--ayah` | int | No | Ayah number (applies to all files if provided; 0 = auto-detect) |
| `--model` | str | No | HuggingFace Whisper model ID (default: `tarteel-ai/whisper-base-ar-quran`) |
| `--device` | choice | No | Device: `auto`, `cpu`, `cuda`, `mps` (default: `auto`) |
| `--backend` | choice | No | ASR backend: `faster-whisper` (default) or `huggingface` |
| `--compute_type` | choice | No | CTranslate2 compute type: `auto`, `int8`, `float16`, `float32` (default: `auto`) |
| `--model_dir` | path | No | CTranslate2 model directory (default: `models/whisper-quran-ct2`) |
| `--verbose` | flag | No | Enable debug logging for all files |

---

## Output Files

### 1. Pipeline Reports

For each WAV file, generates:
- `{stem}.json` — Full structured report with errors, GOP scores, timing
- `{stem}.txt` — Human-readable bordered report (if `--verbose`)

Location: `{output_dir}` (shared with pipeline results)

### 2. Benchmark Results CSV

**Path**: `{output_dir}/benchmark_results.csv`

Contains one row per input file with columns:
- `filename` — Input WAV filename
- `surah`, `ayah` — Detected or provided reference location
- `mode` — Execution mode: `"auto-detect"`, `"explicit"`, or `"provided"` (CSV)
- `pipeline_mode` — Pipeline mode: `FULL`, `NO_GOP`, `TEXT_ONLY`, `TEXT_ONLY_CLEAN`, `ASR_ONLY`, `ERROR`
- `success` — Boolean; True if pipeline completed without errors
- `asr_wer` — Word Error Rate from ASR (0.0–1.0, or NULL if not computed)
- `total_errors` — Count of non-MATCH errors detected
- `confirmed_errors` — Count of CONFIRMED severity errors
- `possible_errors` — Count of POSSIBLE severity errors
- `suppressed_errors` — Count of SUPPRESSED errors
- `processing_time_sec` — Wall-clock time for this file (seconds)

**Example**:
```csv
filename,surah,ayah,mode,pipeline_mode,success,asr_wer,total_errors,confirmed_errors,possible_errors,suppressed_errors,processing_time_sec
mock.wav,1,1,auto-detect,FULL,True,0.1429,2,2,0,0,3.45
sample.wav,2,5,provided,TEXT_ONLY,True,0.0000,0,0,0,0,1.23
error.wav,0,0,auto-detect,ERROR,False,,,,,5.67
```

### 3. Benchmark Summary Report

**Path**: `{output_dir}/benchmark_report.md`

Markdown report containing:

1. **Summary Section**
   - Total files processed
   - Success rate (%)
   - Average ASR WER
   - Average processing time
   - Total errors across all files (broken down by severity)

2. **Failed Files Section** (if any failures)
   - Filename and error message for each failed file

3. **Per-File Results Table**
   - Sortable table with filename, surah, ayah, pipeline mode, WER, error count, processing time, status

**Example**:
```markdown
# Quranic Pipeline Benchmark Report

## Summary

- **Total files**: 5
- **Successful**: 4 (80.0%)
- **Failed**: 1
- **Average ASR WER**: 0.0843
- **Average processing time**: 2.84s
- **Total errors detected**: 12
  - CONFIRMED: 10
  - POSSIBLE: 2

## Failed Files

- **error.wav**: Pipeline failed: CUDA out of memory

## Per-File Results

| Filename | Surah | Ayah | Mode | Pipeline | WER | Errors | Time (s) | Status |
|----------|-------|------|------|----------|-----|--------|----------|--------|
| mock.wav | 1 | 1 | auto-de | FULL | 0.1429 | 2 | 3.45 | ✓ |
| sample.wav | 2 | 5 | provid | TEXT_ONLY | 0.0000 | 0 | 1.23 | ✓ |
| ...
```

---

## Execution Modes

### Mode Detection Logic

For each file, the benchmark runner determines the execution mode:

1. **CSV-provided**: If filename matches an entry in `--reference_csv` → mode = `"provided"`
   - Uses surah/ayah/reference_text from CSV

2. **Explicit**: If `--surah` and `--ayah` are provided (> 0) → mode = `"explicit"`
   - Applies same values to all files

3. **Auto-detect**: Otherwise → mode = `"auto-detect"`
   - Uses built-in fuzzy matching (≥40% word overlap) to detect surah/ayah from ASR transcript
   - Fallback: `ASR_ONLY` mode if no match found

---

## Performance Considerations

### Backend & Device Selection

- **faster-whisper** (default): ~4x faster CPU inference with int8. No MPS required.
- **huggingface**: Original PyTorch backend. Supports MPS/CUDA.

| Backend | Device | Est. per-file (10s audio) | Est. 10-file batch |
|---------|--------|--------------------------|-------------------|
| faster-whisper | CPU (int8) | ~1-2s | ~15-25s |
| huggingface | CPU (float32) | ~5-8s | ~60-90s |
| huggingface | CUDA (float16) | ~0.5-1s | ~10-15s |

Use `--device auto` (default) for automatic fallback: CUDA → MPS → CPU.

### Timeout

Each file has a 5-minute processing timeout. Longer audio may fail; reduce audio length or increase timeout in code.

---

## Error Handling

### Failed File Handling

If a file fails:
- Error message recorded in benchmark results CSV and markdown report
- Pipeline continues to next file
- Partial results available for successful files

Common failure modes:
- **File not found**: Audio file deleted or path wrong
- **Decode error**: Corrupted WAV file
- **Out of memory**: CUDA/MPS GPU ran out of memory; switch to `--device cpu`
- **Timeout**: File too long; reduce audio or increase timeout
- **No reference found**: Auto-detect mode failed (ASR too corrupted or non-Quranic); check reference CSV

### Handling CSV Errors

- Missing or malformed CSV file: Benchmark continues with auto-detect mode for all files
- Invalid row in CSV: Row skipped; file runs in auto-detect mode
- Invalid surah/ayah in CSV: Ignored; file runs with auto-detect or explicit values

---

## Example Workflow

### Setup

```bash
# Create sample directory with test audio
mkdir -p data/samples
cp ~/my_recitations/*.wav data/samples/

# Create reference mapping
cat > data/reference/mapping.csv << 'EOF'
filename,surah,ayah,reference_text
surah1_ayah1.wav,1,1,بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
surah1_ayah2.wav,1,2,الحمد لله رب العالمين
surah2_ayah1.wav,2,1,الم ذلك الكتاب
EOF

mkdir -p results/benchmark/
```

### Run Benchmark

```bash
python scripts/run_benchmark.py \
    --audio_dir data/samples/ \
    --reference_csv data/reference/mapping.csv \
    --output_dir results/benchmark/ \
    --model "tarteel-ai/whisper-base-ar-quran" \
    --device "auto"
```

### Analyze Results

```bash
# View summary report
cat results/benchmark/benchmark_report.md

# Export CSV for spreadsheet analysis
cp results/benchmark/benchmark_results.csv ~/benchmark_results.xlsx

# Check individual reports
cat results/benchmark/surah1_ayah1.json | jq '.summary'
```

---

## Tips & Troubleshooting

### Tip: Use `--verbose` for Detailed Logging

```bash
python scripts/run_benchmark.py \
    --audio_dir data/samples/ \
    --output_dir results/benchmark/ \
    --verbose
```

Writes per-file debug info to stdout and `{stem}.txt` report in output_dir.

### Tip: Subset Large Batches for Testing

```bash
# Test on first 3 files only
ls data/samples/*.wav | head -3 | xargs -I {} cp {} test_subset/
python scripts/run_benchmark.py --audio_dir test_subset/ --output_dir results/test/
```

### Troubleshooting: Memory Errors on GPU

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Switch to CPU
python scripts/run_benchmark.py \
    --audio_dir data/samples/ \
    --output_dir results/benchmark/ \
    --device cpu
```

### Troubleshooting: Very Slow Processing

**Problem**: Batch taking too long

**Solution**:
```bash
# Use faster-whisper backend (default, ~4x faster on CPU)
python scripts/run_benchmark.py \
    --audio_dir data/samples/ \
    --backend faster-whisper \
    --output_dir results/benchmark/

# Or switch to GPU if available
python scripts/run_benchmark.py \
    --audio_dir data/samples/ \
    --backend huggingface \
    --device cuda \
    --output_dir results/benchmark/
```

---

## Data Flow

```
--audio_dir (WAV files)
    ↓
--reference_csv (optional)
    ↓
BenchmarkRunner._get_audio_files()
    ↓
FOR EACH WAV FILE:
    ├─ Determine mode (auto-detect / explicit / provided)
    ├─ Run pipeline (subprocess)
    ├─ Load report JSON
    └─ Extract metrics (WER, errors, timing)
    ↓
Aggregate results
    ↓
Generate CSV + Markdown reports
```

---

*Documentation for scripts/run_benchmark.py*
