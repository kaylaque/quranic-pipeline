# Faster-Whisper Integration Plan

## Overview

This document outlines the plan to integrate [faster-whisper](https://github.com/SYSTRAN/faster-whisper) as the ASR backend, replacing the current HuggingFace `transformers` Whisper implementation. The goal is to achieve significantly faster CPU inference on M1 hardware (no MPS required) while maintaining the existing pipeline's input/output contracts.

---

## Why Faster-Whisper

| Aspect | Current (HF transformers) | Faster-Whisper (CTranslate2) |
|--------|---------------------------|-------------------------------|
| **CPU inference speed** | Baseline (1x) | ~4x faster (int8 quantization) |
| **Memory usage** | ~300MB (base model, float32) | ~75MB (base model, int8) |
| **MPS dependency** | Yes (float16 acceleration) | No (CPU-optimised BLAS/SIMD) |
| **ffmpeg dependency** | No (already removed) | No (bypassed via numpy input) |
| **Word timestamps** | Requires extra generate kwargs | Built-in, high quality |
| **VAD chunking** | Manual fixed-window (10s) | Built-in Silero VAD |
| **Quantization** | float32 (CPU) / float16 (GPU) | int8 / float16 / float32 |
| **Batch inference** | Manual | Native batching support |

### Key Advantage for M1 Without MPS

CTranslate2 uses optimised CPU kernels (Apple Accelerate framework on macOS) with int8 quantization. This delivers near-GPU performance on M1 CPU without requiring MPS or Metal at all.

---

## Requirements

### New Dependencies

```
faster-whisper>=1.1.0         # CTranslate2-based Whisper inference
```

### Removed Dependencies (after full migration)

```
# These become optional (only needed if --backend huggingface is used):
# transformers>=4.40.0
# torch>=2.0.0
# torchaudio>=2.0.0
```

> **Note:** `torch` cannot be fully removed yet because the GOP scoring stage (`src/aligner.py`) uses PyTorch for logit extraction. See [GOP Scoring Considerations](#gop-scoring-considerations) below.

### Retained Dependencies

```
jiwer                         # WER/CER computation (unchanged)
librosa                       # Audio resampling fallback (unchanged)
soundfile                     # Primary audio loading (unchanged)
numpy                         # Array operations (unchanged)
pydantic>=2.0                 # Data models (unchanged)
```

---

## Model Conversion

Faster-whisper uses CTranslate2 format, not HuggingFace format. Models must be converted once.

### Conversion Command

```bash
# Install converter
pip install ctranslate2

# Convert primary model
ct2-transformers-converter \
    --model tarteel-ai/whisper-base-ar-quran \
    --output_dir models/whisper-quran-ct2 \
    --quantization int8

# Convert alternative model (optional, large)
ct2-transformers-converter \
    --model wasimlhr/whisper-quran-v1 \
    --output_dir models/whisper-quran-v1-ct2 \
    --quantization float16
```

### Quantization Options

| Type | Size (base) | Speed (CPU) | Accuracy | Use Case |
|------|-------------|-------------|----------|----------|
| `int8` | ~39MB | Fastest | Minimal loss | **Recommended for M1 CPU** |
| `int8_float16` | ~39MB | Fast | Minimal loss | GPU with int8 compute |
| `float16` | ~74MB | Medium | No loss | GPU inference |
| `float32` | ~148MB | Baseline | Reference | Debugging only |

### Model Storage

```
models/
  whisper-quran-ct2/           # Converted primary model (int8)
    model.bin
    config.json
    tokenizer.json
    vocabulary.txt
  whisper-quran-v1-ct2/        # Converted large model (optional)
    ...
```

> **Note:** Pre-converted models can also be loaded directly from HuggingFace IDs if available. faster-whisper auto-downloads CTranslate2 versions of standard `openai/whisper-*` models. For custom fine-tuned models like `tarteel-ai/whisper-base-ar-quran`, manual conversion is required.

---

## Architecture Changes

### Affected Files

| File | Change Type | Description |
|------|-------------|-------------|
| `src/asr.py` | **Major refactor** | New `FasterWhisperBackend` class; retain HF backend as fallback |
| `scripts/run_pipeline.py` | **Minor update** | Add `--backend` flag; pass backend choice to ASR |
| `scripts/run_benchmark.py` | **Minor update** | Forward `--backend` flag to pipeline subprocess |
| `requirements.txt` | **Update** | Add `faster-whisper`; mark HF deps as optional |
| `src/aligner.py` | **No change** | GOP scoring still uses PyTorch (see below) |
| `src/preprocessor.py` | **No change** | Text processing is ASR-agnostic |
| `src/phonemizer.py` | **No change** | Phonemization is ASR-agnostic |
| `src/error_classifier.py` | **No change** | Classification is ASR-agnostic |
| `src/report.py` | **No change** | Report generation is ASR-agnostic |

### Design: Backend Abstraction

Introduce a minimal backend abstraction so both engines can coexist during migration:

```python
# src/asr.py — proposed structure

class ASRResult:
    """Unified output from any ASR backend."""
    text: str                          # Transcribed Arabic text
    language: str                      # Always "ar"
    duration_sec: float                # Audio duration
    word_timestamps: list[dict] | None # [{word, start, end}, ...] (if available)
    num_chunks: int | None             # Number of chunks (if chunked)


class FasterWhisperBackend:
    """CTranslate2-based Whisper backend (faster-whisper)."""

    def __init__(self, model_path: str, device: str = "auto",
                 compute_type: str = "auto"):
        ...

    def load_model(self) -> None:
        ...

    def transcribe(self, audio_array: np.ndarray, sr: int = 16000) -> ASRResult:
        ...


class HuggingFaceBackend:
    """Original HuggingFace transformers backend (existing code)."""

    def __init__(self, model_name: str, device: str = "auto"):
        ...

    def load_model(self) -> None:
        ...

    def transcribe(self, audio_array: np.ndarray, sr: int = 16000) -> ASRResult:
        ...
```

---

## Implementation Details

### 1. Audio Input (No ffmpeg)

The current pipeline already loads audio via `soundfile` into numpy arrays. faster-whisper's `WhisperModel.transcribe()` accepts either a file path (uses ffmpeg internally) or a numpy array.

**We bypass ffmpeg entirely by passing the numpy array directly:**

```python
from faster_whisper import WhisperModel

model = WhisperModel("models/whisper-quran-ct2", device="cpu", compute_type="int8")

# Current pipeline already provides this:
audio_array, sr = load_audio("recitation.wav")  # float32, 16kHz, mono

# Pass numpy array directly — no ffmpeg needed
segments, info = model.transcribe(
    audio_array,                    # np.ndarray (float32)
    language="ar",                  # Force Arabic
    task="transcribe",              # Not translate
    beam_size=5,                    # Default beam search
    word_timestamps=True,           # Get word-level timing
    vad_filter=False,               # Disable VAD (see chunking section)
)

# Collect results
text_parts = []
word_ts = []
for segment in segments:
    text_parts.append(segment.text)
    if segment.words:
        for w in segment.words:
            word_ts.append({"word": w.word, "start": w.start, "end": w.end})

result = ASRResult(
    text=" ".join(text_parts).strip(),
    language="ar",
    duration_sec=info.duration,
    word_timestamps=word_ts,
)
```

### 2. Device & Compute Type Mapping

| Pipeline `--device` | faster-whisper `device` | `compute_type` |
|---------------------|------------------------|----------------|
| `auto` | `"auto"` | `"auto"` (int8 on CPU, float16 on CUDA) |
| `cpu` | `"cpu"` | `"int8"` |
| `cuda` | `"cuda"` | `"float16"` |
| `mps` | `"cpu"` + warn | `"int8"` (CTranslate2 has no MPS backend) |

> **Important:** CTranslate2 does not support Apple MPS. On M1, it uses highly optimised CPU inference via Apple Accelerate. This is actually faster than PyTorch MPS for Whisper base models due to int8 quantization.

### 3. Chunking Strategy: VAD vs Fixed-Window

**Current approach (fixed-window):**
- 10-second windows with 1-second overlap
- Overlap merge by estimating words-per-second skip
- Can split mid-word or mid-ayah

**faster-whisper built-in VAD (Silero VAD):**
- Detects speech vs silence boundaries
- Natural segmentation at pauses between words/ayahs
- No overlap merge needed
- Better for Quranic recitation (natural pauses between ayahs)

**Recommendation:** Use faster-whisper's internal chunking (30-second Whisper windows) for most cases. Enable Silero VAD only for very long audio (>5 minutes) where memory is a concern:

```python
# Short/medium audio (< 5 min): let Whisper handle internally
segments, info = model.transcribe(audio_array, vad_filter=False)

# Long audio (> 5 min): enable VAD for memory efficiency
segments, info = model.transcribe(
    audio_array,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=300,    # Minimum silence to split
        speech_pad_ms=200,              # Padding around speech
    ),
)
```

This **replaces** the current `chunk_audio()` + `transcribe_chunked()` + `ProcessPoolExecutor` logic with a single call. The fixed-window chunking code in `src/asr.py` (lines 143-406) can be retained but bypassed when using the faster-whisper backend.

### 4. Word Timestamps for Alignment

faster-whisper provides word-level timestamps natively, which can feed directly into Stage 6 (alignment):

```python
# Current approach: re-run Whisper with timestamp generation in aligner.py
# New approach: capture timestamps during ASR and pass downstream

segments, info = model.transcribe(audio_array, word_timestamps=True)

for segment in segments:
    for word in segment.words:
        # word.word  = "بِسْمِ"
        # word.start = 0.0      (seconds)
        # word.end   = 0.48     (seconds)
        # word.probability = 0.97
```

**Benefit:** Eliminates the redundant second Whisper pass currently done in `align_with_fallback()` (Strategy 1 and 2 in `src/aligner.py`). Word timestamps from the initial ASR pass can be reused.

### 5. Parallel/Batch Inference for Benchmarking

CTranslate2 supports batched inference, useful for `run_benchmark.py`:

```python
from faster_whisper import BatchedInferencePipeline

model = WhisperModel("models/whisper-quran-ct2", device="cpu", compute_type="int8")
batched = BatchedInferencePipeline(model=model)

# Process multiple segments in a single batch
segments, info = batched.transcribe(
    audio_array,
    language="ar",
    batch_size=16,        # Adjust based on available memory
    word_timestamps=True,
)
```

This is particularly beneficial for the benchmark runner which currently processes files sequentially via subprocess calls.

---

## GOP Scoring Considerations

### The Problem

The current GOP scoring (`src/aligner.py:compute_gop_score()`) extracts **logits** (raw neural network outputs) from the Whisper encoder via PyTorch:

```python
outputs = model(input_features)       # HF model forward pass
logits = outputs.logits               # (batch, seq, vocab)
log_probs = F.log_softmax(logits)     # Convert to log probabilities
gop_raw = -log_probs.mean()           # GOP formula
```

CTranslate2 does **not** expose encoder logits. It only returns decoded text and token probabilities.

### Options

| Option | Effort | Trade-off |
|--------|--------|-----------|
| **A: Hybrid** — faster-whisper for ASR, HF model for GOP only | Low | Two models loaded; higher memory |
| **B: Token probability proxy** — use faster-whisper's `word.probability` as GOP proxy | Low | Less accurate but may be sufficient |
| **C: Lazy HF loading** — load HF model only when GOP is needed (non-MATCH words) | Medium | Memory spike only during GOP; selective loading |
| **D: Separate lightweight model** — use a small acoustic model (wav2vec2) for GOP | High | Best long-term; decouples ASR from GOP |

### Recommended: Option C (Lazy HF Loading)

```python
# During ASR: use faster-whisper (fast, int8)
asr_backend = FasterWhisperBackend(...)
result = asr_backend.transcribe(audio_array, sr)

# During GOP scoring (Stage 7): lazy-load HF model only if needed
if error_positions:  # Only when there are actual errors to score
    hf_model, hf_processor, device = load_whisper_model(model_name, device)
    aligned_words = score_aligned_words(aligned_words, audio_array, sr,
                                         phone_tokens, hf_model, hf_processor, ...)
```

**Why this works:** GOP scoring only runs on error words (selective GOP). For clean recitations, the HF model is never loaded. For recitations with errors, the HF model load cost is amortised across all error words.

**Migration path:** Start with Option C. Later, evaluate Option B (token probability proxy) — if `word.probability` correlates well with GOP scores, the HF model dependency can be eliminated entirely.

---

## CLI Changes

### New Arguments

```
--backend       faster-whisper | huggingface    (default: faster-whisper)
--compute_type  int8 | float16 | float32 | auto (default: auto)
--model_dir     path to CTranslate2 model dir   (default: models/whisper-quran-ct2)
```

### Updated Argument Behaviour

```
--model         HuggingFace model ID (used for HF backend or for conversion reference)
--device        auto | cpu | cuda               (mps maps to cpu for faster-whisper)
--chunk_seconds (ignored when --backend faster-whisper; uses internal chunking)
--parallel_workers (ignored when --backend faster-whisper; uses native batching)
```

### Example Commands

```bash
# Default (faster-whisper, int8, CPU)
python scripts/run_pipeline.py --audio recitation.wav --surah 1 --ayah 1

# Explicit faster-whisper with custom model dir
python scripts/run_pipeline.py --audio recitation.wav \
    --backend faster-whisper \
    --model_dir models/whisper-quran-ct2 \
    --compute_type int8

# Fallback to original HF backend
python scripts/run_pipeline.py --audio recitation.wav \
    --backend huggingface \
    --model tarteel-ai/whisper-base-ar-quran

# Benchmark with faster-whisper
python scripts/run_benchmark.py --audio_dir data/samples/ \
    --backend faster-whisper \
    --output_dir results/benchmark/
```

---

## Migration Steps

### Phase 1: Add faster-whisper alongside HF (non-breaking)

1. Install `faster-whisper` dependency
2. Convert `tarteel-ai/whisper-base-ar-quran` to CTranslate2 int8 format
3. Implement `FasterWhisperBackend` class in `src/asr.py`
4. Add `--backend` flag to `run_pipeline.py` (default: `huggingface` initially)
5. Verify output parity: run both backends on same audio, compare transcriptions
6. Unit tests for both backends

### Phase 2: Make faster-whisper the default (COMPLETED)

1. ~~Switch default `--backend` to `faster-whisper`~~ Done
2. ~~Update `run_benchmark.py` to forward `--backend` flag~~ Done
3. Pass word timestamps from ASR to alignment stage (eliminate redundant Whisper pass)
4. Update `docs/asr.md` with dual-backend documentation
5. Benchmark: measure speed/accuracy differences on test set

### Phase 2b: HF Parquet Benchmark with dual backend

Add `--backend` / `--compute_type` / `--model_dir` flags to `run_hf_benchmark.py` so
the large-scale parquet benchmark (Buraaq/quran-md-ayahs) can run against either backend.

**Affected files:**

| File | Change | Scope |
|------|--------|-------|
| `scripts/run_hf_benchmark.py` | Add 3 CLI flags, branch model load + transcribe in `_process_row()` and `run()` | ~25 lines |
| `scripts/hf_parquet_utils.py` | No change | — |
| `docs/hf_benchmark.md` | Update usage examples | Minor |

**Implementation detail:**

The HF benchmark script calls ASR functions directly in-process (not via subprocess).
The change is a conditional branch at two points:

1. **Model loading** (`run()` method, line 210):
   ```python
   # Before:
   model, processor, device = load_whisper_model(self.model_name, self.device_pref)

   # After:
   if self.backend == "faster-whisper":
       fw_model = load_faster_whisper_model(self.model_dir, self.device_pref, self.compute_type)
       model, processor, device = fw_model, None, None
   else:
       model, processor, device = load_whisper_model(self.model_name, self.device_pref)
   ```

2. **Transcription** (`_process_row()` method, line 131):
   ```python
   # Before:
   result = transcribe_whisper(model, processor, audio_array, sr, device)

   # After:
   if self.backend == "faster-whisper":
       result = transcribe_faster_whisper(model, audio_array, sr)
   else:
       result = transcribe_whisper(model, processor, audio_array, sr, device)
   ```

Both backends produce `result["text"]` — the rest of the pipeline (WER, reports)
is completely untouched.

**Expected speedup on parquet 0044 (2,635 rows):**

| Backend | Est. per-row | Est. total | Memory |
|---------|-------------|------------|--------|
| HF transformers (CPU float32) | ~5-8s | ~3.5-6 hours | ~600MB |
| **faster-whisper (CPU int8)** | **~1-2s** | **~45-90 min** | **~150MB** |

**Example commands after implementation:**

```bash
# faster-whisper (default, ~4x faster)
python3 scripts/run_hf_benchmark.py \
    --parquet data/benchmark_hf/0044.parquet \
    --backend faster-whisper \
    --reciters alafasy --max_rows 10

# HuggingFace backend (original)
python3 scripts/run_hf_benchmark.py \
    --parquet data/benchmark_hf/0044.parquet \
    --backend huggingface \
    --reciters alafasy --max_rows 10
```

### Phase 3: Optimise and evaluate GOP alternatives

1. Implement lazy HF model loading for GOP scoring (Option C)
2. Evaluate `word.probability` as GOP proxy (Option B)
3. If proxy is sufficient, remove HF/PyTorch from critical path entirely
4. Update `requirements.txt` to mark `transformers`/`torch` as optional

---

## Expected Performance

### Inference Speed (tarteel-ai/whisper-base-ar-quran, 10s audio)

| Backend | Device | Quantization | Time | Memory |
|---------|--------|-------------|------|--------|
| HF transformers | CPU (float32) | None | ~5-8s | ~600MB |
| HF transformers | MPS (float16) | None | ~2-3s | ~400MB |
| **faster-whisper** | **CPU (int8)** | **int8** | **~1-2s** | **~150MB** |
| faster-whisper | CUDA (float16) | float16 | ~0.3-0.5s | ~300MB |

### Benchmark Batch (10 files, ~10s each)

| Backend | Device | Total Time | Notes |
|---------|--------|------------|-------|
| HF transformers (current) | CPU | ~60-90s | Sequential subprocess calls |
| faster-whisper | CPU (int8) | ~15-25s | Sequential, but ~4x faster per file |
| faster-whisper + batched | CPU (int8) | ~10-15s | Native batching, in-process |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| CTranslate2 conversion loses fine-tuned accuracy | High | Compare WER on test set before/after conversion |
| int8 quantization degrades Arabic diacritics | Medium | Benchmark with/without quantization; fallback to float16 |
| faster-whisper API changes in future versions | Low | Pin version; minimal surface area used |
| GOP scoring needs HF model (memory overhead) | Medium | Lazy loading; evaluate probability proxy |
| Silero VAD splits mid-word in Quranic recitation | Low | Disable VAD for short audio; tune `min_silence_duration_ms` |

---

## Validation Checklist

Before switching the default backend:

- [x] Model conversion succeeds for `tarteel-ai/whisper-base-ar-quran` (int8, 73MB)
- [ ] Transcription output matches HF backend (WER delta < 1%)
- [ ] Word timestamps are accurate (compare with current alignment)
- [x] No ffmpeg calls in the entire pipeline (av installed via binary wheel; numpy arrays bypass av)
- [x] `run_pipeline.py` runs with faster-whisper backend (mock + real audio verified)
- [x] `run_benchmark.py` forwards `--backend` flag to pipeline subprocess
- [ ] `run_hf_benchmark.py` supports `--backend` flag for parquet benchmarks
- [ ] CPU inference is measurably faster (target: >= 3x speedup)
- [ ] Memory usage is lower (target: >= 50% reduction)
- [x] GOP scoring still works via lazy HF loading (Option C implemented)
- [x] MPS fallback enabled (`PYTORCH_ENABLE_MPS_FALLBACK=1`)
- [x] All existing pipeline modes work: FULL, NO_GOP, TEXT_ONLY, ASR_ONLY, MOCK
- [ ] Auto-detect mode (fuzzy matching) works with faster-whisper output
- [ ] HF parquet benchmark runs end-to-end with faster-whisper on 0044.parquet

---

## References

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2 Quantization](https://opennmt.net/CTranslate2/quantization.html)
- [CTranslate2 Apple Accelerate Support](https://opennmt.net/CTranslate2/installation.html)
- [Silero VAD](https://github.com/snakers4/silero-vad)
