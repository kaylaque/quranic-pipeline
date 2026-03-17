# ASR -- Automatic Speech Recognition

## Overview

The ASR module (`src/asr.py`) handles audio loading, Whisper-based transcription, and WER/CER computation. It supports two ASR backends:

- **faster-whisper** (default) — CTranslate2-based, ~4x faster CPU inference with int8 quantization, no MPS required.
- **huggingface** — Original PyTorch/transformers backend, supports MPS/CUDA.

Both backends share the same audio loading pipeline (soundfile, no ffmpeg) and produce the same output contract. Models are cached at module level.

---

## ASR Backends

### faster-whisper (default)

Uses [CTranslate2](https://github.com/OpenNMT/CTranslate2) for optimised inference. Requires a one-time model conversion from HuggingFace format.

| Aspect | Value |
|--------|-------|
| Engine | CTranslate2 |
| Quantization | int8 (CPU), float16 (CUDA) |
| MPS support | No — maps to CPU automatically |
| Word timestamps | Built-in, high quality |
| Chunking | Internal Whisper 30s windows |
| Model path | `models/whisper-quran-ct2/` (local) |

**Key functions:**
- `load_faster_whisper_model(model_path, device, compute_type)` — cached model loader
- `transcribe_faster_whisper(fw_model, audio_array, sr)` — returns text + word timestamps

**Device mapping:**

| `--device` | CTranslate2 device | Compute type (auto) |
|------------|-------------------|-------------------|
| `auto` | `cpu` or `cuda` (auto-detect) | int8 (CPU) / float16 (CUDA) |
| `cpu` | `cpu` | `int8` |
| `cuda` | `cuda` | `float16` |
| `mps` | `cpu` (with warning) | `int8` |

### HuggingFace (transformers)

Original backend using PyTorch `AutoModelForSpeechSeq2Seq`. Still required internally for alignment (Stage 6) and GOP scoring (Stage 7).

**Key functions:**
- `load_whisper_model(model_name, device)` — cached model + processor loader with fallback chain
- `transcribe_whisper(model, processor, audio_array, sr, device)` — single-pass transcription
- `transcribe_chunked(...)` — window-based parallel transcription for long audio

---

## Supported Models

| Model ID | Base Architecture | Parameters | WER | Notes |
|----------|------------------|------------|-----|-------|
| `tarteel-ai/whisper-base-ar-quran` | Whisper Base | ~74M | ~15% | Lightweight, fast inference, good for CPU |
| `wasimlhr/whisper-quran-v1` | Whisper Large-v3 | ~1.55B | ~5.35% | High accuracy, requires GPU/MPS recommended |
| `openai/whisper-base` | Whisper Base | ~74M | General | Fallback only, not Quran-specialised |

### Fallback Chain (HuggingFace backend)

```
Requested model (--model flag)
    |
    +--> FAIL --> Other Quranic model
                      |
                      +--> FAIL --> openai/whisper-base
                                        |
                                        +--> FAIL --> RuntimeError
```

---

## Model Conversion (faster-whisper)

The faster-whisper backend requires CTranslate2-format models, converted once from HuggingFace:

```bash
ct2-transformers-converter \
    --model tarteel-ai/whisper-base-ar-quran \
    --output_dir models/whisper-quran-ct2 \
    --quantization int8
```

| Quantization | Size (base model) | CPU Speed | Accuracy |
|-------------|-------------------|-----------|----------|
| `int8` | ~73 MB | Fastest | Minimal loss |
| `float16` | ~148 MB | Medium | No loss |
| `float32` | ~296 MB | Baseline | Reference |

---

## Device Selection

### HuggingFace backend: `get_device(preferred) -> str`

Auto-detection priority:

1. **CUDA** -- `torch.cuda.is_available()` (NVIDIA GPU)
2. **MPS** -- `torch.backends.mps.is_available()` (Apple Metal on M1/M2/M3)
3. **CPU** -- always available

| Device | Dtype    | Notes |
|--------|----------|-------|
| cuda   | float16  | Best performance for large model |
| mps    | float16  | Apple Silicon; some ops fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| cpu    | float32  | Slowest but most compatible |

### faster-whisper backend: `_resolve_fw_device_and_compute(device, compute_type)`

CTranslate2 does not support Apple MPS. On M1, it uses Apple Accelerate for optimised CPU inference with int8 — this is fast without MPS.

---

## Audio Loading

### `load_audio(audio_path, target_sr=16000) -> (np.ndarray, int)`

Shared by both backends. No ffmpeg required.

- Uses **soundfile** library — handles WAV, FLAC, OGG, AIFF natively
- Fallback to **librosa** for edge cases and resampling
- Forces mono by averaging channels, resamples to `target_sr` (16 kHz for Whisper)
- Returns float32 array normalised to [-1, 1]

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV    | `.wav`    | Recommended |
| FLAC   | `.flac`   | Lossless |
| OGG    | `.ogg`    | Lossy |
| AIFF   | `.aiff` / `.aif` | macOS common |

---

## Transcription

### faster-whisper: `transcribe_faster_whisper(fw_model, audio_array, sr, ...) -> dict`

Returns:

| Key | Type | Description |
|-----|------|-------------|
| `text` | str | Transcribed Arabic text |
| `language` | str | Always `"ar"` (forced Arabic) |
| `duration_sec` | float | Audio duration in seconds |
| `word_timestamps` | list\|None | `[{word, start, end, probability}, ...]` |
| `num_chunks` | int | Number of internal Whisper segments |

Audio is passed as a numpy array — ffmpeg/av is never called. Language is forced to Arabic. Word-level timestamps are always collected for downstream alignment reuse.

### HuggingFace: `transcribe_whisper(model, processor, audio_array, sr, device) -> dict`

Returns:

| Key | Type | Description |
|-----|------|-------------|
| `text` | str | Transcribed Arabic text |
| `language` | str | Always `"ar"` (forced Arabic) |
| `duration_sec` | float | Audio duration in seconds |

### HuggingFace: Window-Based Chunked Transcription

`transcribe_chunked(model, processor, audio_array, sr, ...)` — used when audio > `chunk_seconds` (default 10s).

- **CPU mode**: `concurrent.futures.ProcessPoolExecutor(max_workers=N)`
- **GPU/MPS mode**: Sequential chunked processing
- **Merge strategy**: Concatenates words temporally, skipping overlap region words

Not used by the faster-whisper backend (which handles long audio via internal Whisper windows).

---

## WER/CER Computation

### `compute_wer(hypothesis, reference, normalise_fn=None) -> dict`

Uses the `jiwer` library. Shared by both backends.

| Key | Type | Description |
|-----|------|-------------|
| `wer` | float | Word Error Rate [0, 1] |
| `cer` | float | Character Error Rate [0, 1] |
| `insertions` | int | Word-level insertions |
| `deletions` | int | Word-level deletions |
| `substitutions` | int | Word-level substitutions |

---

## Data Flow

```
Audio File (.wav / .flac / .ogg / .aiff)
      |
      v
load_audio() --> np.ndarray (float32, 16kHz, mono)
      |
      v
  --backend?
      |
      +-- faster-whisper --> transcribe_faster_whisper()
      |                          |
      |                          +--> text + word_timestamps
      |
      +-- huggingface --> chunk_audio() (if > chunk_seconds)
                              |
                              v
                         transcribe_whisper() / transcribe_chunked()
                              |
                              +--> text
      |
      v
  compute_wer() --> WER/CER metrics
```

---

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `faster-whisper` | ASR backend: `faster-whisper` or `huggingface` |
| `--model` | `tarteel-ai/whisper-base-ar-quran` | HuggingFace model ID |
| `--model_dir` | `models/whisper-quran-ct2` | CTranslate2 model directory (faster-whisper only) |
| `--compute_type` | `auto` | CTranslate2 compute type: `auto`, `int8`, `float16`, `float32` |
| `--device` | `auto` | Device: `cpu`, `cuda`, `mps`, `auto` |
| `--chunk_seconds` | `10.0` | Window size for chunked ASR (HuggingFace only) |
| `--parallel_workers` | `1` | CPU workers for parallel chunks (HuggingFace only) |
