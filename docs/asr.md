# ASR -- Automatic Speech Recognition

## Overview

The ASR module (`src/asr.py`, 665 lines) handles audio loading, Whisper-based transcription, and WER/CER computation. It supports two ASR backends:

- **faster-whisper** (default) -- CTranslate2-based, ~4x faster CPU inference with int8 quantization, no MPS required.
- **huggingface** -- Original PyTorch/transformers backend, supports MPS/CUDA. Also required internally for alignment (Stage 6) and GOP scoring (Stage 7).

Both backends share the same audio loading pipeline (soundfile, no ffmpeg) and produce the same output contract. Models are cached at module level to avoid reloading across pipeline stages.

---

## Audio Loading

### `load_audio(audio_path, target_sr=16000) -> (np.ndarray, int)`

Shared by both backends. No ffmpeg required.

1. **Primary:** `soundfile.read()` -- handles WAV, FLAC, OGG, AIFF natively
2. **Fallback:** `librosa.load()` -- for edge cases where soundfile fails
3. Converts to mono by averaging channels (`audio.mean(axis=1)`)
4. Resamples to `target_sr` (16 kHz for Whisper) via `librosa.resample()` if needed
5. Returns float32 array normalised to [-1, 1]

Format validation checks the file extension against `SUPPORTED_FORMATS` before loading:

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV    | `.wav`    | Recommended |
| FLAC   | `.flac`   | Lossless |
| OGG    | `.ogg`    | Lossy |
| AIFF   | `.aiff`, `.aif` | macOS common |

### Audio Chunking

For long audio (> `chunk_seconds`), the HuggingFace backend splits audio into overlapping windows before ASR.

**`chunk_audio(audio_array, sr, chunk_seconds=10.0, overlap_seconds=1.0) -> List[AudioChunk]`**

```
|--- chunk 0 (0s-10s) ---|
                    |--- chunk 1 (9s-19s) ---|
                                       |--- chunk 2 (18s-28s) ---|
```

- If audio is shorter than `chunk_seconds`, returns a single chunk covering the full audio
- `AudioChunk` dataclass contains: `array` (np.ndarray), `start_sec` (float), `end_sec` (float)

Not used by the faster-whisper backend (which handles long audio via internal Whisper 30s windows).

---

## ASR Backends

### faster-whisper (default)

Uses [CTranslate2](https://github.com/OpenNMT/CTranslate2) for optimised inference. Requires a one-time model conversion from HuggingFace format.

| Aspect | Value |
|--------|-------|
| Engine | CTranslate2 |
| Quantization | int8 (CPU), float16 (CUDA) |
| MPS support | No -- maps to CPU automatically |
| Word timestamps | Built-in, always collected |
| Chunking | Internal Whisper 30s windows |
| Model path | `models/whisper-quran-ct2/` (local) |

**Key functions:**

**`load_faster_whisper_model(model_path, device, compute_type)`** -- Cached model loader. Returns a `faster_whisper.WhisperModel` instance. Cache key includes model_path + device + compute_type; returns cached instance if all three match.

**`transcribe_faster_whisper(fw_model, audio_array, sr, beam_size=5, vad_filter=False)`** -- Transcribes audio and returns text + word timestamps.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fw_model` | -- | A `faster_whisper.WhisperModel` instance |
| `audio_array` | -- | float32 numpy array, 16 kHz mono |
| `sr` | 16000 | Sample rate (must be 16000 for Whisper) |
| `beam_size` | 5 | Beam search width |
| `vad_filter` | False | Enable Silero VAD silence filtering (useful for audio > 5 min) |

Returns:

| Key | Type | Description |
|-----|------|-------------|
| `text` | str | Transcribed Arabic text |
| `language` | str | Always `"ar"` (forced Arabic) |
| `duration_sec` | float | Audio duration in seconds |
| `word_timestamps` | list\|None | `[{word, start, end, probability}, ...]` |
| `num_chunks` | int | Number of internal Whisper segments |

**Device mapping (`_resolve_fw_device_and_compute`):**

| `--device` | CTranslate2 device | Compute type (auto) |
|------------|-------------------|-------------------|
| `auto` | `cpu` or `cuda` (auto-detect via torch) | int8 (CPU) / float16 (CUDA) |
| `cpu` | `cpu` | `int8` |
| `cuda` | `cuda` | `float16` |
| `mps` | `cpu` (with warning) | `int8` |

CTranslate2 does not support Apple MPS. On M1/M2/M3, CPU + int8 uses Apple Accelerate for fast inference without MPS.

### HuggingFace (transformers)

Original backend using PyTorch `AutoModelForSpeechSeq2Seq`. Required internally for alignment (Stage 6) and GOP scoring (Stage 7) because CTranslate2 doesn't expose encoder logits.

**Key functions:**

**`load_whisper_model(model_name, device) -> (model, processor, device_str)`** -- Cached model + processor loader with fallback chain. Returns a 3-tuple: `(AutoModelForSpeechSeq2Seq, AutoProcessor, device_string)`.

**`transcribe_whisper(model, processor, audio_array, sr, device) -> dict`** -- Single-pass transcription. Forces Arabic language via `forced_decoder_ids` on `model.generation_config`.

Returns:

| Key | Type | Description |
|-----|------|-------------|
| `text` | str | Transcribed Arabic text |
| `language` | str | Always `"ar"` (forced Arabic) |
| `duration_sec` | float | Audio duration in seconds |

**`transcribe_chunked(model, processor, audio_array, sr, device, chunk_seconds=10.0, overlap_seconds=1.0, max_workers=1, model_name=...)`** -- Window-based chunked transcription for long audio.

- **CPU mode** (`max_workers > 1`): `concurrent.futures.ProcessPoolExecutor`
- **GPU/MPS mode** or `max_workers == 1`: Sequential chunk processing
- **Merge strategy**: Keeps first chunk fully, then for each subsequent chunk, skips words that fall within the overlap region at its start (estimated as `overlap_seconds × words_per_sec`) to avoid duplicating transcribed words at chunk boundaries

Returns same keys as `transcribe_whisper` plus `num_chunks`.

---

## Supported Models

| Model ID | Base Architecture | Parameters | Notes |
|----------|------------------|------------|-------|
| `tarteel-ai/whisper-base-ar-quran` | Whisper Base | ~74M | Lightweight, fast on CPU, outputs diacritics |
| `wasimlhr/whisper-quran-v1` | Whisper Large-v3 | ~1.55B | High accuracy (WER ~0.020 on clean audio), no diacritics in output |
| `openai/whisper-base` | Whisper Base | ~74M | Fallback only, not Quran-specialised |

Constants in code:
```python
QURANIC_MODELS = [
    "tarteel-ai/whisper-base-ar-quran",
    "wasimlhr/whisper-quran-v1",
]
FALLBACK_MODEL = "openai/whisper-base"
```

### Fallback Chain (HuggingFace backend)

`load_whisper_model()` builds an ordered candidate list and tries each until one loads:

```
Requested model (--model flag)
    |
    +--> FAIL --> Other Quranic model(s) from QURANIC_MODELS
                      |
                      +--> FAIL --> openai/whisper-base (FALLBACK_MODEL)
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

If a specific device is requested but unavailable, falls back to CPU with a warning.

### `_get_dtype(device) -> torch.dtype`

| Device | Dtype    | Notes |
|--------|----------|-------|
| cuda   | float16  | Best performance for large model |
| mps    | float16  | Apple Silicon; some ops fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| cpu    | float32  | Slowest but most compatible |

---

## Module-Level Caching

Both backends use module-level globals to avoid reloading models on repeated calls within the same process:

**HuggingFace cache (4 globals):**
- `_whisper_model` -- the loaded `AutoModelForSpeechSeq2Seq`
- `_whisper_processor` -- the loaded `AutoProcessor`
- `_whisper_device` -- resolved device string
- `_whisper_model_name` -- model name (cache key)

**faster-whisper cache (4 globals):**
- `_fw_model` -- the loaded `WhisperModel`
- `_fw_model_path` -- model path (cache key)
- `_fw_device_str` -- resolved device string (cache key)
- `_fw_compute_type_str` -- resolved compute type (cache key)

A cached model is returned only if **all** cache key fields match the request. Changing any parameter triggers a full reload.

---

## WER/CER Computation

### `compute_wer(hypothesis, reference, normalise_fn=None) -> dict`

Uses the `jiwer` library. Shared by both backends.

| Key | Type | Description |
|-----|------|-------------|
| `wer` | float | Word Error Rate (can exceed 1.0 -- not clamped) |
| `cer` | float | Character Error Rate |
| `insertions` | int | Word-level insertions |
| `deletions` | int | Word-level deletions |
| `substitutions` | int | Word-level substitutions |

**Edge cases:**
- Empty reference: returns `wer=0.0, cer=0.0` with a warning (WER is undefined when reference has zero words)
- jiwer computation failure: returns `wer=1.0, cer=1.0` as a conservative fallback

---

## Data Flow

```
Audio File (.wav / .flac / .ogg / .aiff / .aif)
      |
      v
load_audio() --> np.ndarray (float32, 16kHz, mono)
      |
      v
  --backend?
      |
      +-- faster-whisper --> load_faster_whisper_model()
      |                          |
      |                     transcribe_faster_whisper()
      |                          |
      |                          +--> text + word_timestamps
      |
      +-- huggingface --> load_whisper_model()
                              |
                         audio > chunk_seconds?
                              |
                         +-- yes --> chunk_audio() --> transcribe_chunked()
                         |                                  |
                         +-- no  --> transcribe_whisper()    |
                                          |                 |
                                          +--> text  <------+
      |
      v
  compute_wer(hypothesis, reference) --> WER/CER metrics
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
