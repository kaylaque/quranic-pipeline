"""
asr.py - Audio loading, Whisper ASR transcription, and WER/CER computation.

Supports two ASR backends:
  - faster-whisper (CTranslate2): ~4x faster CPU inference, int8 quantisation,
    no MPS required — recommended for M1 and production CPU deployments.
  - huggingface (transformers): original backend with PyTorch, supports MPS/CUDA.

Both backends share the same audio loading pipeline (soundfile, no ffmpeg) and
produce the same output contract.  Models are cached at module level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported Quranic models (ordered by preference)
# ---------------------------------------------------------------------------
QURANIC_MODELS = [
    "tarteel-ai/whisper-base-ar-quran",
    "wasimlhr/whisper-quran-v1",
]
FALLBACK_MODEL = "openai/whisper-base"

# Audio formats supported by soundfile
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

# ---------------------------------------------------------------------------
# Module-level model caches (HuggingFace backend)
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_processor = None
_whisper_device = None
_whisper_model_name = None

# ---------------------------------------------------------------------------
# Module-level model cache (faster-whisper backend)
# ---------------------------------------------------------------------------
_fw_model = None
_fw_model_path: Optional[str] = None
_fw_device_str: Optional[str] = None
_fw_compute_type_str: Optional[str] = None


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(preferred: str = "auto") -> str:
    """
    Return the best available PyTorch device string.

    Parameters
    ----------
    preferred : one of "auto", "cpu", "cuda", "mps"

    Returns
    -------
    Device string usable with ``torch.device()``.
    """
    import torch

    if preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but not available, falling back to CPU.")
        return "cpu"

    if preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS requested but not available, falling back to CPU.")
        return "cpu"

    if preferred == "cpu":
        return "cpu"

    # auto
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dtype(device: str):
    """Return the appropriate torch dtype for *device*."""
    import torch
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load an audio file, resample to *target_sr*, and convert to mono.

    Supported formats: WAV, FLAC, OGG, AIFF (via soundfile).

    Returns
    -------
    (audio_array, sample_rate)
        *audio_array* is a float32 numpy array normalised to [-1, 1].
    """
    import soundfile as sf

    path = str(audio_path)
    from pathlib import Path as _Path
    suffix = _Path(path).suffix.lower()  # "" for extensionless files
    if suffix and suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # Primary: soundfile (supports WAV/FLAC/OGG/AIFF natively)
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        # Convert to mono by averaging channels
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
    except Exception as sf_exc:
        # Fallback: librosa (handles more edge cases)
        try:
            import librosa
            audio, sr = librosa.load(path, sr=None, mono=True)
            audio = audio.astype(np.float32)
        except Exception as librosa_exc:
            raise RuntimeError(
                f"Could not load audio '{path}' with soundfile ({sf_exc}) "
                f"or librosa ({librosa_exc})"
            ) from sf_exc

    # Resample if needed
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr


# ---------------------------------------------------------------------------
# Audio chunking for window-based parallel processing
# ---------------------------------------------------------------------------

@dataclass
class AudioChunk:
    array: np.ndarray
    start_sec: float
    end_sec: float


def chunk_audio(
    audio_array: np.ndarray,
    sr: int,
    chunk_seconds: float = 10.0,
    overlap_seconds: float = 1.0,
) -> List[AudioChunk]:
    """
    Split *audio_array* into overlapping windows.

    Returns a list of :class:`AudioChunk` objects.  If the audio is shorter
    than *chunk_seconds*, returns a single chunk covering the full audio.
    """
    total_samples = len(audio_array)
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    step_samples = chunk_samples - overlap_samples

    if total_samples <= chunk_samples:
        return [AudioChunk(
            array=audio_array,
            start_sec=0.0,
            end_sec=float(total_samples) / sr,
        )]

    chunks: List[AudioChunk] = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append(AudioChunk(
            array=audio_array[start:end],
            start_sec=float(start) / sr,
            end_sec=float(end) / sr,
        ))
        if end >= total_samples:
            break
        start += step_samples

    return chunks


# ---------------------------------------------------------------------------
# Whisper model loading
# ---------------------------------------------------------------------------

def load_whisper_model(
    model_name: str = "tarteel-ai/whisper-base-ar-quran",
    device: str = "auto",
) -> tuple:
    """
    Load (or return cached) Whisper model and processor from HuggingFace.

    Tries the requested model first, then other Quranic models, then the
    generic ``openai/whisper-base`` fallback.

    Returns
    -------
    (model, processor, device_str)
    """
    global _whisper_model, _whisper_processor, _whisper_device, _whisper_model_name

    resolved_device = get_device(device)

    if (
        _whisper_model is not None
        and _whisper_processor is not None
        and _whisper_model_name == model_name
    ):
        return _whisper_model, _whisper_processor, _whisper_device

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    # Build ordered candidate list (requested first, then alternatives)
    candidates = [model_name]
    for m in QURANIC_MODELS:
        if m != model_name:
            candidates.append(m)
    candidates.append(FALLBACK_MODEL)

    dtype = _get_dtype(resolved_device)

    for name in candidates:
        try:
            logger.info("Loading Whisper model: %s on %s", name, resolved_device)
            processor = AutoProcessor.from_pretrained(name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                name,
                torch_dtype=dtype,
            )
            model = model.to(resolved_device)
            model.eval()

            _whisper_model = model
            _whisper_processor = processor
            _whisper_device = resolved_device
            _whisper_model_name = name
            logger.info("Loaded Whisper model: %s on %s", name, resolved_device)
            return model, processor, resolved_device
        except Exception as exc:
            logger.warning("Failed to load %s: %s", name, exc)

    raise RuntimeError(
        "Could not load any Whisper model. Check network connection or model cache."
    )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_whisper(
    model,
    processor,
    audio_array: np.ndarray,
    sr: int = 16000,
    device: str = "cpu",
) -> dict:
    """
    Transcribe *audio_array* using the given Whisper *model* and *processor*.

    Returns
    -------
    dict with keys:
        text          -- transcribed Arabic text
        language      -- detected language code (e.g. "ar")
        duration_sec  -- audio duration in seconds
    """
    import torch

    duration_sec = float(len(audio_array)) / sr

    # Prepare input features
    inputs = processor(
        audio_array,
        sampling_rate=sr,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)

    # Build attention mask: use processor-provided mask if available, else all-ones.
    # Whisper pads input_features to 30 s so an all-ones mask is always valid.
    if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
        attention_mask = inputs.attention_mask.to(device)
    else:
        attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=device)

    # Force Arabic language and transcription task.
    # Strategy: patch model.generation_config with forced_decoder_ids so we
    # don't pass 'language'/'task' to generate() (breaks old generation_config)
    # and don't pass forced_decoder_ids as a kwarg (deprecated in newer transformers).
    try:
        forced = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
        if forced:
            model.generation_config.forced_decoder_ids = forced
    except Exception as exc:
        logger.debug("Could not set forced_decoder_ids (non-fatal): %s", exc)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, attention_mask=attention_mask)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcription = text[0].strip() if text else ""

    return {
        "text": transcription,
        "language": "ar",
        "duration_sec": duration_sec,
    }


# ---------------------------------------------------------------------------
# Chunked parallel transcription
# ---------------------------------------------------------------------------

def _transcribe_chunk(args: tuple) -> tuple[float, float, str]:
    """Worker function for parallel chunk transcription.
    Accepts (chunk_array, start_sec, end_sec, model_name, device_str, sr).
    Returns (start_sec, end_sec, text).
    """
    chunk_array, start_sec, end_sec, model_name, device_str, sr_val = args
    model, processor, dev = load_whisper_model(model_name, device_str)
    result = transcribe_whisper(model, processor, chunk_array, sr_val, dev)
    return start_sec, end_sec, result["text"]


def transcribe_chunked(
    model,
    processor,
    audio_array: np.ndarray,
    sr: int = 16000,
    device: str = "cpu",
    chunk_seconds: float = 10.0,
    overlap_seconds: float = 1.0,
    max_workers: int = 1,
    model_name: str = "tarteel-ai/whisper-base-ar-quran",
) -> dict:
    """
    Transcribe *audio_array* using window-based chunking.

    For CPU with max_workers > 1, uses parallel processing.
    For GPU/MPS or max_workers == 1, processes chunks sequentially.

    Returns
    -------
    dict with keys: text, language, duration_sec, num_chunks
    """
    chunks = chunk_audio(audio_array, sr, chunk_seconds, overlap_seconds)

    if len(chunks) == 1:
        result = transcribe_whisper(model, processor, audio_array, sr, device)
        result["num_chunks"] = 1
        return result

    duration_sec = float(len(audio_array)) / sr
    use_parallel = (device == "cpu" and max_workers > 1)

    chunk_results: List[tuple[float, float, str]] = []

    if use_parallel:
        import concurrent.futures
        args_list = [
            (c.array, c.start_sec, c.end_sec, model_name, device, sr)
            for c in chunks
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = pool.map(_transcribe_chunk, args_list)
            chunk_results = list(futures)
    else:
        for c in chunks:
            result = transcribe_whisper(model, processor, c.array, sr, device)
            chunk_results.append((c.start_sec, c.end_sec, result["text"]))

    # Sort by start time and merge
    chunk_results.sort(key=lambda x: x[0])

    # Merge chunks: keep the first chunk fully, then for each subsequent chunk
    # skip words that fall within the overlap region at its start to avoid
    # duplicating transcribed words at chunk boundaries.
    merged_words: List[str] = []
    for i, (start_sec, end_sec, text) in enumerate(chunk_results):
        words = text.split()
        if i > 0 and overlap_seconds > 0 and words:
            chunk_duration = max(end_sec - start_sec, 1e-6)
            words_per_sec = len(words) / chunk_duration
            skip = max(0, int(overlap_seconds * words_per_sec))
            words = words[skip:]
        merged_words.extend(words)

    return {
        "text": " ".join(merged_words),
        "language": "ar",
        "duration_sec": duration_sec,
        "num_chunks": len(chunks),
    }


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------


def _resolve_fw_device_and_compute(
    preferred_device: str = "auto",
    preferred_compute: str = "auto",
) -> tuple[str, str]:
    """
    Resolve CTranslate2 device and compute_type strings.

    CTranslate2 does not support Apple MPS — maps to CPU with a warning.
    On M1, CPU + int8 uses Apple Accelerate and is fast without MPS.

    Returns
    -------
    (device, compute_type) as strings accepted by ``faster_whisper.WhisperModel``.
    """
    if preferred_device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    elif preferred_device == "cuda":
        device = "cuda"
    elif preferred_device == "mps":
        logger.warning(
            "CTranslate2 does not support Apple MPS. "
            "Falling back to CPU with int8 — still fast on M1 via Apple Accelerate."
        )
        device = "cpu"
    else:
        device = "cpu"

    if preferred_compute == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = preferred_compute

    return device, compute_type


def load_faster_whisper_model(
    model_path: str = "models/whisper-quran-ct2",
    device: str = "auto",
    compute_type: str = "auto",
):
    """
    Load (or return cached) a faster-whisper ``WhisperModel``.

    Parameters
    ----------
    model_path :
        Local directory containing a CTranslate2-converted model, or a
        standard ``openai/whisper-*`` HuggingFace model ID (downloaded
        automatically in CTranslate2 format by faster-whisper).
    device :
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"`` (mps maps to cpu).
    compute_type :
        ``"auto"``, ``"int8"``, ``"float16"``, or ``"float32"``.
        ``"auto"`` selects int8 on CPU and float16 on CUDA.

    Returns
    -------
    faster_whisper.WhisperModel
    """
    global _fw_model, _fw_model_path, _fw_device_str, _fw_compute_type_str

    resolved_device, resolved_compute = _resolve_fw_device_and_compute(device, compute_type)

    if (
        _fw_model is not None
        and _fw_model_path == model_path
        and _fw_device_str == resolved_device
        and _fw_compute_type_str == resolved_compute
    ):
        return _fw_model

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper is not installed. Run: pip install faster-whisper>=1.1.0"
        )

    logger.info(
        "Loading faster-whisper model: %s | device=%s | compute_type=%s",
        model_path, resolved_device, resolved_compute,
    )

    model = WhisperModel(
        model_path,
        device=resolved_device,
        compute_type=resolved_compute,
    )

    _fw_model = model
    _fw_model_path = model_path
    _fw_device_str = resolved_device
    _fw_compute_type_str = resolved_compute

    logger.info("Loaded faster-whisper model: %s", model_path)
    return model


def transcribe_faster_whisper(
    fw_model,
    audio_array: np.ndarray,
    sr: int = 16000,
    beam_size: int = 5,
    vad_filter: bool = False,
) -> dict:
    """
    Transcribe *audio_array* using a faster-whisper ``WhisperModel``.

    Audio is passed as a numpy array — no ffmpeg required.  Language is
    forced to Arabic (``"ar"``).  Word-level timestamps are always collected
    so the alignment stage can reuse them without a second Whisper pass.

    Parameters
    ----------
    fw_model :
        A ``faster_whisper.WhisperModel`` instance.
    audio_array :
        float32 numpy array, 16 kHz mono, normalised to [-1, 1].
    sr :
        Sample rate — must be 16 000 Hz (Whisper requirement).
    beam_size :
        Beam search width (default 5).
    vad_filter :
        Enable Silero VAD silence filtering.  Useful for very long audio
        (> 5 min); leave False for typical ayah-length recordings.

    Returns
    -------
    dict with keys:

    =========================================================
    text             str        Transcribed Arabic text
    language         str        Always ``"ar"``
    duration_sec     float      Audio duration in seconds
    word_timestamps  list|None  [{word, start, end, probability}, ...]
    num_chunks       int        Number of internal Whisper segments
    =========================================================
    """
    duration_sec = float(len(audio_array)) / max(sr, 1)

    segments_gen, _ = fw_model.transcribe(
        audio_array,
        language="ar",
        task="transcribe",
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad_filter,
    )

    text_parts: list[str] = []
    word_timestamps: list[dict] = []
    num_segments = 0

    for segment in segments_gen:
        text_parts.append(segment.text.strip())
        num_segments += 1
        if segment.words:
            for w in segment.words:
                word_timestamps.append({
                    "word": w.word.strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": float(w.probability),
                })

    full_text = " ".join(t for t in text_parts if t).strip()

    return {
        "text": full_text,
        "language": "ar",
        "duration_sec": duration_sec,
        "word_timestamps": word_timestamps if word_timestamps else None,
        "num_chunks": num_segments,
    }


# ---------------------------------------------------------------------------
# WER / CER computation
# ---------------------------------------------------------------------------

def compute_wer(
    hypothesis: str,
    reference: str,
    normalise_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """
    Compute Word Error Rate and Character Error Rate.

    Parameters
    ----------
    hypothesis:    ASR output text
    reference:     ground-truth text
    normalise_fn:  optional callable applied to both strings before scoring

    Returns
    -------
    dict with keys: wer, cer, insertions, deletions, substitutions
    """
    import jiwer

    if normalise_fn is not None:
        hypothesis = normalise_fn(hypothesis)
        reference = normalise_fn(reference)

    # Guard against empty strings
    if not reference.strip():
        if hypothesis.strip():
            # WER is undefined when the reference is empty; return 0.0 as a
            # conservative default rather than 1.0 (which implies 100% errors
            # against zero reference words, which is nonsensical).
            logger.warning("compute_wer: reference is empty — WER is undefined, returning 0.0")
        return {
            "wer": 0.0,
            "cer": 0.0,
            "insertions": 0,
            "deletions": 0,
            "substitutions": 0,
        }

    try:
        word_out = jiwer.process_words(reference, hypothesis)
        char_out = jiwer.process_characters(reference, hypothesis)
        return {
            "wer": float(word_out.wer),
            "cer": float(char_out.cer),
            "insertions": int(word_out.insertions),
            "deletions": int(word_out.deletions),
            "substitutions": int(word_out.substitutions),
        }
    except Exception as exc:
        logger.warning("jiwer computation failed: %s", exc)
        return {
            "wer": 1.0,
            "cer": 1.0,
            "insertions": 0,
            "deletions": 0,
            "substitutions": 0,
        }
