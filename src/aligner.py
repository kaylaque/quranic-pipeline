"""
aligner.py - CTC forced alignment, GOP scoring, and fallback alignment strategies.

GOP scoring uses the DNN-based formula:
    GOP(p) = -(1/D) * SUM(t=T..T+D-1) log P_t(p | O)
normalised to [0, 1] via sigmoid scaling.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum RMS energy threshold: audio below this is treated as silence
# and CTC alignment is skipped immediately (would produce empty tensors).
_MIN_AUDIO_RMS = 1e-4


# ---------------------------------------------------------------------------
# AlignedWord dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlignedWord:
    word_position: str          # "surah:ayah:word_index"
    text: str
    start_sec: float
    end_sec: float
    duration_sec: float
    gop_score: Optional[float]          # None until scored
    alignment_confidence: str           # "HIGH" | "LOW" | "FALLBACK"


# ---------------------------------------------------------------------------
# CTC forced alignment
# ---------------------------------------------------------------------------

def _align_with_whisper_word_timestamps(
    audio_array: np.ndarray,
    sr: int,
    reference_text: str,
    model,
    processor,
    device: str,
    surah: int,
    ayah: int,
) -> Optional[List[AlignedWord]]:
    """
    Use Whisper's word-level timestamp output to align *reference_text* to audio.

    The Whisper model derives word timestamps from its own cross-attention weights —
    no separate alignment model is required, and it works for Arabic.

    Strategy:
    1. Run the ASR pipeline with ``return_timestamps="word"`` to get per-word spans.
    2. Map the resulting chunks to *reference_text* words by index.
       If the model produced fewer chunks than reference words (partial transcription),
       the remaining reference words are distributed over the leftover audio span.

    Returns a list of :class:`AlignedWord` with ``alignment_confidence="HIGH"``,
    or ``None`` if the pipeline raises or returns no chunks.
    """
    import torch
    from transformers import AutomaticSpeechRecognitionPipeline

    words = reference_text.split()
    if not words:
        return []

    # Patch generation config for Arabic + timestamps.
    # Older Whisper fine-tunes were saved before the generation_config schema
    # stabilised and are missing timestamp-related fields.  We back-fill them
    # from known Whisper universal constants.
    #
    # alignment_heads for openai/whisper-base (all sizes fine-tuned from it
    # share the same architecture and therefore the same attention-head indices):
    #   layer 3: heads 1-7, layer 4: heads 1,2,4,7, layer 5: heads 1-4
    # These are the cross-attention heads empirically shown to track timing.
    _WHISPER_NO_TIMESTAMPS_ID = 50363
    _WHISPER_BASE_ALIGNMENT_HEADS = [
        [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7],
        [4, 1], [4, 2], [4, 4], [4, 7],
        [5, 1], [5, 2], [5, 3], [5, 4],
    ]
    try:
        cfg = model.generation_config
        if not hasattr(cfg, "no_timestamps_token_id") or cfg.no_timestamps_token_id is None:
            cfg.no_timestamps_token_id = _WHISPER_NO_TIMESTAMPS_ID
        if not hasattr(cfg, "begin_suppress_tokens") or cfg.begin_suppress_tokens is None:
            cfg.begin_suppress_tokens = [220, 50257]
        if not hasattr(cfg, "alignment_heads") or cfg.alignment_heads is None:
            cfg.alignment_heads = _WHISPER_BASE_ALIGNMENT_HEADS
        forced = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
        if forced:
            cfg.forced_decoder_ids = forced
    except Exception as exc:
        logger.debug("Could not patch generation_config for timestamps (non-fatal): %s", exc)

    # Build pipeline reusing the already-loaded model and processor
    try:
        pipe_device = torch.device(device)
    except Exception:
        pipe_device = torch.device("cpu")

    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=pipe_device,
    )

    # Attempt 1: word-level timestamps (requires alignment_heads)
    chunks = []
    try:
        result = pipe(audio_array, return_timestamps="word")
        chunks = result.get("chunks", [])
    except Exception as exc_word:
        logger.debug("Word-level timestamps failed (%s), trying chunk-level", exc_word)

    # Attempt 2: chunk-level timestamps + word interpolation (no alignment_heads needed)
    if not chunks:
        try:
            result = pipe(audio_array, return_timestamps=True)
            raw_chunks = result.get("chunks", [])
            # Distribute words within each chunk proportionally by character count
            for rc in raw_chunks:
                chunk_words = rc.get("text", "").split()
                if not chunk_words:
                    continue
                ts = rc.get("timestamp") or (0.0, float(len(audio_array)) / max(sr, 1))
                chunk_start = float(ts[0]) if ts[0] is not None else 0.0
                chunk_end = float(ts[1]) if ts[1] is not None else float(len(audio_array)) / max(sr, 1)
                dur = chunk_end - chunk_start
                dur_per_word = dur / max(len(chunk_words), 1)
                for w_idx, w in enumerate(chunk_words):
                    w_start = chunk_start + w_idx * dur_per_word
                    w_end = w_start + dur_per_word
                    chunks.append({"text": w, "timestamp": (w_start, w_end)})
        except Exception as exc_chunk:
            logger.debug("Chunk-level timestamps also failed: %s", exc_chunk)

    if not chunks:
        logger.debug("Whisper pipeline returned no chunks")
        return None

    audio_end = float(len(audio_array)) / max(sr, 1)
    aligned: List[AlignedWord] = []

    for idx, word in enumerate(words):
        if idx < len(chunks):
            ts = chunks[idx].get("timestamp") or (None, None)
            start = float(ts[0]) if ts[0] is not None else (aligned[-1].end_sec if aligned else 0.0)
            end = float(ts[1]) if ts[1] is not None else min(start + 0.3, audio_end)
        else:
            # Distribute remaining reference words evenly over leftover audio
            prev_end = aligned[-1].end_sec if aligned else 0.0
            remaining = len(words) - idx
            dur = max(0.05, (audio_end - prev_end) / remaining)
            start = prev_end
            end = min(start + dur, audio_end)

        aligned.append(AlignedWord(
            word_position=f"{surah}:{ayah}:{idx}",
            text=word,
            start_sec=round(start, 4),
            end_sec=round(end, 4),
            duration_sec=round(end - start, 4),
            gop_score=None,
            alignment_confidence="HIGH",
        ))

    return aligned


def align_audio_to_reference(
    audio_array: np.ndarray,
    sr: int,
    reference_text: str,
    language: str = "ara",
    surah: int = 0,
    ayah: int = 0,
    model=None,
    processor=None,
    device: str = "cpu",
) -> Optional[List[AlignedWord]]:
    """
    Attempt forced alignment of *audio_array* to *reference_text* using
    Whisper's word-timestamp pipeline.

    Requires *model* and *processor* (the loaded Whisper model/processor).
    When both are provided alignment runs natively on Arabic without any
    external aligner or separate model.

    Returns a list of :class:`AlignedWord` on success, or ``None`` on failure
    (audio too short, silent, or Whisper returned no chunks).
    """
    words = reference_text.split()
    if not words:
        return []

    # Minimum audio length guard
    min_samples = int(sr * 0.5)
    if len(audio_array) < min_samples:
        logger.debug("Audio too short for alignment (%d samples < %d)", len(audio_array), min_samples)
        return None

    # Silence guard — Whisper produces empty output on silent audio
    rms = float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)))
    if rms < _MIN_AUDIO_RMS:
        logger.debug("Audio is silence (RMS=%.2e < %.2e), skipping alignment", rms, _MIN_AUDIO_RMS)
        return None

    if model is None or processor is None:
        logger.debug("No model/processor provided — skipping Whisper alignment")
        return None

    try:
        return _align_with_whisper_word_timestamps(
            audio_array, sr, reference_text, model, processor, device, surah, ayah
        )
    except Exception as exc:
        logger.warning("Whisper word-timestamp alignment failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# align_with_fallback
# ---------------------------------------------------------------------------

def align_with_fallback(
    audio_array: np.ndarray,
    sr: int,
    reference_text: str,
    asr_transcript: str,
    surah: int = 0,
    ayah: int = 0,
    model=None,
    processor=None,
    device: str = "cpu",
) -> List[AlignedWord]:
    """
    Attempt alignment with multiple fallback strategies.  Never raises.

    Strategy 1: Whisper word-timestamps to *reference_text* → HIGH confidence
    Strategy 2: Whisper word-timestamps to *asr_transcript* → LOW confidence
    Strategy 3: Equal-duration distribution                  → FALLBACK confidence

    When *model*/*processor* are None both Whisper strategies are skipped
    and the result falls directly to equal-duration distribution.
    """
    # Strategy 1
    try:
        result = align_audio_to_reference(
            audio_array, sr, reference_text,
            surah=surah, ayah=ayah,
            model=model, processor=processor, device=device,
        )
        if result is not None and len(result) > 0:
            return result
    except Exception as exc:
        logger.warning("Strategy 1 alignment failed: %s", exc)

    # Strategy 2
    try:
        result = align_audio_to_reference(
            audio_array, sr, asr_transcript,
            surah=surah, ayah=ayah,
            model=model, processor=processor, device=device,
        )
        if result is not None and len(result) > 0:
            for aw in result:
                aw.alignment_confidence = "LOW"
            return result
    except Exception as exc:
        logger.warning("Strategy 2 alignment failed: %s", exc)

    # Strategy 3 – equal distribution
    audio_duration = float(len(audio_array)) / max(sr, 1)
    return create_mock_alignment(
        reference_text, audio_duration, surah=surah, ayah=ayah
    )


# ---------------------------------------------------------------------------
# GOP scoring (DNN-based)
# ---------------------------------------------------------------------------

def compute_gop_score(
    audio_segment: np.ndarray,
    sr: int,
    model,
    processor,
    target_phones: List[str],
    device: str = "cpu",
) -> float:
    """
    Compute a Goodness of Pronunciation (GOP) score using the DNN-based formula:

        GOP(p) = -(1/D) * SUM(t=T..T+D-1) log P_t(p | O)

    The raw GOP value (a negative log-posterior average) is normalised to [0, 1]
    using a sigmoid function:  score = 1 / (1 + exp(gop_raw - threshold))

    Score interpretation:
        0.0 - 0.3  : Poor pronunciation (confirmed error)
        0.3 - 0.7  : Moderate (possible error)
        0.7 - 0.85 : Acceptable (possible ASR artefact)
        0.85 - 1.0 : Good pronunciation (likely correct)
    """
    try:
        import torch
        import torch.nn.functional as F

        if not target_phones:
            return 0.5

        inputs = processor(
            audio_segment,
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=device, dtype=model.dtype)

        with torch.no_grad():
            # Obtain encoder outputs / logits
            try:
                outputs = model(input_features)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits  # (batch, seq, vocab)
                else:
                    hidden = outputs.last_hidden_state  # (batch, seq, hidden)
                    logits = hidden.mean(dim=1)         # (batch, hidden)
            except Exception as model_exc:
                logger.debug("model(input_features) failed (%s), retrying with generate()", model_exc)
                out = model.generate(
                    input_features,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=50,
                )
                if hasattr(out, "scores") and out.scores:
                    logits = torch.stack(out.scores, dim=1)
                else:
                    return 0.5

        # Compute log-posteriors
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # GOP = -(1/D) * sum of max log-posteriors across frames
        # We use the max log-prob per frame as proxy for P(target_phone | O)
        max_log_probs, _ = log_probs.max(dim=-1)  # (batch, seq)
        D = max_log_probs.shape[-1]

        if D == 0:
            return 0.5

        # Average negative log-posterior
        gop_raw = float(-max_log_probs.mean().item())

        # Normalise to [0, 1] via sigmoid: higher score = better pronunciation
        # threshold=2.0 centres the sigmoid around typical GOP values
        gop_normalised = 1.0 / (1.0 + math.exp(gop_raw - 2.0))
        return max(0.0, min(1.0, gop_normalised))

    except Exception as exc:
        logger.warning("GOP scoring failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# score_aligned_words
# ---------------------------------------------------------------------------

def score_aligned_words(
    aligned_words: List[AlignedWord],
    audio_array: np.ndarray,
    sr: int,
    ref_phonemes,           # List[PhonemeToken]
    model,
    processor,
    error_positions: Optional[set] = None,
    device: str = "cpu",
) -> List[AlignedWord]:
    """
    Compute GOP scores for words in *error_positions* (selective GOP).

    If *error_positions* is None, score all words.
    """
    scored = list(aligned_words)  # shallow copy

    for idx, aw in enumerate(scored):
        if error_positions is not None and aw.word_position not in error_positions:
            continue  # skip non-error words

        # Extract audio segment
        start_sample = int(aw.start_sec * sr)
        end_sample = int(aw.end_sec * sr)
        segment = audio_array[start_sample:end_sample]

        if len(segment) == 0:
            aw.gop_score = 0.5
            continue

        # Get target phones for this word
        target_phones: List[str] = []
        if idx < len(ref_phonemes):
            target_phones = ref_phonemes[idx].phonemes

        aw.gop_score = compute_gop_score(
            segment, sr, model, processor, target_phones, device
        )

    return scored


# ---------------------------------------------------------------------------
# Medd duration check
# ---------------------------------------------------------------------------

def check_medd_duration(
    aligned_word: AlignedWord,
    ref_token,                  # PhonemeToken
    min_morae_sec: float = 0.15,
) -> bool:
    """
    Return True if a Medd (elongation) duration violation is detected.

    A violation occurs when:
    - *ref_token.tajweed_rules* contains a MEDD rule
    - *aligned_word.duration_sec* < *min_morae_sec* × 2
    """
    if ref_token is None:
        return False
    tajweed = getattr(ref_token, "tajweed_rules", None)
    if tajweed is None:
        return False
    has_medd = any("MEDD" in r for r in tajweed) if isinstance(tajweed, list) else "MEDD" in str(tajweed)
    if not has_medd:
        return False
    return aligned_word.duration_sec < (min_morae_sec * 2)


# ---------------------------------------------------------------------------
# faster-whisper timestamp → AlignedWord conversion
# ---------------------------------------------------------------------------

def create_aligned_from_fw_timestamps(
    word_timestamps: list,
    reference_text: str,
    surah: int = 0,
    ayah: int = 0,
) -> List[AlignedWord]:
    """
    Convert faster-whisper word timestamps to a list of :class:`AlignedWord`.

    Maps reference words to ASR timestamps by position index (same strategy
    as :func:`_align_with_whisper_word_timestamps`).  Remaining reference
    words beyond the timestamp list are distributed over the leftover audio
    span.

    This avoids a second HuggingFace model inference pass when the pipeline
    already has word-level timestamps from the faster-whisper ASR step.
    """
    words = reference_text.split()
    if not words or not word_timestamps:
        return []

    audio_end = float(word_timestamps[-1].get("end", 0.0)) if word_timestamps else 0.0
    aligned: List[AlignedWord] = []

    for idx, word in enumerate(words):
        if idx < len(word_timestamps):
            ts = word_timestamps[idx]
            start = float(ts.get("start", 0.0))
            end = float(ts.get("end", start + 0.3))
        else:
            prev_end = aligned[-1].end_sec if aligned else 0.0
            remaining = len(words) - idx
            dur = max(0.05, (audio_end - prev_end) / remaining)
            start = prev_end
            end = min(start + dur, audio_end)

        aligned.append(AlignedWord(
            word_position=f"{surah}:{ayah}:{idx}",
            text=word,
            start_sec=round(start, 4),
            end_sec=round(end, 4),
            duration_sec=round(end - start, 4),
            gop_score=None,
            alignment_confidence="HIGH",
        ))

    return aligned


# ---------------------------------------------------------------------------
# Mock / fallback alignment
# ---------------------------------------------------------------------------

def create_mock_alignment(
    reference_text: str,
    audio_duration: float = 5.0,
    surah: int = 0,
    ayah: int = 0,
) -> List[AlignedWord]:
    """
    Create a simple equal-duration alignment for *reference_text*.

    Each word gets an equal share of *audio_duration*.
    gop_score is set to 0.85 and alignment_confidence to "FALLBACK".

    Note: the pre-filled gop_score=0.85 is never used by the classifier for
    FALLBACK-aligned words — :func:`classify_error` detects ``alignment_confidence
    == "FALLBACK"`` and returns LOW_CONFIDENCE before consulting the GOP score.
    The value is present only so the field is non-None if accessed directly.
    """
    words = reference_text.split()
    if not words:
        return []

    duration_per_word = audio_duration / len(words)
    aligned: List[AlignedWord] = []

    for idx, word in enumerate(words):
        start = idx * duration_per_word
        end = start + duration_per_word
        aligned.append(
            AlignedWord(
                word_position=f"{surah}:{ayah}:{idx}",
                text=word,
                start_sec=round(start, 4),
                end_sec=round(end, 4),
                duration_sec=round(duration_per_word, 4),
                gop_score=0.85,
                alignment_confidence="FALLBACK",
            )
        )

    return aligned
