"""
test_aligner.py - Tests for src/aligner.py

Tests 1-9 use synthetic data (np.zeros / small arrays) and require no GPU.
Tests 10-12 load the real sample WAV from data/samples/mock.wav and exercise
the full align_with_fallback pipeline against real audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest

from src.aligner import (
    create_mock_alignment,
    align_with_fallback,
    compute_gop_score,
    check_medd_duration,
    AlignedWord,
)
from src.phonemizer import PhonemeToken

# ---------------------------------------------------------------------------
# Module-level path constant for the real sample audio file
# ---------------------------------------------------------------------------
SAMPLE_WAV = Path(__file__).resolve().parent.parent / "data" / "samples" / "mock.wav"


# ---------------------------------------------------------------------------
# Test 1: create_mock_alignment returns 2 words, each ~2.0 sec
# ---------------------------------------------------------------------------
def test_create_mock_alignment_word_count():
    """create_mock_alignment('بسم الله', 4.0) should return 2 AlignedWords."""
    aligned = create_mock_alignment("بسم الله", 4.0)
    assert len(aligned) == 2
    for aw in aligned:
        assert isinstance(aw, AlignedWord)


def test_create_mock_alignment_duration_per_word():
    """Each word should have duration ≈ 2.0 sec."""
    aligned = create_mock_alignment("بسم الله", 4.0)
    for aw in aligned:
        assert abs(aw.duration_sec - 2.0) < 1e-3


# ---------------------------------------------------------------------------
# Test 2: Durations sum to approximately audio_duration
# ---------------------------------------------------------------------------
def test_create_mock_alignment_total_duration():
    """Sum of durations should approximately equal audio_duration."""
    audio_duration = 6.0
    aligned = create_mock_alignment("بسم الله الرحمن", audio_duration)
    total = sum(aw.duration_sec for aw in aligned)
    assert abs(total - audio_duration) < 0.01


# ---------------------------------------------------------------------------
# Test 3: compute_gop_score with zero-logit tensor → valid float in [0,1]
# ---------------------------------------------------------------------------
def test_compute_gop_score_zero_logits():
    """
    compute_gop_score should return a valid float in [0, 1] even with
    a mock model that outputs uniform (zero-logit) distributions.

    Skipped automatically when torch is not installed.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _MockProcessor:
        def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
            class _Out:
                input_features = torch.zeros(1, 80, 3000)
            return _Out()

    class _MockOutput:
        logits = torch.zeros(1, 10, 51865)   # zero logits → uniform after softmax

    class _MockModel:
        def __call__(self, input_features):
            return _MockOutput()

    audio_segment = np.zeros(1600, dtype=np.float32)
    score = compute_gop_score(
        audio_segment=audio_segment,
        sr=16000,
        model=_MockModel(),
        processor=_MockProcessor(),
        target_phones=["/b/", "/s/", "/m/"],
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test 4: align_with_fallback with invalid audio → returns list, all FALLBACK
# ---------------------------------------------------------------------------
def test_align_with_fallback_invalid_audio():
    """
    align_with_fallback with tiny/invalid audio should not raise and
    should return a list of AlignedWords with FALLBACK confidence.
    """
    audio = np.zeros(100, dtype=np.float32)
    sr = 16000
    reference = "بسم الله"
    asr_transcript = "بسم الله"

    result = align_with_fallback(audio, sr, reference, asr_transcript)
    assert isinstance(result, list)
    assert len(result) > 0
    for aw in result:
        assert aw.alignment_confidence == "FALLBACK"


# ---------------------------------------------------------------------------
# Test 5: check_medd_duration with short segment + MEDD_WAJIB → True
# ---------------------------------------------------------------------------
def test_check_medd_duration_violation():
    """
    An AlignedWord with duration < 0.3 s and a MEDD_WAJIB ref_token
    should return True (violation detected).
    """
    aligned_word = AlignedWord(
        word_position="1:1:1",
        text="الله",
        start_sec=0.5,
        end_sec=0.6,          # duration = 0.1 s < 0.3 s threshold
        duration_sec=0.1,
        gop_score=0.7,
        alignment_confidence="HIGH",
    )
    ref_token = PhonemeToken(
        word_position="1:1:1",
        grapheme="الله",
        phonemes=["/aː/", "/l/", "/l/", "/aː/", "/h/"],
        tajweed_rules=["MEDD_WAJIB_MUTTASIL"],
    )
    violation = check_medd_duration(aligned_word, ref_token, min_morae_sec=0.15)
    assert violation is True


# ---------------------------------------------------------------------------
# Test 6: check_medd_duration with tajweed_rules=[] returns False
# ---------------------------------------------------------------------------
def test_check_medd_duration_empty_rules_returns_false():
    """
    An empty tajweed_rules list contains no MEDD rule, so
    check_medd_duration must return False regardless of duration.
    """
    short_word = AlignedWord(
        word_position="1:1:0",
        text="بسم",
        start_sec=0.0,
        end_sec=0.1,
        duration_sec=0.1,
        gop_score=0.8,
        alignment_confidence="HIGH",
    )
    ref_token = PhonemeToken(
        word_position="1:1:0",
        grapheme="بسم",
        phonemes=["/b/", "/s/", "/m/"],
        tajweed_rules=[],
    )
    assert check_medd_duration(short_word, ref_token, min_morae_sec=0.15) is False


# ---------------------------------------------------------------------------
# Test 7: check_medd_duration with tajweed_rules=["IDGHAM_GHUNNA"] returns False
# ---------------------------------------------------------------------------
def test_check_medd_duration_non_medd_rule_returns_false():
    """
    IDGHAM_GHUNNA is not a MEDD rule; check_medd_duration must return False
    even when the word duration is very short.
    """
    short_word = AlignedWord(
        word_position="1:1:0",
        text="كتاباً",
        start_sec=0.0,
        end_sec=0.05,
        duration_sec=0.05,
        gop_score=0.6,
        alignment_confidence="HIGH",
    )
    ref_token = PhonemeToken(
        word_position="1:1:0",
        grapheme="كتاباً",
        phonemes=["/k/", "/t/", "/aː/", "/b/", "/aː/", "/an/"],
        tajweed_rules=["IDGHAM_GHUNNA"],
    )
    assert check_medd_duration(short_word, ref_token, min_morae_sec=0.15) is False


# ---------------------------------------------------------------------------
# Test 8: silence detection – align_audio_to_reference returns None for
#          audio with RMS < 1e-4
# ---------------------------------------------------------------------------
def test_align_audio_to_reference_returns_none_for_silence():
    """
    A zeroed (silent) audio array has RMS == 0 < _MIN_AUDIO_RMS (1e-4).
    align_audio_to_reference must return None without raising.
    """
    from src.aligner import align_audio_to_reference

    sr = 16000
    # 2 seconds of perfect silence (RMS = 0)
    silent_audio = np.zeros(sr * 2, dtype=np.float32)

    result = align_audio_to_reference(
        audio_array=silent_audio,
        sr=sr,
        reference_text="بسم الله",
        surah=1,
        ayah=1,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Test 9: create_mock_alignment positions follow "surah:ayah:idx" format
# ---------------------------------------------------------------------------
def test_create_mock_alignment_position_format():
    """
    Each AlignedWord from create_mock_alignment must have its word_position
    in the exact "surah:ayah:idx" format where idx is 0-based.
    """
    surah, ayah = 3, 7
    text = "بسم الله الرحمن"
    aligned = create_mock_alignment(text, audio_duration=6.0, surah=surah, ayah=ayah)

    assert len(aligned) == 3
    for idx, aw in enumerate(aligned):
        expected_position = f"{surah}:{ayah}:{idx}"
        assert aw.word_position == expected_position, (
            f"Expected '{expected_position}', got '{aw.word_position}'"
        )


# ---------------------------------------------------------------------------
# Tests 10-12: Integration tests using real sample audio (mock.wav)
#
# mock.wav is stereo 48 kHz ~2.5 s.  load_audio converts it to float32 mono
# at 16 kHz before passing to align_with_fallback.  Because ctc_forced_aligner
# cannot align the general (non-speech) content in this sample, all three
# strategies are expected to fall through to Strategy 3 (equal-duration
# distribution), which always sets alignment_confidence="FALLBACK".
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 10: align_with_fallback on real audio returns a non-empty list
# ---------------------------------------------------------------------------
def test_align_with_fallback_real_audio_returns_nonempty_list():
    """
    align_with_fallback on the real mock.wav must return a non-empty list of
    AlignedWord objects without raising any exception.
    """
    from src.asr import load_audio

    audio, sr = load_audio(str(SAMPLE_WAV))
    reference = "بسم الله"
    asr_transcript = "بسم الله"

    result = align_with_fallback(audio, sr, reference, asr_transcript)

    assert isinstance(result, list), "Result must be a list"
    assert len(result) > 0, "Result must not be empty"
    for aw in result:
        assert isinstance(aw, AlignedWord)


# ---------------------------------------------------------------------------
# Test 11: align_with_fallback on real audio — all entries carry FALLBACK confidence
# ---------------------------------------------------------------------------
def test_align_with_fallback_real_audio_all_fallback_confidence():
    """
    When CTC alignment is unavailable in the test environment, align_with_fallback
    falls through to Strategy 3 (equal-duration distribution).  Every returned
    AlignedWord must have alignment_confidence == "FALLBACK".
    """
    from src.asr import load_audio

    audio, sr = load_audio(str(SAMPLE_WAV))
    reference = "بسم الله"
    asr_transcript = "بسم الله"

    result = align_with_fallback(audio, sr, reference, asr_transcript)

    for aw in result:
        assert aw.alignment_confidence == "FALLBACK", (
            f"Word '{aw.text}' at {aw.word_position}: "
            f"expected FALLBACK, got '{aw.alignment_confidence}'"
        )


# ---------------------------------------------------------------------------
# Test 12: align_with_fallback on real audio — word count matches reference
# ---------------------------------------------------------------------------
def test_align_with_fallback_real_audio_word_count_matches_reference():
    """
    The number of AlignedWords returned by align_with_fallback must equal the
    number of whitespace-delimited words in the reference text.
    """
    from src.asr import load_audio

    audio, sr = load_audio(str(SAMPLE_WAV))
    reference = "بسم الله الرحمن الرحيم"
    asr_transcript = "بسم الله الرحمن الرحيم"
    expected_word_count = len(reference.split())

    result = align_with_fallback(audio, sr, reference, asr_transcript)

    assert len(result) == expected_word_count, (
        f"Expected {expected_word_count} words, got {len(result)}"
    )
