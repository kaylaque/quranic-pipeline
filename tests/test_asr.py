"""
test_asr.py - Tests for src/asr.py

Covers: get_device, load_audio (format validation + real file), chunk_audio,
and SUPPORTED_FORMATS constant.

Tests 1-8 use synthetic data (np.zeros / fake paths) and require no real files.
Tests 9-14 load the real sample WAVs from data/samples/ and verify the full
load_audio pipeline (stereo→mono, 48 kHz→16 kHz, float32 normalisation).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest

from src.asr import (
    SUPPORTED_FORMATS,
    AudioChunk,
    chunk_audio,
    get_device,
    load_audio,
)

# ---------------------------------------------------------------------------
# Module-level path constants for real sample audio files
# ---------------------------------------------------------------------------
SAMPLE_WAV = Path(__file__).resolve().parent.parent / "data" / "samples" / "mock.wav"
SAMPLE_WAV_2 = Path(__file__).resolve().parent.parent / "data" / "samples" / "mock-2.wav"


# ---------------------------------------------------------------------------
# Test 1: get_device("cpu") always returns "cpu"
# ---------------------------------------------------------------------------
def test_get_device_cpu_returns_cpu():
    """get_device('cpu') must return the string 'cpu' unconditionally."""
    result = get_device("cpu")
    assert result == "cpu"


# ---------------------------------------------------------------------------
# Test 2: get_device("auto") returns a valid device string
# ---------------------------------------------------------------------------
def test_get_device_auto_returns_valid_string():
    """get_device('auto') must return one of the recognised device strings."""
    result = get_device("auto")
    assert result in {"cpu", "cuda", "mps"}


# ---------------------------------------------------------------------------
# Test 3: load_audio raises ValueError for .mp3 path
# ---------------------------------------------------------------------------
def test_load_audio_raises_for_mp3():
    """load_audio should raise ValueError immediately for an .mp3 path."""
    with pytest.raises(ValueError, match=r"(?i)unsupported"):
        load_audio("/tmp/fake_audio.mp3")


# ---------------------------------------------------------------------------
# Test 4: load_audio raises ValueError for .aac path
# ---------------------------------------------------------------------------
def test_load_audio_raises_for_aac():
    """load_audio should raise ValueError immediately for an .aac path."""
    with pytest.raises(ValueError, match=r"(?i)unsupported"):
        load_audio("/tmp/fake_audio.aac")


# ---------------------------------------------------------------------------
# Test 5: chunk_audio on short audio returns exactly 1 chunk
# ---------------------------------------------------------------------------
def test_chunk_audio_short_audio_returns_single_chunk():
    """
    Audio shorter than chunk_seconds must produce exactly one chunk
    covering the full audio.
    """
    sr = 16000
    duration_sec = 3.0          # shorter than default chunk_seconds=10
    audio = np.zeros(int(sr * duration_sec), dtype=np.float32)

    chunks = chunk_audio(audio, sr, chunk_seconds=10.0, overlap_seconds=1.0)

    assert len(chunks) == 1
    assert isinstance(chunks[0], AudioChunk)
    # The single chunk must span the whole audio
    assert chunks[0].start_sec == 0.0
    assert abs(chunks[0].end_sec - duration_sec) < 1e-3
    assert len(chunks[0].array) == len(audio)


# ---------------------------------------------------------------------------
# Test 6: chunk_audio on long audio returns multiple chunks with correct
#          start/end times
# ---------------------------------------------------------------------------
def test_chunk_audio_long_audio_returns_multiple_chunks():
    """
    Audio longer than chunk_seconds must produce more than one chunk;
    each chunk's start and end times must be consistent with chunk_seconds
    and the step size.
    """
    sr = 16000
    chunk_sec = 5.0
    overlap_sec = 1.0
    duration_sec = 13.0   # longer than one chunk
    audio = np.zeros(int(sr * duration_sec), dtype=np.float32)

    chunks = chunk_audio(audio, sr, chunk_seconds=chunk_sec, overlap_seconds=overlap_sec)

    assert len(chunks) > 1

    # First chunk starts at 0
    assert chunks[0].start_sec == 0.0

    # Each chunk's duration should not exceed chunk_seconds
    for chunk in chunks:
        assert (chunk.end_sec - chunk.start_sec) <= chunk_sec + 1e-6

    # Last chunk must end at or near the total audio duration
    assert abs(chunks[-1].end_sec - duration_sec) < 1e-3


# ---------------------------------------------------------------------------
# Test 7: chunk_audio overlap — consecutive chunks overlap by overlap_seconds
# ---------------------------------------------------------------------------
def test_chunk_audio_consecutive_chunks_overlap():
    """
    The start of each chunk (except the first) should be
    chunk_seconds - overlap_seconds after the previous chunk's start.
    """
    sr = 16000
    chunk_sec = 4.0
    overlap_sec = 1.0
    step_sec = chunk_sec - overlap_sec   # expected step = 3.0
    duration_sec = 15.0
    audio = np.zeros(int(sr * duration_sec), dtype=np.float32)

    chunks = chunk_audio(audio, sr, chunk_seconds=chunk_sec, overlap_seconds=overlap_sec)

    # There must be at least 3 full chunks for this audio length
    assert len(chunks) >= 3

    for i in range(1, len(chunks) - 1):
        actual_step = chunks[i].start_sec - chunks[i - 1].start_sec
        # Step should equal (chunk_sec - overlap_sec) within floating-point tolerance
        assert abs(actual_step - step_sec) < 0.01, (
            f"Chunk {i}: expected step={step_sec}, got step={actual_step}"
        )


# ---------------------------------------------------------------------------
# Test 8: SUPPORTED_FORMATS contains ".wav" and ".flac"
# ---------------------------------------------------------------------------
def test_supported_formats_contains_wav_and_flac():
    """SUPPORTED_FORMATS must include '.wav' and '.flac'."""
    assert ".wav" in SUPPORTED_FORMATS
    assert ".flac" in SUPPORTED_FORMATS


# ---------------------------------------------------------------------------
# Tests 9-14: Integration tests using real sample audio files
#
# mock.wav is stereo 48 kHz 16-bit PCM (~2.5 s).
# load_audio() must convert it to float32 mono at 16 kHz.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 9: load_audio on mock.wav returns a float32 array
# ---------------------------------------------------------------------------
def test_load_audio_real_wav_returns_float32_array():
    """
    load_audio on the real stereo 48 kHz mock.wav must return a numpy array
    with dtype float32.
    """
    audio, sr = load_audio(str(SAMPLE_WAV))
    assert audio.dtype == np.float32, (
        f"Expected float32, got {audio.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 10: load_audio on mock.wav returns mono (1-D array)
# ---------------------------------------------------------------------------
def test_load_audio_real_wav_is_mono():
    """
    load_audio must mix stereo channels down to mono, producing a 1-D array.
    """
    audio, sr = load_audio(str(SAMPLE_WAV))
    assert audio.ndim == 1, (
        f"Expected 1-D (mono) array, got shape {audio.shape}"
    )


# ---------------------------------------------------------------------------
# Test 11: load_audio on mock.wav resamples to 16 000 Hz
# ---------------------------------------------------------------------------
def test_load_audio_real_wav_resampled_to_16khz():
    """
    mock.wav is recorded at 48 000 Hz.  load_audio must resample it to the
    default target_sr=16 000 Hz and return that value as the second element
    of the tuple.
    """
    audio, sr = load_audio(str(SAMPLE_WAV))
    assert sr == 16000, f"Expected sr=16000, got sr={sr}"


# ---------------------------------------------------------------------------
# Test 12: load_audio on mock.wav yields ~2.5 s duration
# ---------------------------------------------------------------------------
def test_load_audio_real_wav_duration_approx_2_5s():
    """
    mock.wav is ~2.5 seconds long.  After resampling to 16 kHz the array
    length divided by sr must fall within 0.05 s of 2.5.
    """
    audio, sr = load_audio(str(SAMPLE_WAV))
    duration = len(audio) / sr
    assert abs(duration - 2.5) < 0.05, (
        f"Expected duration ~2.5 s, got {duration:.4f} s"
    )


# ---------------------------------------------------------------------------
# Test 13: chunk_audio on real 2.5 s audio with 10 s chunk → single chunk
# ---------------------------------------------------------------------------
def test_chunk_audio_on_real_wav_single_chunk():
    """
    The real mock.wav is ~2.5 s, which is shorter than the default
    chunk_seconds=10.  chunk_audio must return exactly one chunk spanning
    the full audio.
    """
    audio, sr = load_audio(str(SAMPLE_WAV))
    chunks = chunk_audio(audio, sr, chunk_seconds=10.0, overlap_seconds=1.0)

    assert len(chunks) == 1
    assert isinstance(chunks[0], AudioChunk)
    assert chunks[0].start_sec == 0.0
    # End time must equal the actual audio duration
    expected_end = len(audio) / sr
    assert abs(chunks[0].end_sec - expected_end) < 1e-3, (
        f"Expected end_sec~{expected_end:.4f}, got {chunks[0].end_sec:.4f}"
    )
    assert len(chunks[0].array) == len(audio)


# ---------------------------------------------------------------------------
# Test 14: chunk_audio on real audio with 1 s window → multiple chunks
# ---------------------------------------------------------------------------
def test_chunk_audio_on_real_wav_multiple_chunks_with_short_window():
    """
    With chunk_seconds=1.0 and overlap_seconds=0.1 the 2.5 s mock.wav must
    produce more than one chunk.  Each chunk's duration must not exceed
    chunk_seconds and the last chunk must end at the total audio duration.
    """
    audio, sr = load_audio(str(SAMPLE_WAV))
    chunk_sec = 1.0
    overlap_sec = 0.1
    chunks = chunk_audio(audio, sr, chunk_seconds=chunk_sec, overlap_seconds=overlap_sec)

    assert len(chunks) > 1, "Expected multiple chunks for 2.5 s audio with 1 s window"

    for chunk in chunks:
        assert (chunk.end_sec - chunk.start_sec) <= chunk_sec + 1e-6

    expected_end = len(audio) / sr
    assert abs(chunks[-1].end_sec - expected_end) < 1e-3, (
        f"Last chunk end_sec {chunks[-1].end_sec:.4f} != audio duration {expected_end:.4f}"
    )
