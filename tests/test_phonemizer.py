"""
test_phonemizer.py - Tests for src/phonemizer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from src.phonemizer import (
    ARABIC_PHONEME_MAP,
    PhonemeToken,
    PhoneDiff,
    phonemize_verse,
    phonemize_aligned_pair,
    compute_phone_diff,
    phonemize_and_diff,
    _classify_tajweed_diff,
)
from src.preprocessor import DiffResult


# ---------------------------------------------------------------------------
# Test 1: ARABIC_PHONEME_MAP has exactly 29 entries
# ---------------------------------------------------------------------------
def test_phoneme_map_size():
    """ARABIC_PHONEME_MAP must have exactly 29 entries."""
    assert len(ARABIC_PHONEME_MAP) == 29


# ---------------------------------------------------------------------------
# Test 2: phonemize_verse returns one PhonemeToken per word
# ---------------------------------------------------------------------------
def test_phonemize_verse_one_token_per_word():
    """phonemize_verse should return exactly one PhonemeToken per word."""
    text = "بسم الله الرحمن"
    tokens = phonemize_verse(1, 1, text)
    assert len(tokens) == 3
    for tok in tokens:
        assert isinstance(tok, PhonemeToken)
        assert isinstance(tok.phonemes, list)


# ---------------------------------------------------------------------------
# Test 3: compute_phone_diff identical phones → MATCH, distance=0
# ---------------------------------------------------------------------------
def test_compute_phone_diff_match():
    """Identical phoneme lists should produce MATCH with edit_distance=0."""
    phones = ["/b/", "/i/", "/s/", "/m/"]
    ref_token = PhonemeToken(
        word_position="1:1:0",
        grapheme="بسم",
        phonemes=phones,
        tajweed_rules=None,
    )
    diff = compute_phone_diff(ref_token, phones)
    assert diff.diff_type == "MATCH"
    assert diff.edit_distance == 0


# ---------------------------------------------------------------------------
# Test 4: compute_phone_diff with one substituted phone → PHONE_SUB
# ---------------------------------------------------------------------------
def test_compute_phone_diff_substitution():
    """One substituted phoneme should produce PHONE_SUB."""
    ref_phones = ["/b/", "/i/", "/s/"]
    hyp_phones = ["/b/", "/u/", "/s/"]   # /i/ → /u/
    ref_token = PhonemeToken(
        word_position="1:1:0",
        grapheme="بسم",
        phonemes=ref_phones,
        tajweed_rules=None,
    )
    diff = compute_phone_diff(ref_token, hyp_phones)
    assert diff.diff_type == "PHONE_SUB"
    assert diff.edit_distance == 1


# ---------------------------------------------------------------------------
# Test 5: PhonemeToken with tajweed_rule="MEDD_WAJIB" and mismatch → TAJWEED_MEDD
# ---------------------------------------------------------------------------
def test_compute_phone_diff_tajweed_medd():
    """Mismatch on a token with MEDD tajweed rule should produce TAJWEED_MEDD."""
    ref_phones = ["/aː/", "/l/", "/l/", "/aː/", "/h/"]
    hyp_phones = ["/a/", "/l/", "/l/", "/a/", "/h/"]   # short instead of long vowels
    ref_token = PhonemeToken(
        word_position="1:1:1",
        grapheme="الله",
        phonemes=ref_phones,
        tajweed_rules=["MEDD_WAJIB_MUTTASIL"],
    )
    diff = compute_phone_diff(ref_token, hyp_phones)
    assert diff.diff_type == "TAJWEED_MEDD"
    assert diff.edit_distance > 0


# ---------------------------------------------------------------------------
# Helper: build a minimal diff-like object for phonemize_aligned_pair
# ---------------------------------------------------------------------------
class _FakeDiff:
    """Minimal stand-in for a DiffResult used by phonemize_aligned_pair."""
    def __init__(self, position: str, ref_full: str, hyp_full: str):
        self.position = position
        self.reference_full = ref_full
        self.hypothesis_full = hyp_full


# ---------------------------------------------------------------------------
# Test 6: phonemize_verse detects MEDD_TABII on a word with a long vowel
# ---------------------------------------------------------------------------
def test_phonemize_verse_detects_medd_tabii():
    """
    'الكتاب' contains 'ا' (alef, long vowel letter) followed by another
    letter, which must trigger the MEDD_TABII rule in phonemize_verse.
    """
    # Single word with two alef letters; phonemizer should detect MEDD_TABII
    tokens = phonemize_verse(2, 2, "الكتاب")
    assert len(tokens) == 1
    rules = tokens[0].tajweed_rules
    assert rules is not None
    assert "MEDD_TABII" in rules


# ---------------------------------------------------------------------------
# Test 7: phonemize_verse tajweed_rules field is a list or None, never a plain str
# ---------------------------------------------------------------------------
def test_phonemize_verse_tajweed_rules_type():
    """
    tajweed_rules on every token returned by phonemize_verse must be either
    a list of strings or None — never a bare string.
    """
    tokens = phonemize_verse(1, 1, "بسم الله الرحمن الرحيم")
    for tok in tokens:
        assert tok.tajweed_rules is None or isinstance(tok.tajweed_rules, list), (
            f"Token {tok.word_position}: tajweed_rules is {type(tok.tajweed_rules)}"
        )
        if isinstance(tok.tajweed_rules, list):
            for rule in tok.tajweed_rules:
                assert isinstance(rule, str)


# ---------------------------------------------------------------------------
# Test 8: phonemize_aligned_pair returns two PhonemeTokens with matching position
# ---------------------------------------------------------------------------
def test_phonemize_aligned_pair_returns_matching_positions():
    """
    phonemize_aligned_pair must return (ref_token, hyp_token) where both
    tokens share the same word_position as the input diff.
    """
    diff = _FakeDiff(position="1:2:3", ref_full="الله", hyp_full="اله")
    ref_tok, hyp_tok = phonemize_aligned_pair(diff, surah=1, ayah=2)

    assert isinstance(ref_tok, PhonemeToken)
    assert isinstance(hyp_tok, PhonemeToken)
    assert ref_tok.word_position == "1:2:3"
    assert hyp_tok.word_position == "1:2:3"


# ---------------------------------------------------------------------------
# Test 9: phonemize_aligned_pair with empty reference_full returns empty phonemes
# ---------------------------------------------------------------------------
def test_phonemize_aligned_pair_empty_reference_gives_empty_phonemes():
    """
    When reference_full is empty (INSERTION case), the ref PhonemeToken
    must have an empty phonemes list.
    """
    diff = _FakeDiff(position="1:1:5", ref_full="", hyp_full="كلمة")
    ref_tok, hyp_tok = phonemize_aligned_pair(diff, surah=1, ayah=1)

    assert ref_tok.phonemes == []
    # Hypothesis should still have phonemes since hyp_full is non-empty
    assert len(hyp_tok.phonemes) > 0


# ---------------------------------------------------------------------------
# Test 10: _classify_tajweed_diff maps IDGHAM_GHUNNA to TAJWEED_IDGHAM
#          and compute_phone_diff with that rule returns TAJWEED_IDGHAM
# ---------------------------------------------------------------------------
def test_classify_tajweed_diff_idgham_ghunna_maps_to_tajweed_idgham():
    """
    _classify_tajweed_diff(["IDGHAM_GHUNNA"]) must return "TAJWEED_IDGHAM".
    Additionally, compute_phone_diff on a token with IDGHAM_GHUNNA rule and a
    mismatching hypothesis must produce diff_type == "TAJWEED_IDGHAM".
    """
    assert _classify_tajweed_diff(["IDGHAM_GHUNNA"]) == "TAJWEED_IDGHAM"

    ref_phones = ["/k/", "/t/", "/aː/", "/b/", "/aː/", "/an/"]
    hyp_phones = ["/k/", "/t/", "/aː/", "/b/", "/aː/", "/a/"]  # tanween dropped
    ref_token = PhonemeToken(
        word_position="1:1:0",
        grapheme="كتاباً",
        phonemes=ref_phones,
        tajweed_rules=["IDGHAM_GHUNNA"],
    )
    diff = compute_phone_diff(ref_token, hyp_phones)
    assert diff.diff_type == "TAJWEED_IDGHAM"
    assert diff.edit_distance > 0
