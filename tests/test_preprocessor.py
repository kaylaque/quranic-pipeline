"""
test_preprocessor.py - Tests for src/preprocessor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from src.preprocessor import (
    strip_harakat,
    normalise_hamza,
    tokenise_reference,
    two_pass_compare,
    all_match,
    DiffResult,
    WordToken,
)


# ---------------------------------------------------------------------------
# Test 1: strip_harakat basic
# ---------------------------------------------------------------------------
def test_strip_harakat_basic():
    """بِسْمِ should become بسم (remove kasra, sukun, kasra)."""
    result = strip_harakat("بِسْمِ")
    assert result == "بسم"


# ---------------------------------------------------------------------------
# Test 2: strip_harakat with tatweel and superscript alef
# ---------------------------------------------------------------------------
def test_strip_harakat_alef_superscript():
    """ٱلرَّحْمَـٰنِ – strip all harakat including superscript alef (U+0670)."""
    # ٱ is U+0671 (alef wasla) — NOT a harakat, it's a letter
    # ـ is tatweel U+0640
    # ٰ is superscript alef U+0670 (harakat)
    # رَّ has fatha + shadda
    text = "ٱلرَّحْمَـٰنِ"
    result = strip_harakat(text)
    # Should remove: fatha (َ), shadda (ّ), sukun (ْ), fatha (َ), tatweel (ـ),
    # superscript alef (ٰ), kasra (ِ)
    # Letters remaining: ٱ ل ر ح م ن
    assert "ـ" not in result   # tatweel removed
    assert "\u0670" not in result  # superscript alef removed
    assert "\u064e" not in result  # fatha removed
    assert "\u0651" not in result  # shadda removed


# ---------------------------------------------------------------------------
# Test 3: normalise_hamza أ → ا
# ---------------------------------------------------------------------------
def test_normalise_hamza_above():
    """أحمد should become احمد."""
    result = normalise_hamza("أحمد")
    assert result == "احمد"


# ---------------------------------------------------------------------------
# Test 4: normalise_hamza إ → ا
# ---------------------------------------------------------------------------
def test_normalise_hamza_below():
    """إبراهيم should become ابراهيم."""
    result = normalise_hamza("إبراهيم")
    assert result == "ابراهيم"


# ---------------------------------------------------------------------------
# Test 5: two_pass_compare identical → all MATCH
# ---------------------------------------------------------------------------
def test_two_pass_compare_identical():
    """Identical input should produce all MATCH diffs."""
    text = "بسم الله الرحمن الرحيم"
    diffs = two_pass_compare(text, text, 1, 1)
    assert len(diffs) == 4
    assert all_match(diffs)
    assert all(d.error_type == "MATCH" for d in diffs)


# ---------------------------------------------------------------------------
# Test 6: two_pass_compare with one substituted word → SUBSTITUTION
# ---------------------------------------------------------------------------
def test_two_pass_compare_substitution():
    """One substituted word should produce exactly one SUBSTITUTION."""
    reference = "بسم الله الرحمن الرحيم"
    hypothesis = "بسم الله خطأ الرحيم"
    diffs = two_pass_compare(hypothesis, reference, 1, 1)
    sub_diffs = [d for d in diffs if d.error_type == "SUBSTITUTION"]
    assert len(sub_diffs) == 1
    assert sub_diffs[0].hypothesis_rasm == "خطأ" or sub_diffs[0].hypothesis_full == "خطأ"


# ---------------------------------------------------------------------------
# Test 7: HARAKAT_ERROR – rasm matches but full text differs
# ---------------------------------------------------------------------------
def test_two_pass_compare_harakat_error():
    """Correct rasm but wrong harakat → HARAKAT_ERROR."""
    reference = "بِسْمِ اللَّهِ"
    # Strip harakat from first word only → rasm matches but full texts differ
    from src.preprocessor import strip_harakat as sh
    word1_stripped = sh("بِسْمِ")   # = "بسم"
    hypothesis = f"{word1_stripped} اللَّهِ"
    diffs = two_pass_compare(hypothesis, reference, 1, 1)
    harakat_errors = [d for d in diffs if d.error_type == "HARAKAT_ERROR"]
    assert len(harakat_errors) >= 1


# ---------------------------------------------------------------------------
# Test 8: DELETION – missing word
# ---------------------------------------------------------------------------
def test_two_pass_compare_deletion():
    """Hypothesis missing one word → one DELETION."""
    reference = "بسم الله الرحمن الرحيم"
    hypothesis = "بسم الله الرحيم"   # "الرحمن" missing
    diffs = two_pass_compare(hypothesis, reference, 1, 1)
    deletions = [d for d in diffs if d.error_type == "DELETION"]
    assert len(deletions) == 1


# ---------------------------------------------------------------------------
# Test 9: tokenise_reference produces correct WordToken count
# ---------------------------------------------------------------------------
def test_tokenise_reference_count():
    """tokenise_reference should return one WordToken per word."""
    text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    tokens = tokenise_reference(text, 1, 1)
    assert len(tokens) == 4
    for i, tok in enumerate(tokens):
        assert isinstance(tok, WordToken)
        assert tok.surah == 1
        assert tok.ayah == 1
        assert tok.position == i
        # rasm should not contain harakat
        assert "\u064e" not in tok.rasm  # fatha
        assert "\u0650" not in tok.rasm  # kasra
