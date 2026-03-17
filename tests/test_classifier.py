"""
test_classifier.py - Tests for src/error_classifier.py and src/report.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from src.preprocessor import DiffResult
from src.error_classifier import (
    ErrorType,
    Severity,
    ErrorRecord,
    classify_error,
    classify_verse,
)
from src.phonemizer import PhoneDiff
from src.report import generate_report, report_to_json, report_to_text


# ---------------------------------------------------------------------------
# Helper: build a DiffResult
# ---------------------------------------------------------------------------
def _make_diff(
    error_type: str,
    position: str = "1:1:0",
    ref_rasm: str = "بسم",
    hyp_rasm: str = "بسم",
    ref_full: str = "بِسْمِ",
    hyp_full: str = "بِسْمِ",
) -> DiffResult:
    rasm_match = ref_rasm == hyp_rasm
    harakat_match = ref_full == hyp_full and rasm_match
    return DiffResult(
        position=position,
        reference_rasm=ref_rasm,
        hypothesis_rasm=hyp_rasm,
        reference_full=ref_full,
        hypothesis_full=hyp_full,
        rasm_match=rasm_match,
        harakat_match=harakat_match,
        error_type=error_type,
    )


# ---------------------------------------------------------------------------
# Helper: build an AlignedWord
# ---------------------------------------------------------------------------
def _make_aligned(gop: float = 0.5, confidence: str = "HIGH"):
    from src.aligner import AlignedWord
    return AlignedWord(
        word_position="1:1:0",
        text="بسم",
        start_sec=0.0,
        end_sec=0.5,
        duration_sec=0.5,
        gop_score=gop,
        alignment_confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Test 1: classify_error with MATCH → ErrorType.MATCH
# ---------------------------------------------------------------------------
def test_classify_error_match():
    diff = _make_diff("MATCH")
    record = classify_error(diff)
    assert record.error_type == ErrorType.MATCH
    assert record.severity == Severity.CONFIRMED


# ---------------------------------------------------------------------------
# Test 2: SUBSTITUTION + GOP=0.3 → CONFIRMED
# ---------------------------------------------------------------------------
def test_classify_error_substitution_confirmed():
    diff = _make_diff("SUBSTITUTION", hyp_rasm="خطأ", hyp_full="خطأ")
    aligned = _make_aligned(gop=0.3)
    record = classify_error(diff, aligned_word=aligned)
    assert record.error_type == ErrorType.SUBSTITUTION
    assert record.severity == Severity.CONFIRMED


# ---------------------------------------------------------------------------
# Test 3: HARAKAT_ERROR + GOP=0.9 → SUPPRESSED
# ---------------------------------------------------------------------------
def test_classify_error_harakat_suppressed():
    """GOP=0.9 > 0.85 threshold → SUPPRESSED for HARAKAT_ERROR."""
    diff = _make_diff("HARAKAT_ERROR", hyp_full="بسم")  # missing harakat
    aligned = _make_aligned(gop=0.9)
    record = classify_error(diff, aligned_word=aligned)
    assert record.error_type == ErrorType.HARAKAT_ERROR
    assert record.severity == Severity.SUPPRESSED


# ---------------------------------------------------------------------------
# Test 4: FALLBACK alignment → LOW_CONFIDENCE
# ---------------------------------------------------------------------------
def test_classify_error_fallback_alignment():
    """AlignedWord with alignment_confidence=FALLBACK → LOW_CONFIDENCE error type."""
    diff = _make_diff("SUBSTITUTION", hyp_rasm="خطأ", hyp_full="خطأ")
    aligned = _make_aligned(gop=0.5, confidence="FALLBACK")
    record = classify_error(diff, aligned_word=aligned)
    assert record.error_type == ErrorType.LOW_CONFIDENCE
    assert record.severity == Severity.POSSIBLE


# ---------------------------------------------------------------------------
# Test 5: classify_verse with 3 diffs → 3 ErrorRecords
# ---------------------------------------------------------------------------
def test_classify_verse_count():
    diffs = [
        _make_diff("MATCH", position="1:1:0"),
        _make_diff("SUBSTITUTION", position="1:1:1", hyp_rasm="خطأ", hyp_full="خطأ"),
        _make_diff("DELETION", position="1:1:2", hyp_rasm="", hyp_full=""),
    ]
    records = classify_verse(diffs)
    assert len(records) == 3
    assert records[0].error_type == ErrorType.MATCH
    assert records[1].error_type == ErrorType.SUBSTITUTION
    assert records[2].error_type == ErrorType.DELETION


# ---------------------------------------------------------------------------
# Test 6: generate_report summary counts match error list
# ---------------------------------------------------------------------------
def test_generate_report_summary_counts():
    diffs = [
        _make_diff("MATCH", position="1:1:0"),
        _make_diff("SUBSTITUTION", position="1:1:1", hyp_rasm="خطأ", hyp_full="خطأ"),
        _make_diff("DELETION", position="1:1:2", hyp_rasm="", hyp_full=""),
    ]
    errors = classify_verse(diffs)
    report = generate_report(
        audio_id="test_audio",
        surah=1,
        ayah=1,
        reference_text="بسم الله الرحمن",
        hypothesis_text="بسم خطأ",
        errors=errors,
    )
    assert report.summary["total_errors"] == 2
    by_type = report.summary["by_type"]
    assert by_type.get("SUBSTITUTION", 0) == 1
    assert by_type.get("DELETION", 0) == 1


# ---------------------------------------------------------------------------
# Test 7: report_to_json writes valid JSON
# ---------------------------------------------------------------------------
def test_report_to_json_valid():
    errors = classify_verse([_make_diff("MATCH", position="1:1:0")])
    report = generate_report(
        audio_id="json_test",
        surah=1,
        ayah=1,
        reference_text="بسم",
        hypothesis_text="بسم",
        errors=errors,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    report_to_json(report, path)
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["audio_id"] == "json_test"
    assert "errors" in loaded
    assert "summary" in loaded


# ---------------------------------------------------------------------------
# Test 8: report_to_text includes audio_id
# ---------------------------------------------------------------------------
def test_report_to_text_contains_audio_id():
    errors = classify_verse([_make_diff("MATCH", position="1:1:0")])
    report = generate_report(
        audio_id="unique_audio_id_xyz",
        surah=1,
        ayah=1,
        reference_text="بسم",
        hypothesis_text="بسم",
        errors=errors,
    )
    text = report_to_text(report)
    assert "unique_audio_id_xyz" in text


# ---------------------------------------------------------------------------
# Helper: build a PhoneDiff with a specific TAJWEED diff_type
# ---------------------------------------------------------------------------
def _make_phone_diff(diff_type: str) -> PhoneDiff:
    return PhoneDiff(
        position="1:1:0",
        reference_phones=["/q/", "/a/"],
        hypothesis_phones=["/q/"],
        edit_distance=1,
        diff_type=diff_type,
    )


# ---------------------------------------------------------------------------
# Test 9: TAJWEED_QALQALA phone_diff maps to ErrorType.TAJWEED_QALQALA
# ---------------------------------------------------------------------------
def test_classify_error_tajweed_qalqala():
    """
    A phone_diff with diff_type=TAJWEED_QALQALA must produce
    ErrorType.TAJWEED_QALQALA in the resulting ErrorRecord.
    """
    diff = _make_diff("SUBSTITUTION", hyp_rasm="قال", hyp_full="قال")
    phone_diff = _make_phone_diff("TAJWEED_QALQALA")
    record = classify_error(diff, phone_diff=phone_diff)
    assert record.error_type == ErrorType.TAJWEED_QALQALA


# ---------------------------------------------------------------------------
# Test 10: TAJWEED_IQLAB phone_diff maps to ErrorType.TAJWEED_IQLAB
# ---------------------------------------------------------------------------
def test_classify_error_tajweed_iqlab():
    """
    A phone_diff with diff_type=TAJWEED_IQLAB must produce
    ErrorType.TAJWEED_IQLAB in the resulting ErrorRecord.
    """
    diff = _make_diff("SUBSTITUTION", hyp_rasm="من", hyp_full="من")
    phone_diff = _make_phone_diff("TAJWEED_IQLAB")
    record = classify_error(diff, phone_diff=phone_diff)
    assert record.error_type == ErrorType.TAJWEED_IQLAB


# ---------------------------------------------------------------------------
# Test 11: TAJWEED_IZHAR phone_diff maps to ErrorType.TAJWEED_IZHAR
# ---------------------------------------------------------------------------
def test_classify_error_tajweed_izhar():
    """
    A phone_diff with diff_type=TAJWEED_IZHAR must produce
    ErrorType.TAJWEED_IZHAR in the resulting ErrorRecord.
    """
    diff = _make_diff("SUBSTITUTION", hyp_rasm="عليم", hyp_full="عليم")
    phone_diff = _make_phone_diff("TAJWEED_IZHAR")
    record = classify_error(diff, phone_diff=phone_diff)
    assert record.error_type == ErrorType.TAJWEED_IZHAR


# ---------------------------------------------------------------------------
# Test 12: TAJWEED_TAFKHEEM phone_diff maps to ErrorType.TAJWEED_TAFKHEEM
# ---------------------------------------------------------------------------
def test_classify_error_tajweed_tafkheem():
    """
    A phone_diff with diff_type=TAJWEED_TAFKHEEM must produce
    ErrorType.TAJWEED_TAFKHEEM in the resulting ErrorRecord.
    """
    diff = _make_diff("SUBSTITUTION", hyp_rasm="الله", hyp_full="الله")
    phone_diff = _make_phone_diff("TAJWEED_TAFKHEEM")
    record = classify_error(diff, phone_diff=phone_diff)
    assert record.error_type == ErrorType.TAJWEED_TAFKHEEM


# ---------------------------------------------------------------------------
# Test 13: classify_error with DELETION error_type returns ErrorType.DELETION
# ---------------------------------------------------------------------------
def test_classify_error_deletion_returns_deletion_type():
    """
    A diff with error_type='DELETION' and no phone_diff must produce
    ErrorType.DELETION with CONFIRMED severity.
    """
    diff = _make_diff("DELETION", hyp_rasm="", hyp_full="")
    record = classify_error(diff)
    assert record.error_type == ErrorType.DELETION
    assert record.severity == Severity.CONFIRMED
