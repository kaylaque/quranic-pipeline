"""
error_classifier.py - Error classification engine for Quranic recitation errors.

Supports expanded tajweed error types including Qalqala, Iqlab, Izhar, and
Tafkheem violations from the Quranic-Phonemizer reference.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ErrorType(str, Enum):
    MATCH = "MATCH"
    SUBSTITUTION = "SUBSTITUTION"
    DELETION = "DELETION"
    INSERTION = "INSERTION"
    HARAKAT_ERROR = "HARAKAT_ERROR"
    TAJWEED_MEDD = "TAJWEED_MEDD"
    TAJWEED_GHUNNA = "TAJWEED_GHUNNA"
    TAJWEED_IDGHAM = "TAJWEED_IDGHAM"
    TAJWEED_IKHFA = "TAJWEED_IKHFA"
    TAJWEED_QALQALA = "TAJWEED_QALQALA"
    TAJWEED_IQLAB = "TAJWEED_IQLAB"
    TAJWEED_IZHAR = "TAJWEED_IZHAR"
    TAJWEED_TAFKHEEM = "TAJWEED_TAFKHEEM"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class Severity(str, Enum):
    CONFIRMED = "CONFIRMED"
    POSSIBLE = "POSSIBLE"
    SUPPRESSED = "SUPPRESSED"


# ---------------------------------------------------------------------------
# ErrorRecord Pydantic model
# ---------------------------------------------------------------------------

class ErrorRecord(BaseModel):
    position: str
    reference_token: str
    hypothesis_token: str
    error_type: ErrorType
    severity: Severity
    confidence_score: float  # always clamped to [0.0, 1.0]
    gop_score: Optional[float] = None  # clamped to [0.0, 1.0] when set
    alignment_confidence: str = "HIGH"
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Tajweed type mapping
# ---------------------------------------------------------------------------

_TAJWEED_MAP = {
    "TAJWEED_MEDD": ErrorType.TAJWEED_MEDD,
    "TAJWEED_GHUNNA": ErrorType.TAJWEED_GHUNNA,
    "TAJWEED_IDGHAM": ErrorType.TAJWEED_IDGHAM,
    "TAJWEED_IKHFA": ErrorType.TAJWEED_IKHFA,
    "TAJWEED_QALQALA": ErrorType.TAJWEED_QALQALA,
    "TAJWEED_IQLAB": ErrorType.TAJWEED_IQLAB,
    "TAJWEED_IZHAR": ErrorType.TAJWEED_IZHAR,
    "TAJWEED_TAFKHEEM": ErrorType.TAJWEED_TAFKHEEM,
}


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------

def classify_error(
    diff,                       # DiffResult from preprocessor
    phone_diff=None,            # PhoneDiff | None
    aligned_word=None,          # AlignedWord | None
    gop_min_for_possible: float = 0.7,
    gop_min_for_suppressed: float = 0.85,
) -> ErrorRecord:
    """
    Classify a single diff into an :class:`ErrorRecord`.

    Decision logic (in order):
    1. MATCH + no phone mismatch → MATCH, CONFIRMED, confidence=0.0
    2. FALLBACK alignment → downgrade non-MATCH errors to LOW_CONFIDENCE, POSSIBLE
    3. HARAKAT_ERROR → use GOP for severity
    4. TAJWEED_* from phone_diff → map to ErrorType, severity by GOP
    5. SUBSTITUTION/DELETION/INSERTION → map, severity by GOP
    6. GOP >= gop_min_for_suppressed + minor error → SUPPRESSED

    GOP thresholds:
        gop < gop_min_for_possible     (0.7)  → CONFIRMED  (poor pronunciation)
        gop in [0.7, 0.85)             → POSSIBLE   (moderate, may be ASR artefact)
        gop >= gop_min_for_suppressed  (0.85) → SUPPRESSED (likely correct, suppress error)

    confidence_score = 1.0 - gop_score if gop available, else 0.6
    (represents confidence that the flagged item IS an error; high GOP → low confidence in error)
    """
    # Extract basic fields from diff
    error_type_str: str = diff.error_type  # string from DiffResult
    position: str = diff.position
    ref_token: str = diff.reference_full if diff.reference_full else diff.reference_rasm
    hyp_token: str = diff.hypothesis_full if diff.hypothesis_full else diff.hypothesis_rasm

    # Determine GOP score and alignment confidence
    gop: Optional[float] = None
    align_conf: str = "HIGH"
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None

    if aligned_word is not None:
        gop = aligned_word.gop_score
        align_conf = aligned_word.alignment_confidence
        start_sec = aligned_word.start_sec
        end_sec = aligned_word.end_sec

    confidence_score = max(0.0, min(1.0, 1.0 - gop)) if gop is not None else 0.6

    notes = ""

    # ---- 1. MATCH ----
    if error_type_str == "MATCH":
        return ErrorRecord(
            position=position,
            reference_token=ref_token,
            hypothesis_token=hyp_token,
            error_type=ErrorType.MATCH,
            severity=Severity.CONFIRMED,
            confidence_score=0.0,
            gop_score=gop,
            alignment_confidence=align_conf,
            start_sec=start_sec,
            end_sec=end_sec,
            notes="No error detected.",
        )

    # ---- 2. FALLBACK alignment → downgrade to LOW_CONFIDENCE ----
    # Applied to ALL non-MATCH errors when alignment used fallback.
    if align_conf == "FALLBACK":
        return ErrorRecord(
            position=position,
            reference_token=ref_token,
            hypothesis_token=hyp_token,
            error_type=ErrorType.LOW_CONFIDENCE,
            severity=Severity.POSSIBLE,
            confidence_score=confidence_score,
            gop_score=gop,
            alignment_confidence=align_conf,
            start_sec=start_sec,
            end_sec=end_sec,
            notes=f"Underlying error: {error_type_str}. Alignment used fallback; low confidence.",
        )

    # ---- 3. HARAKAT_ERROR ----
    if error_type_str == "HARAKAT_ERROR":
        if gop is not None and gop > gop_min_for_suppressed:
            severity = Severity.SUPPRESSED
            notes = "GOP score high; harakat error likely normalisation mismatch."
        elif gop is not None and gop > gop_min_for_possible:
            severity = Severity.POSSIBLE
            notes = "Harakat mismatch with moderate GOP; possible error."
        else:
            severity = Severity.CONFIRMED
            notes = "Harakat (diacritic) error confirmed."
        return ErrorRecord(
            position=position,
            reference_token=ref_token,
            hypothesis_token=hyp_token,
            error_type=ErrorType.HARAKAT_ERROR,
            severity=severity,
            confidence_score=confidence_score,
            gop_score=gop,
            alignment_confidence=align_conf,
            start_sec=start_sec,
            end_sec=end_sec,
            notes=notes,
        )

    # ---- 4. TAJWEED errors from phone_diff ----
    if phone_diff is not None and phone_diff.diff_type.startswith("TAJWEED_"):
        mapped = _TAJWEED_MAP.get(phone_diff.diff_type, ErrorType.TAJWEED_MEDD)
        if gop is not None and gop > gop_min_for_suppressed:
            taj_severity = Severity.SUPPRESSED
            taj_notes = f"Tajweed rule {phone_diff.diff_type} fired but GOP high; likely correct."
        elif gop is not None and gop > gop_min_for_possible:
            taj_severity = Severity.POSSIBLE
            taj_notes = f"Tajweed rule violation: {phone_diff.diff_type} (moderate GOP)."
        else:
            taj_severity = Severity.CONFIRMED
            taj_notes = f"Tajweed rule violation: {phone_diff.diff_type}"
        return ErrorRecord(
            position=position,
            reference_token=ref_token,
            hypothesis_token=hyp_token,
            error_type=mapped,
            severity=taj_severity,
            confidence_score=confidence_score,
            gop_score=gop,
            alignment_confidence=align_conf,
            start_sec=start_sec,
            end_sec=end_sec,
            notes=taj_notes,
        )

    # ---- 5. SUBSTITUTION / DELETION / INSERTION ----
    error_type_map = {
        "SUBSTITUTION": ErrorType.SUBSTITUTION,
        "DELETION": ErrorType.DELETION,
        "INSERTION": ErrorType.INSERTION,
    }
    mapped_type = error_type_map.get(error_type_str, ErrorType.SUBSTITUTION)

    # ---- 6. Suppression: high GOP on word-level error ----
    # Applies to SUBSTITUTION, DELETION, and INSERTION — high GOP suggests
    # the reciter pronounced something acceptable and the error is an ASR artefact.
    if (
        error_type_str in ("SUBSTITUTION", "DELETION", "INSERTION")
        and gop is not None
        and gop > gop_min_for_suppressed
    ):
        severity = Severity.SUPPRESSED
        notes = "High GOP score suggests ASR artefact; error suppressed."
    else:
        severity = Severity.CONFIRMED
        notes = f"{error_type_str.capitalize()} error detected."

    return ErrorRecord(
        position=position,
        reference_token=ref_token,
        hypothesis_token=hyp_token,
        error_type=mapped_type,
        severity=severity,
        confidence_score=confidence_score,
        gop_score=gop,
        alignment_confidence=align_conf,
        start_sec=start_sec,
        end_sec=end_sec,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# classify_verse
# ---------------------------------------------------------------------------

def classify_verse(
    diffs,                  # List[DiffResult]
    phone_diffs=None,       # List[PhoneDiff] | None
    aligned_words=None,     # List[AlignedWord] | None
) -> List[ErrorRecord]:
    """
    Classify all diffs for an entire verse.

    *phone_diffs* is matched by diff index.
    *aligned_words* is matched by word_position string so that INSERTION diffs
    (which have no reference counterpart and use an INS-prefixed position key)
    do not accidentally consume a reference word's alignment data.
    """
    # Build a position-keyed lookup for aligned words so INSERTION diffs
    # (position "surah:ayah:INS<n>") correctly receive aligned_word=None.
    aligned_by_pos: dict = (
        {aw.word_position: aw for aw in aligned_words} if aligned_words else {}
    )

    records: List[ErrorRecord] = []

    for idx, diff in enumerate(diffs):
        phone_diff = phone_diffs[idx] if phone_diffs and idx < len(phone_diffs) else None
        aligned_word = aligned_by_pos.get(diff.position)

        record = classify_error(diff, phone_diff=phone_diff, aligned_word=aligned_word)
        records.append(record)

    return records
