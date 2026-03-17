"""
report.py - Report generation for Quranic recitation error detection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.error_classifier import ErrorRecord, ErrorType, Severity


# ---------------------------------------------------------------------------
# RecitationReport Pydantic model
# ---------------------------------------------------------------------------

class RecitationReport(BaseModel):
    audio_id: str
    surah: int
    ayah: int
    reference_text: str
    hypothesis_text: str
    errors: List[ErrorRecord]
    summary: dict
    asr_wer: Optional[float] = None
    pipeline_mode: str = "FULL"


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def generate_report(
    audio_id: str,
    surah: int,
    ayah: int,
    reference_text: str,
    hypothesis_text: str,
    errors: List[ErrorRecord],
    asr_wer: Optional[float] = None,
    pipeline_mode: str = "FULL",
) -> RecitationReport:
    """
    Build a :class:`RecitationReport` from classified errors.

    The *summary* dict contains:
    - total_errors   : count of non-MATCH errors
    - by_type        : { ErrorType.value : count }
    - by_severity    : { Severity.value  : count }
    """
    total_errors = sum(1 for e in errors if e.error_type != ErrorType.MATCH)

    by_type: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}

    for rec in errors:
        by_type[rec.error_type.value] = by_type.get(rec.error_type.value, 0) + 1
        by_severity[rec.severity.value] = by_severity.get(rec.severity.value, 0) + 1

    summary = {
        "total_errors": total_errors,
        "by_type": by_type,
        "by_severity": by_severity,
    }

    return RecitationReport(
        audio_id=audio_id,
        surah=surah,
        ayah=ayah,
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        errors=errors,
        summary=summary,
        asr_wer=asr_wer,
        pipeline_mode=pipeline_mode,
    )


# ---------------------------------------------------------------------------
# report_to_json
# ---------------------------------------------------------------------------

def report_to_json(report: RecitationReport, output_path: str) -> None:
    """Serialise *report* to JSON and write to *output_path*."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(report.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# report_to_text
# ---------------------------------------------------------------------------

def _border(title: str, width: int = 60) -> str:
    line = "=" * width
    inner = f"  {title}"
    return f"{line}\n{inner}\n{line}"


def report_to_text(report: RecitationReport) -> str:
    """
    Generate a human-readable bordered text report.

    Shows CONFIRMED and POSSIBLE errors; skips MATCH and SUPPRESSED entries.
    """
    lines: List[str] = []

    lines.append(_border(f"Recitation Report  —  {report.audio_id}"))
    lines.append(f"Surah : {report.surah}   Ayah : {report.ayah}")
    lines.append(f"Mode  : {report.pipeline_mode}")
    if report.asr_wer is not None:
        lines.append(f"ASR WER : {report.asr_wer:.4f}")
    lines.append("")

    lines.append(_border("Reference Text"))
    lines.append(report.reference_text)
    lines.append("")

    lines.append(_border("Hypothesis Text"))
    lines.append(report.hypothesis_text)
    lines.append("")

    lines.append(_border("Summary"))
    lines.append(f"  Total errors  : {report.summary.get('total_errors', 0)}")
    lines.append("  By type:")
    for etype, cnt in report.summary.get("by_type", {}).items():
        lines.append(f"    {etype:<25} {cnt}")
    lines.append("  By severity:")
    for sev, cnt in report.summary.get("by_severity", {}).items():
        lines.append(f"    {sev:<25} {cnt}")
    lines.append("")

    # Filter to CONFIRMED and POSSIBLE only
    visible = [
        e for e in report.errors
        if e.severity in (Severity.CONFIRMED, Severity.POSSIBLE)
        and e.error_type != ErrorType.MATCH
    ]

    lines.append(_border(f"Errors  ({len(visible)} shown)"))
    if not visible:
        lines.append("  No confirmed or possible errors.")
    else:
        for rec in visible:
            lines.append(f"  [{rec.severity.value}] {rec.error_type.value}  @ {rec.position}")
            lines.append(f"    Reference  : {rec.reference_token}")
            lines.append(f"    Hypothesis : {rec.hypothesis_token}")
            if rec.start_sec is not None and rec.end_sec is not None:
                lines.append(
                    f"    Time       : {rec.start_sec:.2f}s – {rec.end_sec:.2f}s"
                )
            if rec.gop_score is not None:
                lines.append(f"    GOP        : {rec.gop_score:.4f}")
            lines.append(f"    Notes      : {rec.notes}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
