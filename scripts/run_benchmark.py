"""
run_benchmark.py - Batch process a folder of WAV files and generate benchmark reports.

Supports three modes:

1. Auto-detect mode (no reference required):
   python scripts/run_benchmark.py --audio_dir data/samples/ --output_dir results/benchmark/

2. With CSV mapping (surah, ayah, or reference text):
   python scripts/run_benchmark.py \\
       --audio_dir data/samples/ \\
       --reference_csv data/reference/mapping.csv \\
       --output_dir results/benchmark/

3. With explicit surah/ayah (applies to all files):
   python scripts/run_benchmark.py \\
       --audio_dir data/samples/ \\
       --surah 1 --ayah 1 \\
       --output_dir results/benchmark/

CSV Format (3 columns, required if --reference_csv provided):
    filename,surah,ayah
    mock.wav,1,1
    sample.wav,2,5

Or with explicit reference text:
    filename,surah,ayah,reference_text
    mock.wav,1,1,بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
    sample.wav,2,5,الحمد لله رب العالمين
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single file benchmark result."""
    filename: str
    surah: int
    ayah: int
    mode: str  # "auto-detect", "explicit", or "provided"
    reference_text: Optional[str]
    hypothesis_text: Optional[str]
    pipeline_mode: str  # FULL, NO_GOP, TEXT_ONLY, etc.
    asr_wer: Optional[float]
    total_errors: int
    confirmed_errors: int
    possible_errors: int
    suppressed_errors: int
    processing_time_sec: float
    success: bool
    error_message: Optional[str] = None


class BenchmarkRunner:
    """Batch runner for pipeline across multiple audio files."""

    def __init__(
        self,
        audio_dir: Path,
        output_dir: Path,
        reference_csv: Optional[Path] = None,
        surah: int = 0,
        ayah: int = 0,
        model: str = "tarteel-ai/whisper-base-ar-quran",
        device: str = "auto",
        backend: str = "faster-whisper",
        compute_type: str = "auto",
        model_dir: str = "models/whisper-quran-ct2",
        verbose: bool = False,
    ):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.reference_csv = Path(reference_csv) if reference_csv else None
        self.surah = surah
        self.ayah = ayah
        self.model = model
        self.device = device
        self.backend = backend
        self.compute_type = compute_type
        self.model_dir = model_dir
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

        # Load reference mapping if provided
        self.reference_map: Dict[str, Dict[str, str]] = {}
        if self.reference_csv and self.reference_csv.exists():
            self._load_reference_csv()

    def _load_reference_csv(self):
        """Load reference_csv into reference_map."""
        logger.info("Loading reference CSV: %s", self.reference_csv)
        try:
            with open(self.reference_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("filename", "").strip()
                    if filename:
                        self.reference_map[filename] = row
        except Exception as exc:
            logger.error("Failed to load reference CSV: %s", exc)
            raise

    def _get_audio_files(self) -> List[Path]:
        """Return sorted list of WAV files in audio_dir."""
        if not self.audio_dir.exists():
            raise ValueError(f"Audio directory not found: {self.audio_dir}")

        wav_files = sorted(self.audio_dir.glob("*.wav"))
        if not wav_files:
            raise ValueError(f"No .wav files found in {self.audio_dir}")

        logger.info("Found %d WAV file(s) in %s", len(wav_files), self.audio_dir)
        return wav_files

    def _run_pipeline(self, audio_file: Path) -> tuple[bool, BenchmarkResult]:
        """
        Run pipeline on a single audio file.

        Returns (success, BenchmarkResult)
        """
        filename = audio_file.name
        start_time = time.time()

        # Determine mode and arguments
        mode = "auto-detect"
        args = [
            "python", "scripts/run_pipeline.py",
            "--audio", str(audio_file),
            "--model", self.model,
            "--device", self.device,
            "--backend", self.backend,
            "--compute_type", self.compute_type,
            "--model_dir", self.model_dir,
            "--output_dir", str(self.output_dir),
        ]

        # Check if reference CSV provides explicit mapping
        ref_info = self.reference_map.get(filename, {})
        if ref_info:
            mode = "provided"
            if "surah" in ref_info and ref_info["surah"].strip():
                args.extend(["--surah", ref_info["surah"]])
            if "ayah" in ref_info and ref_info["ayah"].strip():
                args.extend(["--ayah", ref_info["ayah"]])
            if "reference_text" in ref_info and ref_info["reference_text"].strip():
                args.extend(["--reference", ref_info["reference_text"]])
        elif self.surah > 0 and self.ayah > 0:
            # Explicit surah/ayah provided for all files
            mode = "explicit"
            args.extend(["--surah", str(self.surah), "--ayah", str(self.ayah)])

        if self.verbose:
            args.append("--verbose")

        # Run pipeline as subprocess
        try:
            logger.info("[%s] Running pipeline... (mode=%s)", filename, mode)
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute timeout
            )

            if result.returncode != 0:
                error_msg = f"Pipeline failed: {result.stderr[:200]}"
                logger.error("[%s] %s", filename, error_msg)
                return False, BenchmarkResult(
                    filename=filename,
                    surah=int(ref_info.get("surah", self.surah)) if ref_info else self.surah,
                    ayah=int(ref_info.get("ayah", self.ayah)) if ref_info else self.ayah,
                    mode=mode,
                    reference_text=ref_info.get("reference_text"),
                    hypothesis_text=None,
                    pipeline_mode="ERROR",
                    asr_wer=None,
                    total_errors=0,
                    confirmed_errors=0,
                    possible_errors=0,
                    suppressed_errors=0,
                    processing_time_sec=time.time() - start_time,
                    success=False,
                    error_message=error_msg,
                )

            # Load and parse JSON report
            report_path = self.output_dir / f"{audio_file.stem}.json"
            if not report_path.exists():
                error_msg = f"Report not generated at {report_path}"
                logger.error("[%s] %s", filename, error_msg)
                return False, BenchmarkResult(
                    filename=filename,
                    surah=int(ref_info.get("surah", self.surah)) if ref_info else self.surah,
                    ayah=int(ref_info.get("ayah", self.ayah)) if ref_info else self.ayah,
                    mode=mode,
                    reference_text=ref_info.get("reference_text"),
                    hypothesis_text=None,
                    pipeline_mode="ERROR",
                    asr_wer=None,
                    total_errors=0,
                    confirmed_errors=0,
                    possible_errors=0,
                    suppressed_errors=0,
                    processing_time_sec=time.time() - start_time,
                    success=False,
                    error_message=error_msg,
                )

            report_data = json.loads(report_path.read_text(encoding="utf-8"))

            # Extract metrics
            summary = report_data.get("summary", {})
            by_severity = summary.get("by_severity", {})
            errors = report_data.get("errors", [])
            total_errors = sum(1 for e in errors if e.get("error_type") != "MATCH")

            processing_time = time.time() - start_time
            logger.info(
                "[%s] ✓ Success | pipeline_mode=%s | errors=%d | time=%.2fs",
                filename,
                report_data.get("pipeline_mode", "UNKNOWN"),
                total_errors,
                processing_time,
            )

            return True, BenchmarkResult(
                filename=filename,
                surah=report_data.get("surah", 0),
                ayah=report_data.get("ayah", 0),
                mode=mode,
                reference_text=report_data.get("reference_text"),
                hypothesis_text=report_data.get("hypothesis_text"),
                pipeline_mode=report_data.get("pipeline_mode", "UNKNOWN"),
                asr_wer=report_data.get("asr_wer"),
                total_errors=total_errors,
                confirmed_errors=by_severity.get("CONFIRMED", 0),
                possible_errors=by_severity.get("POSSIBLE", 0),
                suppressed_errors=by_severity.get("SUPPRESSED", 0),
                processing_time_sec=processing_time,
                success=True,
            )

        except subprocess.TimeoutExpired:
            error_msg = "Pipeline timed out (>300s)"
            logger.error("[%s] %s", filename, error_msg)
            return False, BenchmarkResult(
                filename=filename,
                surah=int(ref_info.get("surah", self.surah)) if ref_info else self.surah,
                ayah=int(ref_info.get("ayah", self.ayah)) if ref_info else self.ayah,
                mode=mode,
                reference_text=ref_info.get("reference_text"),
                hypothesis_text=None,
                pipeline_mode="TIMEOUT",
                asr_wer=None,
                total_errors=0,
                confirmed_errors=0,
                possible_errors=0,
                suppressed_errors=0,
                processing_time_sec=time.time() - start_time,
                success=False,
                error_message=error_msg,
            )
        except Exception as exc:
            error_msg = str(exc)[:200]
            logger.error("[%s] Unexpected error: %s", filename, error_msg)
            return False, BenchmarkResult(
                filename=filename,
                surah=int(ref_info.get("surah", self.surah)) if ref_info else self.surah,
                ayah=int(ref_info.get("ayah", self.ayah)) if ref_info else self.ayah,
                mode=mode,
                reference_text=ref_info.get("reference_text"),
                hypothesis_text=None,
                pipeline_mode="ERROR",
                asr_wer=None,
                total_errors=0,
                confirmed_errors=0,
                possible_errors=0,
                suppressed_errors=0,
                processing_time_sec=time.time() - start_time,
                success=False,
                error_message=error_msg,
            )

    def run(self) -> None:
        """Run benchmark on all audio files."""
        audio_files = self._get_audio_files()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("BENCHMARK START | %d file(s) | device=%s", len(audio_files), self.device)
        logger.info("=" * 70)

        for i, audio_file in enumerate(audio_files, 1):
            logger.info("\n[%d/%d] Processing %s", i, len(audio_files), audio_file.name)
            success, result = self._run_pipeline(audio_file)
            self.results.append(result)

        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK COMPLETE | %d/%d successful", sum(1 for r in self.results if r.success), len(self.results))
        logger.info("=" * 70)

        self._generate_reports()

    def _generate_reports(self) -> None:
        """Generate CSV and markdown reports."""
        self._write_csv_report()
        self._write_markdown_report()

    def _write_csv_report(self) -> None:
        """Write results to CSV."""
        csv_path = self.output_dir / "benchmark_results.csv"

        # Fields to include (excluding error_message for CSV brevity)
        fieldnames = [
            "filename",
            "surah",
            "ayah",
            "mode",
            "pipeline_mode",
            "success",
            "asr_wer",
            "total_errors",
            "confirmed_errors",
            "possible_errors",
            "suppressed_errors",
            "processing_time_sec",
        ]

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    row = {k: getattr(result, k) for k in fieldnames}
                    writer.writerow(row)
            logger.info("CSV report written: %s", csv_path)
        except Exception as exc:
            logger.error("Failed to write CSV report: %s", exc)

    def _write_markdown_report(self) -> None:
        """Write summary and per-file results to markdown."""
        md_path = self.output_dir / "benchmark_report.md"

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        # Compute aggregate metrics
        avg_wer = (
            sum(r.asr_wer for r in successful if r.asr_wer is not None)
            / len([r for r in successful if r.asr_wer is not None])
            if any(r.asr_wer is not None for r in successful)
            else None
        )
        avg_time = (
            sum(r.processing_time_sec for r in successful) / len(successful)
            if successful
            else 0.0
        )
        total_errors = sum(r.total_errors for r in successful)
        total_confirmed = sum(r.confirmed_errors for r in successful)
        total_possible = sum(r.possible_errors for r in successful)

        lines = [
            "# Quranic Pipeline Benchmark Report",
            "",
            "## Summary",
            "",
            f"- **Total files**: {len(self.results)}",
            f"- **Successful**: {len(successful)} ({100*len(successful)/len(self.results):.1f}%)",
            f"- **Failed**: {len(failed)}",
            f"- **Average ASR WER**: {avg_wer:.4f}" if avg_wer is not None else "- **Average ASR WER**: N/A",
            f"- **Average processing time**: {avg_time:.2f}s",
            f"- **Total errors detected**: {total_errors}",
            f"  - CONFIRMED: {total_confirmed}",
            f"  - POSSIBLE: {total_possible}",
            "",
        ]

        if failed:
            lines.extend([
                "## Failed Files",
                "",
            ])
            for result in failed:
                lines.append(f"- **{result.filename}**: {result.error_message}")
            lines.append("")

        lines.extend([
            "## Per-File Results",
            "",
            "| Filename | Surah | Ayah | Mode | Pipeline | WER | Errors | Time (s) | Status |",
            "|----------|-------|------|------|----------|-----|--------|----------|--------|",
        ])

        for result in self.results:
            status = "✓" if result.success else "✗"
            wer_str = f"{result.asr_wer:.4f}" if result.asr_wer is not None else "—"
            mode_short = result.mode[:6]  # "auto-de", "explic", "provid"

            lines.append(
                f"| {result.filename} | {result.surah} | {result.ayah} | {mode_short} | "
                f"{result.pipeline_mode} | {wer_str} | {result.total_errors} | "
                f"{result.processing_time_sec:.2f} | {status} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "*Generated by run_benchmark.py*",
        ])

        md_content = "\n".join(lines)
        try:
            md_path.write_text(md_content, encoding="utf-8")
            logger.info("Markdown report written: %s", md_path)
        except Exception as exc:
            logger.error("Failed to write markdown report: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch process a folder of WAV files through the Quranic recitation pipeline."
    )
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Directory containing WAV files to process",
    )
    parser.add_argument(
        "--output_dir",
        default="results/benchmark/",
        help="Output directory for reports and pipeline results (default: results/benchmark/)",
    )
    parser.add_argument(
        "--reference_csv",
        default=None,
        help="Optional CSV mapping filenames to surah/ayah/reference_text",
    )
    parser.add_argument(
        "--surah",
        type=int,
        default=0,
        help="Surah number (if same for all files; 0 = auto-detect)",
    )
    parser.add_argument(
        "--ayah",
        type=int,
        default=0,
        help="Ayah number (if same for all files; 0 = auto-detect)",
    )
    parser.add_argument(
        "--model",
        default="tarteel-ai/whisper-base-ar-quran",
        help="HuggingFace Whisper model ID",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--backend",
        default="faster-whisper",
        choices=["faster-whisper", "huggingface"],
        help="ASR backend: faster-whisper (CTranslate2, int8) or huggingface (PyTorch)",
    )
    parser.add_argument(
        "--compute_type",
        default="auto",
        choices=["auto", "int8", "float16", "float32"],
        help="CTranslate2 compute type for faster-whisper backend (default: auto)",
    )
    parser.add_argument(
        "--model_dir",
        default="models/whisper-quran-ct2",
        help="Path to CTranslate2 model directory for faster-whisper backend",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(
        audio_dir=Path(args.audio_dir),
        output_dir=Path(args.output_dir),
        reference_csv=Path(args.reference_csv) if args.reference_csv else None,
        surah=args.surah,
        ayah=args.ayah,
        model=args.model,
        device=args.device,
        backend=args.backend,
        compute_type=args.compute_type,
        model_dir=args.model_dir,
        verbose=args.verbose,
    )

    try:
        runner.run()
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
