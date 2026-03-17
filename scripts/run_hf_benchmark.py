"""
run_hf_benchmark.py - Benchmark the Quranic recitation pipeline against the
Buraaq/quran-md-ayahs HuggingFace dataset (parquet files).

Usage examples:

  # Quick test — 10 rows, one reciter, already-downloaded parquet
  python3 scripts/run_hf_benchmark.py \\
      --parquet data/benchmark_hf/0044.parquet \\
      --reciters alafasy \\
      --max_rows 10

  # All rows in one parquet, ASR-only mode
  python3 scripts/run_hf_benchmark.py \\
      --parquet data/benchmark_hf/0044.parquet \\
      --mode asr_only \\
      --device auto

  # Filter to specific reciters
  python3 scripts/run_hf_benchmark.py \\
      --parquet data/benchmark_hf/0044.parquet \\
      --reciters alafasy,husary,minshawy_murattal \\
      --max_rows 30
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level imports to avoid re-importing on every row
from src.asr import (
    transcribe_whisper, transcribe_faster_whisper,
    compute_wer, load_whisper_model, load_faster_whisper_model,
)
from src.preprocessor import strip_harakat
from scripts.hf_parquet_utils import mp3_bytes_to_array, iter_parquet_rows


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HFBenchmarkResult:
    parquet_index: str
    surah_id: int
    ayah_id: int
    surah_name_en: str
    reciter_id: str
    reciter_name: str
    reference_text: str
    hypothesis_text: Optional[str]
    wer: Optional[float]
    cer: Optional[float]
    insertions: int
    deletions: int
    substitutions: int
    processing_time_sec: float
    success: bool
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class HFBenchmarkRunner:

    def __init__(
        self,
        parquet_path: Path,
        output_dir: Path,
        mode: str = "asr_only",
        model: str = "tarteel-ai/whisper-base-ar-quran",
        device: str = "auto",
        backend: str = "faster-whisper",
        compute_type: str = "auto",
        model_dir: str = "models/whisper-quran-ct2",
        reciters: list[str] | None = None,
        surahs: list[int] | None = None,
        max_rows: int = 0,
    ):
        self.parquet_path = Path(parquet_path)
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.model_name = model
        self.device_pref = device
        self.backend = backend
        self.compute_type = compute_type
        self.model_dir = model_dir
        self.reciters = reciters
        self.surahs = surahs
        self.max_rows = max_rows
        self.results: list[HFBenchmarkResult] = []

        self.parquet_index = self.parquet_path.stem  # e.g. "0044"

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process_row(
        self,
        row: dict,
        model,
        processor,
        device: str,
    ) -> HFBenchmarkResult:
        start = time.time()
        surah_id = row["surah_id"]
        ayah_id = row["ayah_id"]
        reciter_id = row["reciter_id"]
        reference_raw = row["ayah_ar"]

        try:
            if not row["audio_bytes"]:
                raise ValueError("Empty audio bytes")

            # Decode MP3 → float32 array in-memory (no ffmpeg, no disk write)
            audio_array, sr = mp3_bytes_to_array(row["audio_bytes"], target_sr=16000)

            # Transcribe — branch on backend
            if self.backend == "faster-whisper":
                result = transcribe_faster_whisper(model, audio_array, sr)
            else:
                result = transcribe_whisper(model, processor, audio_array, sr, device)
            hypothesis = result["text"].strip()

            # Normalise both sides before WER
            def normalise(t: str) -> str:
                return strip_harakat(t).strip()

            metrics = compute_wer(hypothesis, reference_raw, normalise_fn=normalise)

            elapsed = time.time() - start
            logger.info(
                "[%s | S%03d:A%03d | %s] WER=%.3f CER=%.3f | %.2fs",
                self.parquet_index, surah_id, ayah_id, reciter_id,
                metrics["wer"], metrics["cer"], elapsed,
            )

            return HFBenchmarkResult(
                parquet_index=self.parquet_index,
                surah_id=surah_id,
                ayah_id=ayah_id,
                surah_name_en=row["surah_name_en"],
                reciter_id=reciter_id,
                reciter_name=row["reciter_name"],
                reference_text=reference_raw,
                hypothesis_text=hypothesis,
                wer=metrics["wer"],
                cer=metrics["cer"],
                insertions=metrics["insertions"],
                deletions=metrics["deletions"],
                substitutions=metrics["substitutions"],
                processing_time_sec=elapsed,
                success=True,
            )

        except Exception as exc:
            elapsed = time.time() - start
            logger.error(
                "[%s | S%03d:A%03d | %s] FAILED: %s",
                self.parquet_index, surah_id, ayah_id, reciter_id, exc,
            )
            return HFBenchmarkResult(
                parquet_index=self.parquet_index,
                surah_id=surah_id,
                ayah_id=ayah_id,
                surah_name_en=row["surah_name_en"],
                reciter_id=reciter_id,
                reciter_name=row["reciter_name"],
                reference_text=reference_raw,
                hypothesis_text=None,
                wer=None,
                cer=None,
                insertions=0,
                deletions=0,
                substitutions=0,
                processing_time_sec=elapsed,
                success=False,
                error_message=str(exc)[:300],
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info(
            "HF BENCHMARK START | parquet=%s | mode=%s | backend=%s | device=%s",
            self.parquet_path.name, self.mode, self.backend, self.device_pref,
        )
        if self.reciters:
            logger.info("Filtering reciters: %s", self.reciters)
        if self.max_rows:
            logger.info("Max rows: %d", self.max_rows)
        logger.info("=" * 70)

        # Load model once — branch on backend
        if self.backend == "faster-whisper":
            model = load_faster_whisper_model(
                self.model_dir, self.device_pref, self.compute_type
            )
            processor = None
            device = None
        else:
            model, processor, device = load_whisper_model(
                self.model_name, self.device_pref
            )

        total = 0
        for row in iter_parquet_rows(
            self.parquet_path,
            reciters=self.reciters,
            surahs=self.surahs,
            max_rows=self.max_rows,
        ):
            total += 1
            result = self._process_row(row, model, processor, device)
            self.results.append(result)

        logger.info("=" * 70)
        success_count = sum(1 for r in self.results if r.success)
        logger.info(
            "BENCHMARK COMPLETE | %d/%d successful | parquet=%s",
            success_count, len(self.results), self.parquet_path.name,
        )
        logger.info("=" * 70)

        self._generate_reports()

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def _aggregate_per_reciter(self) -> dict[str, dict]:
        """Compute per-reciter aggregation (shared by markdown and JSON writers)."""
        reciter_agg: dict[str, dict] = {}
        for r in self.results:
            if not r.success or r.wer is None:
                continue
            s = reciter_agg.setdefault(r.reciter_id, {
                "name": r.reciter_name,
                "wers": [],
                "cers": [],
            })
            s["wers"].append(r.wer)
            s["cers"].append(r.cer or 0.0)
        return reciter_agg

    def _generate_reports(self) -> None:
        self._write_csv()
        self._write_markdown()
        self._write_json_summary()

    def _write_csv(self) -> None:
        path = self.output_dir / f"hf_benchmark_{self.parquet_index}_results.csv"
        fields = [
            "parquet_index", "surah_id", "ayah_id", "surah_name_en",
            "reciter_id", "reciter_name",
            "wer", "cer", "insertions", "deletions", "substitutions",
            "reference_text", "hypothesis_text",
            "processing_time_sec", "success", "error_message",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.results:
                row = asdict(r)
                writer.writerow({k: row.get(k) for k in fields})
        logger.info("CSV written: %s", path)

    def _write_markdown(self) -> None:
        path = self.output_dir / f"hf_benchmark_{self.parquet_index}_report.md"

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        wer_values = [r.wer for r in successful if r.wer is not None]
        cer_values = [r.cer for r in successful if r.cer is not None]
        mean_wer = sum(wer_values) / len(wer_values) if wer_values else None
        mean_cer = sum(cer_values) / len(cer_values) if cer_values else None
        avg_time = sum(r.processing_time_sec for r in successful) / len(successful) if successful else 0.0

        # Per-reciter aggregation (shared with JSON writer)
        reciter_stats = self._aggregate_per_reciter()

        lines = [
            f"# HF Benchmark Report — Parquet {self.parquet_index}",
            "",
            "## Summary",
            "",
            f"- **Parquet**: `{self.parquet_path.name}`",
            f"- **Total rows processed**: {len(self.results)}",
            f"- **Successful**: {len(successful)} ({100*len(successful)/max(len(self.results),1):.1f}%)",
            f"- **Failed**: {len(failed)}",
            f"- **Mean WER**: {mean_wer:.4f}" if mean_wer is not None else "- **Mean WER**: N/A",
            f"- **Mean CER**: {mean_cer:.4f}" if mean_cer is not None else "- **Mean CER**: N/A",
            f"- **Avg processing time**: {avg_time:.2f}s/row",
            "",
        ]

        if reciter_stats:
            lines += [
                "## Per-Reciter Results",
                "",
                "| Reciter ID | Name | Samples | Mean WER | Mean CER |",
                "|------------|------|---------|----------|----------|",
            ]
            for rid, s in sorted(reciter_stats.items(), key=lambda x: sum(x[1]["wers"])/len(x[1]["wers"])):
                mw = sum(s["wers"]) / len(s["wers"])
                mc = sum(s["cers"]) / len(s["cers"])
                lines.append(f"| `{rid}` | {s['name']} | {len(s['wers'])} | {mw:.4f} | {mc:.4f} |")
            lines.append("")

        if failed:
            lines += ["## Failed Rows", ""]
            for r in failed[:20]:
                lines.append(f"- S{r.surah_id:03d}:A{r.ayah_id:03d} `{r.reciter_id}` — {r.error_message}")
            if len(failed) > 20:
                lines.append(f"- *(and {len(failed)-20} more...)*")
            lines.append("")

        # Per-row table (first 50 rows)
        lines += [
            "## Per-Row Results (first 50)",
            "",
            "| Surah | Ayah | Reciter | WER | CER | Ins | Del | Sub | Time(s) | OK |",
            "|-------|------|---------|-----|-----|-----|-----|-----|---------|----|",
        ]
        for r in self.results[:50]:
            wer_s = f"{r.wer:.4f}" if r.wer is not None else "—"
            cer_s = f"{r.cer:.4f}" if r.cer is not None else "—"
            ok = "✓" if r.success else "✗"
            lines.append(
                f"| {r.surah_id} | {r.ayah_id} | `{r.reciter_id}` | {wer_s} | {cer_s} "
                f"| {r.insertions} | {r.deletions} | {r.substitutions} "
                f"| {r.processing_time_sec:.2f} | {ok} |"
            )
        if len(self.results) > 50:
            lines.append(f"\n*... {len(self.results)-50} more rows in CSV*")

        lines += ["", "---", "", "*Generated by run_hf_benchmark.py*"]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown report written: %s", path)

    def _write_json_summary(self) -> None:
        path = self.output_dir / f"hf_benchmark_{self.parquet_index}_summary.json"

        successful = [r for r in self.results if r.success]
        wers = [r.wer for r in successful if r.wer is not None]
        cers = [r.cer for r in successful if r.cer is not None]

        reciter_agg = self._aggregate_per_reciter()

        summary = {
            "parquet": self.parquet_path.name,
            "total_rows": len(self.results),
            "successful": len(successful),
            "failed": len(self.results) - len(successful),
            "overall": {
                "mean_wer": sum(wers) / len(wers) if wers else None,
                "mean_cer": sum(cers) / len(cers) if cers else None,
                "min_wer": min(wers) if wers else None,
                "max_wer": max(wers) if wers else None,
            },
            "per_reciter": {
                rid: {
                    "name": s["name"],
                    "samples": len(s["wers"]),
                    "mean_wer": sum(s["wers"]) / len(s["wers"]),
                    "mean_cer": sum(s["cers"]) / len(s["cers"]),
                }
                for rid, s in reciter_agg.items()
            },
        }

        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("JSON summary written: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Quranic pipeline against HuggingFace parquet dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to a downloaded parquet file (e.g. data/benchmark_hf/0044.parquet)",
    )
    parser.add_argument(
        "--output_dir",
        default="results/hf_benchmark/",
        help="Directory for result files (default: results/hf_benchmark/)",
    )
    parser.add_argument(
        "--mode",
        default="asr_only",
        choices=["asr_only"],
        help="Benchmark mode (default: asr_only)",
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
        help="Device (default: auto)",
    )
    parser.add_argument(
        "--backend",
        default="faster-whisper",
        choices=["faster-whisper", "huggingface"],
        help="ASR backend: faster-whisper (CTranslate2, ~4x faster on CPU) or huggingface (PyTorch)",
    )
    parser.add_argument(
        "--compute_type",
        default="auto",
        choices=["auto", "int8", "float16", "float32"],
        help="CTranslate2 compute type for faster-whisper backend (default: auto → int8 on CPU)",
    )
    parser.add_argument(
        "--model_dir",
        default="models/whisper-quran-ct2",
        help="Path to CTranslate2 model directory for faster-whisper backend",
    )
    parser.add_argument(
        "--reciters",
        default=None,
        help="Comma-separated reciter IDs to include (e.g. alafasy,husary). Default: all.",
    )
    parser.add_argument(
        "--surahs",
        default=None,
        help="Comma-separated surah IDs to include (e.g. 37). Default: all.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Max rows to process (0 = all)",
    )
    args = parser.parse_args()

    reciters = [r.strip() for r in args.reciters.split(",")] if args.reciters else None
    surahs = [int(s.strip()) for s in args.surahs.split(",")] if args.surahs else None

    runner = HFBenchmarkRunner(
        parquet_path=Path(args.parquet),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        model=args.model,
        device=args.device,
        backend=args.backend,
        compute_type=args.compute_type,
        model_dir=args.model_dir,
        reciters=reciters,
        surahs=surahs,
        max_rows=args.max_rows,
    )

    try:
        runner.run()
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
