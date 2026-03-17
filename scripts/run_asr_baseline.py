"""
run_asr_baseline.py - Run ASR baseline evaluation over a directory of audio samples.

Usage
-----
    python scripts/run_asr_baseline.py --samples_dir data/samples/ \
        --reference_csv data/reference/refs.csv --mock

Flags
-----
--samples_dir   : directory containing .wav files
--reference_csv : CSV with columns: filename, reference_text
--mock          : generate synthetic comparisons without loading audio/model
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _mock_results(samples_dir: Path) -> list[dict]:
    """Generate synthetic ASR baseline results with injected errors."""
    synthetic_refs = [
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        "مَالِكِ يَوْمِ الدِّينِ",
    ]
    synthetic_hyps = [
        "بسم الله الرحمن الرحيم",    # missing harakat
        "الحمد لله رب العالمين",      # missing harakat
        "مالك يوم الدين المقدس",      # extra word
    ]
    results = []
    for i, (ref, hyp) in enumerate(zip(synthetic_refs, synthetic_hyps)):
        try:
            from src.asr import compute_wer
            from src.preprocessor import strip_harakat
            metrics = compute_wer(strip_harakat(hyp), strip_harakat(ref))
        except Exception as exc:
            logger.warning("compute_wer failed for mock sample %d: %s", i, exc)
            metrics = {"wer": 0.0, "cer": 0.0, "insertions": 0, "deletions": 0, "substitutions": 0}
        results.append(
            {
                "filename": f"mock_{i:03d}.wav",
                "reference_text": ref,
                "hypothesis_text": hyp,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "insertions": metrics["insertions"],
                "deletions": metrics["deletions"],
                "substitutions": metrics["substitutions"],
            }
        )
    return results


def _real_results(samples_dir: Path, reference_csv: Path) -> list[dict]:
    """Run Whisper on each audio file and compute WER."""
    from src.asr import load_audio, load_whisper_model, transcribe_whisper, compute_wer
    from src.preprocessor import strip_harakat

    refs: dict[str, str] = {}
    if reference_csv.exists():
        with reference_csv.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                refs[row["filename"]] = row["reference_text"]

    model, processor, device = load_whisper_model()

    # Collect supported audio files (WAV/FLAC/OGG/AIFF)
    from src.asr import SUPPORTED_FORMATS
    audio_files = sorted(
        p for p in samples_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_FORMATS
    )
    if not audio_files:
        logger.warning(
            "No supported audio files in %s (supported: %s).",
            samples_dir, ", ".join(sorted(SUPPORTED_FORMATS)),
        )
        return []

    results = []
    for wav_path in audio_files:
        try:
            reference = refs.get(wav_path.name, "")
            audio, sr = load_audio(str(wav_path))
            out = transcribe_whisper(model, processor, audio, sr, device)
            hyp = out["text"]

            ref_norm = strip_harakat(reference) if reference else ""
            hyp_norm = strip_harakat(hyp)
            metrics = compute_wer(hyp_norm, ref_norm) if ref_norm else {
                "wer": 0.0, "cer": 0.0, "insertions": 0, "deletions": 0, "substitutions": 0
            }
            results.append(
                {
                    "filename": wav_path.name,
                    "reference_text": reference,
                    "hypothesis_text": hyp,
                    **metrics,
                }
            )
            logger.info("Processed %s  WER=%.4f", wav_path.name, metrics["wer"])
        except Exception as exc:
            logger.warning("Failed on %s: %s", wav_path.name, exc)
            # Never crash on single file failure
            results.append(
                {
                    "filename": wav_path.name,
                    "reference_text": refs.get(wav_path.name, ""),
                    "hypothesis_text": "ERROR",
                    "wer": -1.0, "cer": -1.0,
                    "insertions": 0, "deletions": 0, "substitutions": 0,
                }
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR baseline evaluation.")
    parser.add_argument("--samples_dir", default="data/samples/", help="Directory with .wav files")
    parser.add_argument("--reference_csv", default="data/reference/refs.csv", help="CSV with references")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock data")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    reference_csv = Path(args.reference_csv)

    if args.mock:
        logger.info("Mock mode enabled – generating synthetic results.")
        results = _mock_results(samples_dir)
    else:
        results = _real_results(samples_dir, reference_csv)

    if not results:
        logger.warning("No results generated.")
        return

    output_path = _ROOT / "results" / "asr_baseline.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Written: %s  (%d rows)", output_path, len(results))

    # Print summary
    wers = [r["wer"] for r in results if r["wer"] >= 0]
    if wers:
        avg_wer = sum(wers) / len(wers)
        logger.info("Average WER: %.4f over %d files", avg_wer, len(wers))


if __name__ == "__main__":
    main()
