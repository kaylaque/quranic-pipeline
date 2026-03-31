"""
run_pipeline.py - End-to-end Quranic recitation error detection pipeline.

Usage
-----
    python scripts/run_pipeline.py \\
        --audio data/samples/recitation.wav \\
        --surah 1 --ayah 1 \\
        --reference "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ" \\
        --output_dir results/reports/ \\
        --model tarteel-ai/whisper-base-ar-quran \\
        --device auto \\
        --verbose

    # Mock mode (no audio file needed)
    python scripts/run_pipeline.py \\
        --surah 1 --ayah 1 \\
        --mock --verbose

    # Audio-only mode (auto-detect surah/ayah from ASR transcript)
    python scripts/run_pipeline.py \\
        --audio data/samples/mock.wav \\
        --verbose
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow MPS ops to silently fall back to CPU for operations not yet implemented
# in Apple Metal (common after OS upgrades where MPS is partially available).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default reference texts for well-known ayahs (used when --reference omitted)
# ---------------------------------------------------------------------------
_DEFAULT_REFERENCES: dict[tuple[int, int], str] = {
    # Surah Al-Fatiha (Uthmani script with full tashkeel)
    (1, 1): "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
    (1, 2): "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ",
    (1, 3): "ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
    (1, 4): "مَـٰلِكِ يَوْمِ ٱلدِّينِ",
    (1, 5): "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
    (1, 6): "ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ",
    (1, 7): "صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ",
    # Surah Al-Baqarah
    (2, 4): "وَٱلَّذِينَ يُؤْمِنُونَ بِمَآ أُنزِلَ إِلَيْكَ وَمَآ أُنزِلَ مِن قَبْلِكَ وَبِٱلْـَٔاخِرَةِ هُمْ يُوقِنُونَ",
    # Ayat al-Kursi (complete, Uthmani script)
    (2, 255): (
        "ٱللَّهُ لَآ إِلَـٰهَ إِلَّا هُوَ ٱلْحَىُّ ٱلْقَيُّومُ"
        " لَا تَأْخُذُهُۥ سِنَةٌ وَلَا نَوْمٌ"
        " لَّهُۥ مَا فِى ٱلسَّمَـٰوَٰتِ وَمَا فِى ٱلْأَرْضِ"
        " مَن ذَا ٱلَّذِى يَشْفَعُ عِندَهُۥٓ إِلَّا بِإِذْنِهِۦ"
        " يَعْلَمُ مَا بَيْنَ أَيْدِيهِمْ وَمَا خَلْفَهُمْ"
        " وَلَا يُحِيطُونَ بِشَىْءٍ مِّنْ عِلْمِهِۦٓ إِلَّا بِمَا شَآءَ"
        " وَسِعَ كُرْسِيُّهُ ٱلسَّمَـٰوَٰتِ وَٱلْأَرْضَ"
        " وَلَا يَـُٔودُهُۥ حِفْظُهُمَا"
        " وَهُوَ ٱلْعَلِىُّ ٱلْعَظِيمُ"
    ),
}


def _step(num: int, name: str, verbose: bool):
    """Context manager-like timing wrapper."""
    label = f"Step {num}/8: {name}"

    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            logger.info("[START] %s", label)
            return self

        def __exit__(self, *_):
            elapsed = time.perf_counter() - self.t0
            logger.info("[DONE ] %s  (%.3f s)", label, elapsed)

    return _Timer()


def _check_audio_format(path: str) -> None:
    """Raise SystemExit with a helpful message for unsupported audio formats."""
    from src.asr import SUPPORTED_FORMATS
    suffix = Path(path).suffix.lower()
    if suffix and suffix not in SUPPORTED_FORMATS:
        logger.error(
            "Unsupported audio format '%s'. Supported: %s",
            suffix, ", ".join(sorted(SUPPORTED_FORMATS)),
        )
        sys.exit(1)


def _ensure_mock_wav(path: Path) -> None:
    """Create a 1-second silence WAV if it doesn't exist."""
    if not path.exists():
        try:
            import numpy as np
            import soundfile as sf
            path.parent.mkdir(parents=True, exist_ok=True)
            silence = np.zeros(16000, dtype=np.float32)
            sf.write(str(path), silence, 16000)
            logger.info("Created mock WAV: %s", path)
        except Exception as exc:
            logger.warning("Could not create mock WAV: %s", exc)


def _mock_transcribe(reference: str, surah: int, ayah: int) -> str:
    """Inject 1-2 errors into the reference to simulate ASR output."""
    from src.preprocessor import strip_harakat
    words = strip_harakat(reference).split()
    if len(words) >= 2:
        # Substitute the second word with a placeholder
        words[1] = "خطأ"
    return " ".join(words)


def _load_reference(surah: int, ayah: int) -> str | None:
    """Try to load reference from data/reference/ or built-in defaults."""
    # Try file-based lookup
    ref_dir = _ROOT / "data" / "reference"
    ref_file = ref_dir / f"{surah}_{ayah}.txt"
    if ref_file.exists():
        return ref_file.read_text(encoding="utf-8").strip()

    # Fall back to built-in defaults
    return _DEFAULT_REFERENCES.get((surah, ayah))


def _detect_ayah_from_transcript(transcript: str) -> tuple[int, int, str] | None:
    """
    Try to match *transcript* against known default references using rasm similarity.

    Returns (surah, ayah, reference_text) on a match, or None.
    """
    from src.preprocessor import strip_harakat, normalise_hamza

    hyp_rasm = normalise_hamza(strip_harakat(transcript)).split()
    if not hyp_rasm:
        return None

    best_match = None
    best_score = 0.0

    for (surah, ayah), ref_text in _DEFAULT_REFERENCES.items():
        ref_rasm = normalise_hamza(strip_harakat(ref_text)).split()
        if not ref_rasm:
            continue

        # Count matching words (order-sensitive overlap)
        matches = 0
        ref_len = len(ref_rasm)
        hyp_len = len(hyp_rasm)
        min_len = min(ref_len, hyp_len)
        max_len = max(ref_len, hyp_len)

        for i in range(min_len):
            if ref_rasm[i] == hyp_rasm[i]:
                matches += 1

        if max_len == 0:
            continue
        score = matches / max_len

        if score > best_score:
            best_score = score
            best_match = (surah, ayah, ref_text)

    # Require at least 40% word overlap to accept a match
    if best_match and best_score >= 0.4:
        logger.info(
            "Auto-detected surah %d ayah %d (%.0f%% match)",
            best_match[0], best_match[1], best_score * 100,
        )
        return best_match

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quranic recitation error detection pipeline.")
    parser.add_argument("--audio", default=None, help="Path to audio file")
    parser.add_argument("--surah", type=int, default=0, help="Surah number (0 = auto-detect from ASR)")
    parser.add_argument("--ayah", type=int, default=0, help="Ayah number (0 = auto-detect from ASR)")
    parser.add_argument("--reference", default=None, help="Reference Arabic text. If omitted, load from data/reference/.")
    parser.add_argument("--output_dir", default="results/reports/", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU usage (deprecated, use --device cpu)")
    parser.add_argument(
        "--model",
        default="tarteel-ai/whisper-base-ar-quran",
        help="HuggingFace model ID (e.g. tarteel-ai/whisper-base-ar-quran, wasimlhr/whisper-quran-v1)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device: auto (detect best), cpu, cuda (NVIDIA), mps (Apple Metal)",
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=10.0,
        help="Window size in seconds for chunked ASR transcription (HuggingFace backend only, default: 10.0)",
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=1,
        help="Parallel workers for CPU chunk transcription (HuggingFace backend only, default: 1)",
    )
    parser.add_argument(
        "--backend",
        default="faster-whisper",
        choices=["faster-whisper", "huggingface"],
        help="ASR backend: faster-whisper (CTranslate2, int8, no MPS needed) or huggingface (PyTorch)",
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
    args = parser.parse_args()

    if args.verbose:
        # Set DEBUG only for our own modules — not root (avoids numba/librosa bytecode dumps)
        for ns in ("src", "eval", "__main__", "scripts"):
            logging.getLogger(ns).setLevel(logging.DEBUG)
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    # Handle device selection
    device_preference = args.device
    if args.no_gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_preference = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surah = args.surah
    ayah = args.ayah
    auto_detect = (surah == 0 and ayah == 0 and args.reference is None and not args.mock)

    # Imports
    from src.preprocessor import tokenise_reference, two_pass_compare, all_match
    from src.error_classifier import classify_verse
    from src.report import generate_report, report_to_json, report_to_text

    # Track pipeline mode
    pipeline_mode = "MOCK" if args.mock else "FULL"

    # -------------------------------------------------------------------------
    # Step 2: Load audio ONCE  (moved before step 1 for auto-detect mode)
    # -------------------------------------------------------------------------
    audio_array = None
    sr = 16000
    audio_id = "mock_audio"

    if args.mock:
        mock_wav = _ROOT / "data" / "samples" / "mock.wav"
        _ensure_mock_wav(mock_wav)
        audio_path = str(mock_wav)
        audio_id = "mock_audio"
    else:
        audio_path = args.audio
        if audio_path is None:
            logger.error("--audio is required in non-mock mode.")
            sys.exit(1)
        _check_audio_format(audio_path)
        audio_id = Path(audio_path).stem

    with _step(2, "Load audio", args.verbose):
        try:
            from src.asr import load_audio
            audio_array, sr = load_audio(audio_path)
            logger.info("Audio loaded: %.2f s, sr=%d", len(audio_array) / sr, sr)
        except Exception as exc:
            logger.warning("Audio load failed (%s), using silence fallback.", exc)
            import numpy as np
            audio_array = np.zeros(16000, dtype=np.float32)
            sr = 16000

    # -------------------------------------------------------------------------
    # Step 3: ASR transcription
    # -------------------------------------------------------------------------
    hypothesis = ""
    asr_wer_dict = None
    # model/processor: HuggingFace objects reused by alignment + GOP stages.
    # When using faster-whisper backend these remain None until lazy-loaded.
    model = None
    processor = None
    # resolved_device starts as "cpu" for the faster-whisper path; updated to
    # the real device once the HuggingFace model loads (lazy load below).
    resolved_device = device_preference if device_preference != "auto" else "cpu"
    # Word timestamps from faster-whisper — reused by alignment to avoid a
    # second HuggingFace Whisper inference pass.
    _fw_word_timestamps = None

    with _step(3, "ASR transcription", args.verbose):
        if args.mock:
            # Mock transcription needs reference — handled after reference resolution
            pass
        else:
            try:
                if args.backend == "faster-whisper":
                    from src.asr import load_faster_whisper_model, transcribe_faster_whisper
                    fw_model = load_faster_whisper_model(
                        args.model_dir,
                        device=device_preference,
                        compute_type=args.compute_type,
                    )
                    result = transcribe_faster_whisper(fw_model, audio_array, sr)
                    hypothesis = result["text"]
                    _fw_word_timestamps = result.get("word_timestamps")
                    logger.info(
                        "faster-whisper | %d segment(s) | transcript: %s",
                        result.get("num_chunks", 0), hypothesis,
                    )
                else:
                    from src.asr import (
                        load_whisper_model, transcribe_whisper,
                        transcribe_chunked,
                    )
                    model, processor, resolved_device = load_whisper_model(
                        args.model, device_preference
                    )
                    logger.info("Using device: %s", resolved_device)

                    audio_duration = float(len(audio_array)) / sr
                    if audio_duration > args.chunk_seconds:
                        logger.info(
                            "Audio %.1fs > chunk_seconds %.1fs, using chunked transcription",
                            audio_duration, args.chunk_seconds,
                        )
                        result = transcribe_chunked(
                            model, processor, audio_array, sr,
                            device=resolved_device,
                            chunk_seconds=args.chunk_seconds,
                            overlap_seconds=1.0,
                            max_workers=args.parallel_workers,
                            model_name=args.model,
                        )
                        logger.info("Chunked into %d segments", result.get("num_chunks", 1))
                    else:
                        result = transcribe_whisper(
                            model, processor, audio_array, sr, resolved_device
                        )

                    hypothesis = result["text"]
                    logger.info("ASR transcript: %s", hypothesis)
            except Exception as exc:
                logger.warning("ASR failed: %s  — using empty hypothesis.", exc)
                hypothesis = ""

    # -------------------------------------------------------------------------
    # ASR failure guard — empty hypothesis in non-mock mode means the model
    # produced no output.  Flag pipeline_mode so the report is not mistaken
    # for a real recitation analysis with all words deleted.
    # -------------------------------------------------------------------------
    asr_failed = not args.mock and not hypothesis
    if asr_failed:
        logger.warning("ASR produced empty transcript — pipeline_mode will be ASR_FAILED.")
        pipeline_mode = "ASR_FAILED"

    # -------------------------------------------------------------------------
    # Auto-detect surah/ayah from transcript when not provided
    # -------------------------------------------------------------------------
    if auto_detect and hypothesis:
        detected = _detect_ayah_from_transcript(hypothesis)
        if detected:
            surah, ayah, _ = detected
        else:
            logger.warning("Could not auto-detect ayah from transcript.")

    # -------------------------------------------------------------------------
    # Step 1: Resolve reference text
    # -------------------------------------------------------------------------
    reference_text = args.reference
    if reference_text is None:
        if surah > 0 and ayah > 0:
            reference_text = _load_reference(surah, ayah)
        if reference_text is None and auto_detect:
            # Use transcript as reference in ASR_ONLY mode
            reference_text = hypothesis
            pipeline_mode = "ASR_ONLY"
            logger.info("No reference found — using ASR transcript as reference (ASR_ONLY mode).")
        elif reference_text is None:
            logger.error(
                "No reference text for surah %d ayah %d. "
                "Provide --reference or place a file at data/reference/%d_%d.txt",
                surah, ayah, surah, ayah,
            )
            sys.exit(1)
        else:
            logger.info("Loaded reference for %d:%d", surah, ayah)

    # Handle mock transcription (needs reference_text)
    if args.mock and not hypothesis:
        hypothesis = _mock_transcribe(reference_text, surah, ayah)
        logger.info("Mock hypothesis: %s", hypothesis)

    with _step(1, "Load reference", args.verbose):
        ref_tokens = tokenise_reference(reference_text, surah, ayah)
        logger.info("Reference: %d words", len(ref_tokens))

    # Compute WER now that we have both hypothesis and reference
    if hypothesis and reference_text and pipeline_mode != "ASR_ONLY":
        try:
            from src.asr import compute_wer
            from src.preprocessor import strip_harakat
            asr_wer_dict = compute_wer(strip_harakat(hypothesis), strip_harakat(reference_text))
            logger.info("WER=%.4f", asr_wer_dict["wer"])
        except Exception:
            pass

    asr_wer = asr_wer_dict["wer"] if asr_wer_dict else None

    # -------------------------------------------------------------------------
    # Step 4: Two-pass text comparison
    # -------------------------------------------------------------------------
    with _step(4, "Text-level diff", args.verbose):
        diffs = two_pass_compare(hypothesis, reference_text, surah, ayah)
        error_count = sum(1 for d in diffs if d.error_type != "MATCH")
        logger.info("Diffs: %d total, %d errors", len(diffs), error_count)

    # -------------------------------------------------------------------------
    # Step 5: EARLY EXIT CHECK
    # -------------------------------------------------------------------------
    if all_match(diffs):
        logger.info("Step 5/8: All words MATCH — early exit (TEXT_ONLY_CLEAN).")
        errors = classify_verse(diffs)
        report = generate_report(
            audio_id=audio_id,
            surah=surah,
            ayah=ayah,
            reference_text=reference_text,
            hypothesis_text=hypothesis,
            errors=errors,
            asr_wer=asr_wer,
            pipeline_mode="TEXT_ONLY_CLEAN",
        )
        _write_report(report, output_dir, audio_id, args.verbose)
        return

    # -------------------------------------------------------------------------
    # Lazy HF model load (faster-whisper backend only)
    # Alignment (Step 6) and GOP scoring (Step 7) still require the HuggingFace
    # Whisper model.  Load it now, only if there are errors that need scoring.
    # -------------------------------------------------------------------------
    if args.backend == "faster-whisper" and model is None and not args.mock:
        try:
            from src.asr import load_whisper_model
            logger.info(
                "Lazy-loading HuggingFace model for alignment + GOP scoring..."
            )
            model, processor, resolved_device = load_whisper_model(
                args.model, device_preference
            )
        except Exception as exc:
            logger.warning(
                "Lazy HF model load failed: %s — alignment and GOP will be skipped.", exc
            )

    # -------------------------------------------------------------------------
    # Step 6: Phonemize + Forced alignment
    # -------------------------------------------------------------------------
    phone_tokens = None
    phone_diffs = None
    aligned_words = None
    error_positions: set[str] = set()

    # Collect error positions for selective GOP
    for diff in diffs:
        if diff.error_type != "MATCH":
            error_positions.add(diff.position)

    with _step(6, "Phonemize + Forced alignment", args.verbose):
        # Phonemize reference and compute phone diffs from aligned pairs
        try:
            from src.phonemizer import phonemize_verse, phonemize_aligned_pair, compute_phone_diff
            phone_tokens = phonemize_verse(surah, ayah, reference_text)

            phone_diffs_list = []
            for diff in diffs:
                ref_ptok, hyp_ptok = phonemize_aligned_pair(diff, surah, ayah)
                pd = compute_phone_diff(ref_ptok, hyp_ptok.phonemes)
                phone_diffs_list.append(pd)
            phone_diffs = phone_diffs_list
            logger.info("Phonemized %d words, %d phone diffs", len(phone_tokens), len(phone_diffs))
        except Exception as exc:
            logger.warning("Phonemization failed: %s", exc)

        # Forced alignment
        try:
            from src.aligner import (
                align_with_fallback, create_mock_alignment,
                create_aligned_from_fw_timestamps,
            )
            if args.mock:
                # Mock mode uses silence WAV — skip CTC entirely, use equal-duration alignment
                audio_duration = float(len(audio_array)) / max(sr, 1)
                aligned_words = create_mock_alignment(
                    reference_text, audio_duration, surah=surah, ayah=ayah
                )
                logger.info("Mock alignment: %d words (equal-duration)", len(aligned_words))
            elif _fw_word_timestamps:
                # Reuse word timestamps already collected by faster-whisper ASR —
                # avoids loading and running the HuggingFace model a second time.
                aligned_words = create_aligned_from_fw_timestamps(
                    _fw_word_timestamps, reference_text, surah=surah, ayah=ayah
                )
                logger.info(
                    "Alignment from faster-whisper timestamps: %d words (HIGH confidence)",
                    len(aligned_words),
                )
            else:
                aligned_words = align_with_fallback(
                    audio_array, sr, reference_text, hypothesis,
                    surah=surah, ayah=ayah,
                    model=model, processor=processor, device=resolved_device,
                )
                logger.info("Aligned %d words (confidence: %s)",
                    len(aligned_words),
                    aligned_words[0].alignment_confidence if aligned_words else "N/A",
                )
        except Exception as exc:
            logger.warning("Alignment failed: %s", exc)

    # -------------------------------------------------------------------------
    # Step 7: Selective GOP scoring (error positions only)
    # -------------------------------------------------------------------------
    gop_available = False
    with _step(7, "Selective GOP scoring", args.verbose):
        if aligned_words and not args.mock:
            try:
                from src.aligner import score_aligned_words
                # model/processor already loaded in Step 3 — reuse the cached instance
                aligned_words = score_aligned_words(
                    aligned_words,
                    audio_array,
                    sr,
                    phone_tokens or [],
                    model,
                    processor,
                    error_positions=error_positions,
                    device=resolved_device,
                )
                gop_available = True
            except Exception as exc:
                logger.warning("GOP scoring failed: %s", exc)
        elif args.mock:
            logger.info("Mock mode — GOP scores from fallback alignment (0.85).")
        else:
            logger.info("No aligned words — skipping GOP.")

    # Determine pipeline_mode (don't overwrite ASR_ONLY or ASR_FAILED)
    if pipeline_mode not in ("ASR_ONLY", "ASR_FAILED"):
        if args.mock:
            pipeline_mode = "MOCK"
        elif gop_available:
            pipeline_mode = "FULL"
        elif aligned_words:
            pipeline_mode = "NO_GOP"
        else:
            pipeline_mode = "TEXT_ONLY"

    # -------------------------------------------------------------------------
    # Step 8: Classify + Report
    # -------------------------------------------------------------------------
    with _step(8, "Classify errors + generate report", args.verbose):
        errors = classify_verse(diffs, phone_diffs=phone_diffs, aligned_words=aligned_words)
        report = generate_report(
            audio_id=audio_id,
            surah=surah,
            ayah=ayah,
            reference_text=reference_text,
            hypothesis_text=hypothesis,
            errors=errors,
            asr_wer=asr_wer,
            pipeline_mode=pipeline_mode,
        )
        _write_report(report, output_dir, audio_id, args.verbose)


def _write_report(report, output_dir: Path, audio_id: str, verbose: bool) -> None:
    from src.report import report_to_json, report_to_text

    json_path = output_dir / f"{audio_id}.json"
    txt_path = output_dir / f"{audio_id}.txt"

    report_to_json(report, str(json_path))
    txt_content = report_to_text(report)
    txt_path.write_text(txt_content, encoding="utf-8")

    logger.info("Report written: %s", json_path)
    logger.info("Report written: %s", txt_path)
    if verbose:
        print(txt_content)


if __name__ == "__main__":
    main()
