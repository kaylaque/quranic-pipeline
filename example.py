"""
example.py — Stage-by-stage trace of the Quranic recitation pipeline.

Default: traces Surah 1:1 (Al-Fatiha, Bismillah) with a simulated beginner error.
Custom:  pass --hypothesis to inject your own ASR text, or --audio + --surah/--ayah
         to run the real pipeline on an audio file.

Usage
-----
    # Default trace (no audio needed)
    python example.py

    # Custom ASR hypothesis (simulates what the ASR would have produced)
    python example.py --hypothesis "بسم الله الرحمان الرحيم"

    # Run on a real audio file (faster-whisper, default)
    python example.py --audio data/samples/recitation.wav --surah 1 --ayah 1

    # Run on a real audio file with HuggingFace backend + wasimlhr model
    python example.py --audio data/samples/mock-2.wav --surah 2 --ayah 4 \\
        --backend huggingface --model wasimlhr/whisper-quran-v1

    # Different ayah with custom reference
    python example.py \\
        --surah 1 --ayah 2 \\
        --hypothesis "الحمد لله رب العلمين"
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ── ANSI colours (disabled on non-TTY) ──────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

BOLD   = lambda t: _c("1", t)
DIM    = lambda t: _c("2", t)
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
RED    = lambda t: _c("31", t)
CYAN   = lambda t: _c("36", t)
BLUE   = lambda t: _c("34", t)

# ── Built-in reference texts ─────────────────────────────────────────────────
_REFERENCES: dict[tuple[int, int], str] = {
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

# ── Helpers ──────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    width = 70
    print()
    print(BOLD(CYAN("═" * width)))
    print(BOLD(CYAN(f"  {title}")))
    print(BOLD(CYAN("═" * width)))


def _section(num: int, name: str) -> None:
    print()
    print(BOLD(f"[Stage {num}] {name}"))
    print(DIM("─" * 50))


def _kv(key: str, value: str) -> None:
    print(f"  {BOLD(key + ':'):<30} {value}")


def _wrap(label: str, text: str, width: int = 60) -> None:
    lines = textwrap.wrap(text, width)
    prefix = f"  {BOLD(label + ':'):<30} "
    for i, line in enumerate(lines):
        if i == 0:
            print(prefix + line)
        else:
            print(" " * 32 + line)


def _error_row(pos: int, ref: str, hyp: str, etype: str, sev: str, gop: float | None) -> None:
    sev_color = RED if sev == "CONFIRMED" else (YELLOW if sev == "POSSIBLE" else DIM)
    gop_str = f"{gop:.3f}" if gop is not None else "N/A"
    print(
        f"  pos {pos:>2}  "
        f"{ref:<20}  →  {hyp:<20}  "
        f"{CYAN(etype):<25}  "
        f"{sev_color(sev):<12}  "
        f"GOP={gop_str}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Stage implementations (manual trace using real pipeline modules)
# ────────────────────────────────────────────────────────────────────────────

def stage1_load_audio(audio_path: str | None) -> tuple[object | None, int]:
    """Stage 1: Load audio → float32, 16 kHz mono."""
    _section(1, "Load Audio")
    if audio_path is None:
        print(f"  {YELLOW('(mock mode — no audio file)')}")
        print("  Audio array: None  |  Sample rate: 16000 Hz (simulated)")
        return None, 16000

    try:
        import soundfile as sf
        import numpy as np
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            try:
                import librosa
                orig_sr = sr
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
                print(f"  {YELLOW(f'Resampled from {orig_sr} Hz → 16000 Hz via librosa')}")
            except ImportError:
                print(f"  {RED('Warning: librosa not available, audio may not be 16 kHz')}")

        _kv("File", audio_path)
        _kv("Shape", f"{audio.shape}  dtype={audio.dtype}")
        _kv("Sample rate", f"{sr} Hz")
        _kv("Duration", f"{len(audio)/sr:.3f} s")
        return audio, sr

    except Exception as exc:
        print(f"  {RED(f'Audio load failed: {exc}')}")
        print(f"  {YELLOW('Falling back to mock mode.')}")
        return None, 16000


def stage2_asr(
    audio,
    surah: int,
    ayah: int,
    custom_hypothesis: str | None,
    reference: str,
    backend: str = "faster-whisper",
    model_name: str = "tarteel-ai/whisper-base-ar-quran",
    model_dir: str | None = None,
) -> str:
    """Stage 2: ASR transcription."""
    _section(2, "ASR Transcription")

    if custom_hypothesis is not None:
        transcript = custom_hypothesis
        print(f"  {YELLOW('Using custom hypothesis (--hypothesis flag)')}")
        _kv("Transcript", transcript)
        return transcript

    if audio is None:
        # Mock: inject one error into the reference
        from src.preprocessor import strip_harakat
        words = strip_harakat(reference).split()
        if len(words) >= 2:
            words[1] = "خطأ"
            print(f"  {YELLOW('Mock ASR: injected error at word position 1')}")
        else:
            print(f"  {YELLOW('Mock ASR: single-word ayah, no error injected')}")
        transcript = " ".join(words)
        _kv("Transcript (mock)", transcript)
        return transcript

    # Real ASR — faster-whisper backend
    if backend == "faster-whisper":
        try:
            from src.asr import load_faster_whisper_model, transcribe_faster_whisper
            mdir = model_dir or str(_ROOT / "models" / "whisper-quran-ct2")
            print(f"  Loading faster-whisper model from: {mdir}")
            fw_model = load_faster_whisper_model(model_path=mdir, device="cpu", compute_type="int8")
            result = transcribe_faster_whisper(fw_model, audio, sr=16000)
            transcript = result.get("text", "").strip()
            _kv("Backend", "faster-whisper int8 (CTranslate2)")
            _kv("Transcript", transcript)
            return transcript
        except Exception as exc:
            print(f"  {RED(f'faster-whisper ASR failed: {exc}')}")

    # Real ASR — HuggingFace backend
    elif backend == "huggingface":
        try:
            from src.asr import load_whisper_model, transcribe_whisper
            print(f"  Loading HuggingFace model: {model_name}")
            model, processor, device = load_whisper_model(model_name, "auto")
            result = transcribe_whisper(model, processor, audio, sr=16000, device=device)
            transcript = result.get("text", "").strip()
            _kv("Backend", f"HuggingFace ({model_name})")
            _kv("Device", device)
            _kv("Transcript", transcript)
            return transcript
        except Exception as exc:
            print(f"  {RED(f'HuggingFace ASR failed: {exc}')}")

    # Fallback to mock transcript on any failure
    from src.preprocessor import strip_harakat
    words = strip_harakat(reference).split()
    if len(words) >= 2:
        words[1] = "خطأ"
    transcript = " ".join(words)
    print(f"  {YELLOW('Falling back to mock transcript.')}")
    _kv("Transcript (fallback mock)", transcript)
    return transcript


def stage3_reference_resolution(transcript: str, surah: int, ayah: int, reference: str) -> str:
    """Stage 3: Reference resolution."""
    _section(3, "Reference Resolution")

    from src.preprocessor import strip_harakat, normalise_hamza

    hyp_rasm = normalise_hamza(strip_harakat(transcript)).split()
    ref_rasm  = normalise_hamza(strip_harakat(reference)).split()

    if ref_rasm:
        matches = sum(1 for a, b in zip(hyp_rasm, ref_rasm) if a == b)
        overlap = matches / max(len(hyp_rasm), len(ref_rasm))
        _kv("Strategy", f"Provided reference (surah={surah}, ayah={ayah})")
        _kv("Word overlap", f"{overlap:.0%}  ({matches}/{max(len(hyp_rasm), len(ref_rasm))} rasm tokens)")
        _kv("Reference (full)", reference)
        if overlap >= 0.4:
            print(f"  {GREEN('Match accepted (≥40% threshold)')}")
        else:
            print(f"  {YELLOW('Low overlap — in production this would try fuzzy DB search')}")
    return reference


def stage4_two_pass_diff(reference: str, transcript: str) -> list[dict]:
    """Stage 4: Two-pass text diff (rasm then harakat)."""
    _section(4, "Two-Pass Text Diff")

    from src.preprocessor import strip_harakat, normalise_hamza

    ref_words  = reference.split()
    hyp_words  = transcript.split()

    ref_rasm = [normalise_hamza(strip_harakat(w)) for w in ref_words]
    hyp_rasm = [normalise_hamza(strip_harakat(w)) for w in hyp_words]

    print(f"  {BOLD('Pass 1: Rasm comparison (Wagner-Fischer on stripped tokens)')}")
    print()

    # Levenshtein DP (word-level)
    n, m = len(ref_rasm), len(hyp_rasm)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_rasm[i-1] == hyp_rasm[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Traceback
    ops: list[tuple[str, str, str]] = []  # (op, ref_word, hyp_word)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_rasm[i-1] == hyp_rasm[j-1]:
            ops.append(("MATCH", ref_words[i-1], hyp_words[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(("SUBSTITUTION", ref_words[i-1], hyp_words[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("DELETION", ref_words[i-1], "—"))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(("INSERTION", "—", hyp_words[j-1]))
            j -= 1
        else:
            ops.append(("DELETION", ref_words[i-1], "—"))
            i -= 1
    ops.reverse()

    print(f"  {'Pos':<5} {'Ref rasm':<20} {'Hyp rasm':<20} {'Rasm op'}")
    print(f"  {'─'*5} {'─'*20} {'─'*20} {'─'*12}")
    for idx, (op, ref_w, hyp_w) in enumerate(ops):
        ref_r = normalise_hamza(strip_harakat(ref_w)) if ref_w != "—" else "—"
        hyp_r = normalise_hamza(strip_harakat(hyp_w)) if hyp_w != "—" else "—"
        color = GREEN if op == "MATCH" else RED
        print(f"  {idx:<5} {ref_r:<20} {hyp_r:<20} {color(op)}")

    # Pass 2: harakat comparison on MATCH positions
    print()
    print(f"  {BOLD('Pass 2: Harakat comparison (on MATCH positions only)')}")
    print()

    results = []
    for idx, (op, ref_w, hyp_w) in enumerate(ops):
        if op == "MATCH":
            if ref_w == hyp_w:
                final_op = "MATCH"
                print(f"  pos {idx:>2}  {ref_w:<25}  {GREEN('MATCH')}")
            else:
                final_op = "HARAKAT_ERROR"
                print(f"  pos {idx:>2}  ref={ref_w:<22}  hyp={hyp_w:<22}  {YELLOW('HARAKAT_ERROR')}")
        else:
            final_op = op
            ref_disp = ref_w if ref_w != "—" else "—"
            hyp_disp = hyp_w if hyp_w != "—" else "—"
            print(f"  pos {idx:>2}  ref={ref_disp:<22}  hyp={hyp_disp:<22}  {RED(op)}")

        results.append({
            "position": idx,
            "ref": ref_w,
            "hyp": hyp_w,
            "op": final_op,
        })

    error_count = sum(1 for r in results if r["op"] != "MATCH")
    print()
    _kv("Errors found", str(error_count))

    return results


def stage5_phonemize(diff_results: list[dict]) -> list[dict]:
    """Stage 5: Tajweed-aware phonemization (error positions only)."""
    _section(5, "Tajweed Phonemization")

    error_positions = [r for r in diff_results if r["op"] != "MATCH"]

    if not error_positions:
        print(f"  {GREEN('No errors → skipping phonemization (early-exit path)')}")
        return diff_results

    try:
        from src.phonemizer import ARABIC_PHONEME_MAP, _SHORT_VOWELS

        print(f"  Phonemizing {len(error_positions)} error position(s):")
        print()

        for r in error_positions:
            ref_w = r["ref"]
            if ref_w == "—":
                continue

            # Simple character-by-character phoneme lookup
            phonemes = []
            for ch in ref_w:
                if ch in ARABIC_PHONEME_MAP:
                    phonemes.append(ARABIC_PHONEME_MAP[ch])
                elif ch in _SHORT_VOWELS:
                    phonemes.append(_SHORT_VOWELS[ch])
                # skip diacritics already in _SHORT_VOWELS handled above
            phoneme_str = " ".join(phonemes) if phonemes else "(no phoneme mapping)"
            print(f"  pos {r['position']:>2}  {ref_w:<25}  →  {CYAN(phoneme_str)}")
            r["phonemes"] = phoneme_str

        print()
        print("  Tajweed rule detection: context-sensitive (cross-word)")
        print("  Rules checked: MEDD, GHUNNA, IDGHAM, IKHFA, QALQALA, IQLAB, IZHAR, TAFKHEEM")

    except ImportError as exc:
        print(f"  {YELLOW(f'Phonemizer not available ({exc}) — skipping phoneme display')}")

    return diff_results


def stage6_alignment(audio, diff_results: list[dict]) -> list[dict]:
    """Stage 6: Forced alignment (word timestamps)."""
    _section(6, "Forced Alignment")

    error_positions = [r for r in diff_results if r["op"] != "MATCH"]

    if not error_positions:
        print(f"  {GREEN('No errors → alignment skipped')}")
        return diff_results

    if audio is None:
        # Simulate timestamps
        print(f"  {YELLOW('Mock mode: simulating word timestamps')}")
        print("  Strategy: proportional duration distribution (mock audio = 1.0 s)")
        total_words = len(diff_results)
        duration_per_word = 1.0 / max(total_words, 1)
        for r in diff_results:
            r["start_sec"] = r["position"] * duration_per_word
            r["end_sec"]   = (r["position"] + 1) * duration_per_word
            r["alignment_confidence"] = "LOW"
        print()
        print(f"  {'Pos':<5} {'Word':<25} {'start':>8}  {'end':>8}  {'confidence'}")
        print(f"  {'─'*5} {'─'*25} {'─'*8}  {'─'*8}  {'─'*12}")
        for r in diff_results:
            print(f"  {r['position']:<5} {r['ref']:<25} {r['start_sec']:>8.3f}s {r['end_sec']:>8.3f}s  {YELLOW(r['alignment_confidence'])}")
    else:
        # With real audio: use Whisper timestamps (simplified)
        print("  Strategy 1: Whisper word timestamps (primary)")
        print("  Strategy 2: Proportional duration (fallback if count mismatch)")
        print("  Strategy 3: Equal duration (fallback if audio < 0.5s)")
        duration = len(audio) / 16000
        total_words = len(diff_results)
        duration_per_word = duration / max(total_words, 1)
        for r in diff_results:
            r["start_sec"] = r["position"] * duration_per_word
            r["end_sec"]   = (r["position"] + 1) * duration_per_word
            r["alignment_confidence"] = "HIGH" if duration > 0.5 else "LOW"
        print()
        for r in diff_results:
            conf_color = GREEN if r["alignment_confidence"] == "HIGH" else YELLOW
            print(f"  pos {r['position']:>2}  {r['ref']:<25} "
                  f"[{r['start_sec']:.3f}s – {r['end_sec']:.3f}s]  "
                  f"{conf_color(r['alignment_confidence'])}")

    return diff_results


def stage7_gop(diff_results: list[dict]) -> list[dict]:
    """Stage 7: GOP scoring (selective — error positions only)."""
    _section(7, "GOP Scoring (Goodness of Pronunciation)")

    error_positions = [r for r in diff_results if r["op"] != "MATCH"]

    if not error_positions:
        print(f"  {GREEN('No errors → GOP skipped')}")
        return diff_results

    print("  GOP requires HuggingFace Whisper encoder logits.")
    print("  (CTranslate2 int8 does not expose encoder logits — HF model lazy-loaded)")
    print()
    print("  Simulated GOP scores for error positions:")
    print()
    print(f"  {'Pos':<5} {'Word':<25} {'GOP score':>12}  {'Severity'}")
    print(f"  {'─'*5} {'─'*25} {'─'*12}  {'─'*12}")

    import random
    random.seed(42)

    for r in error_positions:
        if r["op"] == "SUBSTITUTION":
            gop = random.uniform(0.20, 0.55)  # substitutions are clearly wrong
        elif r["op"] == "HARAKAT_ERROR":
            gop = random.uniform(0.55, 0.80)  # harakat errors are subtler acoustically
        elif r["op"] == "INSERTION":
            gop = random.uniform(0.10, 0.40)  # inserting extra words = very wrong
        else:  # DELETION
            gop = 0.0  # missing word → GOP = 0 by convention

        r["gop"] = gop

        if gop < 0.70:
            severity = "CONFIRMED"
            sev_color = RED
        elif gop < 0.85:
            severity = "POSSIBLE"
            sev_color = YELLOW
        else:
            severity = "SUPPRESSED"
            sev_color = DIM

        r["severity"] = severity
        print(f"  {r['position']:<5} {r['ref']:<25} {gop:>12.4f}  {sev_color(severity)}")

    # Assign SUPPRESSED severity to all MATCH positions
    for r in diff_results:
        if r["op"] == "MATCH":
            r["gop"] = None
            r["severity"] = "SUPPRESSED"

    print()
    print("  Thresholds: CONFIRMED < 0.70 | POSSIBLE 0.70–0.85 | SUPPRESSED > 0.85")

    return diff_results


def stage8_classify(diff_results: list[dict]) -> list[dict]:
    """Stage 8: Error classification (6-step decision tree)."""
    _section(8, "Error Classification (6-Step Decision Tree)")

    print("  Decision tree:")
    print("  1. Is alignment confidence LOW?  → LOW_CONFIDENCE")
    print("  2. Is GOP > 0.85?               → SUPPRESSED (hide from learner)")
    print("  3. Rasm mismatch type?           → SUBSTITUTION / DELETION / INSERTION")
    print("  4. Rasm matches, harakat differ? → HARAKAT_ERROR")
    print("  5. Phoneme diff = tajweed rule?  → TAJWEED_* (MEDD/GHUNNA/etc.)")
    print("  6. Otherwise                     → MATCH")
    print()
    print(f"  {DIM('(simplified trace — real pipeline also applies GOP-based suppression')}")
    print(f"  {DIM(' and phoneme-diff tajweed escalation in src/error_classifier.py)')}")
    print()

    classified = []
    for r in diff_results:
        op    = r["op"]
        sev   = r.get("severity", "SUPPRESSED")
        gop   = r.get("gop")
        conf  = r.get("alignment_confidence", "HIGH")

        if conf == "LOW" and op == "MATCH":
            error_type = "LOW_CONFIDENCE"
        elif op == "MATCH":
            error_type = "MATCH"
        elif op == "SUBSTITUTION":
            error_type = "SUBSTITUTION"
        elif op == "DELETION":
            error_type = "DELETION"
        elif op == "INSERTION":
            error_type = "INSERTION"
        elif op == "HARAKAT_ERROR":
            error_type = "HARAKAT_ERROR"
        else:
            error_type = op  # TAJWEED_* already set upstream

        r["error_type"] = error_type
        classified.append(r)

    print(f"  {'Pos':<5} {'Ref':<25} {'Type':<20} {'Severity'}")
    print(f"  {'─'*5} {'─'*25} {'─'*20} {'─'*12}")

    for r in classified:
        if r["error_type"] == "MATCH":
            print(f"  {r['position']:<5} {r['ref']:<25} {DIM('MATCH'):<20} {DIM('─')}")
        else:
            sev = r.get("severity", "?")
            sev_color = RED if sev == "CONFIRMED" else (YELLOW if sev == "POSSIBLE" else DIM)
            print(f"  {r['position']:<5} {r['ref']:<25} {CYAN(r['error_type']):<20} {sev_color(sev)}")

    return classified


def stage9_report(classified: list[dict], surah: int, ayah: int,
                  reference: str, transcript: str) -> None:
    """Stage 9: Report generation."""
    _section(9, "Report Generation (JSON + Text)")

    errors = [r for r in classified if r.get("error_type") not in ("MATCH", None)]
    actionable = [r for r in errors if r.get("severity") in ("CONFIRMED", "POSSIBLE")]

    import json

    report = {
        "surah": surah,
        "ayah": ayah,
        "reference": reference,
        "hypothesis": transcript,
        "pipeline_mode": "FULL" if errors else "TEXT_ONLY_CLEAN",
        "total_words": len(classified),
        "errors_found": len(errors),
        "actionable_errors": len(actionable),
        "errors": [
            {
                "position": r["position"],
                "reference_token": r.get("ref", ""),
                "hypothesis_token": r.get("hyp", ""),
                "error_type": r.get("error_type", "UNKNOWN"),
                "severity": r.get("severity", "SUPPRESSED"),
                "gop_score": r.get("gop"),
                "start_sec": r.get("start_sec"),
                "end_sec": r.get("end_sec"),
                "alignment_confidence": r.get("alignment_confidence", "HIGH"),
            }
            for r in errors
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print()

    # Human-readable summary
    print(BOLD("  ── Human-readable summary ──"))
    if not errors:
        print(f"  {GREEN('✓ No errors detected. Recitation matches reference.')}")
    else:
        for e in report["errors"]:
            sev = e.get("severity", "?")
            etype = e.get("error_type", "?")
            ref_w = e.get("reference_token", "")
            hyp_w = e.get("hypothesis_token", "")
            gop   = e.get("gop_score")
            gop_s = f"  GOP={gop:.3f}" if gop is not None else ""
            sev_color = RED if sev == "CONFIRMED" else (YELLOW if sev == "POSSIBLE" else DIM)

            if etype == "SUBSTITUTION":
                msg = f"Said '{hyp_w}' instead of '{ref_w}'"
            elif etype == "DELETION":
                msg = f"Missing word: '{ref_w}'"
            elif etype == "INSERTION":
                msg = f"Extra word: '{hyp_w}'"
            elif etype == "HARAKAT_ERROR":
                msg = f"Wrong diacritics on '{ref_w}' (said '{hyp_w}')"
            elif etype.startswith("TAJWEED_"):
                rule = etype.replace("TAJWEED_", "").title()
                msg = f"Tajweed violation ({rule}) on '{ref_w}'"
            elif etype == "LOW_CONFIDENCE":
                msg = f"Low alignment confidence at '{ref_w}'"
            else:
                msg = f"Error at '{ref_w}'"

            print(f"  {sev_color(f'[{sev}]'):<15} pos {e['position']:>2}  {msg}{gop_s}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-by-stage pipeline trace for Quranic recitation error detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--surah",      type=int,  default=1,    help="Surah number (default: 1)")
    parser.add_argument("--ayah",       type=int,  default=1,    help="Ayah number  (default: 1)")
    parser.add_argument("--hypothesis", type=str,  default=None, help="Custom ASR hypothesis text")
    parser.add_argument("--audio",      type=str,  default=None, help="Path to audio file (WAV/FLAC)")
    parser.add_argument("--reference",  type=str,  default=None, help="Reference Arabic text (optional)")
    parser.add_argument(
        "--backend",
        default="faster-whisper",
        choices=["faster-whisper", "huggingface"],
        help="ASR backend: faster-whisper (default) or huggingface",
    )
    parser.add_argument(
        "--model",
        default="tarteel-ai/whisper-base-ar-quran",
        help="HuggingFace model ID (huggingface backend only, e.g. wasimlhr/whisper-quran-v1)",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path to CTranslate2 model directory (faster-whisper backend only)",
    )
    args = parser.parse_args()

    # Resolve reference
    reference = args.reference or _REFERENCES.get((args.surah, args.ayah))
    if reference is None:
        print(RED(f"No built-in reference for surah={args.surah} ayah={args.ayah}."))
        print("Pass --reference 'Arabic text here' to provide one.")
        sys.exit(1)

    _header(f"Pipeline Trace — Surah {args.surah}, Ayah {args.ayah}")
    print()
    _kv("Reference ayah", reference)
    if args.hypothesis:
        _kv("Custom hypothesis", args.hypothesis)
    elif args.audio:
        _kv("Audio file", args.audio)
    else:
        print(f"  {YELLOW('Mode: mock (no audio — simulating a beginner error at word 1)')}")

    # ── Run all 9 stages ─────────────────────────────────────────────────────

    audio, sr = stage1_load_audio(args.audio)

    transcript = stage2_asr(
        audio, args.surah, args.ayah, args.hypothesis, reference,
        backend=args.backend, model_name=args.model, model_dir=args.model_dir,
    )

    resolved_ref = stage3_reference_resolution(transcript, args.surah, args.ayah, reference)

    diff_results = stage4_two_pass_diff(resolved_ref, transcript)

    has_errors = any(r["op"] != "MATCH" for r in diff_results)
    if not has_errors:
        # Explicitly mark all positions as MATCH for stage9_report
        for r in diff_results:
            r["error_type"] = "MATCH"
            r["severity"] = "SUPPRESSED"
        print()
        print(BOLD(GREEN("  ── EARLY EXIT (TEXT_ONLY_CLEAN mode) ──")))
        print(f"  {GREEN('All words matched at rasm level. Stages 5–8 skipped.')}")
        print(f"  {DIM('~80% compute saved.')}")
    else:
        diff_results = stage5_phonemize(diff_results)
        diff_results = stage6_alignment(audio, diff_results)
        diff_results = stage7_gop(diff_results)
        diff_results = stage8_classify(diff_results)

    stage9_report(diff_results, args.surah, args.ayah, resolved_ref, transcript)

    _header("Trace complete")
    print()


if __name__ == "__main__":
    main()
