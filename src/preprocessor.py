"""
preprocessor.py - Text normalisation, tokenisation, and two-pass Levenshtein alignment
for Quranic recitation error detection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Unicode ranges for Arabic diacritics / harakat
# ---------------------------------------------------------------------------
# U+064B–U+065F : fathatan, dammatan, kasratan, fatha, damma, kasra, shadda,
#                 sukun, and extended combining marks
# U+0670        : superscript alef (alef above)
# U+0610–U+061A : Quranic annotation signs
# U+06D6–U+06ED : Quranic marks (sajda, rub-el-hizb, etc.)
_HARAKAT_RE = re.compile(
    "[\u064b-\u065f\u0670\u0610-\u061a\u06d6-\u06ed]"
)
# U+0640 : tatweel / kashida (Arabic letter extension)
_TATWEEL_RE = re.compile("\u0640")


def strip_harakat(text: str) -> str:
    """Remove all Arabic diacritical marks and tatweel from *text*."""
    text = _HARAKAT_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Hamza normalisation table
# ---------------------------------------------------------------------------
_HAMZA_TABLE = str.maketrans(
    {
        "\u0623": "\u0627",  # أ → ا
        "\u0625": "\u0627",  # إ → ا
        "\u0622": "\u0627",  # آ → ا
        "\u0671": "\u0627",  # ٱ → ا
        "\u0626": "\u064a",  # ئ → ي
        "\u0624": "\u0648",  # ؤ → و
    }
)

# Lam-alef ligatures U+FEF5–U+FEFC → ل + ا
_LAM_ALEF = {
    "\ufef5": "\u0644\u0627",  # ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE ISOLATED
    "\ufef6": "\u0644\u0627",  # ... FINAL FORM
    "\ufef7": "\u0644\u0627",  # ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE ISOLATED
    "\ufef8": "\u0644\u0627",  # ... FINAL FORM
    "\ufef9": "\u0644\u0627",  # ARABIC LIGATURE LAM WITH ALEF WITH HAMZA BELOW ISOLATED
    "\ufefa": "\u0644\u0627",  # ... FINAL FORM
    "\ufefb": "\u0644\u0627",  # ARABIC LIGATURE LAM WITH ALEF ISOLATED
    "\ufefc": "\u0644\u0627",  # ... FINAL FORM
}
_LAM_ALEF_RE = re.compile("[" + "".join(_LAM_ALEF.keys()) + "]")


def normalise_hamza(text: str) -> str:
    """Normalise hamza variants and lam-alef ligatures."""
    text = text.translate(_HAMZA_TABLE)
    text = _LAM_ALEF_RE.sub(lambda m: _LAM_ALEF[m.group(0)], text)
    return text


# ---------------------------------------------------------------------------
# WordToken dataclass
# ---------------------------------------------------------------------------

@dataclass
class WordToken:
    surah: int
    ayah: int
    position: int       # 0-based index within the ayah
    rasm: str           # stripped + normalised form (no harakat, normalised hamza)
    full_text: str      # original word as it appears in the reference


# ---------------------------------------------------------------------------
# tokenise_reference
# ---------------------------------------------------------------------------

def tokenise_reference(text: str, surah: int, ayah: int) -> List[WordToken]:
    """
    Split *text* by whitespace and return a list of :class:`WordToken`.
    Each token's *rasm* field is produced by applying
    :func:`strip_harakat` then :func:`normalise_hamza`.
    """
    words = text.split()
    tokens: List[WordToken] = []
    for idx, word in enumerate(words):
        rasm = normalise_hamza(strip_harakat(word))
        tokens.append(
            WordToken(
                surah=surah,
                ayah=ayah,
                position=idx,
                rasm=rasm,
                full_text=word,
            )
        )
    return tokens


# ---------------------------------------------------------------------------
# DiffResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiffResult:
    position: str            # "surah:ayah:word_index"
    reference_rasm: str
    hypothesis_rasm: str
    reference_full: str
    hypothesis_full: str
    rasm_match: bool
    harakat_match: bool
    error_type: str          # MATCH | SUBSTITUTION | DELETION | INSERTION | HARAKAT_ERROR


# ---------------------------------------------------------------------------
# two_pass_compare – Wagner-Fischer DP on word sequences
# ---------------------------------------------------------------------------

def _levenshtein_matrix(ref_words: List[str], hyp_words: List[str]):
    """Return the DP cost matrix (list-of-lists) for word-level Levenshtein."""
    m = len(ref_words)
    n = len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )
    return dp


def _traceback(dp, ref_words: List[str], hyp_words: List[str]):
    """
    Trace back the DP matrix to produce a list of (ref_idx|None, hyp_idx|None) pairs.
    """
    i, j = len(ref_words), len(hyp_words)
    alignment: List[tuple] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if ref_words[i - 1] == hyp_words[j - 1]:
                # match
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                # substitution
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                # deletion (ref word without hypothesis counterpart)
                alignment.append((i - 1, None))
                i -= 1
            else:
                # insertion (hypothesis word without reference counterpart)
                alignment.append((None, j - 1))
                j -= 1
        elif i > 0:
            alignment.append((i - 1, None))
            i -= 1
        else:
            alignment.append((None, j - 1))
            j -= 1

    alignment.reverse()
    return alignment


def two_pass_compare(
    hypothesis: str,
    reference: str,
    surah: int,
    ayah: int,
) -> List[DiffResult]:
    """
    Two-pass comparison of *hypothesis* against *reference*.

    Pass 1 – Wagner-Fischer DP on rasm (stripped + normalised) word forms.
    Pass 2 – For aligned MATCH pairs, compare full forms (including harakat).

    Returns a list of :class:`DiffResult` objects.
    """
    ref_tokens = tokenise_reference(reference, surah, ayah)
    hyp_words = hypothesis.split()

    # Compute rasm forms for hypothesis words
    hyp_rasms = [normalise_hamza(strip_harakat(w)) for w in hyp_words]
    ref_rasms = [t.rasm for t in ref_tokens]

    dp = _levenshtein_matrix(ref_rasms, hyp_rasms)
    alignment = _traceback(dp, ref_rasms, hyp_rasms)

    results: List[DiffResult] = []
    word_index = 0  # tracks reference-word index; advances only when ref_idx is not None

    for ref_idx, hyp_idx in alignment:
        if ref_idx is not None and hyp_idx is not None:
            # Both sides present – could be MATCH, HARAKAT_ERROR, or SUBSTITUTION
            position = f"{surah}:{ayah}:{word_index}"
            ref_tok = ref_tokens[ref_idx]
            hyp_word = hyp_words[hyp_idx]
            ref_rasm = ref_tok.rasm
            hyp_rasm = hyp_rasms[hyp_idx]
            rasm_match = ref_rasm == hyp_rasm

            if rasm_match:
                # Rasm matches – compare full forms (harakat)
                harakat_match = ref_tok.full_text == hyp_word
                error_type = "MATCH" if harakat_match else "HARAKAT_ERROR"
            else:
                harakat_match = False
                error_type = "SUBSTITUTION"

            results.append(
                DiffResult(
                    position=position,
                    reference_rasm=ref_rasm,
                    hypothesis_rasm=hyp_rasm,
                    reference_full=ref_tok.full_text,
                    hypothesis_full=hyp_word,
                    rasm_match=rasm_match,
                    harakat_match=harakat_match,
                    error_type=error_type,
                )
            )
            word_index += 1

        elif ref_idx is not None:
            # Reference word with no hypothesis counterpart → DELETION
            position = f"{surah}:{ayah}:{word_index}"
            ref_tok = ref_tokens[ref_idx]
            results.append(
                DiffResult(
                    position=position,
                    reference_rasm=ref_tok.rasm,
                    hypothesis_rasm="",
                    reference_full=ref_tok.full_text,
                    hypothesis_full="",
                    rasm_match=False,
                    harakat_match=False,
                    error_type="DELETION",
                )
            )
            word_index += 1

        else:
            # Hypothesis word with no reference counterpart → INSERTION.
            # Uses a distinct position key so it never aliases a reference-word position.
            # word_index does NOT advance — insertions have no reference counterpart.
            position = f"{surah}:{ayah}:INS{hyp_idx}"
            hyp_word = hyp_words[hyp_idx]
            results.append(
                DiffResult(
                    position=position,
                    reference_rasm="",
                    hypothesis_rasm=hyp_rasms[hyp_idx],
                    reference_full="",
                    hypothesis_full=hyp_word,
                    rasm_match=False,
                    harakat_match=False,
                    error_type="INSERTION",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def all_match(diffs: List[DiffResult]) -> bool:
    """Return True iff every diff has error_type == 'MATCH'."""
    return all(d.error_type == "MATCH" for d in diffs)
