"""
phonemizer.py - Arabic grapheme-to-phoneme mapping with comprehensive Tajweed rule
detection, following the Quranic-Phonemizer reference
(https://github.com/Hetchy/Quranic-Phonemizer).

Supports context-sensitive phonemization including cross-word tajweed rules
(Idgham, Madd Jaiz Munfasil, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Phoneme map  (29 entries)
# ---------------------------------------------------------------------------

ARABIC_PHONEME_MAP: dict[str, str] = {
    "ب": "/b/",
    "ت": "/t/",
    "ث": "/θ/",
    "ج": "/dʒ/",
    "ح": "/ħ/",
    "خ": "/x/",
    "د": "/d/",
    "ذ": "/ð/",
    "ر": "/r/",
    "ز": "/z/",
    "س": "/s/",
    "ش": "/ʃ/",
    "ص": "/sˤ/",
    "ض": "/dˤ/",
    "ط": "/tˤ/",
    "ظ": "/ðˤ/",
    "ع": "/ʕ/",
    "غ": "/ɣ/",
    "ف": "/f/",
    "ق": "/q/",
    "ك": "/k/",
    "ل": "/l/",
    "م": "/m/",
    "ن": "/n/",
    "ه": "/h/",
    "و": "/w/",
    "ي": "/j/",
    "ا": "/aː/",
    "ء": "/ʔ/",
}

# Short-vowel diacritics → phoneme insertions
_SHORT_VOWELS = {
    "\u064e": "/a/",   # fatha
    "\u064f": "/u/",   # damma
    "\u0650": "/i/",   # kasra
    "\u064b": "/an/",  # fathatan
    "\u064c": "/un/",  # dammatan
    "\u064d": "/in/",  # kasratan
}

# Long-vowel letters that can trigger Madd
_LONG_VOWEL_LETTERS = {"ا", "و", "ي"}

# Shadda (gemination mark)
_SHADDA = "\u0651"

# Sukun
_SUKUN = "\u0652"

# ---------------------------------------------------------------------------
# Tajweed letter groups
# ---------------------------------------------------------------------------

# Throat letters for Izhar
_THROAT_LETTERS = {"ء", "ه", "ع", "ح", "غ", "خ"}

# Letters for Idgham with Ghunna (YNMW)
_IDGHAM_GHUNNA_LETTERS = {"ي", "ن", "م", "و"}

# Letters for Idgham without Ghunna
_IDGHAM_NO_GHUNNA_LETTERS = {"ل", "ر"}

# Iqlab: noon sakinah/tanween before ba
_IQLAB_LETTER = "ب"

# Ikhfa letters (15 remaining letters)
_IKHFA_LETTERS = {"ت", "ث", "ج", "د", "ذ", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ف", "ق", "ك"}

# Qalqala letters
_QALQALA_LETTERS = {"ق", "ط", "ب", "ج", "د"}

# Sun letters (for Lam Shamsiyah)
_SUN_LETTERS = {"ت", "ث", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ن", "ل"}

# Tanween diacritics
_TANWEEN = {"\u064b", "\u064c", "\u064d"}

# All harakat (short vowels + tanween)
_ALL_HARAKAT = set(_SHORT_VOWELS.keys())


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PhonemeToken:
    word_position: str          # e.g. "1:1:0"
    grapheme: str               # the word in its original form
    phonemes: List[str]         # list of IPA phoneme strings
    tajweed_rules: Optional[List[str]] = None   # e.g. ["MEDD_TABII", "IDGHAM_GHUNNA"]


@dataclass
class PhoneDiff:
    position: str
    reference_phones: List[str]
    hypothesis_phones: List[str]
    edit_distance: int
    diff_type: str   # MATCH | PHONE_SUB | PHONE_DEL | PHONE_INS | TAJWEED_*


# ---------------------------------------------------------------------------
# Helper: get next meaningful char (skip diacritics)
# ---------------------------------------------------------------------------

def _next_letter(word: str, idx: int) -> Optional[str]:
    """Return the next base letter after index *idx*, skipping diacritics."""
    for i in range(idx + 1, len(word)):
        ch = word[i]
        if ch not in _ALL_HARAKAT and ch != _SHADDA and ch != _SUKUN and ch != "\u0640":
            if ch in ARABIC_PHONEME_MAP or ch in _LONG_VOWEL_LETTERS:
                return ch
    return None


def _has_sukun_after(word: str, idx: int) -> bool:
    """Check if the character at *idx* is followed by sukun."""
    for i in range(idx + 1, len(word)):
        ch = word[i]
        if ch == _SUKUN:
            return True
        if ch in ARABIC_PHONEME_MAP or ch in _LONG_VOWEL_LETTERS:
            return False
    return False


def _has_shadda_after(word: str, idx: int) -> bool:
    """Check if the character at *idx* is followed by shadda."""
    for i in range(idx + 1, len(word)):
        ch = word[i]
        if ch == _SHADDA:
            return True
        if ch in ARABIC_PHONEME_MAP or ch in _LONG_VOWEL_LETTERS:
            return False
    return False


def _is_noon_sakinah(word: str, idx: int) -> bool:
    """Check if noon at *idx* is sakinah (has sukun or no vowel after it)."""
    if word[idx] != "ن":
        return False
    for i in range(idx + 1, len(word)):
        ch = word[i]
        if ch == _SUKUN:
            return True
        if ch in _ALL_HARAKAT:
            return False  # noon has a vowel, not sakinah
        if ch == _SHADDA:
            return False
        if ch in ARABIC_PHONEME_MAP or ch in _LONG_VOWEL_LETTERS:
            return True  # no vowel found between noon and next letter = sakinah
    return True  # end of word = sakinah


# ---------------------------------------------------------------------------
# phonemize_word (context-sensitive)
# ---------------------------------------------------------------------------

def _phonemize_word(
    word: str,
    next_word_first_letter: Optional[str] = None,
    is_last_word: bool = False,
) -> tuple[List[str], List[str]]:
    """
    Convert a single Arabic word into a list of phoneme strings with
    context-sensitive tajweed rule detection.

    Returns (phonemes, tajweed_rules).
    """
    phonemes: List[str] = []
    tajweed_rules: List[str] = []
    has_long_vowel = False
    chars = list(word)
    length = len(chars)

    i = 0
    while i < length:
        ch = chars[i]

        # --- Shadda: gemination ---
        if ch == _SHADDA:
            if phonemes:
                # Double the last consonant phoneme
                phonemes.append(phonemes[-1])
                if "GHUNNA" not in tajweed_rules:
                    if i > 0 and chars[i - 1] in ("ن", "م"):
                        # Noon/Meem with shadda -> Ghunna
                        tajweed_rules.append("GHUNNA")
            i += 1
            continue

        # --- Sukun ---
        if ch == _SUKUN:
            # Check for Qalqala
            if i > 0 and chars[i - 1] in _QALQALA_LETTERS:
                if is_last_word and i == length - 1:
                    tajweed_rules.append("QALQALA_KUBRA")
                else:
                    tajweed_rules.append("QALQALA_SUGHRA")
            i += 1
            continue

        # --- Short vowel diacritic ---
        if ch in _SHORT_VOWELS:
            phonemes.append(_SHORT_VOWELS[ch])

            # Tanween + next word starts with certain letters
            if ch in _TANWEEN and next_word_first_letter:
                if next_word_first_letter in _THROAT_LETTERS:
                    tajweed_rules.append("IZHAR")
                elif next_word_first_letter in _IDGHAM_GHUNNA_LETTERS:
                    tajweed_rules.append("IDGHAM_GHUNNA")
                elif next_word_first_letter in _IDGHAM_NO_GHUNNA_LETTERS:
                    tajweed_rules.append("IDGHAM_NO_GHUNNA")
                elif next_word_first_letter == _IQLAB_LETTER:
                    tajweed_rules.append("IQLAB")
                elif next_word_first_letter in _IKHFA_LETTERS:
                    tajweed_rules.append("IKHFA")

            i += 1
            continue

        # --- Tatweel: skip ---
        if ch == "\u0640":
            i += 1
            continue

        # --- Skip other diacritical marks ---
        if ch in "\u0610\u0611\u0612\u0613\u0614\u0615\u0616\u0617\u0618\u0619\u061a":
            i += 1
            continue
        if "\u06d6" <= ch <= "\u06ed":
            i += 1
            continue
        if ch == "\u0670":  # superscript alef
            phonemes.append("/aː/")
            has_long_vowel = True
            i += 1
            continue

        # --- Noon Sakinah rules (within word) ---
        if ch == "ن" and _is_noon_sakinah(word, i):
            phonemes.append(ARABIC_PHONEME_MAP[ch])
            next_let = _next_letter(word, i)
            target_letter = next_let if next_let else next_word_first_letter

            if target_letter:
                if target_letter in _THROAT_LETTERS:
                    tajweed_rules.append("IZHAR")
                elif target_letter in _IDGHAM_GHUNNA_LETTERS:
                    tajweed_rules.append("IDGHAM_GHUNNA")
                elif target_letter in _IDGHAM_NO_GHUNNA_LETTERS:
                    tajweed_rules.append("IDGHAM_NO_GHUNNA")
                elif target_letter == _IQLAB_LETTER:
                    tajweed_rules.append("IQLAB")
                    phonemes[-1] = "/m/"  # noon becomes meem
                elif target_letter in _IKHFA_LETTERS:
                    tajweed_rules.append("IKHFA")
                    phonemes[-1] = "/ŋ/"  # nasalised noon

            i += 1
            continue

        # --- Meem Sakinah rules ---
        if ch == "م" and _has_sukun_after(word, i):
            phonemes.append(ARABIC_PHONEME_MAP[ch])
            next_let = _next_letter(word, i)
            target_letter = next_let if next_let else next_word_first_letter

            if target_letter:
                if target_letter == "ب":
                    tajweed_rules.append("IKHFA_SHAFAWI")
                elif target_letter == "م":
                    tajweed_rules.append("IDGHAM_SHAFAWI")
                else:
                    tajweed_rules.append("IZHAR_SHAFAWI")
            i += 1
            continue

        # --- Lam in al- prefix ---
        if ch == "ل" and i == 1 and length > 2 and chars[0] == "ا":
            phonemes.append(ARABIC_PHONEME_MAP[ch])
            next_let = _next_letter(word, i)
            if next_let and next_let in _SUN_LETTERS:
                tajweed_rules.append("LAM_SHAMSIYAH")
            else:
                tajweed_rules.append("LAM_QAMARIYAH")
            i += 1
            continue

        # --- Long vowel letter ---
        if ch in _LONG_VOWEL_LETTERS:
            ph = ARABIC_PHONEME_MAP.get(ch, f"/{ch}/")
            phonemes.append(ph)
            has_long_vowel = True

            # Madd type detection
            next_let = _next_letter(word, i)
            if next_let == "ء" or next_let == "\u0621":
                # Hamza after long vowel in same word -> Madd Wajib Muttasil
                tajweed_rules.append("MEDD_WAJIB_MUTTASIL")
            elif next_let is None and next_word_first_letter:
                if next_word_first_letter == "ء" or next_word_first_letter == "\u0621":
                    # Long vowel at word end + hamza at next word start
                    tajweed_rules.append("MEDD_JAIZ_MUNFASIL")
                else:
                    tajweed_rules.append("MEDD_TABII")
            elif next_let and _has_sukun_after(word, i):
                tajweed_rules.append("MEDD_LAZIM")
            elif next_let and _has_shadda_after(word, i):
                tajweed_rules.append("MEDD_LAZIM")
            else:
                if has_long_vowel:
                    tajweed_rules.append("MEDD_TABII")

            i += 1
            continue

        # --- Qalqala at end of word (no sukun but word-final) ---
        if ch in _QALQALA_LETTERS and i == length - 1:
            phonemes.append(ARABIC_PHONEME_MAP.get(ch, f"/{ch}/"))
            if is_last_word:
                tajweed_rules.append("QALQALA_KUBRA")
            else:
                tajweed_rules.append("QALQALA_SUGHRA")
            i += 1
            continue

        # --- Regular consonant / hamza ---
        if ch in ARABIC_PHONEME_MAP:
            phonemes.append(ARABIC_PHONEME_MAP[ch])

        i += 1

    # Deduplicate tajweed rules while preserving order
    seen = set()
    unique_rules: List[str] = []
    for rule in tajweed_rules:
        if rule not in seen:
            seen.add(rule)
            unique_rules.append(rule)

    return phonemes, unique_rules if unique_rules else []


# ---------------------------------------------------------------------------
# phonemize_verse (cross-word context)
# ---------------------------------------------------------------------------

def phonemize_verse(surah: int, ayah: int, text: str) -> List[PhonemeToken]:
    """
    Phonemize every word in *text* and return one :class:`PhonemeToken` per word.

    Cross-word tajweed rules (Idgham across word boundaries, Madd Jaiz Munfasil)
    are handled by passing the first letter of the next word.
    """
    words = text.split()
    tokens: List[PhonemeToken] = []

    for idx, word in enumerate(words):
        position = f"{surah}:{ayah}:{idx}"
        is_last = (idx == len(words) - 1)

        # Get first letter of next word for cross-word rules
        next_word_first = None
        if idx + 1 < len(words):
            nw = words[idx + 1]
            for ch in nw:
                if ch in ARABIC_PHONEME_MAP or ch in _LONG_VOWEL_LETTERS:
                    next_word_first = ch
                    break

        phones, rules = _phonemize_word(word, next_word_first, is_last)
        tokens.append(
            PhonemeToken(
                word_position=position,
                grapheme=word,
                phonemes=phones,
                tajweed_rules=rules if rules else None,
            )
        )
    return tokens


# ---------------------------------------------------------------------------
# phonemize_aligned_pair
# ---------------------------------------------------------------------------

def phonemize_aligned_pair(
    diff,
    surah: int,
    ayah: int,
) -> tuple[PhonemeToken, PhonemeToken]:
    """
    Produce reference and hypothesis PhonemeTokens from an aligned DiffResult.

    This preserves the alignment relationship for downstream phone_diff
    computation, ensuring both sides are phonemized consistently.
    """
    position = diff.position
    ref_text = diff.reference_full if diff.reference_full else ""
    hyp_text = diff.hypothesis_full if diff.hypothesis_full else ""

    ref_phones, ref_rules = _phonemize_word(ref_text) if ref_text else ([], [])
    hyp_phones, hyp_rules = _phonemize_word(hyp_text) if hyp_text else ([], [])

    ref_token = PhonemeToken(
        word_position=position,
        grapheme=ref_text,
        phonemes=ref_phones,
        tajweed_rules=ref_rules if ref_rules else None,
    )
    hyp_token = PhonemeToken(
        word_position=position,
        grapheme=hyp_text,
        phonemes=hyp_phones,
        tajweed_rules=hyp_rules if hyp_rules else None,
    )
    return ref_token, hyp_token


# ---------------------------------------------------------------------------
# Phoneme-level diff
# ---------------------------------------------------------------------------

def _phone_levenshtein(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein distance between two phoneme sequences."""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[n]


def _classify_tajweed_diff(tajweed_rules: Optional[List[str]]) -> str:
    """Map tajweed rules to the most specific diff type."""
    if not tajweed_rules:
        return "PHONE_SUB"

    # Priority: most specific rules first
    priority = [
        "MEDD_WAJIB_MUTTASIL", "MEDD_LAZIM", "MEDD_JAIZ_MUNFASIL",
        "MEDD_TABII", "MEDD_ARID_LISSUKUN",
        "IDGHAM_GHUNNA", "IDGHAM_NO_GHUNNA", "IDGHAM_SHAFAWI",
        "IKHFA", "IKHFA_SHAFAWI",
        "IQLAB", "IZHAR", "IZHAR_SHAFAWI",
        "QALQALA_KUBRA", "QALQALA_SUGHRA",
        "GHUNNA",
        "LAM_SHAMSIYAH", "LAM_QAMARIYAH",
        "TAFKHEEM",
    ]

    for rule in priority:
        if rule in tajweed_rules:
            # Map to the general TAJWEED_ category
            if "MEDD" in rule:
                return "TAJWEED_MEDD"
            if "IDGHAM" in rule:
                return "TAJWEED_IDGHAM"
            if "IKHFA" in rule:
                return "TAJWEED_IKHFA"
            if rule == "IQLAB":
                return "TAJWEED_IQLAB"
            if "IZHAR" in rule:
                return "TAJWEED_IZHAR"
            if "QALQALA" in rule:
                return "TAJWEED_QALQALA"
            if rule == "GHUNNA":
                return "TAJWEED_GHUNNA"
            if rule == "TAFKHEEM":
                return "TAJWEED_TAFKHEEM"
            if "LAM" in rule:
                return "TAJWEED_IDGHAM"  # Lam Shamsiyah is a form of assimilation

    return "PHONE_SUB"


def compute_phone_diff(
    ref_token: PhonemeToken,
    hyp_phones: List[str],
) -> PhoneDiff:
    """
    Compare *ref_token* phones against *hyp_phones* and return a :class:`PhoneDiff`.

    Diff types
    ----------
    - MATCH         : edit_distance == 0
    - PHONE_SUB     : same length, all substitutions
    - PHONE_DEL     : hypothesis shorter
    - PHONE_INS     : hypothesis longer
    - TAJWEED_*     : ref_token has tajweed rule(s) and distance > 0
    """
    ref_phones = ref_token.phonemes
    dist = _phone_levenshtein(ref_phones, hyp_phones)

    if dist == 0:
        diff_type = "MATCH"
    elif ref_token.tajweed_rules and dist > 0:
        diff_type = _classify_tajweed_diff(ref_token.tajweed_rules)
    elif len(hyp_phones) < len(ref_phones):
        diff_type = "PHONE_DEL"
    elif len(hyp_phones) > len(ref_phones):
        diff_type = "PHONE_INS"
    else:
        diff_type = "PHONE_SUB"

    return PhoneDiff(
        position=ref_token.word_position,
        reference_phones=ref_phones,
        hypothesis_phones=hyp_phones,
        edit_distance=dist,
        diff_type=diff_type,
    )


# ---------------------------------------------------------------------------
# phonemize_and_diff
# ---------------------------------------------------------------------------

def phonemize_and_diff(
    ref_tokens: List[PhonemeToken],
    hyp_phones_per_word: List[List[str]],
) -> List[PhoneDiff]:
    """
    Zip *ref_tokens* with *hyp_phones_per_word* and compute per-word PhoneDiff.

    If lengths differ the shorter side is padded with empty lists / tokens.
    """
    diffs: List[PhoneDiff] = []
    max_len = max(len(ref_tokens), len(hyp_phones_per_word))

    for i in range(max_len):
        if i < len(ref_tokens) and i < len(hyp_phones_per_word):
            diffs.append(compute_phone_diff(ref_tokens[i], hyp_phones_per_word[i]))
        elif i < len(ref_tokens):
            # Reference word with no hypothesis phones → PHONE_DEL
            ref_tok = ref_tokens[i]
            diffs.append(
                PhoneDiff(
                    position=ref_tok.word_position,
                    reference_phones=ref_tok.phonemes,
                    hypothesis_phones=[],
                    edit_distance=len(ref_tok.phonemes),
                    diff_type="PHONE_DEL",
                )
            )
        else:
            # Extra hypothesis phones with no reference → PHONE_INS
            hyp = hyp_phones_per_word[i]
            diffs.append(
                PhoneDiff(
                    position=f"?:?:{i}",
                    reference_phones=[],
                    hypothesis_phones=hyp,
                    edit_distance=len(hyp),
                    diff_type="PHONE_INS",
                )
            )

    return diffs
