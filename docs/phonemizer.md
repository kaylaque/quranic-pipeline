# Phonemizer -- Grapheme-to-Phoneme Mapping & Tajweed Rules

## Overview

The phonemizer module (`src/phonemizer.py`) converts Arabic text into IPA phoneme sequences with comprehensive tajweed rule detection. The design follows the [Quranic-Phonemizer](https://github.com/Hetchy/Quranic-Phonemizer) reference library for tajweed rule coverage while maintaining a self-contained implementation.

---

## Phoneme Inventory

### Consonants (29 base entries)

| Arabic | IPA     | Name    | Arabic | IPA     | Name      |
|--------|---------|---------|--------|---------|-----------|
| ب      | /b/     | Ba      | ص      | /sˤ/    | Sad       |
| ت      | /t/     | Ta      | ض      | /dˤ/    | Dad       |
| ث      | /θ/     | Tha     | ط      | /tˤ/    | Tta       |
| ج      | /dʒ/    | Jim     | ظ      | /ðˤ/    | Dha       |
| ح      | /ħ/     | Hha     | ع      | /ʕ/     | Ain       |
| خ      | /x/     | Kha     | غ      | /ɣ/     | Ghain     |
| د      | /d/     | Dal     | ف      | /f/     | Fa        |
| ذ      | /ð/     | Dhal    | ق      | /q/     | Qaf       |
| ر      | /r/     | Ra      | ك      | /k/     | Kaf       |
| ز      | /z/     | Zay     | ل      | /l/     | Lam       |
| س      | /s/     | Sin     | م      | /m/     | Mim       |
| ش      | /ʃ/     | Shin    | ن      | /n/     | Nun       |
| ه      | /h/     | Ha      | و      | /w/     | Waw       |
| ي      | /j/     | Ya      | ا      | /aː/    | Alef      |
| ء      | /ʔ/     | Hamza   |        |         |           |

### Vowels

| Diacritic      | Unicode  | IPA   | Name       |
|----------------|----------|-------|------------|
| فَتْحَة (fatha)   | U+064E   | /a/   | Short a    |
| ضَمَّة (damma)    | U+064F   | /u/   | Short u    |
| كَسْرَة (kasra)   | U+0650   | /i/   | Short i    |
| فَتْحَتَان (fathatan) | U+064B | /an/  | Tanween a  |
| ضَمَّتَان (dammatan) | U+064C | /un/  | Tanween u  |
| كَسْرَتَان (kasratan) | U+064D | /in/  | Tanween i  |

### Long Vowels

| Letter after haraka | IPA    | Duration      |
|---------------------|--------|---------------|
| ا (alef)            | /aː/   | 2 morae (natural Madd) |
| و (waw)             | /uː/   | 2 morae       |
| ي (ya)              | /iː/   | 2 morae       |

### Tajweed-Specific Phonemes

| Rule            | Phoneme modification         | Example               |
|-----------------|-----------------------------|-----------------------|
| Gemination (shadda) | Doubled consonant: `/bb/`, `/tt/` | رَبِّ → `/r/ /a/ /bb/ /i/` |
| Idgham          | Assimilated nasal: `/ñ/`, `/m̃/` | مَن يَعمَل → idgham with ghunna |
| Iqlab           | Noon → meem: `/m/`           | مِن بَعد → `/m/ /i/ /m/ ...` |
| Ikhfa           | Nasalised noon: `/ŋ/`       | مَن كَان → hidden noon |
| Qalqala         | Echoed release: `Q` / `QQ`  | حَقّ → qalqala kubra   |
| Tafkheem        | Emphasised: `/lˤ/`, `/rˤ/`  | اللَّه → heavy lam     |

---

## Tajweed Rules Detected

### Noon Sakinah / Tanween Rules

| Rule    | Trigger                                        | Following Letters                     |
|---------|------------------------------------------------|---------------------------------------|
| Izhar   | Noon sakinah/tanween + throat letter            | ء ه ع ح غ خ                          |
| Idgham with Ghunna | Noon sakinah/tanween + ي ن م و       | ي ن م و (YNMW)                       |
| Idgham without Ghunna | Noon sakinah/tanween + ل ر          | ل ر                                   |
| Iqlab   | Noon sakinah/tanween + ب                       | ب only                                |
| Ikhfa   | Noon sakinah/tanween + remaining 15 letters     | ت ث ج د ذ ز س ش ص ض ط ظ ف ق ك      |

### Meem Sakinah Rules

| Rule           | Trigger                          |
|----------------|----------------------------------|
| Ikhfa Shafawi  | Meem sakinah + ب                 |
| Idgham Shafawi | Meem sakinah + م                 |
| Izhar Shafawi  | Meem sakinah + any other letter  |

### Madd (Elongation) Rules

| Type                | Duration    | Trigger                                         |
|---------------------|-------------|------------------------------------------------|
| Madd Tabii (Natural)| 2 morae     | Long vowel letter with no hamza/sukun after     |
| Madd Wajib Muttasil | 4-5 morae   | Long vowel + hamza in same word                 |
| Madd Jaiz Munfasil  | 2-4-6 morae | Long vowel at word end + hamza at next word start |
| Madd Lazim          | 6 morae     | Long vowel + sukun/shadda in same word          |
| Madd Arid Lissukun  | 2-4-6 morae | Long vowel before last letter when stopping     |
| Madd Leen           | 2-4-6 morae | Waw/Ya sakinah after fatha before last letter   |

### Other Rules

| Rule             | Description                                           |
|------------------|------------------------------------------------------|
| Qalqala Sughra   | Echoed release on ق ط ب ج د when sakinah mid-word   |
| Qalqala Kubra    | Strong echoed release at end of word/verse            |
| Ghunna           | Nasal sound on noon/meem with shadda (2 morae)       |
| Lam Shamsiyah    | Lam in "al-" assimilates into sun letter             |
| Lam Qamariyah    | Lam in "al-" pronounced before moon letter           |
| Tafkheem         | Heavy/emphatic pronunciation of specific letters     |

---

## Algorithm

### Context-Sensitive Phonemization

Unlike simple character-by-character mapping, the phonemizer uses context windows:

```python
def _phonemize_word(word: str) -> tuple[List[str], List[str]]:
    """
    Scans characters with lookahead/lookbehind for:
    1. Shadda (U+0651) -> gemination (double the preceding consonant)
    2. Sukun (U+0652) -> marks consonant as sakinah (triggers Qalqala, Noon rules)
    3. Noon sakinah/tanween + next letter -> Idgham/Ikhfa/Iqlab/Izhar
    4. Meem sakinah + next letter -> Shafawi rules
    5. Long vowel + following context -> Madd type classification
    6. Lam in al- prefix + next consonant -> Shamsiyah/Qamariyah
    """
```

### Cross-Word Rules

Some tajweed rules span word boundaries (e.g., Idgham, Madd Jaiz Munfasil). The `phonemize_verse()` function processes all words together and applies cross-word lookups:

```python
def phonemize_verse(surah, ayah, text) -> List[PhonemeToken]:
    words = text.split()
    tokens = []
    for idx, word in enumerate(words):
        next_word = words[idx + 1] if idx + 1 < len(words) else None
        phones, tajweed_rules = _phonemize_word(word, next_word_first_letter, is_last)
        tokens.append(PhonemeToken(..., tajweed_rules=tajweed_rules))
    return tokens
```

---

## Data Structures

### PhonemeToken

| Field          | Type               | Description                                |
|----------------|--------------------|--------------------------------------------|
| word_position  | str                | `"surah:ayah:word_index"`                  |
| grapheme       | str                | Original word with diacritics              |
| phonemes       | List[str]          | IPA phoneme sequence                       |
| tajweed_rules  | Optional[List[str]]| List of detected tajweed rule names        |

### PhoneDiff

| Field             | Type       | Description                              |
|-------------------|------------|------------------------------------------|
| position          | str        | `"surah:ayah:word_index"`                |
| reference_phones  | List[str]  | Expected phoneme sequence                |
| hypothesis_phones | List[str]  | Actual phoneme sequence (from ASR/aligner) |
| edit_distance     | int        | Levenshtein distance between sequences   |
| diff_type         | str        | MATCH, PHONE_SUB, PHONE_DEL, PHONE_INS, TAJWEED_* |

---

## Input/Output Contract

**Input:** Aligned `DiffResult` pairs from `two_pass_compare()` -- both reference and hypothesis text with full diacritics.

**Output:** `List[PhonemeToken]` for reference, `List[PhonemeToken]` for hypothesis, and `List[PhoneDiff]` computed via per-word Levenshtein on phoneme sequences.

### `phonemize_aligned_pair(diff, surah, ayah) -> (PhonemeToken, PhonemeToken)`

Produces reference and hypothesis PhonemeTokens from an aligned DiffResult, preserving the alignment relationship for downstream `compute_phone_diff()`.

---

## Phoneme-Level Diff Algorithm

Uses Levenshtein distance on phoneme lists (not character strings):

```
GOP(p) = edit_distance(ref_phonemes, hyp_phonemes) / max(len(ref), len(hyp))
```

Diff type classification:

| Condition                              | diff_type       |
|----------------------------------------|-----------------|
| edit_distance == 0                     | MATCH           |
| tajweed_rule contains MEDD + dist > 0  | TAJWEED_MEDD    |
| any tajweed_rule + dist > 0            | TAJWEED_{rule}  |
| hyp shorter than ref                   | PHONE_DEL       |
| hyp longer than ref                    | PHONE_INS       |
| same length, different phones          | PHONE_SUB       |
