"""FR phonetic class mapping — 32 classes for phonemizer IPA output.

Vowels (0-7) | Stops (8-15) | Fricatives+affricates (16-23) | Nasals/liquids/glides (24-31).
Unknown phonemes → DEFAULT_CLASS (31).
"""

from __future__ import annotations

PHONO_CLASSES: dict[str, int] = {
    # Vowels (0-7)
    "a": 0,
    "ɑ": 0,
    "æ": 0,
    "e": 1,
    "ɛ": 2,
    "i": 3,
    "ɪ": 3,
    "y": 3,
    "o": 4,
    "ɔ": 4,
    "u": 5,
    "ʊ": 5,
    "ø": 6,
    "œ": 6,
    "ə": 6,
    "ã": 7,
    "ɛ̃": 7,
    "ɔ̃": 7,
    "œ̃": 7,
    # Stops (8-15)
    "p": 8,
    "b": 9,
    "t": 10,
    "d": 11,
    "k": 12,
    "g": 13,
    "ʔ": 14,
    # Fricatives + affricates (16-23)
    "f": 16,
    "v": 17,
    "s": 18,
    "z": 19,
    "ʃ": 20,
    "ʒ": 21,
    "ʁ": 22,
    "h": 23,
    "χ": 22,
    # Nasals / liquids / glides (24-31)
    "m": 24,
    "n": 25,
    "ɲ": 26,
    "ŋ": 27,
    "l": 28,
    "j": 29,
    "w": 30,
    "ɥ": 31,
}

N_CLASSES = 32
DEFAULT_CLASS = 31  # fallback
