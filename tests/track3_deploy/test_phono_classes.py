"""Tests for PHONO_CLASSES mapping."""

from __future__ import annotations

from kiki_flow_core.track3_deploy.data.phono_classes import (
    DEFAULT_CLASS,
    N_CLASSES,
    PHONO_CLASSES,
)

N_CLASSES_EXPECTED = 32
MIN_USED_SLOTS = 20


def test_n_classes_is_32() -> None:
    assert N_CLASSES == N_CLASSES_EXPECTED
    assert DEFAULT_CLASS < N_CLASSES


def test_classes_within_bounds() -> None:
    for phoneme, cls in PHONO_CLASSES.items():
        assert 0 <= cls < N_CLASSES, f"{phoneme} -> {cls} out of bounds"


def test_covers_most_slots() -> None:
    used = set(PHONO_CLASSES.values())
    assert len(used) >= MIN_USED_SLOTS


def test_common_fr_phonemes_mapped() -> None:
    for p in ("a", "i", "u", "p", "t", "k", "s", "ʁ", "m", "l"):
        assert p in PHONO_CLASSES
