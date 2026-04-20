"""Tests for CorpusBuilder — assemble + dedup + stratified split."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.track3_deploy.data.corpus_builder import (
    CorpusBuilder,
    CorpusEntry,
)

DEDUP_THRESHOLD = 0.92
EXPECTED_TOTAL = 450
TRAIN_LO, TRAIN_HI = 0.78, 0.82
VAL_LO, VAL_HI = 0.08, 0.12
TEST_LO, TEST_HI = 0.08, 0.12
EXPECTED_STRATA_COUNT = 3


def _entries(source: str, species: str, n: int, prefix: str = "q") -> list[CorpusEntry]:
    return [
        CorpusEntry(text=f"{prefix}_{source}_{i}", source=source, species=species) for i in range(n)
    ]


def test_exact_dedup() -> None:
    builder = CorpusBuilder(dedup_threshold=DEDUP_THRESHOLD)
    entries = [CorpusEntry(text="bonjour", source="B", species="phono")] * 3
    out = builder.dedup(entries)
    assert len(out) == 1


def test_dedup_by_embeddings_cross_source() -> None:
    """With crafted near-duplicate embeddings, the lower-priority source is dropped."""
    builder = CorpusBuilder(dedup_threshold=DEDUP_THRESHOLD)
    e1 = CorpusEntry(text="Bonjour, le monde", source="B", species="phono")
    e2 = CorpusEntry(text="bonjour le monde", source="D", species="phono")
    embs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.001, 0.0],  # cosine ~1 → above threshold
        ],
        dtype=np.float32,
    )
    out = builder.dedup_by_embeddings([e1, e2], embs)
    assert len(out) == 1
    assert out[0].source == "B"  # B kept, D dropped (cross-source rule)


def test_dedup_exact_only_when_no_embedder(caplog) -> None:
    """Without an embedder, only exact dedup runs, and a warning is logged."""
    builder = CorpusBuilder(dedup_threshold=DEDUP_THRESHOLD, embedder=None)
    entries = [
        CorpusEntry(text="hello", source="B", species="phono"),
        CorpusEntry(text="HELLO", source="D", species="phono"),  # exact after normalize
        CorpusEntry(text="different", source="C", species="sem"),
    ]
    with caplog.at_level("WARNING"):
        out = builder.dedup(entries)
    expected_count = 2  # "hello" (normalized) and "different"
    assert len(out) == expected_count
    assert any("no embedder" in msg for msg in caplog.messages)


def test_stratified_split_ratios() -> None:
    builder = CorpusBuilder(dedup_threshold=DEDUP_THRESHOLD)
    entries = (
        _entries("B", "phono", 100)
        + _entries("C", "sem", 200)
        + _entries("D", "lex", 150, prefix="qD")
    )
    splits = builder.split(entries, ratios=(0.8, 0.1, 0.1), seed=0)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == EXPECTED_TOTAL
    assert TRAIN_LO <= len(splits["train"]) / total <= TRAIN_HI
    assert VAL_LO <= len(splits["val"]) / total <= VAL_HI
    assert TEST_LO <= len(splits["test"]) / total <= TEST_HI


def test_stratification_preserves_source_species() -> None:
    builder = CorpusBuilder(dedup_threshold=DEDUP_THRESHOLD)
    entries = _entries("B", "phono", 100) + _entries("C", "sem", 100) + _entries("D", "lex", 100)
    splits = builder.split(entries, ratios=(0.8, 0.1, 0.1), seed=0)
    # each split must contain all 3 (source, species) tuples
    for name, s in splits.items():
        pairs = {(e.source, e.species) for e in s}
        assert len(pairs) == EXPECTED_STRATA_COUNT, f"{name} missing strata: {pairs}"


def test_frozen_test_split_reproducible() -> None:
    """Same entries + same seed → identical test split (for corpus_v1_test tag)."""
    builder = CorpusBuilder(dedup_threshold=DEDUP_THRESHOLD)
    entries = _entries("B", "phono", 100) + _entries("C", "sem", 100)
    s1 = builder.split(entries, ratios=(0.8, 0.1, 0.1), seed=42)
    s2 = builder.split(entries, ratios=(0.8, 0.1, 0.1), seed=42)
    assert [e.text for e in s1["test"]] == [e.text for e in s2["test"]]
