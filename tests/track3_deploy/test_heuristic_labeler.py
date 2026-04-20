"""Tests for HeuristicLabeler."""

from __future__ import annotations

import numpy as np
import pytest

spacy = pytest.importorskip("spacy")
phonemizer = pytest.importorskip("phonemizer")

from kiki_flow_core.track3_deploy.data.heuristic_labeler import HeuristicLabeler  # noqa: E402

N_STACKS = 32
SIMPLEX_SUM_TOL = 1e-5
UNIFORM_TOL = 1e-5
SPECIES_KEYS = {"phono:code", "sem:code", "lex:code", "syntax:code"}


@pytest.fixture(scope="module")
def labeler() -> HeuristicLabeler:
    """Shared labeler instance — SpaCy load is expensive."""
    return HeuristicLabeler()


def test_label_output_structure(labeler: HeuristicLabeler) -> None:
    out = labeler.label("Bonjour le monde.")
    assert set(out.keys()) == SPECIES_KEYS
    for sp, vec in out.items():
        assert vec.shape == (N_STACKS,), f"{sp} shape {vec.shape}"
        assert vec.dtype == np.float32


def test_simplex_constraint(labeler: HeuristicLabeler) -> None:
    out = labeler.label("Voici une phrase française de test.")
    for sp, vec in out.items():
        assert abs(vec.sum() - 1.0) < SIMPLEX_SUM_TOL, f"{sp} sum {vec.sum()}"
        assert (vec >= 0).all()


def test_determinism(labeler: HeuristicLabeler) -> None:
    q = "Une phrase déterministe."
    a = labeler.label(q)
    b = labeler.label(q)
    for sp in SPECIES_KEYS:
        np.testing.assert_array_equal(a[sp], b[sp])


def test_different_queries_different_labels(labeler: HeuristicLabeler) -> None:
    a = labeler.label("mot")
    b = labeler.label("Les chercheurs étudient la cognition humaine.")
    differs = any(not np.allclose(a[sp], b[sp]) for sp in SPECIES_KEYS)
    assert differs


def test_empty_query_returns_uniform(labeler: HeuristicLabeler) -> None:
    out = labeler.label("")
    for _sp, vec in out.items():
        np.testing.assert_allclose(vec, np.ones(N_STACKS) / N_STACKS, atol=UNIFORM_TOL)
