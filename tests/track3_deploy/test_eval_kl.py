"""Tests for EvalKL — KL-per-species, MAPE_Δ, hit@5."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.track3_deploy.eval.kl_species import (
    SHORT_TO_CANONICAL,
    SPECIES_CANONICAL,
    SPECIES_SHORT,
    hit_at_k_routing,
    kl_per_species,
    mape_delta,
)

N_SPECIES = 4
N_STACKS = 32
N_BATCH_UNIFORM = 10
N_BATCH_SMALL = 5
KL_TOL_ZERO = 1e-6
KL_TOL_POS = 0.01
MAPE_TARGET = 0.5
HIT_K = 5
N_ROUTING_BATCH = 20


def test_species_constants_match() -> None:
    assert len(SPECIES_SHORT) == N_SPECIES
    assert len(SPECIES_CANONICAL) == N_SPECIES
    for short, canonical in SHORT_TO_CANONICAL.items():
        assert canonical == f"{short}:code" or (short == "syntax" and canonical == "syntax:code")


def test_kl_zero_when_pred_equals_target() -> None:
    rho = np.full((N_BATCH_UNIFORM, N_SPECIES, N_STACKS), 1.0 / N_STACKS, dtype=np.float32)
    result = kl_per_species(rho, rho)
    assert abs(result["total"]) < KL_TOL_ZERO
    for s in SPECIES_SHORT:
        assert abs(result[s]) < KL_TOL_ZERO


def test_kl_positive_when_differ() -> None:
    target = np.full((N_BATCH_SMALL, N_SPECIES, N_STACKS), 1.0 / N_STACKS, dtype=np.float32)
    pred = target.copy()
    pred[:, 0, 0] += 0.5  # perturb species index 0 = "phono"
    pred /= pred.sum(axis=2, keepdims=True)
    result = kl_per_species(pred, target)
    assert result["phono"] > KL_TOL_POS
    assert result["total"] > 0.0


def test_mape_delta_formula() -> None:
    pred = np.array([[1.0, 2.0]])
    target = np.array([[1.0, 1.0]])
    # |pred - target|_1 / |target|_1 = (0 + 1) / (1 + 1) = 0.5
    assert abs(mape_delta(pred, target) - MAPE_TARGET) < KL_TOL_ZERO


def test_hit_at_5_perfect_agreement() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((N_ROUTING_BATCH, N_STACKS)).astype(np.float32)
    oracle = rng.standard_normal((N_ROUTING_BATCH, N_STACKS)).astype(np.float32)
    # pred == oracle → blend top-5 and oracle_blend top-5 likely intersect
    rate = hit_at_k_routing(base, bridge_pred=oracle, oracle=oracle, k=HIT_K)
    assert rate == 1.0


def test_hit_at_5_zero_when_disjoint() -> None:
    base = np.zeros((N_BATCH_SMALL, N_STACKS), dtype=np.float32)
    oracle = np.zeros((N_BATCH_SMALL, N_STACKS), dtype=np.float32)
    oracle[:, :HIT_K] = 1.0  # oracle top-5 = first 5 stacks
    bridge_pred = np.zeros((N_BATCH_SMALL, N_STACKS), dtype=np.float32)
    bridge_pred[:, -HIT_K:] = 100.0  # bridge pushes blend toward last 5
    rate = hit_at_k_routing(base, bridge_pred=bridge_pred, oracle=oracle, k=HIT_K)
    assert rate == 0.0
