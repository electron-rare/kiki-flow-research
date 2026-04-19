"""Tests for EncoderB_DistilledMiniLM."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.track3_deploy.encoders import ENCODER_REGISTRY
from kiki_flow_core.track3_deploy.encoders.distilled import EncoderB_DistilledMiniLM

EXPECTED_OUTPUT_DIM = 384
EXPECTED_BATCH = 2
PARAM_COUNT_MIN = 1_500_000
PARAM_COUNT_MAX = 3_000_000
SAVE_LOAD_RTOL = 1e-6
DISTILL_STEP_LR = 1e-2
DISTILL_FAKE_BATCH = 8


def test_shape_and_dtype() -> None:
    enc = EncoderB_DistilledMiniLM(seed=0)
    out = enc.encode(["bonjour", "longer query text here"])
    assert out.shape == (EXPECTED_BATCH, EXPECTED_OUTPUT_DIM)
    assert out.dtype == np.float32


def test_param_count_budget() -> None:
    enc = EncoderB_DistilledMiniLM(seed=0)
    assert PARAM_COUNT_MIN < enc.param_count() < PARAM_COUNT_MAX


def test_distillation_step_decreases_loss() -> None:
    """One gradient step against MSE target should reduce loss."""
    enc = EncoderB_DistilledMiniLM(seed=0)
    rng = np.random.default_rng(0)
    fake_texts = [f"query {i}" for i in range(DISTILL_FAKE_BATCH)]
    fake_targets = rng.standard_normal((DISTILL_FAKE_BATCH, EXPECTED_OUTPUT_DIM)).astype(np.float32)
    loss_before = enc.distill_loss(fake_texts, fake_targets)
    enc.distill_step(fake_texts, fake_targets, lr=DISTILL_STEP_LR)
    loss_after = enc.distill_loss(fake_texts, fake_targets)
    assert loss_after < loss_before, f"loss didn't decrease: {loss_before} -> {loss_after}"


def test_save_load_roundtrip(tmp_path) -> None:
    enc = EncoderB_DistilledMiniLM(seed=0)
    original = enc.encode(["query"])
    path = tmp_path / "distilled.safetensors"
    enc.save(path)
    enc2 = EncoderB_DistilledMiniLM(seed=99)
    enc2.load(path)
    np.testing.assert_allclose(enc2.encode(["query"]), original, rtol=SAVE_LOAD_RTOL)


def test_registry_entry() -> None:
    assert "B_distilled" in ENCODER_REGISTRY
    assert ENCODER_REGISTRY["B_distilled"] is EncoderB_DistilledMiniLM
