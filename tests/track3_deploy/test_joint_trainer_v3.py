"""Tests for JointTrainer v3 — MSE + λ·KL loss, JAX backprop through BridgeHead."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

from kiki_flow_core.track3_deploy.encoders.hash_mlp import EncoderC_HashMLP
from kiki_flow_core.track3_deploy.surrogate_trainer_v3 import JointTrainer

N_SPECIES = 4
N_STACKS = 32
STATE_DIM = N_SPECIES * N_STACKS  # 128
ENCODER_DIM = 384
TOY_BATCH = 8
LR = 1e-2
LAMBDA = 0.5
N_STEPS_OVERFIT = 100
LOSS_HALVED = 0.5
CHECKPOINT_LOSS_TOL = 1e-5


def _toy_batch(n: int = TOY_BATCH, seed: int = 0):
    rng = np.random.default_rng(seed)
    texts = [f"query number {i}" for i in range(n)]
    state_pre = rng.standard_normal((n, STATE_DIM)).astype(np.float32)
    state_post = state_pre + rng.standard_normal((n, STATE_DIM)).astype(np.float32) * 0.1
    rho_target = np.abs(rng.standard_normal((n, N_SPECIES, N_STACKS)).astype(np.float32))
    rho_target /= rho_target.sum(axis=2, keepdims=True)
    return texts, state_pre, state_post, rho_target


def test_loss_decreases_overfit_one_batch() -> None:
    encoder = EncoderC_HashMLP(seed=0)
    trainer = JointTrainer(encoder=encoder, lam=LAMBDA, lr=LR, seed=0)
    texts, spre, spost, rho = _toy_batch()
    loss_before = trainer.loss(texts, spre, spost, rho)
    for _ in range(N_STEPS_OVERFIT):
        trainer.step(texts, spre, spost, rho)
    loss_after = trainer.loss(texts, spre, spost, rho)
    assert loss_after < loss_before * LOSS_HALVED, (
        f"overfit 1 batch failed: {loss_before} -> {loss_after}"
    )


def test_kl_component_is_nonneg() -> None:
    encoder = EncoderC_HashMLP(seed=0)
    trainer = JointTrainer(encoder=encoder, lam=LAMBDA, lr=0.0, seed=0)
    texts, spre, spost, rho = _toy_batch(n=4)
    mse, kl = trainer.loss_components(texts, spre, spost, rho)
    assert mse >= 0.0
    assert kl >= 0.0


def test_save_load_checkpoint(tmp_path) -> None:
    encoder = EncoderC_HashMLP(seed=0)
    trainer = JointTrainer(encoder=encoder, lam=LAMBDA, lr=LR, seed=0)
    path = tmp_path / "ckpt.safetensors"
    trainer.save_checkpoint(path)
    trainer2 = JointTrainer(encoder=EncoderC_HashMLP(seed=99), lam=LAMBDA, lr=LR, seed=0)
    trainer2.load_checkpoint(path)
    texts, spre, spost, rho = _toy_batch(n=4)
    loss1 = trainer.loss(texts, spre, spost, rho)
    loss2 = trainer2.loss(texts, spre, spost, rho)
    assert abs(loss1 - loss2) < CHECKPOINT_LOSS_TOL
