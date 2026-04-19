"""Tests for NumpyExporter — JAX params -> NumPy forward with <1e-5 diff."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

import jax.numpy as jnp  # noqa: E402

from kiki_flow_core.track3_deploy.encoders.hash_mlp import EncoderC_HashMLP  # noqa: E402
from kiki_flow_core.track3_deploy.export.to_numpy import (  # noqa: E402
    export_bridge_to_numpy,
    numpy_forward,
)
from kiki_flow_core.track3_deploy.surrogate_trainer_v3 import (  # noqa: E402
    JointTrainer,
    _BridgeHead,
)

PARITY_TOL = 1e-5
N_PARITY_INPUTS = 50
INPUT_DIM = 512
OUTPUT_DIM = 128


def test_export_roundtrip_diff_below_tolerance(tmp_path) -> None:
    """Exported NumPy forward must match JAX to <1e-5 on arbitrary inputs."""
    encoder = EncoderC_HashMLP(seed=0)
    trainer = JointTrainer(encoder=encoder, lam=0.5, lr=1e-3, seed=0)
    path = tmp_path / "winner.safetensors"
    export_bridge_to_numpy(trainer.params, path)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_PARITY_INPUTS, INPUT_DIM)).astype(np.float32)
    jax_out = np.asarray(_BridgeHead.forward(trainer.params, jnp.asarray(x)))
    np_out = numpy_forward(path, x)
    max_diff = float(np.max(np.abs(jax_out - np_out)))
    assert max_diff < PARITY_TOL, f"max diff {max_diff}"


def test_numpy_forward_shape(tmp_path) -> None:
    encoder = EncoderC_HashMLP(seed=0)
    trainer = JointTrainer(encoder=encoder, lam=0.5, lr=1e-3, seed=0)
    path = tmp_path / "winner.safetensors"
    export_bridge_to_numpy(trainer.params, path)
    x = np.zeros((3, INPUT_DIM), dtype=np.float32)
    out = numpy_forward(path, x)
    assert out.shape == (3, OUTPUT_DIM)
    assert out.dtype == np.float32
