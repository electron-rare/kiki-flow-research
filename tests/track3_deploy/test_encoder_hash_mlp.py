"""Tests for EncoderC_HashMLP."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.track3_deploy.encoders import ENCODER_REGISTRY
from kiki_flow_core.track3_deploy.encoders.hash_mlp import EncoderC_HashMLP

EXPECTED_OUTPUT_DIM = 384
EXPECTED_BATCH = 2
PARAM_COUNT_MIN = 300_000
PARAM_COUNT_MAX = 800_000
SAVE_LOAD_RTOL = 1e-6


def test_output_shape() -> None:
    enc = EncoderC_HashMLP(seed=0)
    out = enc.encode(["bonjour", "ceci est une query plus longue"])
    assert out.shape == (EXPECTED_BATCH, EXPECTED_OUTPUT_DIM)
    assert out.dtype == np.float32


def test_determinism() -> None:
    enc = EncoderC_HashMLP(seed=0)
    a = enc.encode(["meme query"])
    b = enc.encode(["meme query"])
    np.testing.assert_array_equal(a, b)


def test_different_inputs_different_outputs() -> None:
    enc = EncoderC_HashMLP(seed=0)
    out = enc.encode(["query un", "query deux"])
    assert not np.allclose(out[0], out[1])


def test_param_count_budget() -> None:
    enc = EncoderC_HashMLP(seed=0)
    assert PARAM_COUNT_MIN < enc.param_count() < PARAM_COUNT_MAX


def test_save_load_roundtrip(tmp_path) -> None:
    enc = EncoderC_HashMLP(seed=0)
    original = enc.encode(["query"])
    path = tmp_path / "hash_mlp.safetensors"
    enc.save(path)
    enc2 = EncoderC_HashMLP(seed=99)  # different init, overwritten by load
    enc2.load(path)
    restored = enc2.encode(["query"])
    np.testing.assert_allclose(restored, original, rtol=SAVE_LOAD_RTOL)


def test_registry_entry() -> None:
    assert "C_hash_mlp" in ENCODER_REGISTRY
    assert ENCODER_REGISTRY["C_hash_mlp"] is EncoderC_HashMLP
