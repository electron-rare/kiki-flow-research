"""Tests for EncoderD_TinyTransformer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

from kiki_flow_core.track3_deploy.encoders import ENCODER_REGISTRY
from kiki_flow_core.track3_deploy.encoders.tiny_tf import EncoderD_TinyTransformer

EXPECTED_OUTPUT_DIM = 384
EXPECTED_BATCH = 2
PARAM_COUNT_MIN = 3_000_000
PARAM_COUNT_MAX = 20_000_000
SAVE_LOAD_RTOL = 1e-5


def test_shape_and_dtype() -> None:
    enc = EncoderD_TinyTransformer(seed=0)
    out = enc.encode(["bonjour le monde", "query deux"])
    assert out.shape == (EXPECTED_BATCH, EXPECTED_OUTPUT_DIM)
    assert out.dtype == np.float32


def test_param_count_budget() -> None:
    enc = EncoderD_TinyTransformer(seed=0)
    assert PARAM_COUNT_MIN < enc.param_count() < PARAM_COUNT_MAX


def test_padding_mask_handling() -> None:
    """Short and long queries in same batch must both produce valid outputs."""
    enc = EncoderD_TinyTransformer(seed=0)
    out = enc.encode(["x", "a b c d e f g h i j"])
    assert np.isfinite(out).all()


def test_save_load_roundtrip(tmp_path) -> None:
    enc = EncoderD_TinyTransformer(seed=0)
    original = enc.encode(["query"])
    path = tmp_path / "tiny_tf.safetensors"
    enc.save(path)
    enc2 = EncoderD_TinyTransformer(seed=99)
    enc2.load(path)
    np.testing.assert_allclose(enc2.encode(["query"]), original, rtol=SAVE_LOAD_RTOL)


def test_registry_entry() -> None:
    assert "D_tiny_tf" in ENCODER_REGISTRY
    assert ENCODER_REGISTRY["D_tiny_tf"] is EncoderD_TinyTransformer
