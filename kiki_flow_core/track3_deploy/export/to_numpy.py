"""Export JAX bridge head weights to pure-NumPy safetensors + forward function.

The NumPy forward replicates `_BridgeHead.forward` (JAX) exactly:
    h1 = gelu(x @ W1 + b1)
    h2 = gelu(h1 @ W2 + b2) + h1     # skip around the second layer
    out = tanh(h2 @ W3 + b3)

JAX's `jax.nn.gelu` defaults to `approximate=True` (tanh approximation):
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

The constant sqrt(2/pi) is cast to float32 inside JAX before the computation,
so we replicate that here to stay within 1e-5 of JAX fp32 output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

# Match JAX's internal float32 cast of sqrt(2/pi) exactly.
_SQRT_2_OVER_PI = np.float32(np.sqrt(2.0 / np.pi))
_COEFF = np.float32(0.044715)
_HALF = np.float32(0.5)
_ONE = np.float32(1.0)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Tanh-approximation GELU — matches jax.nn.gelu(x) (default approximate=True).

    Replicates JAX's float32 path:
        cdf = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return x * cdf
    """
    inner: np.ndarray = _SQRT_2_OVER_PI * (x + _COEFF * x ** np.float32(3))
    cdf: np.ndarray = _HALF * (_ONE + np.tanh(inner))
    out: np.ndarray = np.multiply(x, cdf, dtype=np.float32)
    return out


def export_bridge_to_numpy(jax_params: dict[str, Any], path: Path | str) -> None:
    """Save JAX bridge params as a pure-NumPy safetensors file.

    Args:
        jax_params: Dict of JAX arrays keyed by weight name (W1, b1, W2, b2, W3, b3).
        path: Destination .safetensors file path.
    """
    flat: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float32) for k, v in jax_params.items()
    }
    save_file(flat, str(path))


def numpy_forward(path: Path | str, x: np.ndarray) -> np.ndarray:
    """Pure-NumPy forward that matches `_BridgeHead.forward` to within 1e-5.

    Architecture: 512 -> 256 -> 256 -> 128 (tanh), skip around H2.

    Args:
        path: Path to safetensors file produced by `export_bridge_to_numpy`.
        x: Input array of shape (B, 512), dtype float32.

    Returns:
        Output array of shape (B, 128), dtype float32.
    """
    p = load_file(str(path))
    h1: np.ndarray = _gelu(x @ p["W1"] + p["b1"])
    h2: np.ndarray = _gelu(h1 @ p["W2"] + p["b2"]) + h1  # skip — same as JAX
    out: np.ndarray = np.tanh(h2 @ p["W3"] + p["b3"], dtype=np.float32)
    return out
