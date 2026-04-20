"""DistilledMiniLM encoder: 3-layer MLP over character-bigram bag-of-words.

Teacher targets (MiniLM embeddings) are supplied by the caller at distill time
via distill_step(texts, targets). This keeps the encoder pure-NumPy — no torch
dep, no sentence-transformers dep, respecting the repo's no-torch rule.

Architecture:
    bigram_bow (4096) -> dense 512 (ReLU) -> dense 384 (ReLU) -> dense 384

~2.4M parameters. Deterministic from seed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from kiki_flow_core.track3_deploy.encoders import register
from kiki_flow_core.track3_deploy.encoders.base import TextEncoder

_DEFAULT_INPUT_DIM = 4096
_DEFAULT_HIDDEN1 = 512
_DEFAULT_HIDDEN2 = 384
_DEFAULT_OUTPUT_DIM = 384
_BIGRAM_N = 2


def _bigram_bow(text: str, num_buckets: int) -> np.ndarray:
    """Character bigram bag-of-words, MD5-hashed into num_buckets, L2-normalized."""
    wrapped = f"<{text.lower()}>"
    vec = np.zeros(num_buckets, dtype=np.float32)
    for i in range(len(wrapped) - _BIGRAM_N + 1):
        h = hashlib.md5(wrapped[i : i + _BIGRAM_N].encode("utf-8"), usedforsecurity=False).digest()
        idx = int.from_bytes(h[:8], "big") % num_buckets
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


@register("B_distilled")
class EncoderB_DistilledMiniLM(TextEncoder):  # noqa: N801
    """3-layer MLP: 4096-dim BoW -> 512 -> 384 -> 384 (ReLU activations).

    ~2.4M parameters. Pure-NumPy, no torch dependency. Teacher targets are
    injected by the caller (T3 CorpusBuilder pattern) — sentence-transformers
    stays in a separate offline venv.
    """

    def __init__(
        self,
        input_dim: int = _DEFAULT_INPUT_DIM,
        hidden1: int = _DEFAULT_HIDDEN1,
        hidden2: int = _DEFAULT_HIDDEN2,
        output_dim: int = _DEFAULT_OUTPUT_DIM,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = output_dim
        rng = np.random.default_rng(seed)
        # He init (ReLU)
        self.W1: np.ndarray = (
            rng.standard_normal((input_dim, hidden1)).astype(np.float32) * (2.0 / input_dim) ** 0.5
        )
        self.b1: np.ndarray = np.zeros(hidden1, dtype=np.float32)
        self.W2: np.ndarray = (
            rng.standard_normal((hidden1, hidden2)).astype(np.float32) * (2.0 / hidden1) ** 0.5
        )
        self.b2: np.ndarray = np.zeros(hidden2, dtype=np.float32)
        self.W3: np.ndarray = (
            rng.standard_normal((hidden2, output_dim)).astype(np.float32) * (2.0 / hidden2) ** 0.5
        )
        self.b3: np.ndarray = np.zeros(output_dim, dtype=np.float32)

    def _featurize(self, texts: list[str]) -> np.ndarray:
        result: np.ndarray = np.stack([_bigram_bow(t, self.input_dim) for t in texts])
        return result

    def encode(self, texts: list[str]) -> np.ndarray:
        x = self._featurize(texts)
        h1 = np.maximum(0.0, x @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        out: np.ndarray = (h2 @ self.W3 + self.b3).astype(np.float32)
        return out

    def distill_loss(self, texts: list[str], targets: np.ndarray) -> float:
        """MSE loss between encode(texts) and teacher targets."""
        pred = self.encode(texts)
        return float(np.mean((pred - targets) ** 2))

    def distill_step(
        self,
        texts: list[str],
        targets: np.ndarray,
        lr: float = 1e-3,
    ) -> float:
        """One naive SGD step against MSE(encode(texts), targets).

        Returns the pre-step loss (for monitoring).
        """
        x = self._featurize(texts)  # (B, D_in)
        h1 = np.maximum(0.0, x @ self.W1 + self.b1)  # (B, H1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)  # (B, H2)
        y: np.ndarray = h2 @ self.W3 + self.b3  # (B, D_out)
        b = x.shape[0]
        loss = float(np.mean((y - targets) ** 2))
        # Backprop (MSE, factor 2/B)
        dy = 2.0 * (y - targets) / b  # (B, D_out)
        dw3 = h2.T @ dy
        db3 = dy.sum(axis=0)
        dh2 = dy @ self.W3.T
        dh2_raw = dh2 * (h2 > 0)
        dw2 = h1.T @ dh2_raw
        db2 = dh2_raw.sum(axis=0)
        dh1 = dh2_raw @ self.W2.T
        dh1_raw = dh1 * (h1 > 0)
        dw1 = x.T @ dh1_raw
        db1 = dh1_raw.sum(axis=0)
        # SGD update (direct attribute mutation avoids PLW2901)
        self.W1 -= lr * dw1
        self.b1 -= lr * db1
        self.W2 -= lr * dw2
        self.b2 -= lr * db2
        self.W3 -= lr * dw3
        self.b3 -= lr * db3
        return loss

    def param_count(self) -> int:
        return int(
            self.W1.size + self.b1.size + self.W2.size + self.b2.size + self.W3.size + self.b3.size
        )

    def save(self, path: Path | str) -> None:
        save_file(
            {
                "W1": self.W1,
                "b1": self.b1,
                "W2": self.W2,
                "b2": self.b2,
                "W3": self.W3,
                "b3": self.b3,
            },
            str(path),
        )

    def load(self, path: Path | str) -> None:
        d = load_file(str(path))
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.W3, self.b3 = d["W3"], d["b3"]
