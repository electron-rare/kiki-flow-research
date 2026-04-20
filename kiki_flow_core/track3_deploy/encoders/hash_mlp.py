"""HashMLP encoder: fastText-style n-gram hash -> embedding table -> 2-layer MLP -> 384-dim.

Pure-NumPy, deployable, ~520K params. Deterministic from seed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from kiki_flow_core.track3_deploy.encoders import register
from kiki_flow_core.track3_deploy.encoders.base import TextEncoder

_DEFAULT_NUM_BUCKETS = 4096
_DEFAULT_EMBED_DIM = 96
_DEFAULT_HIDDEN_DIM = 512
_DEFAULT_OUTPUT_DIM = 384
_DEFAULT_NGRAM_N = 3
_EMBED_INIT_SCALE = 0.02


def _ngrams(text: str, n: int) -> list[str]:
    wrapped = f"<{text.lower()}>"
    if len(wrapped) < n:
        return [wrapped]
    return [wrapped[i : i + n] for i in range(len(wrapped) - n + 1)]


def _hash_token(token: str, num_buckets: int) -> int:
    h = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(h[:8], "big") % num_buckets


@register("C_hash_mlp")
class EncoderC_HashMLP(TextEncoder):  # noqa: N801
    """n-gram hash -> sum-pool embedding -> MLP -> 384-dim."""

    def __init__(
        self,
        num_buckets: int = _DEFAULT_NUM_BUCKETS,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        output_dim: int = _DEFAULT_OUTPUT_DIM,
        ngram_n: int = _DEFAULT_NGRAM_N,
        seed: int = 0,
    ) -> None:
        self.num_buckets = num_buckets
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ngram_n = ngram_n
        rng = np.random.default_rng(seed)
        self.embedding = (
            rng.standard_normal((num_buckets, embed_dim)).astype(np.float32) * _EMBED_INIT_SCALE
        )
        self.W1 = (
            rng.standard_normal((embed_dim, hidden_dim)).astype(np.float32)
            * (2.0 / embed_dim) ** 0.5
        )
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (
            rng.standard_normal((hidden_dim, output_dim)).astype(np.float32)
            * (2.0 / hidden_dim) ** 0.5
        )
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def _pool(self, text: str) -> np.ndarray:
        grams = _ngrams(text, self.ngram_n)
        ids = np.array([_hash_token(g, self.num_buckets) for g in grams], dtype=np.int64)
        result: np.ndarray = self.embedding[ids].mean(axis=0)
        return result

    def encode(self, texts: list[str]) -> np.ndarray:
        pooled = np.stack([self._pool(t) for t in texts]).astype(np.float32)
        h = np.maximum(0.0, pooled @ self.W1 + self.b1)  # ReLU
        out: np.ndarray = (h @ self.W2 + self.b2).astype(np.float32)
        return out

    def param_count(self) -> int:
        return int(self.embedding.size + self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def save(self, path: Path | str) -> None:
        save_file(
            {
                "embedding": self.embedding,
                "W1": self.W1,
                "b1": self.b1,
                "W2": self.W2,
                "b2": self.b2,
            },
            str(path),
        )

    def load(self, path: Path | str) -> None:
        d = load_file(str(path))
        self.embedding = d["embedding"]
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
