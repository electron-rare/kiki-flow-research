"""Abstract base class for text encoders in the ablation sweep."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class TextEncoder(ABC):
    """Contract: text (list[str]) -> (B, 384) float32 array."""

    output_dim: int = 384

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Batch forward. Returns shape (len(texts), output_dim), dtype float32."""

    @abstractmethod
    def param_count(self) -> int:
        """Total parameter count."""

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Dump weights to a .safetensors file."""

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Load weights from a .safetensors file (overwrites in place)."""
