"""SHA256-indexed safetensors cache for JKO oracle outputs.

Each entry stores (state_pre, state_post, rho_by_species) for one query.
Keyed by sha256(query_utf8) so repeated queries don't recompute JKO.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

REQUIRED_PAIR_KEYS = frozenset({"state_pre", "state_post", "rho_by_species"})


class JKOCache:
    """Persistent cache of JKO oracle outputs, one .safetensors per query."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def _path(self, query: str) -> Path:
        return self.root / f"{self._key(query)}.safetensors"

    def put(self, query: str, pair: dict[str, Any]) -> None:
        """Store a JKO pair. pair must contain state_pre, state_post, rho_by_species."""
        missing = REQUIRED_PAIR_KEYS - pair.keys()
        if missing:
            raise ValueError(
                f"JKOCache.put: pair missing required keys: {sorted(missing)};"
                f" got {sorted(pair.keys())}"
            )
        flat: dict[str, np.ndarray] = {
            "state_pre": np.asarray(pair["state_pre"], dtype=np.float32),
            "state_post": np.asarray(pair["state_post"], dtype=np.float32),
        }
        for species, rho in pair["rho_by_species"].items():
            flat[f"rho::{species}"] = np.asarray(rho, dtype=np.float32)
        tmp = self._path(query).with_suffix(".safetensors.tmp")
        save_file(flat, str(tmp))
        os.replace(tmp, self._path(query))

    def get(self, query: str) -> dict[str, Any] | None:
        path = self._path(query)
        if not path.exists():
            self._misses += 1
            return None
        self._hits += 1
        flat = load_file(str(path))
        rho_by_species = {k.split("::", 1)[1]: v for k, v in flat.items() if k.startswith("rho::")}
        return {
            "state_pre": flat["state_pre"],
            "state_post": flat["state_post"],
            "rho_by_species": rho_by_species,
        }

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def __contains__(self, query: str) -> bool:
        return self._path(query).exists()

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*.safetensors"))
