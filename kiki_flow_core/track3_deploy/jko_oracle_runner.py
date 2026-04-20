"""CLI: consume a JSONL corpus of queries, run JKO oracle, fill a JKOCache.

When --g-jepa is provided, the oracle uses QueryConditionedF (text-conditioned
dynamics). Otherwise falls back to ZeroF (legacy / test path).
Tests monkeypatch `compute_jko_pair` so unit tests don't need a real JKO run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from kiki_flow_core.master_equation import JKOStep, ZeroF
from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.data.jko_cache import JKOCache
from kiki_flow_core.track3_deploy.query_conditioned_f import (
    SPECIES_CANONICAL,
    QueryConditionedF,
)
from kiki_flow_core.track3_deploy.state_projection import flatten
from kiki_flow_core.track3_deploy.train_g_jepa import load_gjepa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical constants (must match JKOCache and the surrogate training pipeline)
# ---------------------------------------------------------------------------
CANONICAL_SPECIES: tuple[str, ...] = SPECIES_CANONICAL
N_STACKS: int = 32  # per-species resolution
STATE_DIM: int = len(CANONICAL_SPECIES) * N_STACKS  # 128

_N_STACKS = 32
_N_SPECIES = 4
_P_THETA_DIM = 8
_EMBED_DIM = 384
_HASH_SEED_BYTES = 4

# JKO hyper-parameters for the oracle
_JKO_H: float = 0.1
_JKO_N_INNER: int = 20


def _seeded_initial_state(query: str) -> FlowState:
    """Deterministic uniform-ish initial FlowState seeded by sha256 prefix of the query."""
    seed = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:_HASH_SEED_BYTES], "big")
    rng = np.random.default_rng(seed)
    rho: dict[str, np.ndarray] = {}
    for sp in CANONICAL_SPECIES:
        v = rng.random(_N_STACKS).astype(np.float32)
        rho[sp] = v / v.sum()
    return FlowState(
        rho=rho,
        P_theta=np.zeros(_P_THETA_DIM, dtype=np.float32),
        mu_curr=np.zeros(1, dtype=np.float32),
        tau=0,
        metadata={"track_id": "T3"},
    )


def _placeholder_embedder(query: str) -> np.ndarray:
    """Stub embedder; replaced by callers that own the encoder."""
    return np.zeros(_EMBED_DIM, dtype=np.float32)


def _make_pair_computer(
    g_jepa_path: Path | None = None,
    embedder: Callable[[str], np.ndarray] | None = None,
) -> Callable[[str], dict[str, Any]]:
    """Build a compute_jko_pair closure parametrized by g_JEPA weights + embedder.

    When g_jepa_path is None, falls back to ZeroF (legacy / test path).
    The returned closure is safe to call concurrently (no shared mutable state).
    """
    g_jepa_params = load_gjepa(g_jepa_path) if g_jepa_path is not None else None
    embed_fn: Callable[[str], np.ndarray] = (
        embedder if embedder is not None else _placeholder_embedder
    )

    def _compute(query: str) -> dict[str, Any]:
        state_pre = _seeded_initial_state(query)
        if g_jepa_params is None:
            support = np.linspace(0.0, 1.0, _N_STACKS, dtype=np.float32)
            step = JKOStep(
                f_functional=ZeroF(),
                h=_JKO_H,
                support=support,
                n_inner=_JKO_N_INNER,
                apply_w2_prox=False,
            )
        else:
            emb = embed_fn(query)
            f_energy = QueryConditionedF(
                g_jepa_params={k: np.asarray(v) for k, v in g_jepa_params.items()},
                embedding=emb,
            )
            support = np.linspace(0.0, 1.0, _N_STACKS, dtype=np.float32)
            step = JKOStep(
                f_functional=f_energy,
                h=_JKO_H,
                support=support,
                n_inner=_JKO_N_INNER,
                apply_w2_prox=False,
            )
        state_post = step.step(state_pre)
        return {
            "state_pre": flatten(state_pre),
            "state_post": flatten(state_post),
            "rho_by_species": {
                k: np.asarray(v, dtype=np.float32) for k, v in state_post.rho.items()
            },
        }

    return _compute


# Module-level default — tests monkeypatch this name to inject a fake.
# Stored in a mutable container so main() can rebind without a global statement.
_compute_jko_pair_ref: list[Callable[[str], dict[str, Any]]] = [_make_pair_computer(None)]


def compute_jko_pair(query: str) -> dict[str, Any]:  # noqa: D103
    """Dispatch to the active oracle implementation (ZeroF or QueryConditionedF)."""
    return _compute_jko_pair_ref[0](query)


def _iter_corpus(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run JKO oracle over a JSONL corpus.")
    parser.add_argument(
        "--corpus", type=Path, required=True, help="JSONL file; each line must have a 'text' field."
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="Directory for JKOCache safetensors files."
    )
    parser.add_argument(
        "--g-jepa",
        type=Path,
        default=None,
        help=(
            "Path to pre-trained g_JEPA weights for QueryConditionedF;"
            " falls back to ZeroF if omitted."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N new queries (0 = unlimited)."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Rebind the active computer when --g-jepa is provided (production path).
    # Tests that don't pass --g-jepa keep their monkeypatched fake untouched.
    if args.g_jepa is not None:
        _compute_jko_pair_ref[0] = _make_pair_computer(args.g_jepa)

    cache = JKOCache(root=args.cache_dir)
    processed = 0
    skipped = 0
    failed = 0

    for entry in _iter_corpus(args.corpus):
        q: str = entry["text"]
        if q in cache:
            skipped += 1
            continue
        try:
            pair = compute_jko_pair(q)
        except Exception:
            logger.exception("oracle failed on query: %s", q[:80])
            failed += 1
            continue
        cache.put(q, pair)
        processed += 1
        if args.limit and processed >= args.limit:
            break
        if processed % 100 == 0:
            logger.info("processed=%d skipped=%d failed=%d", processed, skipped, failed)

    logger.info(
        "DONE processed=%d skipped=%d failed=%d total_cached=%d",
        processed,
        skipped,
        failed,
        len(cache),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
