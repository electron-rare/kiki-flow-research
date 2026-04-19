"""CLI: consume a JSONL corpus of queries, run JKO oracle, fill a JKOCache.

Integration path (b): JKOStep(ZeroF) with uniform initial FlowState.
No sentence-transformers, no weights files required. Each query receives an
independent uniform state_pre; JKOStep.step() produces state_post via one
JKO iteration (gradient descent on zero free energy = identity-like transport,
giving a valid (pre, post) pair whose rho_by_species follows the simplex).
Species keys are the four canonical Levelt-Baddeley labels:
  {"lex:code", "phono:code", "sem:code", "syntax:code"}
each with 32-stack resolution, flattened to a 128-dim state vector.

Tests monkeypatch `compute_jko_pair` so unit tests don't invoke the real JKO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from kiki_flow_core.master_equation import JKOStep, ZeroF
from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.data.jko_cache import JKOCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical constants (must match JKOCache and the surrogate training pipeline)
# ---------------------------------------------------------------------------
CANONICAL_SPECIES: tuple[str, ...] = ("lex:code", "phono:code", "sem:code", "syntax:code")
N_STACKS: int = 32  # per-species resolution
STATE_DIM: int = len(CANONICAL_SPECIES) * N_STACKS  # 128

# JKO hyper-parameters for the oracle (conservative: many inner steps for accuracy)
_JKO_H: float = 0.1
_JKO_N_INNER: int = 20


def _make_uniform_state(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return a valid uniform-ish initial rho dict for the 4 canonical species."""
    rho: dict[str, np.ndarray] = {}
    for sp in CANONICAL_SPECIES:
        raw = rng.random(N_STACKS).astype(np.float32)
        rho[sp] = raw / raw.sum()
    return rho


def _build_jko_step() -> JKOStep:
    """Construct a JKOStep(ZeroF) for oracle use."""
    support = np.linspace(0.0, 1.0, N_STACKS, dtype=np.float32)
    return JKOStep(
        f_functional=ZeroF(),
        h=_JKO_H,
        support=support,
        n_inner=_JKO_N_INNER,
        apply_w2_prox=False,  # keep it fast; prox requires POT solve
    )


def compute_jko_pair(query: str) -> dict[str, Any]:  # pragma: no cover - replaced by fake in tests
    """Run one JKO step for *query* and return a JKOCache-compatible pair.

    The query string is used as a deterministic seed so repeated calls
    for the same query produce the same pair (oracle idempotence).

    Returns:
        {
          "state_pre":  ndarray shape (128,),
          "state_post": ndarray shape (128,),
          "rho_by_species": {sp: ndarray shape (32,) for sp in CANONICAL_SPECIES},
        }
    """
    # Deterministic seed from query text so reruns are idempotent.
    seed = int.from_bytes(hashlib.sha256(query.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)

    # Build an initial uniform FlowState for T3.
    rho_init = _make_uniform_state(rng)
    state_pre_obj = FlowState(
        rho=rho_init,
        P_theta=np.zeros(8, dtype=np.float32),
        mu_curr=np.array([1.0], dtype=np.float32),
        tau=0,
        metadata={"track_id": "T3"},
    )

    # Flatten pre into a 128-dim vector (sorted-key order = CANONICAL_SPECIES order).
    state_pre = np.concatenate([rho_init[sp] for sp in CANONICAL_SPECIES])

    # One JKO step: ZeroF gradient descent on simplex (fast, deterministic).
    jko = _build_jko_step()
    state_post_obj = jko.step(state_pre_obj)

    # Flatten post.
    state_post = np.concatenate([state_post_obj.rho[sp] for sp in CANONICAL_SPECIES])

    # rho_by_species: canonical species -> post-step distribution (32-dim simplex).
    rho_by_species = {sp: state_post_obj.rho[sp].copy() for sp in CANONICAL_SPECIES}

    return {
        "state_pre": state_pre.astype(np.float32),
        "state_post": state_post.astype(np.float32),
        "rho_by_species": rho_by_species,
    }


# ---------------------------------------------------------------------------
# Corpus iteration
# ---------------------------------------------------------------------------


def _iter_corpus(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run JKO oracle over a JSONL corpus.")
    parser.add_argument(
        "--corpus", type=Path, required=True, help="JSONL file; each line must have a 'text' field."
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="Directory for JKOCache safetensors files."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N new queries (0 = unlimited)."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

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
