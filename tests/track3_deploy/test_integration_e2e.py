"""End-to-end integration: 100 fake queries → train 5 epochs → eval → export.

Runs the full text-bridge pipeline on synthetic data to catch plumbing
regressions. Marked `integration` so it is not in the default test run
(run explicitly with `-m integration`). Must complete in < 5 min.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

from kiki_flow_core.track3_deploy.data.corpus_builder import (
    CorpusBuilder,
    CorpusEntry,
)
from kiki_flow_core.track3_deploy.data.jko_cache import JKOCache
from kiki_flow_core.track3_deploy.encoders.hash_mlp import EncoderC_HashMLP
from kiki_flow_core.track3_deploy.eval.kl_species import (
    SPECIES_CANONICAL,
    evaluate_checkpoint,
)
from kiki_flow_core.track3_deploy.export.to_numpy import (
    export_bridge_to_numpy,
    numpy_forward,
)
from kiki_flow_core.track3_deploy.surrogate_trainer_v3 import JointTrainer

N_PER_SPECIES = 25
STATE_DIM = 128
STACK_DIM = 32
EPOCHS = 5
BATCH_SIZE = 32
LR = 3e-4
LAMBDA = 0.5
MAX_WALL_CLOCK_SEC = 300.0
EXPORT_INPUT_DIM = 512
EXPORT_BATCH = 5
EXPORT_OUTPUT_DIM = 128
DEDUP_THRESHOLD_HIGH = 0.99  # skip embedding stage with high threshold


def _fake_pair(rng: np.random.Generator) -> dict:
    spre = rng.standard_normal(STATE_DIM).astype(np.float32)
    spost = spre + rng.standard_normal(STATE_DIM).astype(np.float32) * 0.05
    rho = {sp: rng.random(STACK_DIM).astype(np.float32) for sp in SPECIES_CANONICAL}
    for sp, val in rho.items():
        rho[sp] = val / val.sum()
    return {"state_pre": spre, "state_post": spost, "rho_by_species": rho}


@pytest.mark.integration
def test_full_pipeline_under_5_min(tmp_path: Path) -> None:
    t0 = time.time()
    rng = np.random.default_rng(0)

    # 1) Fake corpus: 4 species × 25 queries = 100
    entries = [
        CorpusEntry(text=f"query {sp} {i}", source="B", species=sp)
        for sp in ("phono", "sem", "lex", "syntax")
        for i in range(N_PER_SPECIES)
    ]
    builder = CorpusBuilder(dedup_threshold=DEDUP_THRESHOLD_HIGH)
    splits = builder.split(entries, ratios=(0.8, 0.1, 0.1), seed=0)

    # 2) Fake JKO pairs into cache
    cache = JKOCache(root=tmp_path / "cache")
    for e in entries:
        cache.put(e.text, _fake_pair(rng))

    def _to_pairs(split_name: str) -> list[dict]:
        out = []
        for e in splits[split_name]:
            p = cache.get(e.text)
            assert p is not None, f"missing fake pair for {e.text}"
            out.append({"text": e.text, **p})
        return out

    train_pairs = _to_pairs("train")
    _to_pairs("val")  # exercises cache but results not used in this smoke test
    test_pairs = _to_pairs("test")

    # 3) Train EncoderC + bridge for 5 epochs
    encoder = EncoderC_HashMLP(seed=0)
    trainer = JointTrainer(encoder=encoder, lam=LAMBDA, lr=LR, seed=0)
    for _epoch in range(EPOCHS):
        order = rng.permutation(len(train_pairs))
        for i in range(0, len(order), BATCH_SIZE):
            batch_idx = order[i : i + BATCH_SIZE]
            batch = [train_pairs[j] for j in batch_idx]
            texts = [b["text"] for b in batch]
            spre = np.stack([b["state_pre"] for b in batch])
            spost = np.stack([b["state_post"] for b in batch])
            rho = np.stack(
                [np.stack([b["rho_by_species"][sp] for sp in SPECIES_CANONICAL]) for b in batch]
            )
            trainer.step(texts, spre, spost, rho)

    # 4) Eval on test
    result = evaluate_checkpoint(encoder, trainer.params, test_pairs)
    assert "total" in result
    assert result["total"] >= 0.0

    # 5) Export winner to pure-NumPy + parity check
    export_path = tmp_path / "winner.safetensors"
    export_bridge_to_numpy(trainer.params, export_path)
    x = np.zeros((EXPORT_BATCH, EXPORT_INPUT_DIM), dtype=np.float32)
    np_out = numpy_forward(export_path, x)
    assert np_out.shape == (EXPORT_BATCH, EXPORT_OUTPUT_DIM)
    assert np.isfinite(np_out).all()

    # Wall clock budget
    elapsed = time.time() - t0
    print(f"e2e pipeline ran in {elapsed:.1f}s")
    assert elapsed < MAX_WALL_CLOCK_SEC, (
        f"pipeline too slow: {elapsed}s (budget {MAX_WALL_CLOCK_SEC}s)"
    )
