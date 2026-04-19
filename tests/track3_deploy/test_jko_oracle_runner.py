"""Tests for JKO oracle runner CLI (T11)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from kiki_flow_core.track3_deploy import jko_oracle_runner
from kiki_flow_core.track3_deploy import jko_oracle_runner as jor
from kiki_flow_core.track3_deploy.data.jko_cache import JKOCache

CANONICAL_SPECIES = ("phono:code", "sem:code", "lex:code", "syntax:code")
STATE_DIM = 128
STACK_DIM = 32

# Named constants to satisfy PLR2004 (no magic values in assertions)
N_QUERIES_SMALL = 3
N_LIMIT = 3


def _fake_compute_jko_pair(query: str) -> dict:
    """Deterministic fake that avoids running real JKO in tests."""
    rng = np.random.default_rng(abs(hash(query)) % (2**32))
    state_pre = rng.standard_normal(STATE_DIM).astype(np.float32)
    state_post = state_pre + rng.standard_normal(STATE_DIM).astype(np.float32) * 0.05
    rho = {sp: rng.random(STACK_DIM).astype(np.float32) for sp in CANONICAL_SPECIES}
    for sp, val in rho.items():
        rho[sp] = val / val.sum()
    return {"state_pre": state_pre, "state_post": state_post, "rho_by_species": rho}


@pytest.fixture
def fake_jko(monkeypatch):
    monkeypatch.setattr(jko_oracle_runner, "compute_jko_pair", _fake_compute_jko_pair)


def _write_corpus(path: Path, texts: list[str]) -> None:
    with path.open("w") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t, "source": "B", "species": "phono"}) + "\n")


def test_processes_new_queries(tmp_path: Path, fake_jko) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, ["q1", "q2", "q3"])
    cache_dir = tmp_path / "cache"
    rc = jko_oracle_runner.main(["--corpus", str(corpus), "--cache-dir", str(cache_dir)])
    assert rc == 0
    cache = JKOCache(root=cache_dir)
    assert len(cache) == N_QUERIES_SMALL
    assert cache.get("q1") is not None


def test_skips_already_cached(tmp_path: Path, fake_jko) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, ["q1", "q2"])
    cache_dir = tmp_path / "cache"
    # First run populates
    jko_oracle_runner.main(["--corpus", str(corpus), "--cache-dir", str(cache_dir)])
    # Second run: track calls to confirm skip path
    called: list[str] = []

    def _track_calls(query: str) -> dict:
        called.append(query)
        return _fake_compute_jko_pair(query)

    orig = jko_oracle_runner.compute_jko_pair
    jko_oracle_runner.compute_jko_pair = _track_calls  # type: ignore[assignment]
    try:
        rc = jko_oracle_runner.main(["--corpus", str(corpus), "--cache-dir", str(cache_dir)])
    finally:
        jko_oracle_runner.compute_jko_pair = orig  # type: ignore[assignment]
    assert rc == 0
    assert called == [], f"expected no new oracle calls on second run, got: {called}"


def test_respects_limit(tmp_path: Path, fake_jko) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, [f"q{i}" for i in range(10)])
    cache_dir = tmp_path / "cache"
    rc = jko_oracle_runner.main(
        [
            "--corpus",
            str(corpus),
            "--cache-dir",
            str(cache_dir),
            "--limit",
            str(N_LIMIT),
        ]
    )
    assert rc == 0
    assert len(JKOCache(root=cache_dir)) == N_LIMIT


def test_runner_accepts_g_jepa_flag(tmp_path: Path, monkeypatch) -> None:
    """Runner accepts --g-jepa flag without crashing.

    Patches _make_pair_computer so main()'s rebind uses the fake instead of
    loading real weights. Cache ends up with the expected 2 entries.
    """
    fake_path = tmp_path / "g_jepa.safetensors"
    save_file(
        {
            "W1": np.zeros((128, 256), dtype=np.float32),
            "b1": np.zeros(256, dtype=np.float32),
            "W2": np.zeros((256, 384), dtype=np.float32),
            "b2": np.zeros(384, dtype=np.float32),
        },
        str(fake_path),
    )

    # Patch the factory so the rebind inside main() returns our fake callable.
    monkeypatch.setattr(jor, "_make_pair_computer", lambda p: _fake_compute_jko_pair)

    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, ["q1", "q2"])
    cache_dir = tmp_path / "cache"
    rc = jor.main(
        [
            "--corpus",
            str(corpus),
            "--cache-dir",
            str(cache_dir),
            "--g-jepa",
            str(fake_path),
        ]
    )
    assert rc == 0
    cache = JKOCache(root=cache_dir)
    assert len(cache) == 2  # noqa: PLR2004
