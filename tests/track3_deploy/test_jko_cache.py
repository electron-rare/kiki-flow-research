"""Tests for JKOCache — SHA256-indexed safetensors storage."""

from __future__ import annotations

import numpy as np
import pytest

from kiki_flow_core.track3_deploy.data.jko_cache import JKOCache


@pytest.fixture
def cache(tmp_path) -> JKOCache:
    return JKOCache(root=tmp_path / "jko_cache")


def _make_pair() -> dict:
    return {
        "state_pre": np.ones(128, dtype=np.float32),
        "state_post": np.ones(128, dtype=np.float32) * 0.9,
        "rho_by_species": {
            "phono:code": np.full(32, 0.25, dtype=np.float32),
            "sem:code": np.full(32, 0.25, dtype=np.float32),
            "lex:code": np.full(32, 0.25, dtype=np.float32),
            "syntax:code": np.full(32, 0.25, dtype=np.float32),
        },
    }


def test_put_get_roundtrip(cache: JKOCache) -> None:
    pair = _make_pair()
    cache.put("hello world", pair)
    restored = cache.get("hello world")
    assert restored is not None
    np.testing.assert_array_equal(restored["state_pre"], pair["state_pre"])
    np.testing.assert_array_equal(restored["state_post"], pair["state_post"])
    for sp, rho in pair["rho_by_species"].items():
        np.testing.assert_array_equal(restored["rho_by_species"][sp], rho)


def test_miss_returns_none(cache: JKOCache) -> None:
    assert cache.get("never seen") is None


def test_sha256_collision_safety(cache: JKOCache) -> None:
    """Different queries must produce different cache keys."""
    cache.put("query one", _make_pair())
    assert cache.get("query two") is None


EXPECTED_HITS = 2
EXPECTED_MISSES = 1


def test_hit_stats(cache: JKOCache) -> None:
    cache.put("a", _make_pair())
    cache.get("a")  # hit
    cache.get("a")  # hit
    cache.get("b")  # miss
    stats = cache.stats()
    assert stats["hits"] == EXPECTED_HITS
    assert stats["misses"] == EXPECTED_MISSES


def test_persistence_across_instances(tmp_path) -> None:
    c1 = JKOCache(root=tmp_path / "c")
    c1.put("persistent", _make_pair())
    c2 = JKOCache(root=tmp_path / "c")
    assert c2.get("persistent") is not None


def test_put_rejects_malformed_pair(cache: JKOCache) -> None:
    """put() must raise ValueError with clear message when pair is missing required keys."""
    with pytest.raises(ValueError, match="missing required keys"):
        cache.put(
            "bad", {"state_pre": np.zeros(128, dtype=np.float32)}
        )  # missing state_post, rho_by_species
    # cache should still be empty (no partial file created)
    assert cache.get("bad") is None


def test_put_cleans_up_tmp_on_success(cache: JKOCache, tmp_path) -> None:
    """After put() succeeds, no .tmp file should linger."""
    cache.put(
        "ok",
        {
            "state_pre": np.ones(128, dtype=np.float32),
            "state_post": np.ones(128, dtype=np.float32),
            "rho_by_species": {"phono:code": np.full(32, 0.25, dtype=np.float32)},
        },
    )
    tmps = list(cache.root.glob("*.tmp"))
    assert tmps == []
