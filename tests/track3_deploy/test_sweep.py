"""Smoke test for SweepRunner.pick_top_k R3 kill-switch logic."""

from __future__ import annotations

from kiki_flow_core.track3_deploy.sweep import pick_top_k

FLIP_TOLERANCE = 0.15


def _fake_summary(kls: dict[str, float]) -> dict[str, object]:
    return {
        "phase": "pilot10k",
        "archs": {arch: {"test": {"total": kl}} for arch, kl in kls.items()},
    }


def test_pick_top_k_clear_winner() -> None:
    summary = _fake_summary({"A": 0.10, "B": 0.50, "C": 1.00})
    # 1.00 vs 0.10 gap is 900%, well above flip tolerance — normal Top-2
    assert pick_top_k(summary, k=2, flip_tolerance=FLIP_TOLERANCE) == ["A", "B"]


def test_pick_top_k_promotes_all_on_small_gap() -> None:
    summary = _fake_summary({"A": 0.100, "B": 0.105, "C": 0.110})
    # gap (0.110 - 0.100) / 0.100 = 10% < 15% → promote all
    assert pick_top_k(summary, k=2, flip_tolerance=FLIP_TOLERANCE) == ["A", "B", "C"]


def test_pick_top_k_empty() -> None:
    summary: dict[str, object] = {"phase": "pilot10k", "archs": {}}
    assert pick_top_k(summary) == []
