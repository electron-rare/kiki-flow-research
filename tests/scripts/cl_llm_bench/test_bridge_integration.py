# tests/scripts/cl_llm_bench/test_bridge_integration.py
from pathlib import Path

import numpy as np
import pytest

from scripts.cl_llm_bench.bridge_integration import AdvisoryRecorder, blend_advisory

EXPECTED_LINES = 2


def test_advisory_recorder_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIKI_FLOW_ENABLED", "1")
    out = tmp_path / "advisory.jsonl"
    weights_path = Path("kiki_flow_core/track3_deploy/weights/v0.2-d128.safetensors")
    recorder = AdvisoryRecorder(weights_path=weights_path, out_path=out)
    recorder.record("train sample 1", task_name="phono_sst2", step=0)
    recorder.record("train sample 2", task_name="phono_sst2", step=1)
    recorder.flush()
    lines = out.read_text().splitlines()
    assert len(lines) == EXPECTED_LINES


PRIOR_WEIGHT = 0.2


def test_blend_advisory_with_none_returns_base_unchanged() -> None:
    base = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out = blend_advisory(base, advisory=None, prior_weight=0.1)
    np.testing.assert_array_equal(out, base)


def test_blend_advisory_weighted_sum() -> None:
    base = np.full(4, 0.25, dtype=np.float32)
    adv = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = blend_advisory(base, advisory=adv, prior_weight=PRIOR_WEIGHT)
    expected = (1.0 - PRIOR_WEIGHT) * base + PRIOR_WEIGHT * adv
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_blend_advisory_shape_mismatch_returns_base() -> None:
    base = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    adv = np.array([1.0, 0.0], dtype=np.float32)
    out = blend_advisory(base, advisory=adv, prior_weight=0.5)
    np.testing.assert_array_equal(out, base)
