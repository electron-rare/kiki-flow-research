import json
from pathlib import Path

import numpy as np

from kiki_flow_core.state import FlowState
from kiki_flow_core.track1_perf.checkpoint import load_checkpoint, save_checkpoint


def make_state() -> FlowState:
    return FlowState(
        rho={"phono:code": np.array([0.5, 0.5])},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=3,
        metadata={"track_id": "T1", "step_id": "abc"},
    )


def test_save_roundtrip(tmp_path: Path):
    state = make_state()
    path = tmp_path / "ckpt"
    save_checkpoint(state, path)
    assert (path.with_suffix(".safetensors")).exists()
    assert (path.with_suffix(".json")).exists()
    loaded = load_checkpoint(path)
    assert loaded.tau == 3  # noqa: PLR2004
    np.testing.assert_array_equal(loaded.rho["phono:code"], state.rho["phono:code"])


def test_manifest_contains_git_sha(tmp_path: Path):
    state = make_state()
    path = tmp_path / "ckpt"
    save_checkpoint(state, path)
    manifest = json.loads((path.with_suffix(".json")).read_text())
    assert "git_sha" in manifest
    assert "timestamp" in manifest
