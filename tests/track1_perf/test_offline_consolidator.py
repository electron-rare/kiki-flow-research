from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from kiki_flow_core.track1_perf.offline_consolidator import run_once


def test_run_once_smoke(tmp_path: Path):
    def aeon_fetcher(h: int) -> list[dict]:
        return [{"id": 1, "concepts": ["phono"]}]

    def moe_snapshotter() -> dict[str, np.ndarray]:
        return {"code": np.zeros(4), "math": np.zeros(4)}

    routing = MagicMock()
    cfg = {
        "stack_names": ["code", "math"],
        "n_grid": 16,
        "checkpoint_dir": tmp_path,
    }
    manifest_out = run_once(
        config=cfg,
        aeon_fetcher=aeon_fetcher,
        moe_snapshotter=moe_snapshotter,
        advisory_publisher=routing.publish_advisory,
    )
    assert manifest_out["status"] == "ok"
    assert routing.publish_advisory.called
    assert (tmp_path / "latest.safetensors").exists()
