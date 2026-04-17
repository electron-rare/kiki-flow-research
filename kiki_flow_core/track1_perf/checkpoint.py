"""Safetensors + JSON manifest serialization for FlowState."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from kiki_flow_core.state import FlowState


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def save_checkpoint(state: FlowState, path: Path) -> None:
    path = Path(path)
    tensors: dict[str, np.ndarray] = {f"rho::{k}": v for k, v in state.rho.items()}
    tensors["P_theta"] = state.P_theta
    tensors["mu_curr"] = state.mu_curr
    save_file(tensors, str(path.with_suffix(".safetensors")))
    manifest: dict[str, Any] = {
        "tau": state.tau,
        "metadata": state.metadata,
        "git_sha": _git_sha(),
        "timestamp": time.time(),
        "rho_keys": list(state.rho.keys()),
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))


def load_checkpoint(path: Path) -> FlowState:
    path = Path(path)
    tensors = load_file(str(path.with_suffix(".safetensors")))
    manifest = json.loads(path.with_suffix(".json").read_text())
    rho = {k.split("::", 1)[1]: v for k, v in tensors.items() if k.startswith("rho::")}
    return FlowState(
        rho=rho,
        P_theta=tensors["P_theta"],
        mu_curr=tensors["mu_curr"],
        tau=manifest["tau"],
        metadata=manifest["metadata"],
    )
