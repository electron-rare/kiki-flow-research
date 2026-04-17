"""Flatten / unflatten FlowState between tensor format and dict-of-arrays."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.state import FlowState


def flatten(state: FlowState) -> np.ndarray:
    """Concatenate all rho species arrays in sorted-key order into a single 1D vector."""
    pieces = [state.rho[k] for k in sorted(state.rho.keys())]
    return np.concatenate(pieces)


def unflatten(arr: np.ndarray, reference: FlowState) -> FlowState:
    """Cut a flat array back into per-species rho using the reference's shape template."""
    new_rho: dict[str, np.ndarray] = {}
    offset = 0
    for k in sorted(reference.rho.keys()):
        n = reference.rho[k].size
        new_rho[k] = arr[offset : offset + n].copy()
        offset += n
    return reference.model_copy(update={"rho": new_rho})
