"""Reaction-diffusion solver: upwind advection + explicit diffusion."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.species.base import SpeciesBase


class AdvectionDiffusion:
    """1D first-order upwind advection + explicit diffusion with mass conservation."""

    def __init__(
        self,
        species: SpeciesBase | None,
        x_grid: np.ndarray,
        diffusion: float = 0.0,
    ) -> None:
        self.species = species
        self.x = x_grid
        self.dx = float(x_grid[1] - x_grid[0])
        self.d_coef = diffusion

    def step_1d(
        self,
        rho: np.ndarray,
        v_field: np.ndarray,
        dt: float,
        source: np.ndarray | None = None,
    ) -> np.ndarray:
        """One explicit step: upwind advection + explicit diffusion + optional source."""
        if rho.shape != v_field.shape:
            raise ValueError(f"rho {rho.shape} and v_field {v_field.shape} must match")

        # Upwind advection with zero-flux boundary
        v_pos = np.maximum(v_field, 0.0)
        v_neg = np.minimum(v_field, 0.0)
        rho_left = np.roll(rho, 1)
        rho_right = np.roll(rho, -1)
        rho_left[0] = rho[0]
        rho_right[-1] = rho[-1]
        flux = -(v_pos * (rho - rho_left) + v_neg * (rho_right - rho)) / self.dx
        rho_adv = rho + dt * flux

        # Explicit diffusion (CFL-bounded)
        if self.d_coef > 0.0:
            laplacian = np.roll(rho_adv, -1) - 2 * rho_adv + np.roll(rho_adv, 1)
            laplacian[0] = laplacian[1]
            laplacian[-1] = laplacian[-2]
            rho_diff = rho_adv + dt * self.d_coef * laplacian / (self.dx**2)
        else:
            rho_diff = rho_adv

        if source is not None:
            rho_diff = rho_diff + dt * source

        # Clip negatives from numerical roundoff and renormalize to unit mass
        rho_diff = np.clip(rho_diff, 0.0, None)
        total = rho_diff.sum()
        if total > 0:
            rho_diff = rho_diff / total
        out: np.ndarray = rho_diff
        return out
