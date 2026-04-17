"""Vygotskyan scaffolding scheduler: h_tau and curriculum potential mu_curr."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _default_zpd_oracle(errs: np.ndarray) -> float:
    """Default ZPD oracle: mean absolute error as a proxy for learning difficulty."""
    return float(errs.mean())


class ScaffoldingScheduler:
    """Adaptive step size and curriculum based on the Zone of Proximal Development (ZPD).

    The ZPD is encoded as a scalar in [0, 1]:
      - 0 = nothing to learn (system masters everything) -> large step h_max
      - 1 = everything beyond reach (errors saturate) -> small step h_min
    Curriculum potential mu_curr concentrates where errors are highest,
    via a softmax with a temperature parameter.
    """

    def __init__(
        self,
        h_min: float = 1e-3,
        h_max: float = 1.0,
        zpd_oracle: Callable[[np.ndarray], float] | None = None,
        temperature: float = 1.0,
    ) -> None:
        if h_min >= h_max:
            raise ValueError("h_min must be < h_max")
        self.h_min = h_min
        self.h_max = h_max
        self.zpd_oracle = zpd_oracle if zpd_oracle is not None else _default_zpd_oracle
        self.temperature = temperature

    def next_step(self, error_profile: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (h_tau, mu_curr) given the recent error profile."""
        zpd = float(np.clip(self.zpd_oracle(error_profile), 0.0, 1.0))
        h = self.h_max - (self.h_max - self.h_min) * zpd
        h = float(np.clip(h, self.h_min, self.h_max))

        # Softmax over errors (concentrate curriculum where errors are high)
        if error_profile.size == 0:
            mu = np.array([1.0])
        else:
            scaled = error_profile / max(self.temperature, 1e-6)
            scaled = scaled - scaled.max()  # numerical stability
            mu = np.exp(scaled)
            mu = mu / mu.sum()
        return h, mu
