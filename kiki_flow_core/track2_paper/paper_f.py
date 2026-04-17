"""T2 free energy: potential + KL prior + Levelt-Baddeley reaction + Turing cross-diffusion."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy
from kiki_flow_core.species import OrthoSpecies
from kiki_flow_core.state import FlowState


class T2FreeEnergy(FreeEnergy):
    """F_T2 = sum_i <rho_i, V_i> + sum_i KL(rho_i || prior_i) + reaction + turing."""

    def __init__(
        self,
        species: OrthoSpecies,
        potentials: dict[str, np.ndarray],
        prior: dict[str, np.ndarray],
        turing_strength: float = 0.1,
    ) -> None:
        self.species = species
        self.potentials = potentials
        self.prior = prior
        self.turing_strength = turing_strength
        self._coupling = species.coupling_matrix()

    def value(self, state: FlowState) -> float:
        total = 0.0
        names = self.species.species_names()
        rhos = [state.rho[n] for n in names]

        for n, rho in zip(names, rhos, strict=True):
            total += float(np.dot(rho, self.potentials[n]))
            rho_safe = np.clip(rho, 1e-12, None)
            prior_safe = np.clip(self.prior[n], 1e-12, None)
            total += float(np.sum(rho_safe * np.log(rho_safe / prior_safe)))

        j = self._coupling
        for i, ri in enumerate(rhos):
            for k, rk in enumerate(rhos):
                total += float(j[i, k] * np.dot(ri, rk))

        if self.turing_strength > 0.0:
            turing = 0.0
            for i, ri in enumerate(rhos):
                for k in range(i + 1, len(rhos)):
                    rk = rhos[k]
                    turing += float(np.sum(np.abs(np.gradient(ri) * np.gradient(rk))))
            total += self.turing_strength * turing

        return total
