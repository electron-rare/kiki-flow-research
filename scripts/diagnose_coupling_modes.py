"""One-shot diagnostic: does each of the five coupling modes instantiate and
run a minimal sweep cell without crashing?

Mirrors hyperparam_sweep_coupling_modes.py but restricted to a single
(alpha, beta, seed) so the full pipeline is exercised end-to-end in < 2 min.

Run with: ``PYTHONPATH=. uv run python scripts/diagnose_coupling_modes.py``.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.paper_f_separable import SeparableEnergy

GRID = 16
ALPHA = 10.0
BETA = 1.0
SEED = 0
MODES = ["separable", "symmetric-dell", "dell-full", "symmetric-levelt", "levelt-full"]


def build_energy(mode: str) -> object:
    variant = "levelt" if "levelt" in mode else "dell"
    species = CanonicalSpecies(coupling_variant=variant)
    names = species.species_names()
    potentials = {n: ALPHA * np.linspace(-1.0, 1.0, GRID) for n in names}
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    if mode == "separable":
        return species, SeparableEnergy(species=species, potentials=potentials, prior=prior)
    f = T2FreeEnergy(
        species=species,
        potentials=potentials,
        prior=prior,
        turing_strength=0.0,
    )
    if mode.startswith("symmetric-"):
        zero = np.zeros_like(f._J_asym)
        f._J_asym = zero
        f._coupling = f._J_sym.copy()
    return species, f


def run_diagnostic(mode: str) -> tuple[bool, str, float]:
    t0 = time.time()
    try:
        species, energy = build_energy(mode)
        # Minimal forward evaluation: compute F on a uniform state.
        rhos = {n: np.full(GRID, 1.0 / GRID) for n in species.species_names()}
        state = FlowState(
            rho=rhos,
            P_theta=np.zeros(4),
            mu_curr=np.full(GRID, 1.0 / GRID),
            tau=0,
            metadata={"track_id": "T2"},
        )
        val = energy.value(state)
        elapsed = time.time() - t0
        if not np.isfinite(val):
            return False, f"non-finite F = {val}", elapsed
        return True, f"F = {val:+.6f}", elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc(limit=2).strip().splitlines()[-1]
        return False, f"{type(exc).__name__}: {exc} ({tb})", elapsed


def main() -> int:
    print(f"Diagnostic: 5 coupling modes at (alpha={ALPHA}, beta={BETA}, seed={SEED})")
    print(f"Grid = {GRID}, species = 4 (phono, lex, syntax, sem)\n")
    failures = []
    for mode in MODES:
        ok, msg, elapsed = run_diagnostic(mode)
        tag = "OK " if ok else "FAIL"
        print(f"  [{tag}] {mode:<20s}  {msg}  ({elapsed * 1000:.1f} ms)")
        if not ok:
            failures.append(mode)
    print()
    if failures:
        print(f"DIAGNOSTIC FAILED on {len(failures)} mode(s): {failures}")
        return 1
    print("DIAGNOSTIC PASSED: all 5 modes instantiate and evaluate F without error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
