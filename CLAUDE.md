# CLAUDE.md

Wasserstein-gradient-flow engine for LLM consolidation under the Levelt-Baddeley
four-species model. Research code: results tables in `README.md` are claims that
must stay numerically reproducible.

## Three tracks, one core

All tracks share `kiki_flow_core/` (state, master equation, Wasserstein ops,
species, modules, hooks, telemetry). Track-specific code lives in subpackages:

| Track | Purpose | Entry point |
|---|---|---|
| T1 `track1_perf` | nightly offline consolidation (40 hybrid species, Eulerian grid) | `track1_perf.offline_consolidator` |
| T2 `track2_paper` | paper figures, N-particle Langevin x JKO, 4 Levelt-Baddeley species | `track2_paper.paper_run.run_paper` |
| T3 `track3_deploy` | pure-NumPy streaming surrogate, advisory routing | `track3_deploy.streaming_runner` |

`metadata.track_id` on every `FlowState` must be exactly `"T1" | "T2" | "T3"`
(Pydantic validator enforces this).

## Where to look

| Need | Read |
|---|---|
| Math / numerical contracts | `kiki_flow_core/CLAUDE.md` |
| Paper sources, LaTeX, fast-path figures | `paper/CLAUDE.md` |
| 84-min rigorous verification run artifacts | `paper_rigorous/CLAUDE.md` |
| Reproducibility scripts (deterministic, seeded) | `scripts/CLAUDE.md` |
| Real-LLM CL benchmark (remote GPU, manual) | `docs/superpowers/runbooks/real-cl-bench.md` |
| Test conventions, golden regressions | `tests/CLAUDE.md` |
| SLO / latency / backend-speedup ledger | `bench/CLAUDE.md` |
| Published claims to preserve | `README.md` quantitative-results table |
| Full formalism | `docs/superpowers/specs/2026-04-17-kiki-flow-core-design.md` |

## Core invariants (never break)

- `FlowState.rho[*]` are simplex vectors: finite, non-negative, sum = 1 (tol 1e-4).
  `assert_invariants` is the ground truth; add it to any new pipeline step.
- `tau` is monotonic and non-negative. One JKO step -> `tau += 1`.
- `track_id` in metadata is T1/T2/T3 only. Do not invent T4.
- Sinkhorn in log domain (`method="sinkhorn_log"`) — vanilla kernel underflows
  at the epsilons we use (0.001-0.05).

## Agent workflow

1. Before editing math code: read `kiki_flow_core/CLAUDE.md` and the relevant
   test file — golden fixtures in `tests/golden/*.npz` pin numerical behaviour.
2. Before changing anything reachable from `paper/main.tex` or a figure
   generator: check whether the change invalidates a claim in
   `README.md` -> "Quantitative results". If yes, the fix is to re-run the
   scripts and update JSON + figures + stats, not to hand-edit numbers.
3. Prefer adding new species / modules / hooks to touching `state.py`,
   `master_equation.py`, or `wasserstein_ops.py`. Those three are load-bearing
   for all three tracks simultaneously.
4. Track-isolation rule: a change in `track1_perf/` must not import from
   `track2_paper/` or `track3_deploy/` (and vice versa). Shared code goes up
   into `kiki_flow_core/` top-level.

## Reproducibility non-negotiables

- Every script and test that does stochastic work takes an explicit seed.
  `tests/conftest.py` sets `np.random.seed(42)` per test via autouse fixture;
  do not rely on it for new stochastic code — use `np.random.default_rng(seed)`
  locally and thread the seed through.
- Published figures correspond to `SEEDS = [0, 1, 2, 3, 4]`. Don't silently
  change this list.
- Numerical tolerances in tests are tuned — if a test starts failing because
  of a legitimate algorithmic change, update the golden NPZ and document it
  in the commit; never just widen the tolerance.

## Anti-patterns (this repo)

- Using PyTorch. Stack is NumPy + MLX + JAX + POT on Python 3.14. No torch.
- Adding `warnings.filterwarnings("ignore")` to hide Sinkhorn non-convergence.
  Either raise `n_iter`, raise `epsilon`, or switch to the rigorous path.
- Writing results into `paper/*.json` or `paper_rigorous/stats.json` by hand.
  They are script outputs.
- Editing `bench/*.jsonl` in place — those are append-only ledgers (see
  `bench/CLAUDE.md`).
