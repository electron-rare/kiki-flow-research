# AGENTS.md

Guidance for AI coding agents (Claude Code, Aider, Cursor, etc.) working in this repo.

## Project

`kiki-flow-research` — Wasserstein-gradient-flow engine for LLM consolidation under the Levelt-Baddeley four-species model. **Hypneum Lab** research code (workshop CLOSED per audit; paper figures + claims must remain numerically reproducible). Package: `kiki-flow-core` v0.0.1. Default branch `main`, currently on `docs/readme-stratification-link`.

## Tech stack

- Language: Python **3.14**
- Runtime: `uv` (PEP 668)
- Test: `pytest` (+ `pytest-cov`, `hypothesis`)
- Build: `hatchling`; packages `kiki_flow_core/`
- Core deps: `mlx>=0.20`, `jax>=0.4.30`, `jaxlib>=0.4.30`, `POT>=0.9`, `pydantic>=2.6`, `numpy>=2.1`, `optax`, `flax`, `safetensors`, `pyyaml`, `prometheus-client`, `scikit-learn`, `spacy`, `phonemizer`
- **No PyTorch** (see anti-patterns)

## Commands

```bash
uv sync
uv run pytest
uv run pytest tests/golden/                      # golden regressions
uv run python -m track2_paper.paper_run          # paper figures (T2)
uv run python -m track1_perf.offline_consolidator # nightly consolidator (T1)
```

## Conventions

- Commits: subject ≤ 50 chars, body ≤ 72, no underscore in scope, no AI attribution, never `--no-verify`.
- Branches: `feat/<name>`, `fix/<name>`, `docs/<name>`, `paper/<name>`.
- Track-isolation rule: code in `track1_perf/` must not import from `track2_paper/` or `track3_deploy/` (and vice versa). Shared code goes into `kiki_flow_core/`.
- Critic-review mandatory for ship-impacting commits — see `~/.claude/projects/-Users-electron/memory/feedback_critic_before_ship.md`.

## File layout

- `kiki_flow_core/` — shared core (state, master equation, Wasserstein ops, species, modules, hooks, telemetry). **Load-bearing for all three tracks.**
- `track1_perf/` — T1: nightly offline consolidation (40 hybrid species, Eulerian grid). Entry `track1_perf.offline_consolidator`.
- `track2_paper/` — T2: paper figures, N-particle Langevin × JKO, 4 Levelt-Baddeley species. Entry `track2_paper.paper_run.run_paper`.
- `track3_deploy/` — T3: pure-NumPy streaming surrogate, advisory routing. Entry `track3_deploy.streaming_runner`.
- `paper/` — LaTeX sources + fast-path figure generators.
- `paper_rigorous/` — 84-min rigorous verification run artifacts (`stats.json` is generated, do not hand-edit).
- `bench/` — SLO / latency / backend-speedup ledger (`bench/*.jsonl` is append-only).
- `tests/golden/*.npz` — golden numerical fixtures.
- `scripts/` — reproducibility / deterministic seeded runs.
- `docs/superpowers/` — specs, runbooks (incl. `docs/superpowers/runbooks/real-cl-bench.md`).

## Domain-specific gotchas

- **No PyTorch.** Stack is NumPy + MLX + JAX + POT on Python 3.14. Do not introduce `torch` deps.
- **Sinkhorn must run in log domain** (`method="sinkhorn_log"`) — vanilla kernel underflows at the project's epsilons (0.001-0.05). Do not "fix" non-convergence by suppressing warnings; raise `n_iter`, raise `epsilon`, or switch to the rigorous path.
- **`FlowState.rho[*]` are simplex vectors** (finite, non-negative, sum = 1, tol 1e-4). `assert_invariants` is the ground truth — add it to any new pipeline step.
- **`tau` is monotonic and non-negative**; one JKO step → `tau += 1`.
- **`metadata.track_id` is exactly `"T1" | "T2" | "T3"`** (Pydantic validator). Do not invent `T4`.
- **Do not hand-edit `paper/*.json` or `paper_rigorous/stats.json`** — they are script outputs. Re-run the generators.
- **`bench/*.jsonl` is append-only** (see `bench/CLAUDE.md`).
- **Stochastic code must take an explicit seed**; the `np.random.seed(42)` autouse fixture in `tests/conftest.py` is for tests only — use `np.random.default_rng(seed)` and thread the seed through.
- **Published figures use `SEEDS = [0, 1, 2, 3, 4]`** — don't silently change this.
- **Test tolerances are tuned** — on a legitimate algorithmic change, update the golden NPZ + document it; never just widen the tolerance.
- **Changes touching README "Quantitative results" claims** require re-running scripts + updating JSON/figures/stats. The README table is a contract.

## When in doubt

- Read `CLAUDE.md` (rich, with nested files in `kiki_flow_core/`, `paper/`, `paper_rigorous/`, `scripts/`, `tests/`, `bench/`).
- Spec: `docs/superpowers/specs/2026-04-17-kiki-flow-core-design.md`.
- Recent commits: `git log --oneline -20`.
- Cluster context: `~/CLAUDE.md`.
- Run `uv run pytest` (incl. `tests/golden/`) before non-trivial commits.
