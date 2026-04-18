# scripts — paper reproducibility entry points

Every script here is a deterministic producer of one or more artifacts
consumed by `paper/` (JSON + PDFs). Running a script must be idempotent
given the same seeds.

## Script -> artifact map

| Script | Produces | Referenced in |
|---|---|---|
| `cl_benchmark.py` | `paper/cl_benchmark.json`, `paper/figures/fig5_cl_gap.*` | README results table |
| `cl_benchmark_ewc.py` | `paper/cl_benchmark_ewc.json`, `fig5_cl_gap_ewc.*` | paper appendix |
| `epsilon_sweep.py` | `paper/epsilon_sweep.json`, `fig4_kl_vs_epsilon.*` | Sinkhorn-bias claim |
| `hyperparam_sweep.py` | `paper/hyperparam_sweep.json`, `fig6_heatmap.*` | specialization claim |
| `hyperparam_sweep_dense.py` | `paper/hyperparam_sweep_dense.json` | denser follow-up sweep |
| `dump_t2_pairs.py` | `bench/runs/T2_pairs/*.safetensors` | T3 surrogate trainer input |
| `dump_hybrid_pairs.py` | `bench/runs/T2_pairs_d128/*.safetensors` | T3 v0.2-d128 trainer input |
| `cl_llm_bench/` (subpackage) | `bench/cl_llm/runs/<name>/result.json` + `paper/figures/fig7_cl_forgetting.*` | real-LLM CL §3.4 + issue #1 |

Figures live under `paper/figures/`, emitted via the generators in
`kiki_flow_core/track2_paper/figures/` — scripts call those, never write
PDFs directly.

## `cl_llm_bench/` subpackage — real-LLM CL with three modes

Unlike the other scripts above (pure Python, single machine, deterministic),
`cl_llm_bench/` orchestrates a real LoRA fine-tuning loop on a remote GPU
host. It exposes a CLI with three modes:

- `--mode stub` — synthetic forgetting summary, runs in CI. Plausibility
  numbers derived from the distributional-proxy runs above. No network.
- `--mode preflight` — read-only SSH probe of the remote host (weights,
  datasets, uv, disk, GPU). Returns a structured `preflight.json` with
  per-check status and a single `ready_for_real` boolean.
- `--mode real` — gated by `--i-confirm-heavy-training`, launches the
  full CL sequence: per-task JSONL prep on the local machine, rsync up,
  SSH-invoke the remote trainer (`scripts/cl_llm_bench/kxkm_trainer/`
  with PEP-723 inline deps and 4-bit QLoRA), parse manifests, post-sequence
  evals on prior tasks, compute forgetting, write `result.json`. See
  `docs/superpowers/runbooks/real-cl-bench.md` for the end-to-end recipe.

Artifacts live under `bench/cl_llm/runs/<run-name>/`; the aggregated
5-seed summary at `bench/cl_llm/runs/e2_5seeds_summary.json` is the
single source of truth for the paper §3.4 real-LLM claim.

## Rules

- Seeds are declared at module top (`SEEDS = [0, 1, 2, 3, 4]`). Don't
  randomize silently; don't shrink the list to "make it faster".
- Invocation form is `PYTHONPATH=. uv run python scripts/<name>.py` —
  `pyproject.toml` already has `pythonpath = ["."]` for pytest, but
  scripts run outside pytest and need the explicit env var.
- Paths are relative to repo root (e.g. `Path("paper/cl_benchmark.json")`).
  Never use absolute paths; never compute paths from `__file__` gymnastics.
- Output JSON schema must stay backward-compatible with existing files in
  `paper/` and `paper_rigorous/` — the LaTeX source reads them. If a
  schema change is unavoidable, update `paper/main.tex` in the same commit.

## Anti-patterns (domain-specific)

- Adding a script that mutates an existing JSON file rather than rewriting
  it. JSON outputs are full snapshots of a run; partial updates break
  provenance.
- Embedding seeds inside the loop body (`rng = np.random.default_rng(0)`
  hard-coded). Accept `seed: int` as a parameter, expose `SEEDS` at the
  top of the module.
- Using `print(...)` for the result and forgetting the JSON write — the
  terminal output is a diagnostic, the JSON file is the artifact.
- Calling `plt.show()` in a reproducibility script. Save to disk only.
- Long-running sweeps with no checkpointing. If the script takes more
  than ~10 min, write incremental partial results so a crash doesn't
  destroy N hours of compute (see `hyperparam_sweep_dense.py` for the
  pattern).
