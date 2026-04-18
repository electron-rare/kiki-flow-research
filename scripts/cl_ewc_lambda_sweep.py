"""EWC lambda sensitivity sweep on the embedding-driven CL benchmark.

Follow-up to scripts/cl_benchmark_embeddings.py. Varies the EWC
regularization strength across a grid to surface the stability /
plasticity frontier and identify an operating point that matches the
naive baseline on later tasks while keeping the task-0 benefit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.cl_benchmark_embeddings as emb_mod
from scripts.cl_benchmark_embeddings import (
    DOMAINS,
    SEEDS,
    build_target_histograms,
    run_sequence,
)

LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def main() -> None:
    all_results: dict[float, dict[str, dict[str, float]]] = {}
    for lam in LAMBDAS:
        emb_mod.EWC_LAMBDA = lam  # monkey-patch the embeddings module constant
        per_domain: dict[str, list[float]] = {name: [] for name in DOMAINS}
        for seed in SEEDS:
            task_targets = build_target_histograms(seed)
            result = run_sequence("with_ewc", seed, task_targets)
            for name in DOMAINS:
                per_domain[name].append(result[name])
        all_results[lam] = {
            name: {
                "mean": float(np.mean(per_domain[name])),
                "std": float(np.std(per_domain[name])),
            }
            for name in DOMAINS
        }
        means = [all_results[lam][name]["mean"] for name in DOMAINS]
        min_task = min(means)
        mean_all = float(np.mean(means))
        print(
            f"lambda={lam:>5.1f}  phono={means[0]:.3f}  "
            f"lex={means[1]:.3f}  syn={means[2]:.3f}  "
            f"mean={mean_all:.3f}  min={min_task:.3f}"
        )

    best_lam = max(
        LAMBDAS,
        key=lambda lam: min(all_results[lam][name]["mean"] for name in DOMAINS),
    )
    best_min = min(all_results[best_lam][name]["mean"] for name in DOMAINS)
    print(f"\nBest max-min: lambda={best_lam}  min-task={best_min:.4f}")

    Path("paper/cl_ewc_lambda_sweep.json").write_text(
        json.dumps(
            {
                "n_seeds": len(SEEDS),
                "lambdas": LAMBDAS,
                "results": {str(lam): all_results[lam] for lam in LAMBDAS},
                "best_lambda_by_max_min": best_lam,
                "best_min_task_score": best_min,
            },
            indent=2,
        )
    )
    print("Wrote paper/cl_ewc_lambda_sweep.json")


if __name__ == "__main__":
    main()
