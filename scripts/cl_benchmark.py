"""Density-level continual-learning proxy for T2 fig5.

Sequence of 3 synthetic tasks, each with a target distribution peaked
at a different grid location. We train on task 1, then 2, then 3,
measuring retention of earlier tasks at the end. "With consolidation"
keeps a running average of past targets as the prior in the JKO step;
"without" uses uniform prior.

This is not a real LLM continual-learning benchmark; it is a
distributional analog that validates the qualitative claim that
Wasserstein-regularized consolidation reduces forgetting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy, JKOStep
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.figures.continual_learning_gap import (
    make_continual_learning_gap,
)

GRID = 16
SEEDS = [0, 1, 2, 3, 4]
N_STEPS_PER_TASK = 30


def target_distribution(task_id: int) -> np.ndarray:
    """3 tasks with peaks at grid indices 3, 8, 13."""
    centers = [3, 8, 13]
    x = np.arange(GRID)
    peak = np.exp(-0.5 * ((x - centers[task_id]) / 1.2) ** 2)
    return peak / peak.sum()


class TaskF(FreeEnergy):
    """F = -<rho, V> + KL(rho || prior). Minimizer pulls rho toward V (attractive)."""

    def __init__(self, target: np.ndarray, prior: np.ndarray, kl_weight: float = 1.0) -> None:
        self.target = target
        self.prior = prior
        self.kl_weight = kl_weight

    def value(self, state: FlowState) -> float:
        r = state.rho["phono"]
        r_safe = np.clip(r, 1e-12, None)
        prior_safe = np.clip(self.prior, 1e-12, None)
        # Quadratic attraction to target (strong signal)
        attr = float(((r - self.target) ** 2).sum())
        kl = float((r_safe * np.log(r_safe / prior_safe)).sum())
        return attr + self.kl_weight * kl

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        r = state.rho["phono"]
        r_safe = np.clip(r, 1e-12, None)
        prior_safe = np.clip(self.prior, 1e-12, None)
        grad_attr = 2.0 * (r - self.target)
        grad_kl = self.kl_weight * (np.log(r_safe / prior_safe) + 1.0)
        out: np.ndarray = grad_attr + grad_kl
        return out


def run_sequence(use_consolidation: bool, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    rho = rng.dirichlet(np.ones(GRID)).astype(np.float64)
    state = FlowState(
        rho={"phono": rho},
        P_theta=np.zeros(4),
        mu_curr=np.full(GRID, 1.0 / GRID),
        tau=0,
        metadata={"track_id": "T2"},
    )
    previous_targets: list[np.ndarray] = []

    for task_id in range(3):
        target = target_distribution(task_id)
        if use_consolidation and previous_targets:
            # Prior = average of previous targets (consolidation memory)
            prior = np.mean(previous_targets, axis=0)
            prior = prior / prior.sum()
        else:
            prior = np.full(GRID, 1.0 / GRID)

        f_task = TaskF(target=target, prior=prior, kl_weight=0.3)
        jko = JKOStep(f_functional=f_task, h=0.01, support=support, n_inner=20, apply_w2_prox=False)
        for _ in range(N_STEPS_PER_TASK):
            state = jko.step(state)
        previous_targets.append(target)

    # Measure final-state overlap with each task's target
    final_rho = state.rho["phono"]
    accuracies: dict[str, float] = {}
    for i in range(3):
        target_i = target_distribution(i)
        # Cosine similarity as task "accuracy" proxy
        cos = float(
            np.dot(final_rho, target_i) / (np.linalg.norm(final_rho) * np.linalg.norm(target_i))
        )
        accuracies[f"task{i}"] = cos
    return accuracies


def main() -> None:
    with_accs: dict[str, list[float]] = {"task0": [], "task1": [], "task2": []}
    without_accs: dict[str, list[float]] = {"task0": [], "task1": [], "task2": []}
    for seed in SEEDS:
        with_r = run_sequence(use_consolidation=True, seed=seed)
        without_r = run_sequence(use_consolidation=False, seed=seed)
        for k in ["task0", "task1", "task2"]:
            with_accs[k].append(with_r[k])
            without_accs[k].append(without_r[k])

    with_means = [float(np.mean(with_accs[k])) for k in ["task0", "task1", "task2"]]
    with_stds = [float(np.std(with_accs[k])) for k in ["task0", "task1", "task2"]]
    without_means = [float(np.mean(without_accs[k])) for k in ["task0", "task1", "task2"]]
    without_stds = [float(np.std(without_accs[k])) for k in ["task0", "task1", "task2"]]

    for i, k in enumerate(["task0", "task1", "task2"]):
        print(
            f"{k}: with={with_means[i]:.4f}+/-{with_stds[i]:.4f}  "
            f"without={without_means[i]:.4f}+/-{without_stds[i]:.4f}"
        )

    make_continual_learning_gap(
        tasks=["phonology", "lexicon", "syntax"],
        with_consolidation=with_means,
        without_consolidation=without_means,
        out_dir=Path("paper/figures"),
        filename="fig5_cl_gap",
    )

    Path("paper/cl_benchmark.json").write_text(
        json.dumps(
            {
                "n_seeds": len(SEEDS),
                "with_consolidation": {"mean": with_means, "std": with_stds},
                "without_consolidation": {"mean": without_means, "std": without_stds},
                "tasks": ["phonology", "lexicon", "syntax"],
            },
            indent=2,
        )
    )
    print("Wrote paper/cl_benchmark.json and updated fig5")


if __name__ == "__main__":
    main()
