"""Wasserstein-2 operations via entropic Sinkhorn (POT-backed)."""

from __future__ import annotations

import numpy as np
import ot


def sinkhorn_cost(
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray,
    epsilon: float = 0.01,
    n_iter: int = 1000,
) -> float:
    """Entropic Sinkhorn approximation of OT cost.

    Uses log-domain Sinkhorn (``method='sinkhorn_log'``) for numerical stability
    when ``epsilon`` is small relative to the cost-matrix scale — the vanilla
    kernel ``exp(-M/epsilon)`` underflows otherwise and the transport plan
    collapses toward zero.
    """
    transport = ot.sinkhorn(
        a,
        b,
        cost_matrix,
        reg=epsilon,
        numItermax=n_iter,
        stopThr=1e-9,
        method="sinkhorn_log",
    )
    return float((transport * cost_matrix).sum())


def w2_distance(
    p: np.ndarray,
    q: np.ndarray,
    support: np.ndarray,
    epsilon: float = 0.005,
    n_iter: int = 2000,
) -> float:
    """Wasserstein-2 distance between p and q on shared support."""
    cost_matrix = ot.dist(support, support, metric="sqeuclidean")
    sq = sinkhorn_cost(p, q, cost_matrix, epsilon=epsilon, n_iter=n_iter)
    return float(np.sqrt(max(sq, 0.0)))


def prox_w2(
    distribution: np.ndarray,
    reference: np.ndarray,
    epsilon: float,
    support: np.ndarray,
    n_iter: int = 500,
    step_size: float = 0.1,
) -> np.ndarray:
    """W2 proximal operator: argmin_{q} 0.5 * W2^2(q, reference) + epsilon * KL(q || distribution).

    Uses iterative gradient descent on the simplex via projected gradient steps.
    """
    cost_matrix = ot.dist(support, support, metric="sqeuclidean")
    q = distribution.copy()
    reg = max(epsilon, 1e-3)
    for _ in range(n_iter):
        transport = ot.sinkhorn(
            q,
            reference,
            cost_matrix,
            reg=reg,
            numItermax=200,
            method="sinkhorn_log",
        )
        grad = (transport @ cost_matrix.diagonal()) - epsilon * np.log(q / distribution + 1e-12)
        q = q - step_size * grad
        q = np.clip(q, 1e-12, None)
        q = q / q.sum()
    return q
