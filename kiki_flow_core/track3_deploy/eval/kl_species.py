"""Per-species KL, MAPE_Δ, routing hit@5, and paper ablation figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-8

SPECIES_SHORT: tuple[str, ...] = ("phono", "sem", "lex", "syntax")
SPECIES_CANONICAL: tuple[str, ...] = (
    "phono:code",
    "sem:code",
    "lex:code",
    "syntax:code",
)
SHORT_TO_CANONICAL: dict[str, str] = dict(zip(SPECIES_SHORT, SPECIES_CANONICAL, strict=True))

# Alias for downstream code indexing rho_by_species dicts with JKO-canonical keys
SPECIES: tuple[str, ...] = SPECIES_CANONICAL


def kl_per_species(rho_pred: np.ndarray, rho_target: np.ndarray) -> dict[str, float]:
    """KL(target || pred), keyed by SHORT species names for readability.

    Inputs shaped (B, 4, 32). Axis 1 is positional (order = SPECIES_SHORT).
    Returns {phono, sem, lex, syntax, total} where total = mean over species.
    """
    if rho_pred.shape != rho_target.shape:
        raise ValueError(f"shape mismatch: {rho_pred.shape} vs {rho_target.shape}")
    b, s, k = rho_pred.shape
    if s != len(SPECIES_SHORT):
        raise ValueError(f"expected {len(SPECIES_SHORT)} species, got {s}")
    out: dict[str, float] = {}
    for i, name in enumerate(SPECIES_SHORT):
        p = rho_pred[:, i, :]
        q = rho_target[:, i, :]
        kl = (q * (np.log(q + EPS) - np.log(p + EPS))).sum(axis=-1).mean()
        out[name] = float(kl)
    out["total"] = float(np.mean([out[name] for name in SPECIES_SHORT]))
    return out


def mape_delta(delta_pred: np.ndarray, delta_target: np.ndarray) -> float:
    """Mean absolute percentage error on predicted deltas."""
    num = np.abs(delta_pred - delta_target).sum(axis=-1)
    den = np.abs(delta_target).sum(axis=-1) + EPS
    return float(np.mean(num / den))


def hit_at_k_routing(
    base: np.ndarray,
    bridge_pred: np.ndarray,
    oracle: np.ndarray,
    k: int = 5,
    blend_weight: float = 0.1,
) -> float:
    """Blend (1-w)*base + w*bridge_pred; intersection of top-k with top-k(base + oracle)."""
    blended = (1 - blend_weight) * base + blend_weight * bridge_pred
    oracle_blend = base + oracle
    blended_top = np.argpartition(-blended, kth=k - 1, axis=-1)[:, :k]
    oracle_top = np.argpartition(-oracle_blend, kth=k - 1, axis=-1)[:, :k]
    hits = [len(set(b) & set(o)) > 0 for b, o in zip(blended_top, oracle_top, strict=True)]
    return float(np.mean(hits))


def evaluate_checkpoint(
    encoder: Any,
    bridge_params: dict[str, Any],
    pairs: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    """Run a checkpoint over pairs and return all metrics.

    Each pair: {text, state_pre, state_post, rho_by_species (canonical keys)}.
    Optionally {base_scores, oracle_advisory} for hit@k.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from kiki_flow_core.track3_deploy.surrogate_trainer_v3 import (  # noqa: PLC0415
        _BridgeHead,
        _softmax_per_species,
    )

    texts = [p["text"] for p in pairs]
    spre = np.stack([p["state_pre"] for p in pairs])
    spost = np.stack([p["state_post"] for p in pairs])
    rho_target = np.stack(
        [
            np.stack([p["rho_by_species"][canonical] for canonical in SPECIES_CANONICAL])
            for p in pairs
        ]
    )  # (B, 4, 32)

    enc_out = encoder.encode(texts)
    feats = jnp.concatenate([jnp.asarray(spre), jnp.asarray(enc_out)], axis=-1)
    delta_pred = np.asarray(_BridgeHead.forward(bridge_params, feats))
    pred_state = spre + delta_pred
    rho_pred = np.asarray(_softmax_per_species(jnp.asarray(pred_state)))
    delta_target = spost - spre

    result = kl_per_species(rho_pred, rho_target)
    result["mape_delta"] = mape_delta(delta_pred, delta_target)
    if all("base_scores" in p and "oracle_advisory" in p for p in pairs):
        base = np.stack([p["base_scores"] for p in pairs])
        oracle = np.stack([p["oracle_advisory"] for p in pairs])
        bridge_adv = delta_pred[:, :_N_STACKS_FOR_ROUTING]
        result["hit_at_5"] = hit_at_k_routing(base, bridge_adv, oracle, k=k)
    return result


_N_STACKS_FOR_ROUTING = 32


def plot_ablation_figure(
    results_10k: dict[str, dict[str, float]],
    results_50k: dict[str, dict[str, float]],
    baseline_v02: dict[str, float] | None,
    output_path: Path | str,
) -> None:
    """Render Figure 4.x (2 panels): species breakdown + scaling curves."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: stacked species bars
    labels = list(results_50k.keys())
    if baseline_v02:
        labels = ["v0.2 (no text)", *labels]
    bottoms = np.zeros(len(labels))
    colors = {
        "phono": "#440154",
        "sem": "#3b528b",
        "lex": "#21918c",
        "syntax": "#5ec962",
    }
    for species in SPECIES_SHORT:
        heights = []
        for lab in labels:
            if lab == "v0.2 (no text)":
                heights.append(baseline_v02[species] if baseline_v02 else 0.0)
            else:
                heights.append(results_50k[lab][species])
        ax_a.bar(labels, heights, bottom=bottoms, label=species, color=colors[species])
        bottoms += np.asarray(heights)
    ax_a.set_ylabel("KL divergence")
    ax_a.set_title("(a) Species breakdown at 50k")
    ax_a.legend(loc="upper right", fontsize=8)

    # Panel B: scaling curves
    for arch, res10 in results_10k.items():
        if arch in results_50k:
            ax_b.plot(
                [10_000, 50_000],
                [res10["total"], results_50k[arch]["total"]],
                marker="o",
                label=arch,
            )
        else:
            ax_b.plot(
                [10_000], [res10["total"]], marker="x", linestyle="", label=f"{arch} (10k only)"
            )
    if baseline_v02:
        ax_b.axhline(baseline_v02["total"], color="gray", linestyle=":", label="v0.2 baseline")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Corpus size")
    ax_b.set_ylabel("KL_total (test)")
    ax_b.set_title("(b) Scaling behavior")
    ax_b.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
