"""Orchestrate the 3-architecture ablation sweep: pilot 10k → rank → scale 50k on Top-k."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Side-effect imports to populate ENCODER_REGISTRY via @register decorators
import kiki_flow_core.track3_deploy.encoders.distilled  # noqa: F401
import kiki_flow_core.track3_deploy.encoders.hash_mlp  # noqa: F401
import kiki_flow_core.track3_deploy.encoders.tiny_tf  # noqa: F401
from kiki_flow_core.track3_deploy.data.jko_cache import JKOCache
from kiki_flow_core.track3_deploy.encoders import ENCODER_REGISTRY
from kiki_flow_core.track3_deploy.eval.kl_species import SPECIES_CANONICAL, evaluate_checkpoint
from kiki_flow_core.track3_deploy.surrogate_trainer_v3 import JointTrainer

logger = logging.getLogger(__name__)

ARCH_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "B_distilled": {"lr": 3e-4, "batch": 128, "epochs": 20},
    "C_hash_mlp": {"lr": 3e-4, "batch": 128, "epochs": 20},
    "D_tiny_tf": {"lr": 1e-4, "batch": 64, "epochs": 30},
}

_EARLY_STOP_PATIENCE = 3
_MIN_VAL_IMPROVEMENT = 1e-4
_DEFAULT_SEED = 0
_DEFAULT_LAM = 0.5
_DEFAULT_TOP_K = 2
_FLIP_TOLERANCE = 0.15


def _load_split(corpus_dir: Path, split: str) -> list[dict[str, Any]]:
    path = corpus_dir / f"{split}.jsonl"
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _pairs_from_cache(entries: list[dict[str, Any]], cache: JKOCache) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        pair = cache.get(e["text"])
        if pair is None:
            continue
        out.append({"text": e["text"], **pair})
    return out


def _batch_stack_rho(batch: list[dict[str, Any]]) -> np.ndarray:
    """Stack rho_by_species into (B, 4, 32), species axis in SPECIES_CANONICAL order."""
    return np.stack(
        [np.stack([b["rho_by_species"][sp] for sp in SPECIES_CANONICAL]) for b in batch]
    )


def train_one_arch(
    arch: str,
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    hyper: dict[str, Any],
    output_dir: Path,
    seed: int = _DEFAULT_SEED,
    lam: float = _DEFAULT_LAM,
) -> dict[str, Any]:
    """Train one encoder+bridge combo, save the best-val-KL checkpoint to output_dir."""
    enc_cls = ENCODER_REGISTRY[arch]
    encoder = enc_cls(seed=seed)  # type: ignore[call-arg]
    trainer = JointTrainer(encoder=encoder, lam=lam, lr=hyper["lr"], seed=seed)

    best_val_kl = float("inf")
    patience_counter = 0
    last_epoch = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{arch}.safetensors"
    for epoch in range(int(hyper["epochs"])):
        last_epoch = epoch
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(train_pairs))
        for i in range(0, len(order), int(hyper["batch"])):
            batch_idx = order[i : i + int(hyper["batch"])]
            batch = [train_pairs[int(j)] for j in batch_idx]
            texts = [b["text"] for b in batch]
            spre = np.stack([b["state_pre"] for b in batch])
            spost = np.stack([b["state_post"] for b in batch])
            rho = _batch_stack_rho(batch)
            trainer.step(texts, spre, spost, rho)
        val_res = evaluate_checkpoint(encoder, trainer.params, val_pairs)
        val_kl = float(val_res["total"])
        logger.info("arch=%s epoch=%d val_kl=%.4f", arch, epoch, val_kl)
        if val_kl < best_val_kl - _MIN_VAL_IMPROVEMENT:
            best_val_kl = val_kl
            patience_counter = 0
            trainer.save_checkpoint(ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= _EARLY_STOP_PATIENCE:
                logger.info("early stop arch=%s at epoch=%d", arch, epoch)
                break
    return {"arch": arch, "best_val_kl": best_val_kl, "epochs_trained": last_epoch + 1}


def run_phase(
    phase: str,
    archs: list[str],
    corpus_dir: Path,
    cache_dir: Path,
    output_root: Path,
    seed: int = _DEFAULT_SEED,
    lam: float = _DEFAULT_LAM,
) -> dict[str, Any]:
    """Train all archs on corpus+cache split, evaluate on test, write summary.json."""
    cache = JKOCache(root=cache_dir)
    train_entries = _load_split(corpus_dir, "train")
    val_entries = _load_split(corpus_dir, "val")
    test_entries = _load_split(corpus_dir, "test")
    train_pairs = _pairs_from_cache(train_entries, cache)
    val_pairs = _pairs_from_cache(val_entries, cache)
    test_pairs = _pairs_from_cache(test_entries, cache)
    logger.info("pairs: train=%d val=%d test=%d", len(train_pairs), len(val_pairs), len(test_pairs))

    phase_dir = output_root / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"phase": phase, "archs": {}}
    for arch in archs:
        hyper = ARCH_HYPERPARAMS[arch]
        logger.info("=== training %s (%s) ===", arch, hyper)
        train_stats = train_one_arch(
            arch, train_pairs, val_pairs, hyper, phase_dir, seed=seed, lam=lam
        )
        encoder = ENCODER_REGISTRY[arch](seed=seed)  # type: ignore[call-arg]
        trainer = JointTrainer(encoder=encoder, lam=lam, lr=float(hyper["lr"]), seed=seed)
        trainer.load_checkpoint(phase_dir / f"{arch}.safetensors")
        test_res = evaluate_checkpoint(encoder, trainer.params, test_pairs)
        summary["archs"][arch] = {**train_stats, "test": test_res}
    (phase_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_manifest(phase_dir)
    return summary


def _write_manifest(phase_dir: Path) -> None:
    lines: list[str] = []
    for p in sorted(phase_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.relative_to(phase_dir)}")
    (phase_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def pick_top_k(
    summary: dict[str, Any],
    k: int = _DEFAULT_TOP_K,
    flip_tolerance: float = _FLIP_TOLERANCE,
) -> list[str]:
    """Rank archs by test KL_total, return Top-k.

    R3 kill-switch: promote all if gap < flip_tolerance (ranking too noisy to trust).
    """
    ranked = sorted(summary["archs"].items(), key=lambda kv: kv[1]["test"]["total"])
    names = [name for name, _ in ranked]
    if len(ranked) == 0:
        return []
    kl1 = float(ranked[0][1]["test"]["total"])
    kln = float(ranked[-1][1]["test"]["total"])
    if kl1 > 0 and (kln - kl1) / kl1 < flip_tolerance:
        logger.warning(
            "R3 kill-switch: gap %.1f%% < %.0f%% — promoting all %d archs",
            (kln - kl1) / kl1 * 100,
            flip_tolerance * 100,
            len(ranked),
        )
        return names
    return names[:k]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ablation sweep: pilot10k or scale50k phase.")
    parser.add_argument("--phase", choices=["pilot10k", "scale50k"], required=True)
    parser.add_argument(
        "--archs",
        type=lambda s: s.split(","),
        default=list(ARCH_HYPERPARAMS.keys()),
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--pick-top", type=int, default=_DEFAULT_TOP_K)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    summary = run_phase(
        args.phase, args.archs, args.corpus, args.cache, args.output, seed=args.seed
    )
    print(json.dumps(summary, indent=2))
    if args.phase == "pilot10k":
        top = pick_top_k(summary, k=args.pick_top)
        print(f"TOP-{args.pick_top}: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
