"""LoRA trainer wrapper around the standalone ~/kiki-flow-research-kxkm/train_cl_task.py.

Runs in two modes:
- ``stub``: returns a synthetic manifest without touching any LLM. For
  CI and for initial pipeline wiring.
- ``real``: invokes the remote trainer on kxkm-ai via SSH + ``uv run``
  (the remote script uses PEP 723 inline metadata to resolve torch /
  transformers / peft / datasets on the fly). Requires the user's
  confirmation every time (per user memory: never launch heavy training
  on kxkm-ai without asking).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoRATrainingConfig:
    base_model: str
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    n_steps: int
    batch_size: int
    output_dir: Path
    seed: int
    extra_args: dict[str, Any] = field(default_factory=dict)


class LoRATrainerStub:
    """Deterministic stub for CI and pipeline wiring. Never touches a real LLM."""

    def __init__(self, config: LoRATrainingConfig) -> None:
        self.config = config

    def train(self, dataset_stub: list[dict[str, Any]]) -> dict[str, Any]:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "ok",
            "mode": "stub",
            "base_model": self.config.base_model,
            "n_steps": self.config.n_steps,
            "n_samples": len(dataset_stub),
            "seed": self.config.seed,
            "timestamp": time.time(),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


class LoRATrainerReal:
    """Invokes the real trainer on a remote SSH host via the standalone
    ~/kiki-flow-research-kxkm/train_cl_task.py entry point. The remote
    script declares its heavy ML deps via PEP 723 inline metadata, so we
    run it with ``uv run``. Every invocation requires explicit user
    confirmation when dry_run is False."""

    REMOTE_SCRIPT = "~/kiki-flow-research-kxkm/train_cl_task.py"
    # Absolute path — non-interactive SSH sessions don't load ~/.profile on
    # kxkm-ai, so PATH doesn't include ~/.local/bin and a bare ``uv`` would
    # trigger rc=127 ("command not found").
    REMOTE_UV = "$HOME/.local/bin/uv"

    def __init__(
        self,
        config: LoRATrainingConfig,
        ssh_host: str,
        dry_run: bool = True,
    ) -> None:
        self.config = config
        self.ssh_host = ssh_host
        self.dry_run = dry_run

    def build_command(self, dataset_path: Path) -> list[str]:
        return [
            self.REMOTE_UV,
            "run",
            self.REMOTE_SCRIPT,
            "--base-model",
            self.config.base_model,
            "--lora-rank",
            str(self.config.lora_rank),
            "--lora-alpha",
            str(self.config.lora_alpha),
            "--learning-rate",
            str(self.config.learning_rate),
            "--n-steps",
            str(self.config.n_steps),
            "--batch-size",
            str(self.config.batch_size),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(self.config.output_dir),
            "--seed",
            str(self.config.seed),
        ]

    def train(self, dataset_path: Path) -> dict[str, Any]:
        if self.dry_run:
            return {"status": "dry-run", "command": self.build_command(dataset_path)}
        raise RuntimeError(
            "Real-mode training on kxkm-ai requires explicit user "
            "confirmation. Run this script interactively, not automated."
        )
