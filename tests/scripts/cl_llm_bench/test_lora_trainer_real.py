# tests/scripts/cl_llm_bench/test_lora_trainer_real.py
from pathlib import Path

import pytest

from scripts.cl_llm_bench.lora_trainer import LoRATrainerReal, LoRATrainingConfig


def test_lora_trainer_real_builds_correct_command(tmp_path: Path) -> None:
    cfg = LoRATrainingConfig(
        base_model="qwen3.5-4b",
        lora_rank=8,
        lora_alpha=16,
        learning_rate=1e-4,
        n_steps=100,
        batch_size=4,
        output_dir=tmp_path / "real_run",
        seed=0,
    )
    trainer = LoRATrainerReal(cfg, ssh_host="kxkm-ai", dry_run=True)
    cmd = trainer.build_command(dataset_path=tmp_path / "data.jsonl")
    assert cmd[0].endswith("uv"), f"expected uv path, got {cmd[0]!r}"
    assert cmd[1] == "run"
    assert "train_cl_task.py" in " ".join(cmd)
    assert "--lora-rank" in cmd
    assert "8" in cmd


def test_lora_trainer_real_raises_when_not_dry_run(tmp_path: Path) -> None:
    cfg = LoRATrainingConfig(
        base_model="qwen3.5-4b",
        lora_rank=8,
        lora_alpha=16,
        learning_rate=1e-4,
        n_steps=100,
        batch_size=4,
        output_dir=tmp_path / "real_run",
        seed=0,
    )
    trainer = LoRATrainerReal(cfg, ssh_host="kxkm-ai", dry_run=False)
    with pytest.raises(RuntimeError, match="explicit user confirmation"):
        trainer.train(dataset_path=tmp_path / "data.jsonl")
