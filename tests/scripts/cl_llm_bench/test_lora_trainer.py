# tests/scripts/cl_llm_bench/test_lora_trainer.py
from scripts.cl_llm_bench.lora_trainer import LoRATrainerStub, LoRATrainingConfig


def test_lora_trainer_stub_produces_manifest(tmp_path):
    cfg = LoRATrainingConfig(
        base_model="qwen3.5-4b",
        lora_rank=8,
        lora_alpha=16,
        learning_rate=1e-4,
        n_steps=2,
        batch_size=2,
        output_dir=tmp_path / "stub_run",
        seed=0,
    )
    trainer = LoRATrainerStub(cfg)
    manifest = trainer.train(dataset_stub=[{"text": "hello"}, {"text": "world"}])
    assert manifest["status"] == "ok"
    assert manifest["n_steps"] == 2  # noqa: PLR2004
    assert (tmp_path / "stub_run" / "manifest.json").exists()
