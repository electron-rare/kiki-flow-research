# Real-mode LLM CL benchmark runbook

**Never run automatically.** Follow manually, under user supervision.

## Prerequisites
- SSH passwordless access to kxkm-ai (RTX 4090 24 GB)
- ~30 GB free disk under `~/KIKI-Mac_tunner/`
- Qwen3.5-4B weights cached locally
- HuggingFace datasets pre-cached (or online access)

## Steps
1. Confirm with user: "OK to launch LoRA fine-tuning on kxkm-ai for ~2 hours?"
2. ssh kxkm-ai; cd ~/KIKI-Mac_tunner; git pull origin main
3. Prepare task datasets:
   `PYTHONPATH=. uv run python scripts/cl_llm_bench/task_sequences.py --prepare --out ~/kxkm_data/cl_tasks/`
4. Run baseline (no bridge):
   `KIKI_FLOW_ENABLED=0 uv run python scripts/cl_llm_bench/run_cl_bench.py --mode real --tasks phono_sst2,lex_cola,syn_boolq --seed 0 --output ~/bench_runs/baseline_seed0/`
5. Run with bridge:
   `KIKI_FLOW_ENABLED=1 uv run python scripts/cl_llm_bench/run_cl_bench.py --mode real --tasks phono_sst2,lex_cola,syn_boolq --seed 0 --output ~/bench_runs/bridge_seed0/`
6. Repeat for seeds 1-4.
7. rsync results back to GrosMac M5, regenerate fig7+fig8, update `paper/` and PERFORMANCE.md.

## Acceptance
- `bench/cl_llm/summary.json` shows measurable difference between the two arms
- fig7 bars differ by at least 1 σ on at least one task
- Paper §5 adds a paragraph citing these numbers
