# Real-mode LLM CL benchmark runbook

**Never run automatically.** Follow manually, under user supervision.

## Prerequisites
- SSH passwordless access to kxkm-ai (RTX 4090 24 GB)
- ~30 GB free disk on kxkm-ai home partition
- Qwen3-4B base weights cached on kxkm-ai (`Qwen/Qwen3-4B` or equivalent)
- HuggingFace datasets pre-cached (`glue/sst2`, `glue/cola`, `super_glue/boolq`, `glue/rte`)
- `uv` installed on kxkm-ai
- `scripts/cl_llm_bench/kxkm_trainer/train_cl_task.py` synced to kxkm-ai (see P2 below)

Always run preflight first — it confirms every prereq in one shot:

```bash
PYTHONPATH=. uv run python -m scripts.cl_llm_bench.run_cl_bench \
  --mode preflight \
  --ssh-host kxkm-ai \
  --output bench/cl_llm/runs/preflight-check \
  --seed 0 \
  --tasks phono_sst2,lex_cola,syn_boolq
```

The JSON report shows `ready_for_real: true|false` with per-check status.

## Known gaps (audit 2026-04-18)

The original plan assumed `~/KIKI-Mac_tunner/scripts/train_stack.py` exposed a
`--base-model / --lora-rank / --learning-rate / --dataset / --output-dir`
interface. In reality that script is hardwired for Qwen3.5-35B-A3B MoE domain
stacks and uses a different CLI (`--domain`). A compatible trainer is shipped
separately under `scripts/cl_llm_bench/kxkm_trainer/train_cl_task.py`; sync it
to kxkm-ai before running real mode.

Until `train_cl_task.py` is pushed to kxkm-ai and preflight goes fully green,
**stub mode is the only supported path** and is the chain validated by issue
#1 closure.

## Setup sequence (one-time, before any real-mode run)

1. Install `uv` on kxkm-ai:
   `ssh kxkm-ai 'curl -LsSf https://astral.sh/uv/install.sh | sh'`

2. Sync the trainer from this repo to kxkm-ai:
   `rsync -av scripts/cl_llm_bench/kxkm_trainer/ kxkm-ai:~/kiki-flow-research-kxkm/`

3. Pre-cache base weights (one-off, ~8 GB):
   `ssh kxkm-ai 'uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3-4B\")"'`

4. Pre-cache GLUE/SuperGLUE subsets (~500 MB):
   `ssh kxkm-ai 'uv run --with datasets python -c "from datasets import load_dataset; [load_dataset(*x) for x in [(\"glue\",\"sst2\"),(\"glue\",\"cola\"),(\"super_glue\",\"boolq\"),(\"glue\",\"rte\")]]"'`

5. Re-run preflight to confirm all six checks pass.

## Execution sequence (per seed, both arms)

1. Confirm with user: "OK to launch LoRA fine-tuning on kxkm-ai for ~2 hours?"
2. Run baseline (no bridge):
   ```
   PYTHONPATH=. uv run python -m scripts.cl_llm_bench.run_cl_bench \
     --mode real --i-confirm-heavy-training \
     --tasks phono_sst2,lex_cola,syn_boolq --seed 0 \
     --output bench/cl_llm/runs/baseline_seed0 \
     --ssh-host kxkm-ai
   ```
3. Run with bridge (same command, plus `KIKI_FLOW_ENABLED=1` exported locally
   before the run — the bridge is consumed by `AdvisoryRecorder` in-process).
4. Repeat for seeds 1-4.
5. `rsync kxkm-ai:~/bench_runs/ bench/cl_llm/runs/` back to GrosMac.
6. Regenerate fig7+fig8 from the collected `summary.json`s (see
   `scripts/cl_llm_bench/run_cl_bench.py` `--mode stub` for an example of how
   the figure generators plug into the summary).
7. Update `paper/` and `PERFORMANCE.md` with the measured numbers.

## Acceptance
- Every `bench/cl_llm/runs/*/summary.json` is valid and covers the 3 tasks.
- The with-bridge vs without-bridge forgetting scores differ by at least 1 σ
  on at least one task (across 5 seeds).
- Paper §5 adds a paragraph citing these numbers, replacing the current
  distributional-proxy language.

## Why this is still manual

The full real-mode sequence consumes ~20 h of RTX 4090 wall time and ~80 GB
of bandwidth for the weight download. User memory rule
`feedback_no_launch_kxkm_without_ask.md` requires explicit per-run consent.
The CLI honors this via the `--i-confirm-heavy-training` gate; the runbook
formalizes the wrapping process.
