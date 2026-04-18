"""Orchestrator for the full CL benchmark.

Stub mode computes a synthetic summary based on the distributional-proxy
results already validated in the repo (cl_benchmark_ewc.json etc.) so
the wiring can be exercised in CI without a real LLM. Preflight mode probes
kxkm-ai SSH host for prerequisites (trainer script, weights, uv, disk space).
Real mode invokes the SSH-based LoRA trainer on kxkm-ai (requires explicit
--i-confirm-heavy-training flag per user memory feedback).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

from scripts.cl_llm_bench.eval_forgetting import forgetting_score
from scripts.cl_llm_bench.lora_trainer import LoRATrainerReal, LoRATrainingConfig
from scripts.cl_llm_bench.task_sequences import TASK_REGISTRY, load_task_sequence

Mode = Literal["stub", "preflight", "real"]

# Plausibility stub numbers derived from the distributional-proxy
# results in paper/cl_benchmark_ewc.json so the wiring produces
# non-trivial but deterministic output in CI.
_STUB_BEFORE = 0.80
_STUB_AFTER_WITHOUT = (0.29, 0.44, 0.81, 0.35)
_STUB_AFTER_WITH = (0.81, 0.26, 0.24, 0.30)

# Minimum disk space required for training (GB)
_MIN_DISK_GB = 50

# Timeouts (seconds) for remote subprocess calls in real mode.
_RSYNC_TIMEOUT_S = 120
_SSH_TRAIN_TIMEOUT_S = 3600  # 60-minute cap for single-task training

# Remote layout on kxkm-ai.
_REMOTE_DATASET_DIR = "~/bench_runs/datasets"

# Preflight check script: READ-ONLY SSH probe for kxkm-ai prerequisites.
# Uses `set +e` to capture all results without early exit.
_PREFLIGHT_SCRIPT = r"""
set +e
echo "===train_cl_task==="
test -f ~/kiki-flow-research-kxkm/train_cl_task.py && echo "ok" || echo "missing"
echo "===qwen_weights==="
_qwen=$(ls -d ~/.cache/huggingface/hub/models--Qwen--* 2>/dev/null | head -1)
[ -n "$_qwen" ] && echo "ok" || echo "missing"
echo "===hf_datasets==="
test -d ~/.cache/huggingface/datasets && echo "ok" || echo "missing"
echo "===uv==="
(command -v uv || test -x ~/.local/bin/uv) >/dev/null 2>&1 && echo "ok" || echo "missing"
echo "===disk_gb==="
df -BG ~/ | awk 'NR==2 {gsub("G","",$4); print $4}'
echo "===gpu==="
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "no gpu"
"""


def preflight_report(ssh_host: str) -> dict[str, Any]:
    """Probe SSH host for CL training prerequisites.

    Returns a structured dict with:
    - host: str
    - checks: {name: {status: "ok"|"fail", detail: str}}
    - ready_for_real: bool (True iff all checks pass)

    If SSH fails, all checks are marked "fail" without raising.
    """
    report: dict[str, Any] = {
        "host": ssh_host,
        "checks": {},
        "ready_for_real": False,
    }

    try:
        result = subprocess.run(
            ["ssh", ssh_host, _PREFLIGHT_SCRIPT],
            capture_output=True,
            timeout=15,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired:
        report["checks"]["ssh_timeout"] = {
            "status": "fail",
            "detail": f"SSH to {ssh_host} timed out (15s)",
        }
        return report
    except Exception as e:
        report["checks"]["ssh_error"] = {
            "status": "fail",
            "detail": f"SSH error: {e!s}",
        }
        return report

    if result.returncode != 0:
        report["checks"]["ssh_exit"] = {
            "status": "fail",
            "detail": f"SSH exited with code {result.returncode}\nstderr: {result.stderr[:200]}",
        }
        return report

    lines = result.stdout.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("===") and line.endswith("==="):
            check_name = line.strip("=")
            i += 1
            if i < len(lines):
                value = lines[i].strip()
                # Parse the value
                if check_name == "disk_gb":
                    try:
                        disk_gb = int(value)
                        status = "ok" if disk_gb > _MIN_DISK_GB else "fail"
                        detail = f"{disk_gb} GB free (need > {_MIN_DISK_GB} GB)"
                    except ValueError:
                        status = "fail"
                        detail = f"Could not parse disk output: {value}"
                elif check_name == "gpu":
                    status = "ok" if value != "no gpu" else "fail"
                    detail = value
                else:
                    status = "ok" if value == "ok" else "fail"
                    detail = value

                report["checks"][check_name] = {
                    "status": status,
                    "detail": detail,
                }
            i += 1
        else:
            i += 1

    # ready_for_real iff all checks pass
    all_pass = all(check["status"] == "ok" for check in report["checks"].values())
    report["ready_for_real"] = all_pass

    return report


def _prepare_task_jsonl(task_dict: dict[str, Any], local_path: Path) -> Path:
    """Transform a task-sequence dict into a JSONL file of ``{"text", "label"}``.

    The remote trainer expects exactly those two keys. Task-registry tasks
    carry arbitrary ``text_field`` names (``sentence``, ``question``, ...),
    so we normalize here. Train + eval are concatenated — the trainer does
    its own 80/20 split.
    """
    name = task_dict["name"]
    entry = TASK_REGISTRY[name]
    text_field = entry["text_field"]
    label_field = entry["label_field"]

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(task_dict.get("train", [])) + list(task_dict.get("eval", []))
    with local_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            text = rec.get(text_field, "")
            label = rec.get(label_field, 0)
            fh.write(
                json.dumps({"text": str(text), "label": int(label)}, ensure_ascii=False) + "\n"
            )
    return local_path


def _rsync_up(local_path: Path, ssh_host: str, remote_dir: str) -> None:
    """Copy ``local_path`` to ``ssh_host:remote_dir/``. Raises on failure."""
    cmd = [
        "rsync",
        "-a",
        "--mkpath",
        str(local_path),
        f"{ssh_host}:{remote_dir}/",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_RSYNC_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"rsync up failed (rc={result.returncode}): "
            f"{result.stderr[:300] or result.stdout[:300]}"
        )
        raise RuntimeError(msg)


def _rsync_down(ssh_host: str, remote_path: str, local_path: Path) -> None:
    """Copy ``ssh_host:remote_path`` down to ``local_path``. Raises on failure."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-a",
        "--mkpath",
        f"{ssh_host}:{remote_path}",
        str(local_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_RSYNC_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"rsync down failed (rc={result.returncode}): "
            f"{result.stderr[:300] or result.stdout[:300]}"
        )
        raise RuntimeError(msg)


def _parse_manifest_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract the last top-level JSON object from the trainer's stdout.

    ``train_cl_task.py`` prints the manifest last via ``json.dumps(..., indent=2)``.
    We scan for ``{...}`` candidates and return the last one that decodes.
    """
    candidates = re.findall(r"\{[\s\S]*?\}(?=\s*$|\s*\{)", stdout) or re.findall(
        r"\{[\s\S]*\}", stdout
    )
    for blob in reversed(candidates):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    msg = "Could not find a JSON object in trainer stdout"
    raise ValueError(msg)


def _fail(stage: str, error: str, seed: int) -> dict[str, Any]:
    """Build a structured ``status=failed`` error dict for real-mode stages."""
    return {
        "mode": "real",
        "status": "failed",
        "stage": stage,
        "error": error,
        "seed": seed,
    }


def _run_real(  # noqa: PLR0911 (each stage returns a structured error dict by design)
    task_names: list[str],
    output_dir: Path,
    seed: int,
    ssh_host: str,
    confirmed: bool,
    max_samples: int = 500,
    base_model: str = "Qwen/Qwen3-4B",
) -> dict[str, Any]:
    """Real mode execution: run LoRA trainer on kxkm-ai for a single task.

    This function is ONLY reachable if confirmed=True. Otherwise it raises.
    First runs preflight checks; aborts if any fail. E1_min scope: SINGLE
    task only. If multiple task names are passed, extras are ignored with
    a stderr WARN; multi-task CL is E2_alt.
    """
    if not confirmed:
        raise RuntimeError(
            "Real mode requires explicit --i-confirm-heavy-training flag.\n"
            "Use: python -m scripts.cl_llm_bench.run_cl_bench --mode real "
            "--i-confirm-heavy-training ...\n"
            "Or run --mode preflight first to check prerequisites."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preflight safety gate (all network calls happen AFTER this returns ok).
    preflight = preflight_report(ssh_host)
    if not preflight["ready_for_real"]:
        return {
            "mode": "real",
            "status": "preflight_failed",
            "preflight": preflight,
            "seed": seed,
        }
    if not task_names:
        return _fail("input", "task_names is empty", seed)

    first_task = task_names[0]
    if len(task_names) > 1:
        print(
            f"WARN: E1_min is single-task only; ignoring extra tasks {task_names[1:]}",
            file=sys.stderr,
        )

    t_start = time.time()

    # 1. Load the task (train + eval).
    try:
        tasks = load_task_sequence([first_task], max_samples=max_samples)
    except Exception as e:  # noqa: BLE001
        return _fail("load_task", str(e), seed)
    task = tasks[0]
    n_samples = len(task.get("train", [])) + len(task.get("eval", []))

    # 2. Write normalized JSONL locally.
    local_jsonl = output_dir / f"{first_task}.jsonl"
    try:
        _prepare_task_jsonl(task, local_jsonl)
    except Exception as e:  # noqa: BLE001
        return _fail("prepare_jsonl", str(e), seed)

    # 3. rsync JSONL up to kxkm-ai.
    try:
        _rsync_up(local_jsonl, ssh_host, _REMOTE_DATASET_DIR)
    except Exception as e:  # noqa: BLE001
        return _fail("rsync_up", str(e), seed)

    # 4. Build the SSH + trainer invocation.
    remote_dataset_jsonl = f"{_REMOTE_DATASET_DIR}/{local_jsonl.name}"
    remote_output_dir = Path(f"~/bench_runs/{seed}_{first_task}")
    cfg = LoRATrainingConfig(
        base_model=base_model,
        lora_rank=8,
        lora_alpha=16,
        learning_rate=2e-4,
        n_steps=500,
        batch_size=4,
        output_dir=remote_output_dir,
        seed=seed,
    )
    trainer = LoRATrainerReal(cfg, ssh_host, dry_run=False)
    cmd = trainer.build_command(dataset_path=Path(remote_dataset_jsonl))

    # 5. Execute on the remote host.
    try:
        ssh_result = subprocess.run(
            ["ssh", ssh_host, *cmd],
            capture_output=True,
            text=True,
            timeout=_SSH_TRAIN_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return _fail("ssh_train", f"ssh train timed out after {_SSH_TRAIN_TIMEOUT_S}s: {e!s}", seed)
    except Exception as e:  # noqa: BLE001
        return _fail("ssh_train", str(e), seed)
    if ssh_result.returncode != 0:
        return _fail(
            "ssh_train", f"ssh rc={ssh_result.returncode}: {ssh_result.stderr[:500]}", seed
        )

    # 6. Parse the manifest from stdout.
    try:
        manifest = _parse_manifest_from_stdout(ssh_result.stdout)
    except Exception as e:  # noqa: BLE001
        return _fail("manifest_parse", str(e), seed)

    # 7. rsync the manifest back for provenance.
    remote_manifest = f"{remote_output_dir}/manifest.json"
    local_manifest = output_dir / "manifest.json"
    try:
        _rsync_down(ssh_host, remote_manifest, local_manifest)
    except Exception as e:  # noqa: BLE001
        return _fail("rsync_down", str(e), seed)

    return {
        "mode": "real",
        "status": "ok",
        "task": first_task,
        "seed": seed,
        "n_samples": n_samples,
        "manifest": manifest,
        "wall_time_s": time.time() - t_start,
    }


def run_cl_bench(
    task_names: list[str],
    mode: Mode,
    output_dir: Path,
    seed: int,
    ssh_host: str = "kxkm-ai",
    confirmed: bool = False,
    max_samples: int = 500,
    base_model: str = "Qwen/Qwen3-4B",
) -> dict[str, Any]:
    """Orchestrate CL benchmark in stub, preflight, or real mode.

    Args:
        task_names: List of task identifiers (e.g. ["phono_sst2", "lex_cola"]).
        mode: "stub" (synthetic), "preflight" (SSH probe), or "real" (LoRA trainer).
        output_dir: Directory for output artifacts.
        seed: Random seed for reproducibility.
        ssh_host: SSH host for real mode (default "kxkm-ai").
        confirmed: Must be True to unlock real mode (requires --i-confirm-heavy-training).
        max_samples: Per-task sample cap used by ``load_task_sequence`` (real mode only).
        base_model: HF repo ID passed to the remote trainer (real mode only).

    Returns:
        dict[str, Any]: Summary or structured error report.

    Raises:
        RuntimeError: If real mode invoked without confirmed=True.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "stub":
        before = {name: _STUB_BEFORE for name in task_names}
        after_without = dict(zip(task_names, _STUB_AFTER_WITHOUT, strict=False))
        after_with = dict(zip(task_names, _STUB_AFTER_WITH, strict=False))

        summary = {
            "mode": mode,
            "seed": seed,
            "forgetting_without_bridge": forgetting_score(before, after_without),
            "forgetting_with_bridge": forgetting_score(before, after_with),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    if mode == "preflight":
        preflight = preflight_report(ssh_host)
        (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
        return preflight

    if mode == "real":
        result = _run_real(
            task_names,
            output_dir,
            seed,
            ssh_host,
            confirmed,
            max_samples=max_samples,
            base_model=base_model,
        )
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result

    msg = f"Unknown mode: {mode}"
    raise ValueError(msg)


def main() -> int:
    """CLI entry point for CL benchmark orchestrator."""
    parser = argparse.ArgumentParser(
        description="CL benchmark orchestrator: stub / preflight / real modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.cl_llm_bench.run_cl_bench --mode stub "
            "--tasks phono_sst2,lex_cola --output bench/runs/stub_0/\n"
            "  python -m scripts.cl_llm_bench.run_cl_bench --mode preflight "
            "--ssh-host kxkm-ai --output bench/runs/preflight_check/\n"
            "  python -m scripts.cl_llm_bench.run_cl_bench --mode real "
            "--i-confirm-heavy-training --ssh-host kxkm-ai "
            "--tasks phono_sst2 --output bench/runs/real_0/\n"
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["stub", "preflight", "real"],
        default="stub",
        help="Execution mode: stub (synthetic), preflight (SSH probe), or real (LoRA trainer).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="phono_sst2,lex_cola,syn_boolq",
        help="Comma-separated task names (default: phono_sst2,lex_cola,syn_boolq).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for summary.json / preflight.json / result.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--ssh-host",
        type=str,
        default="kxkm-ai",
        help="SSH host for preflight/real modes (default: kxkm-ai).",
    )
    parser.add_argument(
        "--i-confirm-heavy-training",
        action="store_true",
        help="REQUIRED for real mode. Confirms user intends to launch GPU training on kxkm-ai.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Per-task sample cap for real mode (default: 500).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HF repo ID for the base model in real mode (default: Qwen/Qwen3-4B).",
    )

    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",")]

    try:
        result = run_cl_bench(
            task_names=task_names,
            mode=args.mode,  # type: ignore[arg-type]
            output_dir=args.output,
            seed=args.seed,
            ssh_host=args.ssh_host,
            confirmed=args.i_confirm_heavy_training,
            max_samples=args.max_samples,
            base_model=args.base_model,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
