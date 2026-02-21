#!/usr/bin/env python3
"""Full Training Pipeline: Pretrain -> Fuse -> DPO -> Evaluate

Orchestrates the complete darshana training validation pipeline.

For each base model:
  Phase A — Pretraining (3 runs): samkhya, bloom, random
  Phase B — DPO on base model (4 runs): yoga, reverse, complexity, random
  Phase C — DPO on pretrained models (3 runs): samkhya+yoga, bloom+complexity, random+random

Total: 10 training runs per model size.

Usage:
  python training/run_full_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct
  python training/run_full_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct --phase A
  python training/run_full_pipeline.py --model Qwen/Qwen2.5-3B-Instruct --phase C
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_SIZE_MAP = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "3b",
}


def run_cmd(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED: {description} (exit code {result.returncode}, {elapsed/60:.1f} min)")
        return False

    print(f"\n  SUCCESS: {description} ({elapsed/60:.1f} min)")
    return True


def phase_a_pretraining(model_name):
    """Phase A: Continued pretraining with 3 orderings."""
    print(f"\n{'#'*60}")
    print(f"  PHASE A: Pretraining ({model_name})")
    print(f"{'#'*60}")

    py = sys.executable

    # Pretrain all 3 orderings
    success = run_cmd(
        [py, "training/pretrain_samkhya.py", "--mode", "all", "--model", model_name],
        "Pretrain: samkhya + bloom + random"
    )
    if not success:
        return False

    # Fuse all adapters
    success = run_cmd(
        [py, "training/fuse_adapters.py", "--model", model_name],
        "Fuse LoRA adapters"
    )
    return success


def phase_b_dpo_base(model_name):
    """Phase B: DPO on base model with 4 orderings."""
    print(f"\n{'#'*60}")
    print(f"  PHASE B: DPO on base model ({model_name})")
    print(f"{'#'*60}")

    py = sys.executable

    return run_cmd(
        [py, "training/train_dpo.py", "--all-modes", "--base-model", model_name],
        "DPO: yoga + reverse + complexity + random"
    )


def phase_c_dpo_pretrained(model_name):
    """Phase C: DPO on pretrained models (combined configs)."""
    print(f"\n{'#'*60}")
    print(f"  PHASE C: DPO on pretrained models ({model_name})")
    print(f"{'#'*60}")

    py = sys.executable
    size = MODEL_SIZE_MAP.get(model_name, model_name.split("/")[-1].lower())

    combined_configs = [
        ("samkhya", "yoga"),       # darshana stack
        ("bloom", "complexity"),   # Western stack
        ("random", "random"),      # null hypothesis
    ]

    all_success = True
    for pt_config, dpo_mode in combined_configs:
        fused_path = MODELS_DIR / f"{size}-{pt_config}-fused"
        if not fused_path.exists():
            print(f"  [SKIP] Fused model not found: {fused_path}")
            all_success = False
            continue

        success = run_cmd(
            [py, "training/train_dpo.py",
             "--mode", dpo_mode,
             "--base-model", model_name,
             "--pretrained-path", str(fused_path)],
            f"DPO: {pt_config}+{dpo_mode} (combined)"
        )
        if not success:
            all_success = False

    return all_success


def main():
    parser = argparse.ArgumentParser(description="Full Training Pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model")
    parser.add_argument("--phase", type=str, default=None,
                        choices=["A", "B", "C"],
                        help="Run specific phase only (default: all)")
    args = parser.parse_args()

    t0 = time.time()
    size = MODEL_SIZE_MAP.get(args.model, args.model.split("/")[-1].lower())

    print(f"\n{'#'*60}")
    print(f"  DARSHANA TRAINING PIPELINE")
    print(f"  Model: {args.model} ({size})")
    print(f"  Phase: {args.phase or 'ALL (A+B+C)'}")
    print(f"{'#'*60}")

    phases = {
        "A": ("Pretraining", phase_a_pretraining),
        "B": ("DPO on base", phase_b_dpo_base),
        "C": ("DPO on pretrained", phase_c_dpo_pretrained),
    }

    if args.phase:
        run_phases = [args.phase]
    else:
        run_phases = ["A", "B", "C"]

    results = {}
    for phase in run_phases:
        name, fn = phases[phase]
        results[phase] = fn(args.model)

    total_elapsed = time.time() - t0

    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE ({total_elapsed/3600:.1f} hours)")
    print(f"{'#'*60}")
    for phase, success in results.items():
        name = phases[phase][0]
        status = "SUCCESS" if success else "FAILED"
        print(f"  Phase {phase} ({name}): {status}")

    print(f"\nNext steps:")
    print(f"  python experiments/exp6_samkhya_pretraining.py --model-size {size} --questions all --judge --judge-model sonnet")
    print(f"  python experiments/exp7_yoga_dpo.py --model-size {size} --questions all --judge --judge-model sonnet")
    print(f"  python experiments/analyze_training.py --model-size {size}")


if __name__ == "__main__":
    main()
