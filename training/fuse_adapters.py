#!/usr/bin/env python3
"""Fuse MLX LoRA Adapters into Full Models

After pretraining with LoRA, fuse adapters into standalone models
for downstream DPO training.

Usage:
  python training/fuse_adapters.py --model Qwen/Qwen2.5-1.5B-Instruct
  python training/fuse_adapters.py --model Qwen/Qwen2.5-3B-Instruct
  python training/fuse_adapters.py --config samkhya --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_SIZE_MAP = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "3b",
}

PRETRAIN_CONFIGS = ["samkhya", "bloom", "random"]

SFT_CONFIGS = {
    "yoga-sft-small": "yoga-sft-small",
    "yoga-sft": "yoga-sft",
    "reverse-sft": "reverse-sft",
    "random-sft": "random-sft",
}


def fuse_adapter(model_name, config_name):
    """Fuse a single LoRA adapter into a full model."""
    size = MODEL_SIZE_MAP.get(model_name, model_name.split("/")[-1].lower())
    adapter_dir = MODELS_DIR / f"{size}-{config_name}-pretrained" / "adapters"
    output_dir = MODELS_DIR / f"{size}-{config_name}-fused"

    if not adapter_dir.exists():
        print(f"  [SKIP] Adapter not found: {adapter_dir}")
        return False

    if output_dir.exists():
        print(f"  [SKIP] Already fused: {output_dir}")
        return True

    print(f"  Fusing: {config_name}")
    print(f"    Base model: {model_name}")
    print(f"    Adapter: {adapter_dir}")
    print(f"    Output: {output_dir}")

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", model_name,
        "--adapter-path", str(adapter_dir),
        "--save-path", str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: Fuse failed")
        if result.stderr:
            print(f"    {result.stderr[:500]}")
        return False

    print(f"    Fused successfully -> {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fuse MLX LoRA Adapters")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model name")
    parser.add_argument("--config", type=str, default=None,
                        choices=PRETRAIN_CONFIGS + list(SFT_CONFIGS.keys()),
                        help="Specific config to fuse (default: all)")
    parser.add_argument("--sft", action="store_true",
                        help="Fuse SFT adapters instead of pretrain adapters")
    args = parser.parse_args()

    configs = [args.config] if args.config else PRETRAIN_CONFIGS

    # Also fuse SFT adapters if --sft flag
    if args.sft:
        sft_names = [args.config] if args.config and args.config in SFT_CONFIGS else list(SFT_CONFIGS.keys())
        configs = sft_names

    print(f"\nFusing LoRA adapters for {args.model}")
    successes = 0
    for config in configs:
        if fuse_adapter(args.model, config):
            successes += 1

    size = MODEL_SIZE_MAP.get(args.model, args.model.split("/")[-1].lower())
    print(f"\nFused {successes}/{len(configs)} adapters.")
    print(f"Fused models in: {MODELS_DIR}/{size}-*-fused/")


if __name__ == "__main__":
    main()
