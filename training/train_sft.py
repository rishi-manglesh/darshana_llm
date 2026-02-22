#!/usr/bin/env python3
"""SFT Training with Yoga Curriculum Ordering

Trains LoRA adapters on SFT data using MLX.

Phase 1 (small test — yoga vs base only):
  --mode yoga   : Stage 1→5 ordering via mlx_ordered_lora.py

Phase 2 (scale up — if yoga shows signal vs base):
  --mode yoga     : Stage 1→5 ordering (Ashtanga)
  --mode reverse  : Stage 5→1 ordering
  --mode complexity: Same progression, no Yoga framing (generic)
  --mode random   : Shuffled (default mlx_lm, null hypothesis)

Usage:
  python training/train_sft.py --mode yoga --phase small
  python training/train_sft.py --mode yoga --phase full
  python training/train_sft.py --mode random --phase full
"""

import argparse
import json
import random as rnd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- Config --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sft_yoga"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_SIZE_MAP = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "3b",
}

# Training hyperparams
PHASE_CONFIGS = {
    "small": {"iters": 100, "batch_size": 2, "lr": "1e-5"},
    "full": {"iters": 400, "batch_size": 2, "lr": "1e-5"},
}


def get_sft_data_path(phase):
    """Get the SFT data file for a given phase."""
    if phase == "small":
        return DATA_DIR / "small_test.jsonl"
    elif phase == "full":
        return DATA_DIR / "full_train.jsonl"
    else:
        raise ValueError(f"Unknown phase: {phase}")


def prepare_sft_training_data(sft_data_path, output_dir, mode="yoga", seed=42):
    """Convert SFT JSONL to mlx_lm train.jsonl + valid.jsonl format.

    Modes control ordering:
      yoga:       Sort by stage 1→5 (Ashtanga order)
      reverse:    Sort by stage 5→1
      complexity: Same stage order as yoga (1→5) — tests if Yoga framing matters
      random:     Shuffle all examples

    Returns:
        int: number of training examples
    """
    rng = rnd.Random(seed)

    # Load SFT records
    records = []
    with open(sft_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"ERROR: No records in {sft_data_path}")
        sys.exit(1)

    # Apply ordering
    if mode == "yoga":
        # Ashtanga order: stage 1→5, preserve order within stage
        records.sort(key=lambda r: r.get("stage", 0))
    elif mode == "reverse":
        # Reverse: stage 5→1
        records.sort(key=lambda r: -r.get("stage", 0))
    elif mode == "bookend":
        # Yoga progression 1→5 then anchor back to stage 1 at end
        # Tests: does yoga ordering help if you end on epistemic honesty?
        records.sort(key=lambda r: r.get("stage", 0))
        stage1_copies = [r.copy() for r in records if r.get("stage") == 1]
        records.extend(stage1_copies)
    elif mode == "complexity":
        # Same 1→5 order but we note it's "generic complexity" not Yoga
        # (The data itself is the same — this tests framing vs ordering)
        records.sort(key=lambda r: r.get("stage", 0))
    elif mode == "random":
        rng.shuffle(records)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Convert to mlx_lm chat format
    # mlx_lm expects {"messages": [...]} for chat fine-tuning
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratified 90/10 split — sample every 10th for validation
    n_val = max(1, len(records) // 10)
    val_indices = set(range(0, len(records), max(1, len(records) // n_val)))
    train = [r for i, r in enumerate(records) if i not in val_indices]
    valid = [r for i, r in enumerate(records) if i in val_indices]

    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for r in train:
            # mlx_lm chat format: {"messages": [...]}
            entry = {"messages": r["messages"]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    valid_path = output_dir / "valid.jsonl"
    with open(valid_path, "w") as f:
        for r in valid:
            entry = {"messages": r["messages"]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Prepared {len(train)} train + {len(valid)} valid -> {output_dir}")
    print(f"  Mode: {mode} | Stages in train: {sorted(set(r.get('stage',0) for r in train))}")
    return len(train)


def run_sft_training(mode, phase, model_name, num_layers=16, seed=42):
    """Run SFT LoRA training using MLX."""
    import subprocess

    size = MODEL_SIZE_MAP.get(model_name, model_name.split("/")[-1].lower())
    sft_data_path = get_sft_data_path(phase)

    if not sft_data_path.exists():
        print(f"ERROR: {sft_data_path} not found.")
        print(f"Run: python training/generate_sft_data.py --mode {phase}")
        sys.exit(1)

    # Prepare ordered training data
    train_data_dir = DATA_DIR / f"{mode}_{phase}_data"
    n_train = prepare_sft_training_data(sft_data_path, train_data_dir, mode=mode, seed=seed)

    # Output directory
    output_dir = MODELS_DIR / f"{size}-yoga-sft" if mode == "yoga" else MODELS_DIR / f"{size}-{mode}-sft"
    if phase == "small":
        output_dir = MODELS_DIR / f"{size}-yoga-sft-small" if mode == "yoga" else MODELS_DIR / f"{size}-{mode}-sft-small"

    hparams = PHASE_CONFIGS[phase]

    print(f"\nStarting SFT training: {mode} ({phase})")
    print(f"  Model: {model_name}")
    print(f"  Data: {train_data_dir} ({n_train} train examples)")
    print(f"  Output: {output_dir}")
    print(f"  Hyperparams: iters={hparams['iters']}, batch_size={hparams['batch_size']}, lr={hparams['lr']}")

    # Use ordered wrapper for yoga/reverse/complexity, default mlx_lm for random
    if mode in ("yoga", "reverse", "bookend", "complexity"):
        script = str(Path(__file__).parent / "mlx_ordered_lora.py")
    else:
        script = "-m mlx_lm lora"

    if mode == "random":
        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", model_name,
            "--data", str(train_data_dir),
            "--train",
            "--adapter-path", str(output_dir / "adapters"),
            "--iters", str(hparams["iters"]),
            "--batch-size", str(hparams["batch_size"]),
            "--learning-rate", hparams["lr"],
            "--num-layers", str(num_layers),
            "--seed", str(seed),
        ]
    else:
        ordered_wrapper = str(Path(__file__).parent / "mlx_ordered_lora.py")
        cmd = [
            sys.executable, ordered_wrapper,
            "--model", model_name,
            "--data", str(train_data_dir),
            "--train",
            "--adapter-path", str(output_dir / "adapters"),
            "--iters", str(hparams["iters"]),
            "--batch-size", str(hparams["batch_size"]),
            "--learning-rate", hparams["lr"],
            "--num-layers", str(num_layers),
            "--seed", str(seed),
        ]

    print(f"\n  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: Training failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n  Training complete. Adapter saved to: {output_dir / 'adapters'}")
    return output_dir


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT Training with Yoga Curriculum")
    parser.add_argument("--mode", choices=["yoga", "reverse", "bookend", "complexity", "random"],
                        default="yoga",
                        help="Training ordering mode (default: yoga)")
    parser.add_argument("--phase", choices=["small", "full"], default="small",
                        help="Phase: small (50 examples) or full (375)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--num-layers", type=int, default=16,
                        help="Number of LoRA layers (default: 16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    run_sft_training(
        mode=args.mode,
        phase=args.phase,
        model_name=args.model,
        num_layers=args.num_layers,
        seed=args.seed,
    )

    print(f"\nNext: python experiments/exp8_yoga_sft.py --phase {args.phase} --judge --judge-model haiku")


if __name__ == "__main__":
    main()
