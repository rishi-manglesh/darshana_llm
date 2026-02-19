#!/usr/bin/env python3
"""Continued Pretraining with Samkhya-Organized Data

Continues pretraining Qwen2.5-0.5B on:
  1. Samkhya-ordered corpus (tattva categories in ontological order)
  2. Random-ordered corpus (same data, shuffled)

Uses MLX for Apple Silicon training.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- Config --------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "samkhya_corpus"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAINING_CONFIGS = {
    "samkhya": {
        "data": "samkhya_corpus.txt",
        "output": "models/qwen25-samkhya-pretrained",
    },
    "random": {
        "data": "random_corpus.txt",
        "output": "models/qwen25-random-pretrained",
    },
}


def prepare_training_data(corpus_path, output_path):
    """Convert raw text corpus to JSONL training format."""
    texts = []
    with open(corpus_path) as f:
        current = []
        for line in f:
            if line.strip() == "" and current:
                text = " ".join(current).strip()
                if len(text) > 100:  # Skip very short passages
                    texts.append(text)
                current = []
            else:
                current.append(line.strip())
        if current:
            text = " ".join(current).strip()
            if len(text) > 100:
                texts.append(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")

    print(f"  Prepared {len(texts)} training passages -> {output_path}")
    return len(texts)


def run_pretraining(config_name):
    """Run continued pretraining using MLX."""
    config = TRAINING_CONFIGS[config_name]
    corpus_path = DATA_DIR / config["data"]
    output_dir = Path(__file__).resolve().parent.parent / config["output"]

    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found. Run training/prepare_samkhya_data.py first.")
        sys.exit(1)

    # Prepare JSONL training data
    train_path = DATA_DIR / f"{config_name}_train.jsonl"
    n_passages = prepare_training_data(corpus_path, train_path)

    # MLX training command
    print(f"\nStarting continued pretraining: {config_name}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Data: {train_path} ({n_passages} passages)")
    print(f"  Output: {output_dir}")

    import subprocess
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", MODEL_NAME,
        "--data", str(DATA_DIR),
        "--train",
        "--adapter-path", str(output_dir / "adapters"),
        "--iters", "200",
        "--batch-size", "2",
        "--learning-rate", "5e-5",
        "--lora-layers", "16",
    ]

    print(f"\n  Command: {' '.join(cmd)}")
    print(f"\n  (This will take ~1 hour on M4)")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: Training failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n  Training complete. Adapter saved to: {output_dir / 'adapters'}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Samkhya Continued Pretraining")
    parser.add_argument("--mode", choices=["samkhya", "random", "both"], default="both",
                        help="Which ordering to train (default: both)")
    args = parser.parse_args()

    modes = ["samkhya", "random"] if args.mode == "both" else [args.mode]

    for mode in modes:
        run_pretraining(mode)

    print("\nAll training complete. Update MODEL_PATHS in exp1_samkhya_pretraining.py.")


if __name__ == "__main__":
    main()
