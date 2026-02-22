#!/usr/bin/env python3
"""Continued Pretraining with Samkhya/Bloom/Random-Organized Data

Continues pretraining a model on:
  1. Samkhya-ordered corpus (tattva categories in ontological order)
  2. Bloom-ordered corpus (Bloom's taxonomy order — Western control)
  3. Random-ordered corpus (same data, shuffled — null hypothesis)

Uses MLX for Apple Silicon training.

Usage:
  python training/pretrain_samkhya.py --mode all --model Qwen/Qwen2.5-1.5B-Instruct
  python training/pretrain_samkhya.py --mode samkhya --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- Config --------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "samkhya_corpus"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Model size shorthand for output directories
MODEL_SIZE_MAP = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "3b",
}


def get_training_configs(model_name):
    """Build training configs with model-size-aware output paths."""
    size = MODEL_SIZE_MAP.get(model_name, model_name.split("/")[-1].lower())
    return {
        "samkhya": {
            "data": "samkhya_corpus.txt",
            "output": f"models/{size}-samkhya-pretrained",
        },
        "bloom": {
            "data": "bloom_corpus.txt",
            "output": f"models/{size}-bloom-pretrained",
        },
        "random": {
            "data": "random_corpus.txt",
            "output": f"models/{size}-random-pretrained",
        },
    }


def prepare_training_data(corpus_path, output_dir):
    """Convert raw text corpus to train.jsonl + valid.jsonl in output_dir.

    mlx_lm expects a directory with train.jsonl and valid.jsonl files.

    Fix #6: Uses stratified sampling for validation split instead of taking
    the last 10% — taking the last 10% would bias validation toward later
    categories in an ordered corpus, defeating the purpose of ordering.
    """
    import random as _rnd
    _rnd.seed(42)

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

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fix #6: Stratified 90/10 split — sample every 10th item for validation
    # This preserves the ordering in training set while getting representative validation
    n_val = max(1, len(texts) // 10)
    val_indices = set(range(0, len(texts), max(1, len(texts) // n_val)))
    train_texts = [t for i, t in enumerate(texts) if i not in val_indices]
    val_texts = [t for i, t in enumerate(texts) if i in val_indices]

    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for text in train_texts:
            f.write(json.dumps({"text": text}) + "\n")

    valid_path = output_dir / "valid.jsonl"
    with open(valid_path, "w") as f:
        for text in val_texts:
            f.write(json.dumps({"text": text}) + "\n")

    print(f"  Prepared {len(train_texts)} train + {len(val_texts)} valid passages -> {output_dir}")
    return len(train_texts)


def run_pretraining(config_name, model_name, iters=200, batch_size=2, lr="5e-5", num_layers=16):
    """Run continued pretraining using MLX."""
    configs = get_training_configs(model_name)
    config = configs[config_name]
    corpus_path = DATA_DIR / config["data"]
    output_dir = PROJECT_ROOT / config["output"]

    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found. Run training/prepare_samkhya_data.py first.")
        sys.exit(1)

    # Prepare JSONL training data — mlx_lm expects train.jsonl + valid.jsonl in a dir
    train_data_dir = DATA_DIR / f"{config_name}_data"
    n_passages = prepare_training_data(corpus_path, train_data_dir)

    # MLX training command
    print(f"\nStarting continued pretraining: {config_name}")
    print(f"  Model: {model_name}")
    print(f"  Data: {train_data_dir} ({n_passages} train passages)")
    print(f"  Output: {output_dir}")

    import subprocess
    # Fix #3: Use ordered wrapper that preserves corpus ordering
    # mlx_lm sorts by token length + shuffles batches, destroying our ordering.
    # mlx_ordered_lora.py monkey-patches iterate_batches to iterate sequentially.
    ordered_wrapper = str(Path(__file__).parent / "mlx_ordered_lora.py")
    cmd = [
        sys.executable, ordered_wrapper,
        "--model", model_name,
        "--data", str(train_data_dir),
        "--train",
        "--adapter-path", str(output_dir / "adapters"),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", lr,
        "--num-layers", str(num_layers),
        "--seed", "42",
    ]

    print(f"\n  Command: {' '.join(cmd)}")
    print(f"\n  (This will take ~1 hour per config on M4)")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: Training failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n  Training complete. Adapter saved to: {output_dir / 'adapters'}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Samkhya/Bloom/Random Continued Pretraining")
    parser.add_argument("--mode", choices=["samkhya", "bloom", "random", "all"],
                        default="all",
                        help="Which ordering to train (default: all)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--iters", type=int, default=200,
                        help="Training iterations (default: 200)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size (default: 2)")
    parser.add_argument("--lr", type=str, default="5e-5",
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--num-layers", type=int, default=16,
                        help="Number of LoRA layers (default: 16)")
    args = parser.parse_args()

    modes = ["samkhya", "bloom", "random"] if args.mode == "all" else [args.mode]

    for mode in modes:
        run_pretraining(mode, args.model, args.iters, args.batch_size, args.lr, args.num_layers)

    print(f"\nAll training complete.")
    print(f"Next: python training/fuse_adapters.py --model {args.model}")


if __name__ == "__main__":
    main()
