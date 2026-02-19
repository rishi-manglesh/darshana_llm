#!/usr/bin/env python3
"""DPO Training with Yoga Curriculum vs Random vs Standard

Trains three DPO variants on Qwen2.5-0.5B:
  1. yoga_dpo: Pairs in Yoga stage order (Yama -> ... -> Samadhi)
  2. random_dpo: Same 150 Yoga pairs, shuffled
  3. standard_dpo: 150 generic preference pairs

Uses TRL's DPOTrainer (HuggingFace) for DPO training.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- Config --------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dpo_yoga_pairs"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_dpo_pairs(path):
    """Load DPO pairs from JSONL."""
    pairs = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("preferred") and rec.get("rejected"):
                pairs.append(rec)
    return pairs


def prepare_dpo_dataset(pairs):
    """Convert pairs to DPO dataset format."""
    dataset = []
    for p in pairs:
        dataset.append({
            "prompt": p["query"],
            "chosen": p["preferred"],
            "rejected": p["rejected"],
        })
    return dataset


def train_dpo(mode, pairs, output_dir):
    """Run DPO training using TRL."""
    print(f"\n{'='*60}")
    print(f"DPO Training: {mode}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        from datasets import Dataset
    except ImportError:
        print("ERROR: Requires trl and datasets packages.")
        print("  pip install trl datasets")
        sys.exit(1)

    # Prepare dataset
    dataset = prepare_dpo_dataset(pairs)
    ds = Dataset.from_list(dataset)

    # Load model
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # DPO config
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        max_length=512,
        max_prompt_length=256,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("  Training...")
    trainer.train()

    # Save final model
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"  Saved to: {output_dir / 'final'}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--mode", choices=["yoga", "random", "standard", "all"], default="all")
    args = parser.parse_args()

    yoga_path = DATA_DIR / "yoga_pairs.jsonl"
    generic_path = DATA_DIR / "generic_pairs.jsonl"

    modes = ["yoga", "random", "standard"] if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "yoga":
            if not yoga_path.exists():
                print(f"ERROR: {yoga_path} not found. Run generate_dpo_pairs.py first.")
                continue
            pairs = load_dpo_pairs(yoga_path)
            # Sort by stage (curriculum order)
            pairs.sort(key=lambda p: p.get("stage", 0))
            train_dpo(mode, pairs, MODELS_DIR / "qwen25-yoga-dpo")

        elif mode == "random":
            if not yoga_path.exists():
                print(f"ERROR: {yoga_path} not found. Run generate_dpo_pairs.py first.")
                continue
            pairs = load_dpo_pairs(yoga_path)
            random.shuffle(pairs)
            train_dpo(mode, pairs, MODELS_DIR / "qwen25-random-dpo")

        elif mode == "standard":
            if not generic_path.exists():
                print(f"ERROR: {generic_path} not found. Run generate_dpo_pairs.py first.")
                continue
            pairs = load_dpo_pairs(generic_path)
            random.shuffle(pairs)
            train_dpo(mode, pairs, MODELS_DIR / "qwen25-standard-dpo")

    print("\nAll training complete. Update MODEL_PATHS in exp2_yoga_posttraining.py.")


if __name__ == "__main__":
    main()
