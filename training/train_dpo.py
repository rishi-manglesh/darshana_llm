#!/usr/bin/env python3
"""DPO Training with Multiple Ordering Strategies

Trains DPO variants using combined real pairs from experiments + extended generation.

Ordering modes:
  - yoga: Pairs sorted by Yoga stage 1->5 (darshana hypothesis)
  - reverse: Pairs sorted by stage 5->1 (tests direction)
  - complexity: Pairs sorted by score_delta ascending (easy->hard, Western control)
  - random: Pairs shuffled (null hypothesis)

Supports:
  - Custom base model via --base-model
  - Pretrained model as base via --pretrained-path (for combined configs)
  - All modes in one run via --all-modes

Uses TRL's DPOTrainer (HuggingFace) for DPO training.

Usage:
  python training/train_dpo.py --mode yoga --base-model Qwen/Qwen2.5-1.5B-Instruct
  python training/train_dpo.py --all-modes --base-model Qwen/Qwen2.5-1.5B-Instruct
  python training/train_dpo.py --mode yoga --pretrained-path models/1.5b-samkhya-fused
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- Config --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMBINED_PAIRS_PATH = PROJECT_ROOT / "data" / "dpo_combined" / "train.jsonl"
REAL_PAIRS_PATH = PROJECT_ROOT / "data" / "dpo_real_pairs.jsonl"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_SIZE_MAP = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "3b",
}

DPO_MODES = ["yoga", "reverse", "complexity", "random"]


def load_dpo_pairs(path):
    """Load DPO pairs from JSONL."""
    pairs = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("preferred") and rec.get("rejected"):
                pairs.append(rec)
    return pairs


def order_pairs(pairs, mode):
    """Order pairs according to the specified mode."""
    ordered = list(pairs)

    if mode == "yoga":
        # Sort by Yoga stage 1->5 (darshana hypothesis)
        ordered.sort(key=lambda p: p.get("stage", 3))
    elif mode == "reverse":
        # Sort by Yoga stage 5->1 (tests direction)
        ordered.sort(key=lambda p: p.get("stage", 3), reverse=True)
    elif mode == "complexity":
        # Sort by score_delta ascending (easy->hard, Western control)
        ordered.sort(key=lambda p: p.get("total_delta", p.get("score_delta", 0)))
    elif mode == "random":
        # Shuffle (null hypothesis)
        random.shuffle(ordered)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return ordered


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


def get_output_dir(mode, base_model, pretrained_path=None):
    """Determine output directory based on mode and model."""
    size = MODEL_SIZE_MAP.get(base_model, base_model.split("/")[-1].lower())

    if pretrained_path:
        # Extract pretrain config from path (e.g., "1.5b-samkhya-fused" -> "samkhya")
        pt_name = Path(pretrained_path).name
        parts = pt_name.split("-")
        if len(parts) >= 3:
            pt_config = parts[1]  # e.g., "samkhya", "bloom", "random"
        else:
            pt_config = pt_name
        return MODELS_DIR / f"{size}-{pt_config}-{mode}-dpo"
    else:
        return MODELS_DIR / f"{size}-{mode}-dpo"


def train_dpo(mode, pairs, output_dir, base_model, pretrained_path=None):
    """Run DPO training using TRL."""
    model_path = pretrained_path if pretrained_path else base_model

    print(f"\n{'='*60}")
    print(f"DPO Training: {mode}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Base model: {model_path}")
    if pretrained_path:
        print(f"  (Pretrained from: {pretrained_path})")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        from datasets import Dataset
    except ImportError:
        print("ERROR: Requires trl and datasets packages.")
        print("  pip install trl datasets transformers torch")
        sys.exit(1)

    # Prepare dataset
    dataset = prepare_dpo_dataset(pairs)
    ds = Dataset.from_list(dataset)

    # Load model
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)

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
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"  Saved to: {final_dir}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPO Training with Multiple Orderings")
    parser.add_argument(
        "--mode",
        choices=DPO_MODES,
        default=None,
        help="DPO ordering mode",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Train all 4 modes (yoga, reverse, complexity, random)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help="Path to pretrained (fused) model as base for combined configs",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=f"Path to DPO pairs JSONL (default: {COMBINED_PAIRS_PATH})",
    )
    args = parser.parse_args()

    # Determine data source
    if args.data:
        data_path = Path(args.data)
    elif COMBINED_PAIRS_PATH.exists():
        data_path = COMBINED_PAIRS_PATH
    elif REAL_PAIRS_PATH.exists():
        print(f"  Combined data not found, falling back to: {REAL_PAIRS_PATH}")
        data_path = REAL_PAIRS_PATH
    else:
        print("ERROR: No DPO data found. Run prepare_dpo_data.py first.")
        sys.exit(1)

    if not data_path.exists():
        print(f"ERROR: Data not found: {data_path}")
        sys.exit(1)

    # Load pairs
    all_pairs = load_dpo_pairs(data_path)
    print(f"Loaded {len(all_pairs)} DPO pairs from {data_path}")

    # Determine modes
    if args.all_modes:
        modes = DPO_MODES
    elif args.mode:
        modes = [args.mode]
    else:
        print("ERROR: Specify --mode or --all-modes")
        sys.exit(1)

    # Resolve pretrained path
    pretrained = None
    if args.pretrained_path:
        pretrained = str(PROJECT_ROOT / args.pretrained_path) \
            if not Path(args.pretrained_path).is_absolute() \
            else args.pretrained_path

    # Train each mode
    for mode in modes:
        pairs = order_pairs(all_pairs, mode)
        output_dir = get_output_dir(mode, args.base_model, pretrained)
        train_dpo(mode, pairs, output_dir, args.base_model, pretrained)

    print(f"\nAll DPO training complete.")
    print(f"Models saved in: {MODELS_DIR}/")


if __name__ == "__main__":
    main()
