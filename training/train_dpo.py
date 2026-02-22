#!/usr/bin/env python3
"""DPO Training with Multiple Ordering Strategies

Trains DPO variants using combined real pairs from experiments + extended generation.

Ordering modes:
  - yoga: Pairs sorted by Yoga stage 1->5 (darshana hypothesis)
  - reverse: Pairs sorted by stage 5->1 (tests direction)
  - complexity: Pairs sorted by score_delta ascending (easy->hard, Western control)
  - random: Pairs shuffled (null hypothesis)

CRITICAL: Uses SequentialSampler to preserve pair ordering — the whole point
of the experiment is to test whether ORDER matters. HuggingFace Trainer uses
RandomSampler by default, which would destroy our ordering.

Uses LoRA (via peft) for memory-efficient training on Apple Silicon.

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

SEED = 42


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


def validate_stages(pairs, mode):
    """Validate that pairs have stage fields when ordering by stage.

    Fix #4: Warn about missing stages instead of silently defaulting.
    """
    if mode not in ("yoga", "reverse"):
        return

    missing = [i for i, p in enumerate(pairs) if "stage" not in p]
    if missing:
        print(f"  WARNING: {len(missing)}/{len(pairs)} pairs missing 'stage' field")
        print(f"  These will default to stage=3 (middle). First 5: {missing[:5]}")
        if len(missing) > len(pairs) * 0.5:
            print(f"  ERROR: >50% of pairs missing stage — ordering will be meaningless!")
            print(f"  Check your data pipeline (prepare_dpo_data.py)")
            sys.exit(1)


def order_pairs(pairs, mode):
    """Order pairs according to the specified mode."""
    random.seed(SEED)
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
    """Run DPO training using TRL with SequentialSampler and LoRA."""
    model_path = pretrained_path if pretrained_path else base_model

    print(f"\n{'='*60}")
    print(f"DPO Training: {mode}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Base model: {model_path}")
    if pretrained_path:
        print(f"  (Pretrained from: {pretrained_path})")
    print(f"  Output: {output_dir}")
    print(f"  Using: SequentialSampler (ordering preserved)")
    print(f"  Using: LoRA (memory-efficient)")
    print(f"  Seed: {SEED}")
    print(f"{'='*60}")

    # Verify ordering
    if mode in ("yoga", "reverse"):
        stages = [p.get("stage", 3) for p in pairs]
        print(f"  Stage sequence (first 10): {stages[:10]}")
        print(f"  Stage sequence (last 10):  {stages[-10:]}")

    try:
        import torch
        from torch.utils.data import SequentialSampler
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        print(f"ERROR: Missing package: {e}")
        print("  pip install trl datasets transformers torch peft")
        sys.exit(1)

    # Fix #1: Custom trainer with SequentialSampler to preserve ordering
    class OrderedDPOTrainer(DPOTrainer):
        """DPO trainer that preserves dataset ordering."""
        def _get_train_sampler(self, dataset=None):
            return SequentialSampler(dataset if dataset is not None else self.train_dataset)

    # Prepare dataset
    dataset = prepare_dpo_dataset(pairs)
    ds = Dataset.from_list(dataset)

    # Load model
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )

    # Fix #12: LoRA config for memory-efficient training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # DPO config — Fix #2: explicit seed
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
        seed=SEED,
        data_seed=SEED,
    )

    trainer = OrderedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    # Verify sampler type
    sampler = trainer._get_train_sampler()
    print(f"  Sampler type: {type(sampler).__name__}")
    assert "Sequential" in type(sampler).__name__, \
        f"Expected SequentialSampler but got {type(sampler).__name__}"

    print("  Training...")
    trainer.train()

    # Save final model (merge LoRA weights)
    final_dir = output_dir / "final"
    merged = model.merge_and_unload()
    merged.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"  Saved merged model to: {final_dir}")


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

    # Set global seeds
    random.seed(SEED)

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
        validate_stages(all_pairs, mode)
        pairs = order_pairs(all_pairs, mode)
        output_dir = get_output_dir(mode, args.base_model, pretrained)
        train_dpo(mode, pairs, output_dir, args.base_model, pretrained)

    print(f"\nAll DPO training complete.")
    print(f"Models saved in: {MODELS_DIR}/")


if __name__ == "__main__":
    main()
