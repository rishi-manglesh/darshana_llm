#!/usr/bin/env python3
"""Exp 1: Samkhya — Pretraining Data Organization

LAYER: Pretraining
VALIDATION: Does organizing training data by Samkhya's 25-tattva categories
improve a small model vs random data organization?

Method:
  1. Continue-pretrain Qwen2.5-0.5B on Samkhya-organized data
  2. Continue-pretrain Qwen2.5-0.5B on same data in random order
  3. Evaluate both + base on 30 questions with LLM judge

Configs:
  - base: Qwen2.5-0.5B-Instruct, no continued pretraining
  - samkhya_pretrained: + Samkhya-organized data (~5MB, tattva-ordered)
  - random_pretrained: + same data, random order

Success: samkhya > random on reasoning depth
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, run_experiment, mean,
    load_jsonl, append_jsonl,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp1_samkhya"
MODEL_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 512

CONFIGS = ["base", "samkhya_pretrained", "random_pretrained"]

# Model paths (set after training)
MODEL_PATHS = {
    "base": MODEL_BASE,
    "samkhya_pretrained": None,  # Set after running training/pretrain_samkhya.py
    "random_pretrained": None,   # Set after running training/pretrain_samkhya.py
}


# -- Generation ----------------------------------------------------------------

def generate_local(config, question):
    """Generate a response using a local model via MLX."""
    import re
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model_path = MODEL_PATHS[config]
    if model_path is None:
        return {
            "response": f"[SKIP: {config} model not yet trained. Run training/pretrain_samkhya.py first.]",
            "word_count": 0,
        }

    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=0.7, top_p=0.9)

    messages = [{"role": "user", "content": question["query"]}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = generate(
        model, tokenizer, prompt=text,
        max_tokens=MAX_TOKENS, sampler=sampler, verbose=False
    )
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

    return {
        "response": result,
        "word_count": len(result.split()),
        "model_path": model_path,
    }


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 1: Samkhya Pretraining")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions (for smoke test)")
    parser.add_argument("--judge", action="store_true", help="Run judging after generation")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    args = parser.parse_args()

    # Check which models are available
    available_configs = []
    for config in CONFIGS:
        if config == "base" or MODEL_PATHS.get(config) is not None:
            available_configs.append(config)

    if len(available_configs) < len(CONFIGS):
        missing = set(CONFIGS) - set(available_configs)
        print(f"WARNING: Models not yet trained: {missing}")
        print("Run training/pretrain_samkhya.py first for full experiment.")
        print(f"Running with available configs: {available_configs}")

    # We need to cache loaded models to avoid reloading per question
    _model_cache = {}

    def generate_fn(config, question):
        return generate_local(config, question)

    results = run_experiment(
        name=EXPERIMENT_NAME,
        configs=available_configs,
        generate_fn=generate_fn,
        limit=args.limit,
    )

    if args.judge and len(available_configs) > 1:
        experimental = [c for c in available_configs if c != "base"]
        run_pairwise_judging(
            EXPERIMENT_NAME, "base", experimental,
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
