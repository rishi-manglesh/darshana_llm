#!/usr/bin/env python3
"""Exp 2: Yoga — Post-Training (DPO) Curriculum

LAYER: Post-training
VALIDATION: Does Yoga's 8-limb training ORDER improve DPO outcomes
vs standard random-order DPO?

Method:
  1. Generate 150 DPO preference pairs via Claude (30 questions x 5 stages)
  2. Train DPO three ways on Qwen2.5-0.5B:
     - Yoga curriculum: stages in order (Yama -> Niyama -> ... -> Samadhi)
     - Random DPO: same 150 pairs, shuffled
     - Standard DPO: 150 generic preference pairs
  3. Evaluate all 4 on 30 questions

Configs:
  - base: Qwen2.5-0.5B-Instruct, no DPO
  - yoga_dpo: Yoga-ordered DPO (150 pairs, curriculum order)
  - random_dpo: same 150 pairs, shuffled
  - standard_dpo: 150 generic preference pairs

Success: yoga_dpo > random_dpo > base. Yoga order matters.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, run_experiment,
    load_jsonl, append_jsonl,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp2_yoga"
MODEL_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 512

CONFIGS = ["base", "yoga_dpo", "random_dpo", "standard_dpo"]

MODEL_PATHS = {
    "base": MODEL_BASE,
    "yoga_dpo": None,      # Set after running training/train_dpo.py --mode yoga
    "random_dpo": None,     # Set after running training/train_dpo.py --mode random
    "standard_dpo": None,   # Set after running training/train_dpo.py --mode standard
}


# -- Generation ----------------------------------------------------------------

def generate_local(config, question):
    """Generate a response using a local model via MLX."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model_path = MODEL_PATHS[config]
    if model_path is None:
        return {
            "response": f"[SKIP: {config} model not yet trained. Run training/train_dpo.py first.]",
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
    parser = argparse.ArgumentParser(description="Exp 2: Yoga DPO Curriculum")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument("--judge", action="store_true", help="Run judging")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    args = parser.parse_args()

    available_configs = []
    for config in CONFIGS:
        if config == "base" or MODEL_PATHS.get(config) is not None:
            available_configs.append(config)

    if len(available_configs) < len(CONFIGS):
        missing = set(CONFIGS) - set(available_configs)
        print(f"WARNING: Models not yet trained: {missing}")
        print("Run training/generate_dpo_pairs.py then training/train_dpo.py first.")
        print(f"Running with available configs: {available_configs}")

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
