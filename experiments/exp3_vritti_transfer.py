#!/usr/bin/env python3
"""Exp 3: Vritti — Transfer to Target Open-Source Model

LAYER: System prompt
STATUS: Already VALIDATED on Qwen3-4B (63%) and Sonnet (100%).
        Replicate on Qwen2.5-7B-Instruct.

Configs (3 x 30 = 90 generations, LOCAL):
  - qwen25_baseline: Qwen2.5-7B-Instruct, no system prompt
  - qwen25_vritti: + Vritti inline prompt
  - qwen25_vritti_cot: + Vritti + CoT

Success: >60% win rate
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.vritti import VRITTI_INLINE_PROMPT
from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, run_experiment,
    count_vritti_tags, count_hedging, count_udaharana,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp3_vritti"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 512

SYSTEM_PROMPTS = {
    "qwen25_baseline": None,
    "qwen25_vritti": VRITTI_INLINE_PROMPT,
    "qwen25_vritti_cot": (
        "Think step by step. Consider the question carefully, break it into parts, "
        "and reason through each part before giving your conclusion.\n\n"
        + VRITTI_INLINE_PROMPT
    ),
}

CONFIGS = list(SYSTEM_PROMPTS.keys())

# Model cache (avoid reloading per question)
_model = None
_tokenizer = None
_sampler = None


def _get_model():
    global _model, _tokenizer, _sampler
    if _model is None:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        print(f"Loading model: {MODEL_NAME}", flush=True)
        _model, _tokenizer = load(MODEL_NAME)
        _sampler = make_sampler(temp=0.7, top_p=0.9)
    return _model, _tokenizer, _sampler


# -- Generation ----------------------------------------------------------------

def generate_fn(config, question):
    """Generate response using local Qwen2.5-7B with optional Vritti prompt."""
    from mlx_lm import generate

    model, tokenizer, sampler = _get_model()
    system_prompt = SYSTEM_PROMPTS[config]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question["query"]})

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
        "metrics": {
            "vritti_tags": count_vritti_tags(result),
            "hedging_count": count_hedging(result),
            "udaharana_count": count_udaharana(result),
        },
    }


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 3: Vritti Transfer to Qwen2.5-7B")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument("--judge", action="store_true", help="Run judging")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    args = parser.parse_args()

    results = run_experiment(
        name=EXPERIMENT_NAME,
        configs=CONFIGS,
        generate_fn=generate_fn,
        limit=args.limit,
    )

    # Print metrics summary
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print(f"{'='*60}")
    for cfg in CONFIGS:
        recs = by_config.get(cfg, [])
        if not recs:
            continue
        n = len(recs)
        avg_tags = sum(r.get("metrics", {}).get("vritti_tags", {}).get("total", 0) for r in recs) / n
        avg_hedge = sum(r.get("metrics", {}).get("hedging_count", 0) for r in recs) / n
        avg_words = sum(r.get("word_count", 0) for r in recs) / n
        print(f"  {cfg:<25} | tags: {avg_tags:.1f} | hedges: {avg_hedge:.1f} | words: {avg_words:.0f}")

    if args.judge:
        experimental = [c for c in CONFIGS if c != "qwen25_baseline"]
        run_pairwise_judging(
            EXPERIMENT_NAME, "qwen25_baseline", experimental,
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
