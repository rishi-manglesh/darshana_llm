#!/usr/bin/env python3
"""Exp 6: Samkhya — Pretraining Data Organization

LAYER: Pretraining
RESEARCH FRAMEWORK: Nyaya Pancha-avayava (5-Step Syllogism)
STATUS: NEVER TESTED — most speculative experiment

PRATIJNA (Thesis):
  Organizing pretraining data by Samkhya's tattva categories (Purusha/Prakriti ->
  Gunas -> Tanmatras) produces a better small model than the same data in random order.

HETU (Reason):
  Samkhya's categories map to a knowledge hierarchy: meta-knowledge (Purusha) ->
  raw facts (Prakriti) -> analytical mode (Sattva/Rajas/Tamas) -> evidence types
  (Tanmatras). This order teaches model knowledge ABOUT knowledge before the
  knowledge itself — similar to curriculum learning in ML.

UDAHARANA (Prior Evidence):
  - No prior evidence from vedic_llm (never built)
  - Curriculum learning literature: data ordering can improve small model training
  - samkhya.py exists with keyword-based categorization

UPANAYA (Experiment Design):
  4 configs:
  - base: Qwen2.5-0.5B-Instruct, no continued pretraining
  - samkhya_ordered: + Samkhya-organized 5MB corpus (tattva-ordered)
  - random_ordered: + same corpus, shuffled
  - bloom_ordered: + same corpus, ordered by Bloom's taxonomy (Western control)

  The bloom_ordered control tests: is it ORDERING that helps, or specifically
  SAMKHYA's ordering?

NIGAMANA (Success Criteria):
  - PROVEN: samkhya_ordered > random_ordered AND samkhya_ordered > bloom_ordered
  - PARTIALLY PROVEN: samkhya ≈ bloom > random (ordering helps, Samkhya isn't special)
  - DISPROVEN: random ≈ samkhya (order doesn't matter at this scale)

Compute: ~4 hrs M4 | Cost: ~$0.30 judge
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, run_experiment,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp6_samkhya"
MODEL_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 512

CONFIGS = ["base", "samkhya_ordered", "random_ordered", "bloom_ordered"]

# Model paths (set after training)
MODEL_PATHS = {
    "base": MODEL_BASE,
    "samkhya_ordered": None,   # Set after running training/pretrain_samkhya.py --mode samkhya
    "random_ordered": None,    # Set after running training/pretrain_samkhya.py --mode random
    "bloom_ordered": None,     # Set after running training/pretrain_samkhya.py --mode bloom
}


# -- Generation ----------------------------------------------------------------

def generate_local(config, question):
    """Generate a response using a local model via MLX."""
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
    parser = argparse.ArgumentParser(description="Exp 6: Samkhya Pretraining Data Organization")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions (smoke test)")
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
