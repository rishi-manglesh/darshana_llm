#!/usr/bin/env python3
"""Exp 7: Yoga — DPO Curriculum Ordering

LAYER: Post-training (DPO alignment)
RESEARCH FRAMEWORK: Nyaya Pancha-avayava (5-Step Syllogism)
STATUS: NEVER TESTED — second most speculative experiment

PRATIJNA (Thesis):
  Training DPO preference pairs in Yoga's Ashtanga order (ethics -> stability ->
  focus -> depth -> integration) produces better-aligned models than the same
  pairs in random order.

HETU (Reason):
  Yoga's 8 limbs encode a developmental sequence: learn honesty (Yama) before
  structure (Asana) before focus (Pratyahara) before depth (Dharana) before
  synthesis (Samadhi). Likewise, an LLM should learn "don't hallucinate" before
  "reason deeply" before "synthesize coherently."

UDAHARANA (Prior Evidence):
  - No prior evidence from vedic_llm (never built)
  - DPO literature: data ordering can matter on small models
  - yoga_dpo.py exists with 5-stage mapping

UPANAYA (Experiment Design):
  5 configs:
  - base: Qwen2.5-0.5B-Instruct, no DPO
  - yoga_ordered: 150 pairs in Ashtanga stage order
  - random_ordered: Same 150 pairs, shuffled
  - reverse_ordered: Same 150 pairs, REVERSE Ashtanga order (synthesis -> ethics)
  - generic_curriculum: 150 pairs ordered by complexity (simple -> complex) without Yoga

  reverse_ordered directly tests whether the ORDER matters.
  generic_curriculum tests whether Yoga's specific ordering adds value beyond
  "easy first."

NIGAMANA (Success Criteria):
  - PROVEN: yoga_ordered > random_ordered AND yoga_ordered > generic_curriculum
  - PARTIALLY PROVEN: yoga ≈ generic > random (ordering helps, Yoga isn't special)
  - DISPROVEN: random ≈ yoga (DPO pair order doesn't matter on small models)

Compute: ~8 hrs M4 (4 DPO runs) | Cost: ~$1.05
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

EXPERIMENT_NAME = "exp7_yoga"
MODEL_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 512

CONFIGS = ["base", "yoga_ordered", "random_ordered", "reverse_ordered", "generic_curriculum"]

MODEL_PATHS = {
    "base": MODEL_BASE,
    "yoga_ordered": None,         # Set after training/train_dpo.py --mode yoga
    "random_ordered": None,       # Set after training/train_dpo.py --mode random
    "reverse_ordered": None,      # Set after training/train_dpo.py --mode reverse
    "generic_curriculum": None,   # Set after training/train_dpo.py --mode generic
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
    parser = argparse.ArgumentParser(description="Exp 7: Yoga DPO Curriculum Ordering")
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
