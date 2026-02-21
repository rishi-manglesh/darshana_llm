#!/usr/bin/env python3
"""Exp 6: Samkhya — Pretraining Data Organization

LAYER: Pretraining
RESEARCH FRAMEWORK: Nyaya Pancha-avayava (5-Step Syllogism)

PRATIJNA (Thesis):
  Organizing pretraining data by Samkhya's tattva categories (Purusha/Prakriti ->
  Gunas -> Tanmatras) produces a better small model than the same data in random
  order or Bloom's taxonomy order.

HETU (Reason):
  Samkhya's categories map to a knowledge hierarchy: meta-knowledge (Purusha) ->
  raw facts (Prakriti) -> analytical mode (Sattva/Rajas/Tamas) -> evidence types
  (Tanmatras). This order teaches model knowledge ABOUT knowledge before the
  knowledge itself.

NIGAMANA (Success Criteria):
  - PROVEN: samkhya > random AND samkhya > bloom
  - PARTIALLY PROVEN: samkhya ~ bloom > random (ordering helps, Samkhya not special)
  - DISPROVEN: random ~ samkhya (order doesn't matter)

Usage:
  python experiments/exp6_samkhya_pretraining.py --model-size 1.5b --questions all --judge --judge-model sonnet
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, run_experiment,
    get_extended_questions, get_all_questions,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp6_samkhya"
MAX_TOKENS = 512

MODEL_BASES = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}

CONFIGS = ["base", "samkhya_only", "bloom_only", "random_pt_only"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def get_model_paths(model_size):
    """Build model paths for the given model size."""
    base = MODEL_BASES.get(model_size, f"Qwen/Qwen2.5-{model_size.upper()}-Instruct")

    # Check for fused models
    samkhya_fused = MODELS_DIR / f"{model_size}-samkhya-fused"
    bloom_fused = MODELS_DIR / f"{model_size}-bloom-fused"
    random_fused = MODELS_DIR / f"{model_size}-random-fused"

    return {
        "base": base,
        "samkhya_only": str(samkhya_fused) if samkhya_fused.exists() else None,
        "bloom_only": str(bloom_fused) if bloom_fused.exists() else None,
        "random_pt_only": str(random_fused) if random_fused.exists() else None,
    }


# -- Generation ----------------------------------------------------------------

_model_cache = {}


def generate_local(config, question, model_paths):
    """Generate a response using a local model via MLX."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model_path = model_paths[config]
    if model_path is None:
        return {
            "response": f"[SKIP: {config} model not yet trained. Run training/pretrain_samkhya.py first.]",
            "word_count": 0,
        }

    # Cache loaded models
    if model_path not in _model_cache:
        _model_cache[model_path] = load(model_path)
    model, tokenizer = _model_cache[model_path]

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
    parser = argparse.ArgumentParser(description="Exp 6: Samkhya Pretraining")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions (smoke test)")
    parser.add_argument("--judge", action="store_true", help="Run judging after generation")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--model-size", choices=["0.5b", "1.5b", "3b"], default="1.5b",
                        help="Model size (default: 1.5b)")
    parser.add_argument("--questions", choices=["original", "extended", "all"],
                        default="original",
                        help="Question set: original (30), extended (45), all (75)")
    args = parser.parse_args()

    model_paths = get_model_paths(args.model_size)
    exp_name = f"{EXPERIMENT_NAME}_{args.model_size}" if args.model_size != "1.5b" else EXPERIMENT_NAME

    # Select question set
    if args.questions == "original":
        questions = TRANSFER_QUESTIONS
    elif args.questions == "extended":
        questions = get_extended_questions()
    elif args.questions == "all":
        questions = get_all_questions()

    # Check which models are available
    available_configs = []
    for config in CONFIGS:
        if config == "base" or model_paths.get(config) is not None:
            available_configs.append(config)

    if len(available_configs) < len(CONFIGS):
        missing = set(CONFIGS) - set(available_configs)
        print(f"WARNING: Models not yet trained: {missing}")
        print("Run training/pretrain_samkhya.py and training/fuse_adapters.py first.")
        print(f"Running with available configs: {available_configs}")

    def generate_fn(config, question):
        return generate_local(config, question, model_paths)

    results = run_experiment(
        name=exp_name,
        configs=available_configs,
        generate_fn=generate_fn,
        limit=args.limit,
        questions=questions,
    )

    if args.judge and len(available_configs) > 1:
        experimental = [c for c in available_configs if c != "base"]
        run_pairwise_judging(
            exp_name, "base", experimental,
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
