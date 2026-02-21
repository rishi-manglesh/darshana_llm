#!/usr/bin/env python3
"""Exp 7: Yoga DPO + Combined Darshana Stack Evaluation

LAYER: Post-training (DPO alignment) + Combined (pretrain x DPO)
RESEARCH FRAMEWORK: Nyaya Pancha-avayava (5-Step Syllogism)

Tests 11 configs covering:
  Exp 6 (pretraining only): samkhya, bloom, random pretrained
  Exp 7 (DPO only): yoga, reverse, complexity, random DPO
  Combined: samkhya+yoga, bloom+complexity, random+random

NIGAMANA (Success Criteria):
  - PROVEN: yoga > complexity > random (DPO) AND samkhya_yoga >> bloom_complexity
  - PARTIALLY PROVEN: ordering helps but darshana not special
  - DISPROVEN: all ~ base

Usage:
  python experiments/exp7_yoga_dpo.py --model-size 1.5b --questions all --judge --judge-model sonnet
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

EXPERIMENT_NAME = "exp7_yoga"
MAX_TOKENS = 512

MODEL_BASES = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}

# All 11 configs for the 2-factor experiment
CONFIGS = [
    # Control
    "base",
    # Exp 6: Pretraining only (no DPO)
    "samkhya_only",
    "bloom_only",
    "random_pt_only",
    # Exp 7: DPO only (no pretraining)
    "yoga_only",
    "reverse_only",
    "complexity_only",
    "random_dpo_only",
    # Combined: pretrain + DPO
    "samkhya_yoga",
    "bloom_complexity",
    "random_random",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def get_model_paths(model_size):
    """Build model paths for all 11 configs."""
    base = MODEL_BASES.get(model_size, f"Qwen/Qwen2.5-{model_size.upper()}-Instruct")
    s = model_size  # shorthand

    def model_path(subdir):
        p = MODELS_DIR / subdir
        # Check for DPO final model, then fused model, then adapter dir
        if (p / "final").exists():
            return str(p / "final")
        if p.exists():
            return str(p)
        return None

    return {
        "base": base,
        # Pretraining only (fused models)
        "samkhya_only": model_path(f"{s}-samkhya-fused"),
        "bloom_only": model_path(f"{s}-bloom-fused"),
        "random_pt_only": model_path(f"{s}-random-fused"),
        # DPO only (base + DPO)
        "yoga_only": model_path(f"{s}-yoga-dpo"),
        "reverse_only": model_path(f"{s}-reverse-dpo"),
        "complexity_only": model_path(f"{s}-complexity-dpo"),
        "random_dpo_only": model_path(f"{s}-random-dpo"),
        # Combined: pretrained + DPO
        "samkhya_yoga": model_path(f"{s}-samkhya-yoga-dpo"),
        "bloom_complexity": model_path(f"{s}-bloom-complexity-dpo"),
        "random_random": model_path(f"{s}-random-random-dpo"),
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
            "response": f"[SKIP: {config} model not yet trained.]",
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
    parser = argparse.ArgumentParser(description="Exp 7: Yoga DPO + Combined Stack")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument("--judge", action="store_true", help="Run judging")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--model-size", choices=["0.5b", "1.5b", "3b"], default="1.5b",
                        help="Model size (default: 1.5b)")
    parser.add_argument("--questions", choices=["original", "extended", "all"],
                        default="original",
                        help="Question set: original (30), extended (45), all (75)")
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        help="Specific configs to run (default: all available)")
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

    # Determine which configs to run
    requested_configs = args.configs or CONFIGS
    available_configs = []
    for config in requested_configs:
        if config == "base" or model_paths.get(config) is not None:
            available_configs.append(config)

    if len(available_configs) < len(requested_configs):
        missing = set(requested_configs) - set(available_configs)
        print(f"WARNING: Models not yet trained: {missing}")
        print("Run training/run_full_pipeline.py first.")
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
