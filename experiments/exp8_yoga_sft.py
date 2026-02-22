#!/usr/bin/env python3
"""Exp 8: Yoga SFT Curriculum Evaluation

LAYER: Post-training (SFT alignment with Yoga curriculum ordering)
RESEARCH FRAMEWORK: Progressive faculty development via Ashtanga limbs

Phase 1 (small test — yoga vs base only):
  2 configs: base, yoga_sft
  10 held-out questions, pairwise judge vs base using Haiku
  Decision gate: does yoga_sft beat base?

Phase 2 (scale up — if Phase 1 shows signal):
  4 configs: base, yoga_sft, reverse_sft, random_sft
  60 held-out questions, pairwise judge with Sonnet
  Bootstrap 95% CIs, effect sizes, per-dimension analysis

NIGAMANA (Success Criteria):
  Phase 1: yoga_sft > base (>10% win rate gap)
  Phase 2: yoga > random, yoga-random gap >10%, CI lower bound >50%

Usage:
  python experiments/exp8_yoga_sft.py --phase small --judge --judge-model haiku
  python experiments/exp8_yoga_sft.py --phase full --judge --judge-model sonnet
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, run_experiment,
    load_jsonl,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp8_yoga_sft"
MAX_TOKENS = 512

MODEL_BASES = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SFT_DATA_DIR = PROJECT_ROOT / "data" / "sft_yoga"

# Phase 1: yoga vs base only
PHASE1_CONFIGS = ["base", "yoga_sft"]

# Phase 2: full comparison (only if Phase 1 shows signal)
PHASE2_CONFIGS = ["base", "yoga_sft", "reverse_sft", "random_sft"]


def get_model_paths(model_size, phase):
    """Build model paths for the current phase.

    Returns dict mapping config -> (base_model, adapter_path_or_None).
    For 'base' config, adapter is None. For SFT configs, adapter points
    to the adapters/ dir so mlx_lm.load() can apply the LoRA weights.
    """
    base = MODEL_BASES.get(model_size, f"Qwen/Qwen2.5-{model_size.upper()}-Instruct")
    s = model_size

    def adapter_info(name):
        """Return (base, adapter_path) if adapter exists, else None."""
        p = MODELS_DIR / name / "adapters"
        if p.exists():
            return (base, str(p))
        return None

    if phase == "small":
        return {
            "base": (base, None),
            "yoga_sft": adapter_info(f"{s}-yoga-sft-small"),
        }
    else:
        paths = {
            "base": (base, None),
            "yoga_sft": adapter_info(f"{s}-yoga-sft"),
            "reverse_sft": adapter_info(f"{s}-reverse-sft"),
            "random_sft": adapter_info(f"{s}-random-sft"),
        }
        # Add 1-epoch variants if they exist
        for mode in ["yoga", "reverse", "random", "bookend"]:
            key = f"{mode}_1ep"
            info = adapter_info(f"{s}-{mode}-sft-1ep")
            if info:
                paths[key] = info
        return paths


def load_holdout_questions(phase):
    """Load held-out questions for evaluation."""
    if phase == "small":
        holdout_path = SFT_DATA_DIR / "small_holdout.json"
    else:
        holdout_path = SFT_DATA_DIR / "full_holdout.json"

    if not holdout_path.exists():
        print(f"  [WARN] Holdout not found: {holdout_path}")
        print(f"  Using first 10 TRANSFER_QUESTIONS instead")
        return TRANSFER_QUESTIONS[:10]

    with open(holdout_path) as f:
        return json.load(f)


# -- Generation ----------------------------------------------------------------

_model_cache = {}
_gen_temp = 0.7
_gen_top_p = 0.9


def generate_local(config, question, model_paths):
    """Generate a response using a local model via MLX.

    model_paths values are (base_model, adapter_path_or_None) tuples.
    """
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    path_info = model_paths[config]
    if path_info is None:
        raise FileNotFoundError(
            f"Model not found for config '{config}'. "
            f"Run training first or remove this config."
        )

    base_model, adapter_path = path_info
    cache_key = f"{base_model}|{adapter_path}"

    # Cache loaded models
    if cache_key not in _model_cache:
        if adapter_path:
            _model_cache[cache_key] = load(base_model, adapter_path=adapter_path)
        else:
            _model_cache[cache_key] = load(base_model)
    model, tokenizer = _model_cache[cache_key]

    sampler = make_sampler(temp=_gen_temp, top_p=_gen_top_p)
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
        "model_path": f"{base_model}+{adapter_path}" if adapter_path else base_model,
    }


# -- Per-Dimension Analysis ----------------------------------------------------

def per_dimension_analysis(experiment_name):
    """Analyze judge results per quality dimension.

    Maps judge dimensions back to Yoga stages to see if
    stage 1 (epistemic_honesty) specifically improves factual_accuracy, etc.
    """
    judge_path = RESULTS_DIR / experiment_name / "judge.jsonl"
    if not judge_path.exists():
        print("  No judge results found for per-dimension analysis")
        return

    judgments = load_jsonl(judge_path)
    pairwise = [r for r in judgments if r.get("type") == "pairwise"]

    if not pairwise:
        print("  No pairwise judgments found")
        return

    # Yoga stage → judge dimension mapping
    stage_dimension_map = {
        "epistemic_honesty": "factual_accuracy",
        "structural_stability": "completeness",
        "focus": "usefulness",
        "analytical_depth": "reasoning_depth",
        "synthesis": "usefulness",
    }

    dimensions = ["factual_accuracy", "reasoning_depth", "completeness", "calibration", "usefulness"]

    print(f"\n  Per-Dimension Analysis ({len(pairwise)} judgments):")
    print(f"  {'Dimension':<22} {'Exp Mean':>10} {'Base Mean':>10} {'Delta':>8}")
    print(f"  {'-'*52}")

    for dim in dimensions:
        exp_scores = []
        base_scores = []
        for j in pairwise:
            # scores_a = baseline scores, scores_b = experimental scores
            sa = j.get("scores_a", {}).get(dim)
            sb = j.get("scores_b", {}).get(dim)
            if sa is not None and sb is not None:
                base_scores.append(sa)
                exp_scores.append(sb)

        if exp_scores:
            exp_mean = sum(exp_scores) / len(exp_scores)
            base_mean = sum(base_scores) / len(base_scores)
            delta = exp_mean - base_mean
            marker = " *" if abs(delta) > 0.3 else ""
            print(f"  {dim:<22} {exp_mean:>10.2f} {base_mean:>10.2f} {delta:>+8.2f}{marker}")

    print()


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 8: Yoga SFT Curriculum")
    parser.add_argument("--phase", choices=["small", "full"], default="small",
                        help="Phase: small (yoga vs base) or full (4-way comparison)")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument("--judge", action="store_true", help="Run pairwise judging")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku",
                        help="Judge model (default: haiku)")
    parser.add_argument("--model-size", choices=["0.5b", "1.5b", "3b"], default="1.5b",
                        help="Model size (default: 1.5b)")
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        help="Specific configs to run (default: phase-appropriate)")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Generation temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Generation top-p (default: 0.9)")
    args = parser.parse_args()

    global _gen_temp, _gen_top_p
    _gen_temp = args.temp
    _gen_top_p = args.top_p

    model_paths = get_model_paths(args.model_size, args.phase)
    exp_name = f"{EXPERIMENT_NAME}_{args.phase}"
    if args.model_size != "1.5b":
        exp_name = f"{exp_name}_{args.model_size}"

    # Load held-out questions
    questions = load_holdout_questions(args.phase)
    if args.limit:
        questions = questions[:args.limit]

    # Determine configs
    if args.configs:
        requested_configs = args.configs
    elif args.phase == "small":
        requested_configs = PHASE1_CONFIGS
    else:
        requested_configs = PHASE2_CONFIGS

    # Filter to available models
    available_configs = []
    for config in requested_configs:
        if config == "base" or model_paths.get(config) is not None:
            available_configs.append(config)

    if len(available_configs) < len(requested_configs):
        missing = set(requested_configs) - set(available_configs)
        print(f"WARNING: Models not yet trained: {missing}")
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

        # Per-dimension breakdown
        per_dimension_analysis(exp_name)

        # Print decision gate
        print("\n" + "=" * 60)
        if args.phase == "small":
            print("DECISION GATE (Phase 1):")
            print("  yoga_sft > base (>10% gap) → Proceed to Phase 2")
            print("  yoga_sft ≈ base            → Check data quality")
        else:
            print("DECISION GATE (Phase 2):")
            print("  yoga > random >10% gap, CI >50%  → PROVEN")
            print("  (yoga ≈ complexity) > random      → PARTIALLY PROVEN")
            print("  yoga ≈ random ≈ base              → DISPROVEN")
        print("=" * 60)


if __name__ == "__main__":
    main()
