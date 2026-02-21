#!/usr/bin/env python3
"""Analyze Darshana Training Validation Results

Produces a comprehensive report covering:
  - Exp 6: Samkhya pretraining (samkhya vs bloom vs random)
  - Exp 7: Yoga DPO (yoga vs reverse vs complexity vs random)
  - Combined: darshana stack vs Western stack vs random
  - Generalization: train queries vs held-out test queries
  - Scale validation: 1.5B vs 3B consistency
  - Per-dimension analysis

Usage:
  python experiments/analyze_training.py
  python experiments/analyze_training.py --model-size 1.5b
  python experiments/analyze_training.py --model-size 3b
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import load_jsonl, RESULTS_DIR, DATA_DIR
from experiments.stats import (
    bootstrap_ci, format_win_rate, cohens_h, effect_size_label,
    pairwise_comparison_table, compute_verdict, is_significant,
)


# -- Config --------------------------------------------------------------------

# Configs grouped by experiment
EXP6_CONFIGS = ["samkhya_only", "bloom_only", "random_pt_only"]
EXP7_CONFIGS = ["yoga_only", "reverse_only", "complexity_only", "random_dpo_only"]
COMBINED_CONFIGS = ["samkhya_yoga", "bloom_complexity", "random_random"]

DIMENSION_NAMES = {
    "factual_accuracy": "Factual Accuracy",
    "reasoning_depth": "Reasoning Depth",
    "completeness": "Completeness",
    "calibration": "Calibration",
    "usefulness": "Usefulness",
}


# -- Analysis Functions --------------------------------------------------------

def load_judge_results(exp_name):
    """Load pairwise judge results for an experiment."""
    judge_path = RESULTS_DIR / exp_name / "judge.jsonl"
    if not judge_path.exists():
        return []
    records = load_jsonl(judge_path)
    return [r for r in records if r.get("type") == "pairwise"]


def compute_win_rates(judgments, baseline="base"):
    """Compute win rates for each config vs baseline.

    Returns:
        dict of config -> {"wins": N, "losses": N, "ties": N, "total": N}
    """
    results = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "total": 0})

    for j in judgments:
        config_a = j.get("config_a", "")
        config_b = j.get("config_b", "")
        winner = j.get("winner", "TIE")

        # We expect config_a = baseline, config_b = experimental
        if config_a == baseline:
            exp_config = config_b
        elif config_b == baseline:
            exp_config = config_a
        else:
            continue

        results[exp_config]["total"] += 1
        if winner == exp_config:
            results[exp_config]["wins"] += 1
        elif winner == baseline:
            results[exp_config]["losses"] += 1
        else:
            results[exp_config]["ties"] += 1

    return dict(results)


def compute_per_dimension_rates(judgments, config, baseline="base"):
    """Compute average score difference per dimension for a config vs baseline.

    Returns:
        dict of dimension -> avg_delta
    """
    deltas = defaultdict(list)

    for j in judgments:
        config_a = j.get("config_a", "")
        config_b = j.get("config_b", "")

        if config_a == baseline and config_b == config:
            scores_exp = j.get("scores_b", {})
            scores_base = j.get("scores_a", {})
        elif config_b == baseline and config_a == config:
            scores_exp = j.get("scores_a", {})
            scores_base = j.get("scores_b", {})
        else:
            continue

        for dim in DIMENSION_NAMES:
            s_exp = scores_exp.get(dim, 0)
            s_base = scores_base.get(dim, 0)
            if isinstance(s_exp, (int, float)) and isinstance(s_base, (int, float)):
                deltas[dim].append(s_exp - s_base)

    return {dim: sum(ds)/len(ds) if ds else 0 for dim, ds in deltas.items()}


def compute_generalization(judgments, test_queries, baseline="base"):
    """Split win rates into train vs test queries.

    Returns:
        dict of config -> {"train": {"wins": N, "total": N}, "test": {"wins": N, "total": N}}
    """
    test_set = set(test_queries) if test_queries else set()
    results = defaultdict(lambda: {
        "train": {"wins": 0, "total": 0},
        "test": {"wins": 0, "total": 0},
    })

    for j in judgments:
        config_a = j.get("config_a", "")
        config_b = j.get("config_b", "")
        winner = j.get("winner", "TIE")
        query = j.get("query", "")

        if config_a == baseline:
            exp_config = config_b
        elif config_b == baseline:
            exp_config = config_a
        else:
            continue

        split = "test" if query in test_set else "train"
        results[exp_config][split]["total"] += 1
        if winner == exp_config:
            results[exp_config][split]["wins"] += 1

    return dict(results)


# -- Report Generation ---------------------------------------------------------

def print_section(title, char="="):
    width = max(60, len(title) + 4)
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_win_rates(win_rates, configs, label=""):
    """Print win rates with CIs for a set of configs."""
    if label:
        print(f"\n  {label}")
    for config in configs:
        stats = win_rates.get(config, {})
        wins = stats.get("wins", 0)
        total = stats.get("total", 0)
        losses = stats.get("losses", 0)
        ties = stats.get("ties", 0)

        wr_str = format_win_rate(wins, total)
        h = cohens_h(wins / total, 0.5) if total > 0 else 0
        sig = "*" if is_significant(wins, total) else " "

        print(f"    {config:<25} vs base: {wr_str:<20} "
              f"(W:{wins} L:{losses} T:{ties}) h={h:+.2f} {sig}")


def generate_report(model_size="1.5b"):
    """Generate the full analysis report."""
    exp6_name = f"exp6_samkhya_{model_size}" if model_size != "1.5b" else "exp6_samkhya"
    exp7_name = f"exp7_yoga_{model_size}" if model_size != "1.5b" else "exp7_yoga"

    # Load judgments from both experiments
    # exp7 now contains all 11 configs (exp6 pretraining + exp7 DPO + combined)
    exp6_judgments = load_judge_results(exp6_name)
    exp7_judgments = load_judge_results(exp7_name)

    # Merge all judgments for combined analysis
    all_judgments = exp6_judgments + exp7_judgments

    if not all_judgments:
        print(f"No judge results found for model size {model_size}.")
        print(f"Run experiments/exp6_samkhya_pretraining.py and/or experiments/exp7_yoga_dpo.py first.")
        return

    # Compute win rates
    all_win_rates = compute_win_rates(all_judgments)

    # Load test queries for generalization analysis
    test_queries_path = DATA_DIR / "dpo_combined" / "test_queries.json"
    test_queries = []
    if test_queries_path.exists():
        with open(test_queries_path) as f:
            test_queries = json.load(f)

    # =========================================================================
    # REPORT
    # =========================================================================

    print_section(f"DARSHANA TRAINING VALIDATION RESULTS ({model_size.upper()})")

    # -- Exp 6: Samkhya Pretraining --
    print_section("EXP 6: SAMKHYA PRETRAINING", "-")
    exp6_rates = {c: all_win_rates[c] for c in EXP6_CONFIGS if c in all_win_rates}
    if exp6_rates:
        print_win_rates(all_win_rates, EXP6_CONFIGS)
        verdict, explanation = compute_verdict(
            exp6_rates, "samkhya_only", "bloom_only", "random_pt_only"
        )
        print(f"\n  Verdict: {verdict}")
        print(f"  {explanation}")
    else:
        print("  No pretraining results available.")

    # -- Exp 7: Yoga DPO --
    print_section("EXP 7: YOGA DPO", "-")
    exp7_rates = {c: all_win_rates[c] for c in EXP7_CONFIGS if c in all_win_rates}
    if exp7_rates:
        print_win_rates(all_win_rates, EXP7_CONFIGS)
        verdict, explanation = compute_verdict(
            exp7_rates, "yoga_only", "complexity_only", "random_dpo_only"
        )
        print(f"\n  Verdict: {verdict}")
        print(f"  {explanation}")
    else:
        print("  No DPO results available.")

    # -- Combined: Darshana Stack --
    print_section("COMBINED: DARSHANA STACK", "-")
    combined_rates = {c: all_win_rates[c] for c in COMBINED_CONFIGS if c in all_win_rates}
    if combined_rates:
        print_win_rates(all_win_rates, COMBINED_CONFIGS)
        verdict, explanation = compute_verdict(
            combined_rates, "samkhya_yoga", "bloom_complexity", "random_random"
        )
        print(f"\n  Verdict: {verdict}")
        print(f"  {explanation}")

        # Check compounding effect
        samkhya_wr = all_win_rates.get("samkhya_only", {})
        yoga_wr = all_win_rates.get("yoga_only", {})
        combined_wr = all_win_rates.get("samkhya_yoga", {})

        s_rate = samkhya_wr.get("wins", 0) / samkhya_wr.get("total", 1) if samkhya_wr.get("total", 0) > 0 else 0
        y_rate = yoga_wr.get("wins", 0) / yoga_wr.get("total", 1) if yoga_wr.get("total", 0) > 0 else 0
        c_rate = combined_wr.get("wins", 0) / combined_wr.get("total", 1) if combined_wr.get("total", 0) > 0 else 0

        if s_rate > 0 and y_rate > 0 and c_rate > 0:
            expected_additive = (s_rate + y_rate) / 2
            print(f"\n  Compounding analysis:")
            print(f"    Samkhya alone: {s_rate*100:.0f}%")
            print(f"    Yoga alone:    {y_rate*100:.0f}%")
            print(f"    Combined:      {c_rate*100:.0f}%")
            print(f"    Expected (avg): {expected_additive*100:.0f}%")
            if c_rate > expected_additive + 0.05:
                print(f"    Compounding: YES (combined > average of individual)")
            elif c_rate > max(s_rate, y_rate):
                print(f"    Compounding: WEAK (combined > either individual)")
            else:
                print(f"    Compounding: NO")
    else:
        print("  No combined results available.")

    # -- Generalization --
    if test_queries:
        print_section("GENERALIZATION (Train vs Test Queries)", "-")
        gen_results = compute_generalization(all_judgments, test_queries)
        all_configs = EXP6_CONFIGS + EXP7_CONFIGS + COMBINED_CONFIGS
        available = [c for c in all_configs if c in gen_results]

        if available:
            print(f"\n  {'Config':<25} {'Train WR':>10} {'Test WR':>10} {'Gap':>8}")
            print(f"  {'-'*55}")
            for config in available:
                train = gen_results[config]["train"]
                test = gen_results[config]["test"]
                tr = train["wins"] / train["total"] if train["total"] > 0 else 0
                te = test["wins"] / test["total"] if test["total"] > 0 else 0
                gap = tr - te
                print(f"  {config:<25} {tr*100:>6.0f}%    {te*100:>6.0f}%    {gap*100:>+5.0f}%")

            # Overall generalization verdict
            gaps = []
            for c in available:
                train = gen_results[c]["train"]
                test = gen_results[c]["test"]
                if train["total"] > 0 and test["total"] > 0:
                    gaps.append(
                        train["wins"]/train["total"] - test["wins"]/test["total"]
                    )
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                if avg_gap > 0.15:
                    print(f"\n  Generalization: POOR (avg gap {avg_gap*100:.0f}%, likely overfitting)")
                elif avg_gap > 0.05:
                    print(f"\n  Generalization: MODERATE (avg gap {avg_gap*100:.0f}%)")
                else:
                    print(f"\n  Generalization: GOOD (avg gap {avg_gap*100:.0f}%)")

    # -- Per-Dimension Analysis --
    print_section("PER-DIMENSION ANALYSIS", "-")
    key_configs = ["samkhya_yoga", "yoga_only", "bloom_complexity"]
    available_key = [c for c in key_configs if c in all_win_rates]

    if available_key:
        print(f"\n  Avg score delta vs base (positive = better than base):")
        print(f"\n  {'Config':<20}", end="")
        for dim, name in DIMENSION_NAMES.items():
            print(f" {name[:8]:>9}", end="")
        print()
        print(f"  {'-'*70}")

        for config in available_key:
            dim_deltas = compute_per_dimension_rates(all_judgments, config)
            print(f"  {config:<20}", end="")
            for dim in DIMENSION_NAMES:
                d = dim_deltas.get(dim, 0)
                print(f" {d:>+8.2f}", end="")
            print()

    # -- Full Comparison Table --
    print_section("FULL COMPARISON TABLE", "-")
    all_configs = EXP6_CONFIGS + EXP7_CONFIGS + COMBINED_CONFIGS
    available_all = {c: all_win_rates[c] for c in all_configs if c in all_win_rates}
    if available_all:
        print()
        print(pairwise_comparison_table(available_all))

    print(f"\n{'='*60}")
    print(f"  * = significantly above 50% (95% CI lower bound > 50%)")
    print(f"  h = Cohen's h effect size vs 50% baseline")
    print(f"{'='*60}\n")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze Darshana Training Results")
    parser.add_argument("--model-size", type=str, default="1.5b",
                        choices=["0.5b", "1.5b", "3b"],
                        help="Model size to analyze (default: 1.5b)")
    parser.add_argument("--compare-scales", action="store_true",
                        help="Compare results across model sizes")
    args = parser.parse_args()

    if args.compare_scales:
        for size in ["1.5b", "3b"]:
            generate_report(size)
    else:
        generate_report(args.model_size)


if __name__ == "__main__":
    main()
