#!/usr/bin/env python3
"""Extract Real DPO Pairs from Judged Experiment Results

Extracts preference pairs from:
  - darshana_llm experiments 1, 3, 5 (Sonnet + cross-model)
  - vedic_llm eval_results_darshana (pipeline judge)
  - vedic_llm eval_results_dharmic (dharmic principles judge)

Each pair has a winner (preferred) and loser (rejected) determined by blind
pairwise judging, making these stronger training signal than synthetic pairs.

Dimension -> Yoga Stage mapping:
  factual_accuracy  -> Stage 1 (Yama: honesty)
  completeness      -> Stage 2 (Asana: structure)
  calibration       -> Stage 3 (Pratyahara: restraint)
  reasoning_depth   -> Stage 4 (Dharana: deep analysis)
  usefulness        -> Stage 5 (Dhyana: synthesis)
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import load_jsonl

# -- Config --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = PROJECT_ROOT / "data" / "dpo_real_pairs.jsonl"

VEDIC_LLM_ROOT = Path("/Users/rishimanglesh/Projects/vedic_llm")

DIMENSION_TO_STAGE = {
    "factual_accuracy": 1,
    "completeness": 2,
    "calibration": 3,
    "reasoning_depth": 4,
    "usefulness": 5,
}

STAGE_NAMES = {
    1: "Yama (honesty)",
    2: "Asana (structure)",
    3: "Pratyahara (restraint)",
    4: "Dharana (deep analysis)",
    5: "Dhyana (synthesis)",
}

# Each source: (experiment_name, results_path, judge_path, source_model)
DATA_SOURCES = [
    # darshana_llm Sonnet experiments
    (
        "exp1_vritti",
        RESULTS_DIR / "exp1_vritti" / "results.jsonl",
        RESULTS_DIR / "exp1_vritti" / "judge.jsonl",
        "sonnet",
    ),
    (
        "exp3_mimamsa",
        RESULTS_DIR / "exp3_mimamsa" / "results.jsonl",
        RESULTS_DIR / "exp3_mimamsa" / "judge.jsonl",
        "sonnet",
    ),
    (
        "exp5_vedanta",
        RESULTS_DIR / "exp5_vedanta" / "results.jsonl",
        RESULTS_DIR / "exp5_vedanta" / "judge.jsonl",
        "sonnet",
    ),
    # Cross-model: exp1 vritti
    (
        "exp1_vritti",
        RESULTS_DIR / "cross_model" / "exp1_vritti" / "qwen3_8b_results.jsonl",
        RESULTS_DIR / "cross_model" / "exp1_vritti" / "judge_qwen3_8b.jsonl",
        "qwen3_8b",
    ),
    (
        "exp1_vritti",
        RESULTS_DIR / "cross_model" / "exp1_vritti" / "qwen3_32b_results.jsonl",
        RESULTS_DIR / "cross_model" / "exp1_vritti" / "judge_qwen3_32b.jsonl",
        "qwen3_32b",
    ),
    # Cross-model: exp3 mimamsa
    (
        "exp3_mimamsa",
        RESULTS_DIR / "cross_model" / "exp3_mimamsa" / "qwen3_8b_results.jsonl",
        RESULTS_DIR / "cross_model" / "exp3_mimamsa" / "judge_qwen3_8b.jsonl",
        "qwen3_8b",
    ),
    # Cross-model: exp5 vedanta
    (
        "exp5_vedanta",
        RESULTS_DIR / "cross_model" / "exp5_vedanta" / "qwen3_8b_results.jsonl",
        RESULTS_DIR / "cross_model" / "exp5_vedanta" / "judge_qwen3_8b.jsonl",
        "qwen3_8b",
    ),
    (
        "exp5_vedanta",
        RESULTS_DIR / "cross_model" / "exp5_vedanta" / "qwen3_32b_results.jsonl",
        RESULTS_DIR / "cross_model" / "exp5_vedanta" / "judge_qwen3_32b.jsonl",
        "qwen3_32b",
    ),
    # vedic_llm: darshana pipeline results
    (
        "vedic_darshana",
        VEDIC_LLM_ROOT / "eval_results_darshana" / "pipeline_results.jsonl",
        VEDIC_LLM_ROOT / "eval_results_darshana" / "darshana_judge.jsonl",
        "sonnet_vedic",
    ),
    # vedic_llm: dharmic principles results
    (
        "vedic_dharmic",
        VEDIC_LLM_ROOT / "eval_results_dharmic" / "dharmic_principles.jsonl",
        VEDIC_LLM_ROOT / "eval_results_dharmic" / "dharmic_judge.jsonl",
        "sonnet_vedic",
    ),
]


# -- Extraction ----------------------------------------------------------------

def compute_dominant_stage(winner_scores, loser_scores):
    """Find the dimension with the largest score delta and map to Yoga stage.

    Args:
        winner_scores: dict of {dimension: score} for the winner
        loser_scores: dict of {dimension: score} for the loser

    Returns:
        (stage_number, dimension_name, delta)
    """
    best_dim = None
    best_delta = -1

    for dim, stage in DIMENSION_TO_STAGE.items():
        w = winner_scores.get(dim, 0)
        l = loser_scores.get(dim, 0)
        if not isinstance(w, (int, float)):
            continue
        if not isinstance(l, (int, float)):
            continue
        delta = w - l
        if delta > best_delta:
            best_delta = delta
            best_dim = dim

    if best_dim is None:
        return 3, "calibration", 0  # default to middle stage

    return DIMENSION_TO_STAGE[best_dim], best_dim, best_delta


def compute_total_delta(winner_scores, loser_scores):
    """Compute total score delta across all dimensions."""
    total = 0
    for dim in DIMENSION_TO_STAGE:
        w = winner_scores.get(dim, 0)
        l = loser_scores.get(dim, 0)
        if isinstance(w, (int, float)) and isinstance(l, (int, float)):
            total += w - l
    return total


def extract_pairs_from_source(exp_name, results_path, judge_path, source_model, min_delta=0):
    """Extract DPO pairs from a single source.

    Returns:
        list of pair dicts
    """
    if not results_path.exists():
        print(f"  [SKIP] Results not found: {results_path}")
        return []
    if not judge_path.exists():
        print(f"  [SKIP] Judge not found: {judge_path}")
        return []

    results = load_jsonl(results_path)
    judgments = load_jsonl(judge_path)

    # Index results by (config, query_prefix) for lookup
    response_index = {}
    for r in results:
        key = (r.get("config", ""), r.get("query", "")[:60])
        response_index[key] = r

    pairs = []
    skipped_ties = 0
    skipped_missing = 0
    skipped_weak = 0

    for j in judgments:
        if j.get("type") != "pairwise":
            continue

        winner = j.get("winner", "TIE")
        if winner == "TIE":
            skipped_ties += 1
            continue

        config_a = j.get("config_a", "")
        config_b = j.get("config_b", "")
        query = j.get("query", "")
        query_prefix = query[:60]

        # Determine winner and loser configs
        if winner == config_a:
            winner_config = config_a
            loser_config = config_b
        elif winner == config_b:
            winner_config = config_b
            loser_config = config_a
        else:
            # Winner doesn't match either config (shouldn't happen)
            continue

        # Look up responses
        winner_rec = response_index.get((winner_config, query_prefix))
        loser_rec = response_index.get((loser_config, query_prefix))

        if not winner_rec or not loser_rec:
            skipped_missing += 1
            continue

        winner_response = winner_rec.get("response", "")
        loser_response = loser_rec.get("response", "")

        if not winner_response or not loser_response:
            skipped_missing += 1
            continue

        # Get scores for winner and loser
        # scores_a always corresponds to config_a, scores_b to config_b
        scores_a = j.get("scores_a", {})
        scores_b = j.get("scores_b", {})

        if winner == config_a:
            winner_scores = scores_a
            loser_scores = scores_b
        else:
            winner_scores = scores_b
            loser_scores = scores_a

        # Apply min_delta filter
        total_delta = compute_total_delta(winner_scores, loser_scores)
        if total_delta < min_delta:
            skipped_weak += 1
            continue

        stage, dominant_dim, delta = compute_dominant_stage(winner_scores, loser_scores)

        pairs.append({
            "query": query,
            "preferred": winner_response,
            "rejected": loser_response,
            "stage": stage,
            "dominant_dimension": dominant_dim,
            "score_delta": delta,
            "total_delta": total_delta,
            "source": exp_name,
            "source_model": source_model,
            "winner_config": winner_config,
            "loser_config": loser_config,
        })

    if skipped_ties or skipped_missing or skipped_weak:
        print(f"  Skipped: {skipped_ties} ties, {skipped_missing} missing, {skipped_weak} weak signal")

    return pairs


def extract_all(sources=None, min_delta=0):
    """Extract pairs from all or selected sources.

    Args:
        sources: "sonnet", "cross_model", "vedic", or None for all
        min_delta: minimum total score delta to include

    Returns:
        list of all pairs
    """
    all_pairs = []

    for exp_name, results_path, judge_path, source_model in DATA_SOURCES:
        # Filter by source type
        if sources == "sonnet" and source_model != "sonnet":
            continue
        if sources == "cross_model" and source_model == "sonnet":
            continue
        if sources == "vedic" and not source_model.startswith("sonnet_vedic"):
            continue

        label = f"{exp_name} ({source_model})"
        print(f"\n  Extracting: {label}")

        pairs = extract_pairs_from_source(
            exp_name, results_path, judge_path, source_model, min_delta=min_delta
        )
        print(f"  Extracted: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    return all_pairs


def write_pairs(pairs, output_path=None):
    """Write pairs to JSONL file."""
    output_path = Path(output_path or OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  Written: {len(pairs)} pairs to {output_path}")


def print_stats(pairs):
    """Print detailed statistics about extracted pairs."""
    print(f"\n{'='*60}")
    print(f"DPO PAIR STATISTICS")
    print(f"{'='*60}")
    print(f"\n  Total pairs: {len(pairs)}")

    # By source experiment
    by_source = Counter(p["source"] for p in pairs)
    print(f"\n  By experiment:")
    for src, count in sorted(by_source.items()):
        print(f"    {src}: {count}")

    # By source model
    by_model = Counter(p["source_model"] for p in pairs)
    print(f"\n  By source model:")
    for model, count in sorted(by_model.items()):
        print(f"    {model}: {count}")

    # By stage
    by_stage = Counter(p["stage"] for p in pairs)
    print(f"\n  By Yoga stage:")
    for stage in sorted(by_stage.keys()):
        name = STAGE_NAMES.get(stage, f"Stage {stage}")
        print(f"    Stage {stage} ({name}): {by_stage[stage]}")

    # By dominant dimension
    by_dim = Counter(p["dominant_dimension"] for p in pairs)
    print(f"\n  By dominant dimension:")
    for dim, count in sorted(by_dim.items(), key=lambda x: -x[1]):
        print(f"    {dim}: {count}")

    # By winner config
    by_winner = Counter(p["winner_config"] for p in pairs)
    print(f"\n  By winner config:")
    for config, count in sorted(by_winner.items(), key=lambda x: -x[1]):
        print(f"    {config}: {count}")

    # Score delta distribution
    deltas = [p["score_delta"] for p in pairs]
    if deltas:
        avg_delta = sum(deltas) / len(deltas)
        print(f"\n  Score delta (dominant dim): avg={avg_delta:.2f}, "
              f"min={min(deltas)}, max={max(deltas)}")

    total_deltas = [p.get("total_delta", 0) for p in pairs]
    if total_deltas:
        avg_td = sum(total_deltas) / len(total_deltas)
        print(f"  Total delta (all dims):    avg={avg_td:.2f}, "
              f"min={min(total_deltas)}, max={max(total_deltas)}")

    # Stage x source_model cross-tab
    print(f"\n  Stage x Source Model:")
    models = sorted(set(p["source_model"] for p in pairs))
    header = f"    {'Stage':<30}" + "".join(f"{m:>12}" for m in models) + f"{'Total':>12}"
    print(header)
    print(f"    {'-'*(len(header)-4)}")
    cross = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        cross[p["stage"]][p["source_model"]] += 1
    for stage in sorted(cross.keys()):
        name = STAGE_NAMES.get(stage, f"Stage {stage}")
        row = f"    {stage}. {name:<27}"
        total = 0
        for m in models:
            c = cross[stage][m]
            total += c
            row += f"{c:>12}"
        row += f"{total:>12}"
        print(row)

    print(f"\n{'='*60}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract real DPO pairs from judged experiment results"
    )
    parser.add_argument(
        "--sources",
        choices=["sonnet", "cross_model", "vedic"],
        default=None,
        help="Filter by source type (default: all)",
    )
    parser.add_argument(
        "--min-delta",
        type=int,
        default=0,
        help="Minimum total score delta to include a pair (default: 0)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics (extract if needed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_PATH

    # If --stats and file exists, just print stats
    if args.stats and output_path.exists() and not args.sources and args.min_delta == 0:
        pairs = load_jsonl(output_path)
        if pairs:
            print_stats(pairs)
            return

    print(f"\nExtracting real DPO pairs...")
    if args.min_delta > 0:
        print(f"  Filtering: min_delta >= {args.min_delta}")
    pairs = extract_all(sources=args.sources, min_delta=args.min_delta)

    if not pairs:
        print("\nNo pairs extracted. Check that experiment results exist.")
        sys.exit(1)

    write_pairs(pairs, output_path)
    print_stats(pairs)


if __name__ == "__main__":
    main()
