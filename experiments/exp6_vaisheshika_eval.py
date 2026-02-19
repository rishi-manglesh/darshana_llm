#!/usr/bin/env python3
"""Exp 6: Vaisheshika — 7-Padartha Evaluation

LAYER: Evaluation / metrics
VALIDATION: Does padartha-based judging discriminate better than the
current 5-dimension judge?

Method:
  Re-judge Phase 7's 180 existing generations with BOTH judge prompts:
  - Current: factual_accuracy, reasoning_depth, completeness, calibration, usefulness
  - Vaisheshika: dravya, guna, karma, samanya, vishesha, samavaya, abhava

Metric: Higher discrimination power (score variance, separation between configs)
Cost: ~$0.30 | Success: Vaisheshika judge produces wider score spread
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.vaisheshika_judge import judge_with_padarthas
from experiments.utils import (
    RESULTS_DIR, MODELS, DATA_DIR,
    get_client, load_jsonl, append_jsonl, load_existing_keys, mean,
)
from experiments.judge import PAIRWISE_SYSTEM, call_api_json


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp6_vaisheshika"
PHASE7_DATA = DATA_DIR / "phase7_outputs" / "pipeline_results.jsonl"


def load_phase7_records():
    """Load Phase 7 pipeline results."""
    records = load_jsonl(PHASE7_DATA)
    if not records:
        print(f"ERROR: {PHASE7_DATA} not found or empty.")
        print("Copy from vedic_llm/eval_results_darshana/pipeline_results.jsonl")
        sys.exit(1)
    return records


# -- Evaluation ----------------------------------------------------------------

def run_dual_judging(records, judge_model="haiku", limit=None):
    """Judge each response with BOTH the current and Vaisheshika frameworks."""
    client = get_client()
    model = MODELS[judge_model]

    results_dir = RESULTS_DIR / EXPERIMENT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"
    existing = load_existing_keys(results_path)

    if limit:
        records = records[:limit]

    print(f"  Records to judge: {len(records)}")
    print(f"  Existing: {len(existing)} (will skip)")

    t0 = time.time()
    new_count = 0

    for i, rec in enumerate(records):
        key = f"dual|{rec['config']}|{rec['query'][:60]}"
        if key in existing:
            continue

        # Judge with Vaisheshika 7-padartha framework
        padartha_scores = judge_with_padarthas(
            client, model, rec["query"], rec["response"]
        )

        # Judge with current 5-dimension framework (single response scoring)
        current_system = """You are an expert evaluator. Score this response on 5 dimensions (1-5 each):

1. FACTUAL_ACCURACY: 1=major errors, 3=mostly correct, 5=fully accurate
2. REASONING_DEPTH: 1=surface only, 3=some explanation, 5=genuine insight
3. COMPLETENESS: 1=misses most, 3=covers basics, 5=comprehensive
4. CALIBRATION: 1=overconfident, 3=some hedging, 5=accurate certainty signals
5. USEFULNESS: 1=not helpful, 3=decent, 5=significantly improves understanding

Respond with ONLY a JSON object:
{
  "factual_accuracy": <1-5>,
  "reasoning_depth": <1-5>,
  "completeness": <1-5>,
  "calibration": <1-5>,
  "usefulness": <1-5>,
  "total": <sum>
}"""

        current_scores = call_api_json(
            client, model, current_system,
            f"QUESTION: {rec['query']}\n\nRESPONSE:\n{rec['response']}"
        )

        result = {
            "key": key,
            "query": rec["query"],
            "config": rec["config"],
            "target_op": rec.get("target_op", ""),
            "domain": rec.get("domain", ""),
            "padartha_scores": padartha_scores,
            "current_scores": current_scores,
        }

        append_jsonl(results_path, result)
        existing.add(key)
        new_count += 1

        if new_count % 20 == 0:
            elapsed = time.time() - t0
            print(f"    [{new_count} done] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Dual judging complete: {new_count} new in {elapsed:.0f}s")


def analyze_discrimination():
    """Compare discrimination power of both judge frameworks."""
    results_dir = RESULTS_DIR / EXPERIMENT_NAME
    records = load_jsonl(results_dir / "results.jsonl")

    if not records:
        print("No results to analyze.")
        return

    print(f"\n{'='*60}")
    print("DISCRIMINATION ANALYSIS")
    print(f"{'='*60}")

    # Score variance per config (higher = more discriminating)
    by_config = defaultdict(list)
    for r in records:
        by_config[r["config"]].append(r)

    # Padartha dimensions
    padartha_dims = ["dravya", "guna", "karma", "samanya", "vishesha", "samavaya", "abhava"]
    current_dims = ["factual_accuracy", "reasoning_depth", "completeness", "calibration", "usefulness"]

    # Compute per-config means for each framework
    print(f"\n  {'Config':<22} | Padartha total | Current total | P-variance | C-variance")
    print(f"  {'-'*80}")

    all_padartha_totals = []
    all_current_totals = []

    for cfg in sorted(by_config.keys()):
        recs = by_config[cfg]

        p_totals = []
        c_totals = []
        for r in recs:
            if r.get("padartha_scores"):
                pt = sum(r["padartha_scores"].get(d, 0) for d in padartha_dims)
                p_totals.append(pt)
            if r.get("current_scores"):
                ct = sum(r["current_scores"].get(d, 0) for d in current_dims)
                c_totals.append(ct)

        p_mean = mean(p_totals) if p_totals else 0
        c_mean = mean(c_totals) if c_totals else 0
        p_var = mean([(x - p_mean)**2 for x in p_totals]) if len(p_totals) > 1 else 0
        c_var = mean([(x - c_mean)**2 for x in c_totals]) if len(c_totals) > 1 else 0

        all_padartha_totals.extend(p_totals)
        all_current_totals.extend(c_totals)

        print(f"  {cfg:<22} | {p_mean:>14.1f} | {c_mean:>13.1f} | {p_var:>10.1f} | {c_var:>10.1f}")

    # Overall variance (across ALL records)
    if all_padartha_totals and all_current_totals:
        p_overall_mean = mean(all_padartha_totals)
        c_overall_mean = mean(all_current_totals)
        p_overall_var = mean([(x - p_overall_mean)**2 for x in all_padartha_totals])
        c_overall_var = mean([(x - c_overall_mean)**2 for x in all_current_totals])

        print(f"\n  Overall variance (higher = more discriminating):")
        print(f"    Padartha (7-dim):  {p_overall_var:.1f}")
        print(f"    Current (5-dim):   {c_overall_var:.1f}")
        print(f"    Winner: {'Padartha' if p_overall_var > c_overall_var else 'Current'}")

    # Config separation (distance between best and worst config)
    if len(by_config) > 1:
        config_p_means = {}
        config_c_means = {}
        for cfg, recs in by_config.items():
            p_vals = []
            c_vals = []
            for r in recs:
                if r.get("padartha_scores"):
                    p_vals.append(sum(r["padartha_scores"].get(d, 0) for d in padartha_dims))
                if r.get("current_scores"):
                    c_vals.append(sum(r["current_scores"].get(d, 0) for d in current_dims))
            config_p_means[cfg] = mean(p_vals) if p_vals else 0
            config_c_means[cfg] = mean(c_vals) if c_vals else 0

        p_spread = max(config_p_means.values()) - min(config_p_means.values())
        c_spread = max(config_c_means.values()) - min(config_c_means.values())

        print(f"\n  Config separation (spread between best and worst):")
        print(f"    Padartha spread: {p_spread:.1f}")
        print(f"    Current spread:  {c_spread:.1f}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 6: Vaisheshika 7-Padartha Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit records to judge")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--analyze-only", action="store_true", help="Skip judging, just analyze")
    args = parser.parse_args()

    if not args.analyze_only:
        records = load_phase7_records()
        print(f"Loaded {len(records)} Phase 7 records from {PHASE7_DATA}")
        run_dual_judging(records, judge_model=args.judge_model, limit=args.limit)

    analyze_discrimination()


if __name__ == "__main__":
    main()
