#!/usr/bin/env python3
"""Exp 4: Vaisheshika — 7-Padartha Judge vs 5-Dimension Judge

LAYER: Evaluation / metrics
RESEARCH FRAMEWORK: Nyaya Pancha-avayava (5-Step Syllogism)

PRATIJNA (Thesis):
  Judging LLM responses using Vaisheshika's 7 padarthas provides better
  discrimination power than the current 5-dimension judge.

HETU (Reason):
  The 7 padarthas cover dimensions the current judge misses:
  - Karma (actionability): Can you DO something with this?
  - Samanya/Vishesha (general/particular): Explicit tension between principles and specifics
  - Abhava (absence): What's MISSING?
  7 dimensions > 5 dimensions should spread scores more, reducing ties.

UDAHARANA (Prior Evidence):
  - Phase 7: 180 existing generations with 5-dim judge scores
  - vaisheshika_judge.py exists with 7-padartha framework
  - Vaisheshika in vedic_llm: only used as regex counter — metrics DECOUPLED from quality
  - Gap: 7-padartha judge never tested, no comparison vs generic equivalent

UPANAYA (Experiment Design):
  Re-judge Phase 7's 180 existing generations with 3 judge frameworks:
  - 5dim_current: factual_accuracy, reasoning_depth, completeness, calibration, usefulness
  - 7padartha: dravya, guna, karma, samanya, vishesha, samavaya, abhava
  - generic_7dim: accuracy, reasoning, completeness, usefulness, specificity, coherence, gaps

  The generic_7dim is the KEY control — same # dimensions, similar concepts, but
  without Sanskrit/philosophical framing.

  Metrics: Discrimination (variance), Agreement with pairwise, Inter-dimension correlation

NIGAMANA (Success Criteria):
  - PROVEN: 7padartha has higher discrimination AND lower inter-dim correlation than BOTH
  - DISPROVEN: generic_7dim ≈ 7padartha (more dimensions help, Padarthas aren't special)
  - INCONCLUSIVE: 5dim still discriminates better (more isn't always better)

Cost: ~$0.50
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.vaisheshika_judge import judge_with_padarthas, judge_with_generic_7dim
from experiments.utils import (
    RESULTS_DIR, MODELS, DATA_DIR,
    get_client, call_api_json, load_jsonl, append_jsonl, load_existing_keys, mean,
)


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp4_vaisheshika"
PHASE7_DATA = DATA_DIR / "phase7_outputs" / "pipeline_results.jsonl"


def load_phase7_records():
    """Load Phase 7 pipeline results."""
    records = load_jsonl(PHASE7_DATA)
    if not records:
        print(f"ERROR: {PHASE7_DATA} not found or empty.")
        print("Copy from vedic_llm/eval_results_darshana/pipeline_results.jsonl")
        sys.exit(1)
    return records


# -- 5-Dimension Judge (current) -----------------------------------------------

CURRENT_5DIM_SYSTEM = """You are an expert evaluator. Score this response on 5 dimensions (1-5 each):

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


# -- Evaluation ----------------------------------------------------------------

def run_triple_judging(records, judge_model="haiku", limit=None):
    """Judge each response with ALL THREE judge frameworks."""
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
        key = f"triple|{rec['config']}|{rec['query'][:60]}"
        if key in existing:
            continue

        # 1. Judge with Vaisheshika 7-padartha framework
        padartha_scores = judge_with_padarthas(
            client, model, rec["query"], rec["response"]
        )

        # 2. Judge with current 5-dimension framework
        current_scores = call_api_json(
            client, model, CURRENT_5DIM_SYSTEM,
            f"QUESTION: {rec['query']}\n\nRESPONSE:\n{rec['response']}"
        )

        # 3. Judge with generic 7-dimension framework
        generic_7dim_scores = judge_with_generic_7dim(
            client, model, rec["query"], rec["response"]
        )

        result = {
            "key": key,
            "query": rec["query"],
            "config": rec["config"],
            "target_op": rec.get("target_op", ""),
            "domain": rec.get("domain", ""),
            "padartha_scores": padartha_scores,
            "current_scores": current_scores,
            "generic_7dim_scores": generic_7dim_scores,
        }

        append_jsonl(results_path, result)
        existing.add(key)
        new_count += 1

        if new_count % 20 == 0:
            elapsed = time.time() - t0
            print(f"    [{new_count} done] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Triple judging complete: {new_count} new in {elapsed:.0f}s")


def analyze_discrimination():
    """Compare discrimination power of all three judge frameworks."""
    results_dir = RESULTS_DIR / EXPERIMENT_NAME
    records = load_jsonl(results_dir / "results.jsonl")

    if not records:
        print("No results to analyze.")
        return

    print(f"\n{'='*60}")
    print("DISCRIMINATION ANALYSIS (3 Judge Frameworks)")
    print(f"{'='*60}")

    by_config = defaultdict(list)
    for r in records:
        by_config[r["config"]].append(r)

    # Dimension names for each framework
    padartha_dims = ["dravya", "guna", "karma", "samanya", "vishesha", "samavaya", "abhava"]
    current_dims = ["factual_accuracy", "reasoning_depth", "completeness", "calibration", "usefulness"]
    generic_dims = ["accuracy", "reasoning", "completeness", "usefulness", "specificity", "coherence", "gaps"]

    # Compute per-config means for each framework
    print(f"\n  {'Config':<22} | Padartha(7) | Current(5) | Generic(7) | P-var  | C-var  | G-var")
    print(f"  {'-'*95}")

    all_p_totals, all_c_totals, all_g_totals = [], [], []

    for cfg in sorted(by_config.keys()):
        recs = by_config[cfg]
        p_totals, c_totals, g_totals = [], [], []

        for r in recs:
            if r.get("padartha_scores"):
                p_totals.append(sum(r["padartha_scores"].get(d, 0) for d in padartha_dims))
            if r.get("current_scores"):
                c_totals.append(sum(r["current_scores"].get(d, 0) for d in current_dims))
            if r.get("generic_7dim_scores"):
                g_totals.append(sum(r["generic_7dim_scores"].get(d, 0) for d in generic_dims))

        p_mean = mean(p_totals) if p_totals else 0
        c_mean = mean(c_totals) if c_totals else 0
        g_mean = mean(g_totals) if g_totals else 0
        p_var = mean([(x - p_mean)**2 for x in p_totals]) if len(p_totals) > 1 else 0
        c_var = mean([(x - c_mean)**2 for x in c_totals]) if len(c_totals) > 1 else 0
        g_var = mean([(x - g_mean)**2 for x in g_totals]) if len(g_totals) > 1 else 0

        all_p_totals.extend(p_totals)
        all_c_totals.extend(c_totals)
        all_g_totals.extend(g_totals)

        print(f"  {cfg:<22} | {p_mean:>11.1f} | {c_mean:>10.1f} | {g_mean:>10.1f} | "
              f"{p_var:>6.1f} | {c_var:>6.1f} | {g_var:>6.1f}")

    # Overall variance
    if all_p_totals and all_c_totals and all_g_totals:
        p_m = mean(all_p_totals)
        c_m = mean(all_c_totals)
        g_m = mean(all_g_totals)
        p_v = mean([(x - p_m)**2 for x in all_p_totals])
        c_v = mean([(x - c_m)**2 for x in all_c_totals])
        g_v = mean([(x - g_m)**2 for x in all_g_totals])

        print(f"\n  Overall variance (higher = more discriminating):")
        print(f"    Padartha (7-dim):  {p_v:.1f}")
        print(f"    Current (5-dim):   {c_v:.1f}")
        print(f"    Generic (7-dim):   {g_v:.1f}")

        winner = max([("Padartha", p_v), ("Current", c_v), ("Generic", g_v)], key=lambda x: x[1])
        print(f"    Winner: {winner[0]}")

    # Inter-dimension correlation analysis
    print(f"\n  Inter-dimension correlation (lower = dimensions capture different things):")
    for name, dims, score_key in [
        ("Padartha", padartha_dims, "padartha_scores"),
        ("Generic", generic_dims, "generic_7dim_scores"),
        ("Current", current_dims, "current_scores"),
    ]:
        all_dim_scores = {d: [] for d in dims}
        for r in records:
            scores = r.get(score_key)
            if scores:
                for d in dims:
                    all_dim_scores[d].append(scores.get(d, 0))

        # Average pairwise correlation
        correlations = []
        for i, d1 in enumerate(dims):
            for d2 in dims[i+1:]:
                v1, v2 = all_dim_scores[d1], all_dim_scores[d2]
                if len(v1) > 2 and len(v2) > 2:
                    m1, m2 = mean(v1), mean(v2)
                    num = sum((a - m1) * (b - m2) for a, b in zip(v1, v2))
                    d1_sq = sum((a - m1)**2 for a in v1)
                    d2_sq = sum((b - m2)**2 for b in v2)
                    denom = (d1_sq * d2_sq) ** 0.5
                    if denom > 0:
                        correlations.append(abs(num / denom))

        avg_corr = mean(correlations) if correlations else 0
        print(f"    {name:<10} avg |r|: {avg_corr:.3f} ({len(dims)} dims, {len(correlations)} pairs)")

    # Config separation
    if len(by_config) > 1:
        print(f"\n  Config separation (spread between best and worst avg total):")
        for name, dims, score_key in [
            ("Padartha", padartha_dims, "padartha_scores"),
            ("Generic", generic_dims, "generic_7dim_scores"),
            ("Current", current_dims, "current_scores"),
        ]:
            cfg_means = {}
            for cfg, recs in by_config.items():
                vals = []
                for r in recs:
                    s = r.get(score_key)
                    if s:
                        vals.append(sum(s.get(d, 0) for d in dims))
                cfg_means[cfg] = mean(vals) if vals else 0
            spread = max(cfg_means.values()) - min(cfg_means.values())
            print(f"    {name:<10} spread: {spread:.1f}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 4: Vaisheshika 7-Padartha Judge")
    parser.add_argument("--limit", type=int, default=None, help="Limit records to judge")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--analyze-only", action="store_true", help="Skip judging, just analyze")
    args = parser.parse_args()

    if not args.analyze_only:
        records = load_phase7_records()
        print(f"Loaded {len(records)} Phase 7 records from {PHASE7_DATA}")
        run_triple_judging(records, judge_model=args.judge_model, limit=args.limit)

    analyze_discrimination()


if __name__ == "__main__":
    main()
