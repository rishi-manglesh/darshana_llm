#!/usr/bin/env python3
"""Generate Extended DPO Pairs from 45 New Questions

Generates responses to 45 extended questions using Sonnet with 3 configs:
  - bare_baseline (no system prompt)
  - vritti_contemporary (exp1's best darshana config)
  - generic_confidence (exp1's best generic config)

Then judges pairwise (baseline vs each experimental) using Sonnet.
Output: ~90 new judged pairs across 45 diverse queries.

Cost: ~$3-4 (135 generation calls + 90 judge calls)

Usage:
  python training/generate_extended_pairs.py
  python training/generate_extended_pairs.py --generate-only
  python training/generate_extended_pairs.py --judge-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.vritti import VRITTI_CONTEMPORARY_PROMPT, GENERIC_CONFIDENCE_PROMPT
from experiments.utils import (
    get_extended_questions, MODELS, get_client, call_api,
    load_jsonl, append_jsonl, load_existing_keys,
)
from experiments.judge import judge_pairwise

# -- Config --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "extended_pairs"
RESULTS_PATH = OUTPUT_DIR / "results.jsonl"
JUDGE_PATH = OUTPUT_DIR / "judge.jsonl"
PAIRS_PATH = OUTPUT_DIR / "pairs.jsonl"

MODEL = "sonnet"

SYSTEM_PROMPTS = {
    "bare_baseline": None,
    "vritti_contemporary": VRITTI_CONTEMPORARY_PROMPT,
    "generic_confidence": GENERIC_CONFIDENCE_PROMPT,
}

CONFIGS = list(SYSTEM_PROMPTS.keys())

DIMENSION_TO_STAGE = {
    "factual_accuracy": 1,
    "completeness": 2,
    "calibration": 3,
    "reasoning_depth": 4,
    "usefulness": 5,
}


# -- Generation ----------------------------------------------------------------

def generate_responses(client, questions, limit=None):
    """Generate responses for all configs x questions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing_keys(RESULTS_PATH)

    if limit:
        questions = questions[:limit]

    total = len(questions) * len(CONFIGS)
    done = len(existing)
    t0 = time.time()

    print(f"\nGenerating responses: {len(questions)} questions x {len(CONFIGS)} configs = {total}")
    print(f"  Existing: {len(existing)} (will skip)")

    model_id = MODELS[MODEL]

    for qi, q in enumerate(questions):
        for config in CONFIGS:
            key = f"{config}|{q['query'][:60]}"
            if key in existing:
                continue

            done += 1
            elapsed = time.time() - t0
            rate = elapsed / (done - len(existing)) if done > len(existing) else 0
            remaining = rate * (total - done)
            print(f"  [{done}/{total}] {config:<25} | {q.get('domain', ''):<15} | "
                  f"~{remaining:.0f}s left", flush=True)

            system = SYSTEM_PROMPTS[config] or ""
            response = call_api(client, model_id, system, q["query"], max_tokens=1024)

            if response is None:
                print(f"    [WARN] API returned None, skipping")
                continue

            record = {
                "key": key,
                "query": q["query"],
                "target_op": q.get("target_op", ""),
                "domain": q.get("domain", ""),
                "config": config,
                "response": response,
                "word_count": len(response.split()),
            }
            append_jsonl(RESULTS_PATH, record)
            existing.add(key)

    elapsed = time.time() - t0
    print(f"\n  Generation complete: {done - len(existing)} new in {elapsed:.0f}s")


# -- Judging -------------------------------------------------------------------

def run_judging(client, questions, limit=None):
    """Judge baseline vs each experimental config."""
    existing = load_existing_keys(JUDGE_PATH)
    results = load_jsonl(RESULTS_PATH)

    if limit:
        questions = questions[:limit]

    # Index by (config, query_prefix)
    by_query = {}
    for r in results:
        qp = r["query"][:60]
        if qp not in by_query:
            by_query[qp] = {}
        by_query[qp][r["config"]] = r

    model_id = MODELS[MODEL]
    experimental = [c for c in CONFIGS if c != "bare_baseline"]

    pairs_to_judge = []
    for q in questions:
        qp = q["query"][:60]
        if qp not in by_query or "bare_baseline" not in by_query[qp]:
            continue
        for exp_config in experimental:
            if exp_config in by_query[qp]:
                pairs_to_judge.append((q, qp, exp_config))

    print(f"\n  Judging: {len(pairs_to_judge)} pairs ({len(existing)} existing)")
    t0 = time.time()
    new_count = 0

    for q, qp, exp_config in pairs_to_judge:
        key = f"pairwise|{qp}|bare_baseline|{exp_config}"
        if key in existing:
            continue

        resp_base = by_query[qp]["bare_baseline"]["response"]
        resp_exp = by_query[qp][exp_config]["response"]

        result = judge_pairwise(
            client, model_id, q["query"],
            resp_base, resp_exp,
            "bare_baseline", exp_config,
        )

        if result is None:
            continue

        result.update({
            "key": key,
            "type": "pairwise",
            "query": q["query"],
            "config_a": "bare_baseline",
            "config_b": exp_config,
            "target_op": q.get("target_op", ""),
            "domain": q.get("domain", ""),
        })

        append_jsonl(JUDGE_PATH, result)
        existing.add(key)
        new_count += 1

        if new_count % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{new_count} done] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Judging complete: {new_count} new in {elapsed:.0f}s")


# -- Extract Pairs -------------------------------------------------------------

def extract_pairs():
    """Convert judge results to DPO pairs."""
    judgments = load_jsonl(JUDGE_PATH)
    results = load_jsonl(RESULTS_PATH)

    # Index responses
    response_index = {}
    for r in results:
        key = (r["config"], r["query"][:60])
        response_index[key] = r

    pairs = []
    for j in judgments:
        if j.get("type") != "pairwise":
            continue

        winner = j.get("winner", "TIE")
        if winner == "TIE":
            continue

        config_a = j.get("config_a", "")
        config_b = j.get("config_b", "")
        query = j.get("query", "")

        if winner == config_a:
            winner_config, loser_config = config_a, config_b
        elif winner == config_b:
            winner_config, loser_config = config_b, config_a
        else:
            continue

        winner_rec = response_index.get((winner_config, query[:60]))
        loser_rec = response_index.get((loser_config, query[:60]))

        if not winner_rec or not loser_rec:
            continue

        winner_response = winner_rec.get("response", "")
        loser_response = loser_rec.get("response", "")

        if not winner_response or not loser_response:
            continue

        scores_a = j.get("scores_a", {})
        scores_b = j.get("scores_b", {})

        if winner == config_a:
            winner_scores, loser_scores = scores_a, scores_b
        else:
            winner_scores, loser_scores = scores_b, scores_a

        # Compute dominant stage
        best_dim, best_delta = None, -1
        for dim, stage in DIMENSION_TO_STAGE.items():
            w = winner_scores.get(dim, 0)
            l = loser_scores.get(dim, 0)
            if isinstance(w, (int, float)) and isinstance(l, (int, float)):
                d = w - l
                if d > best_delta:
                    best_delta = d
                    best_dim = dim
        if best_dim is None:
            best_dim = "calibration"
            best_delta = 0

        stage = DIMENSION_TO_STAGE[best_dim]

        # Total delta
        total_delta = 0
        for dim in DIMENSION_TO_STAGE:
            w = winner_scores.get(dim, 0)
            l = loser_scores.get(dim, 0)
            if isinstance(w, (int, float)) and isinstance(l, (int, float)):
                total_delta += w - l

        pairs.append({
            "query": query,
            "preferred": winner_response,
            "rejected": loser_response,
            "stage": stage,
            "dominant_dimension": best_dim,
            "score_delta": best_delta,
            "total_delta": total_delta,
            "source": "extended_generation",
            "source_model": "sonnet",
            "winner_config": winner_config,
            "loser_config": loser_config,
        })

    # Write pairs
    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  Extracted {len(pairs)} DPO pairs -> {PAIRS_PATH}")
    return pairs


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Extended DPO Pairs")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate responses (skip judging)")
    parser.add_argument("--judge-only", action="store_true",
                        help="Only run judging (skip generation)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract pairs from existing judgments")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit questions (smoke test)")
    args = parser.parse_args()

    questions = get_extended_questions()
    if not questions:
        print("ERROR: No extended questions found. Check data/questions_extended.json.")
        sys.exit(1)

    print(f"Extended questions: {len(questions)}")

    if args.extract_only:
        extract_pairs()
        return

    client = get_client()

    if not args.judge_only:
        generate_responses(client, questions, limit=args.limit)

    if not args.generate_only:
        run_judging(client, questions, limit=args.limit)
        extract_pairs()


if __name__ == "__main__":
    main()
