#!/usr/bin/env python3
"""Combine, Filter, and Balance DPO Data for Training

Combines all DPO data sources:
  - 833 pairs from darshana_llm exps 1,3,5 + cross-model (dpo_real_pairs.jsonl)
  - ~286 pairs from vedic_llm (included in dpo_real_pairs.jsonl after re-extraction)
  - ~70-90 pairs from extended question generation (extended_pairs/pairs.jsonl)

Processing:
  1. Filter to --min-delta (strong signal only)
  2. Deduplicate (same query + winner response = same pair)
  3. Balance stages: cap each at N pairs (N = min stage count, or configurable)
  4. Split: training queries (60) + held-out test queries (15)

Output: data/dpo_combined/{train.jsonl, test_queries.json, stats.json}

Usage:
  python training/prepare_dpo_data.py
  python training/prepare_dpo_data.py --min-delta 2 --stats
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import load_jsonl, TRANSFER_QUESTIONS, get_extended_questions

# -- Config --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "dpo_combined"

# Input sources
REAL_PAIRS_PATH = DATA_DIR / "dpo_real_pairs.jsonl"
EXTENDED_PAIRS_PATH = DATA_DIR / "extended_pairs" / "pairs.jsonl"

STAGE_NAMES = {
    1: "Yama (honesty)",
    2: "Asana (structure)",
    3: "Pratyahara (restraint)",
    4: "Dharana (deep analysis)",
    5: "Dhyana (synthesis)",
}


# -- Processing ----------------------------------------------------------------

def load_all_pairs(min_delta=0):
    """Load pairs from all sources."""
    all_pairs = []

    # Real pairs (darshana_llm + vedic_llm)
    if REAL_PAIRS_PATH.exists():
        pairs = load_jsonl(REAL_PAIRS_PATH)
        print(f"  Real pairs: {len(pairs)}")
        all_pairs.extend(pairs)
    else:
        print(f"  [WARN] Real pairs not found: {REAL_PAIRS_PATH}")

    # Extended generation pairs
    if EXTENDED_PAIRS_PATH.exists():
        pairs = load_jsonl(EXTENDED_PAIRS_PATH)
        print(f"  Extended pairs: {len(pairs)}")
        all_pairs.extend(pairs)
    else:
        print(f"  [WARN] Extended pairs not found: {EXTENDED_PAIRS_PATH}")

    print(f"  Total loaded: {len(all_pairs)}")

    # Filter by min_delta
    if min_delta > 0:
        before = len(all_pairs)
        all_pairs = [p for p in all_pairs if p.get("total_delta", p.get("score_delta", 0)) >= min_delta]
        print(f"  After min_delta >= {min_delta}: {len(all_pairs)} (filtered {before - len(all_pairs)})")

    return all_pairs


def deduplicate(pairs):
    """Remove duplicate pairs (same query + same preferred response)."""
    seen = set()
    unique = []
    for p in pairs:
        # Hash on query prefix + first 200 chars of preferred
        key = hashlib.md5(
            (p["query"][:80] + "|" + p["preferred"][:200]).encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    print(f"  After deduplication: {len(unique)} (removed {len(pairs) - len(unique)} dupes)")
    return unique


def balance_stages(pairs, max_per_stage=None):
    """Balance pairs across Yoga stages.

    If max_per_stage is None, uses the minimum stage count (so all stages equal).
    """
    by_stage = defaultdict(list)
    for p in pairs:
        by_stage[p.get("stage", 3)].append(p)

    stage_counts = {s: len(ps) for s, ps in sorted(by_stage.items())}
    print(f"  Stage distribution before balancing: {stage_counts}")

    if max_per_stage is None:
        max_per_stage = min(len(ps) for ps in by_stage.values()) if by_stage else 0

    print(f"  Balancing to {max_per_stage} per stage")

    balanced = []
    for stage in sorted(by_stage.keys()):
        stage_pairs = by_stage[stage]
        # Sort by total_delta descending — keep strongest signal pairs
        stage_pairs.sort(key=lambda p: p.get("total_delta", p.get("score_delta", 0)), reverse=True)
        balanced.extend(stage_pairs[:max_per_stage])

    print(f"  After balancing: {len(balanced)}")
    return balanced


def split_train_test(pairs, test_ratio=0.2, seed=42):
    """Split by query into train and test sets.

    Ensures all pairs for a given query go to either train or test (no leakage).
    Returns (train_pairs, test_queries).
    """
    # Group by query
    by_query = defaultdict(list)
    for p in pairs:
        by_query[p["query"]].append(p)

    queries = list(by_query.keys())
    random.seed(seed)
    random.shuffle(queries)

    n_test = max(1, int(len(queries) * test_ratio))
    test_queries = set(queries[:n_test])
    train_queries = set(queries[n_test:])

    train_pairs = [p for p in pairs if p["query"] in train_queries]
    test_pair_count = sum(len(by_query[q]) for q in test_queries)

    print(f"  Split: {len(train_queries)} train queries ({len(train_pairs)} pairs), "
          f"{len(test_queries)} test queries ({test_pair_count} pairs held out)")

    return train_pairs, sorted(test_queries)


def print_stats(pairs, label=""):
    """Print statistics for a set of pairs."""
    if label:
        print(f"\n  --- {label} ---")
    print(f"  Total pairs: {len(pairs)}")

    by_stage = Counter(p.get("stage", 0) for p in pairs)
    print(f"  By stage:")
    for stage in sorted(by_stage.keys()):
        name = STAGE_NAMES.get(stage, f"Stage {stage}")
        print(f"    {stage}. {name}: {by_stage[stage]}")

    by_source = Counter(p.get("source", "unknown") for p in pairs)
    print(f"  By source:")
    for src, count in sorted(by_source.items()):
        print(f"    {src}: {count}")

    queries = set(p["query"] for p in pairs)
    print(f"  Unique queries: {len(queries)}")

    deltas = [p.get("total_delta", p.get("score_delta", 0)) for p in pairs]
    if deltas:
        print(f"  Delta: avg={sum(deltas)/len(deltas):.2f}, "
              f"min={min(deltas)}, max={max(deltas)}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Combined DPO Data")
    parser.add_argument("--min-delta", type=int, default=2,
                        help="Minimum total score delta (default: 2)")
    parser.add_argument("--max-per-stage", type=int, default=None,
                        help="Max pairs per stage (default: auto-balance)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of queries for test (default: 0.2)")
    parser.add_argument("--stats", action="store_true",
                        help="Print stats of existing combined data")
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip stage balancing")
    args = parser.parse_args()

    # If --stats and data exists, just print
    train_path = OUTPUT_DIR / "train.jsonl"
    if args.stats and train_path.exists():
        pairs = load_jsonl(train_path)
        print_stats(pairs, "Training Data")
        test_path = OUTPUT_DIR / "test_queries.json"
        if test_path.exists():
            with open(test_path) as f:
                test_qs = json.load(f)
            print(f"\n  Test queries: {len(test_qs)}")
        return

    print(f"\nPreparing combined DPO data...")

    # Load
    pairs = load_all_pairs(min_delta=args.min_delta)
    if not pairs:
        print("ERROR: No pairs found. Run extract_real_pairs.py and generate_extended_pairs.py first.")
        sys.exit(1)

    # Deduplicate
    pairs = deduplicate(pairs)

    # Balance stages
    if not args.no_balance:
        pairs = balance_stages(pairs, max_per_stage=args.max_per_stage)

    # Split train/test
    train_pairs, test_queries = split_train_test(pairs, test_ratio=args.test_ratio)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n  Written train: {train_path} ({len(train_pairs)} pairs)")

    test_path = OUTPUT_DIR / "test_queries.json"
    with open(test_path, "w") as f:
        json.dump(test_queries, f, indent=2, ensure_ascii=False)
    print(f"  Written test queries: {test_path} ({len(test_queries)} queries)")

    # Stats
    stats = {
        "total_raw": len(load_all_pairs(min_delta=0)),
        "after_filter": len(load_all_pairs(min_delta=args.min_delta)),
        "after_dedup": len(pairs),
        "train_pairs": len(train_pairs),
        "test_queries": len(test_queries),
        "min_delta": args.min_delta,
        "stage_distribution": dict(Counter(p.get("stage", 0) for p in train_pairs)),
        "source_distribution": dict(Counter(p.get("source", "") for p in train_pairs)),
    }
    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Written stats: {stats_path}")

    print_stats(train_pairs, "Final Training Data")


if __name__ == "__main__":
    main()
