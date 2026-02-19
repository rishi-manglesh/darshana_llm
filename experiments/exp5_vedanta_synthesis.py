#!/usr/bin/env python3
"""Exp 5: Vedanta — Deep Synthesis vs Generic Cleanup

LAYER: Output synthesis / post-processing
RESEARCH FRAMEWORK: Nyaya Pancha-avayava (5-Step Syllogism)

PRATIJNA (Thesis):
  Vedantic synthesis (Brahman/Maya/Atman) produces higher-quality output from
  multi-stage pipeline responses than generic "clean this up" post-processing.

HETU (Reason):
  Vedanta's three operations are SEMANTICALLY distinct:
  - Brahman (unity): Find the single truth across all evidence — INTEGRATION, not summarization
  - Maya (scaffolding removal): Distinguish process artifacts from content
  - Atman (core insight): Extract what persists after details fade

UDAHARANA (Prior Evidence):
  - Phase 7: raw full_pipeline = 53% (worse than coin flip)
  - Phase 7: pipeline_clean (regex formatter) = 100%
  - The 53% -> 100% jump was pure formatting, not intellectual synthesis
  - vedanta_synth.py exists but never tested
  - Key question: Does Vedanta synthesis ADD intellectual value?

UPANAYA (Experiment Design):
  Use Phase 7's 30 full_pipeline responses as input, process 4 ways:
  - raw: No post-processing (the 53% version)
  - regex_format: Regex cleanup via formatter.py (the 100% version)
  - vedanta_synth: Vedantic synthesis (Brahman/Maya/Atman)
  - generic_synth: "Synthesize into clear, unified answer. Remove scaffolding."

  Critical test: vedanta_synth vs generic_synth.

NIGAMANA (Success Criteria):
  - PROVEN: vedanta_synth > generic_synth by >10% AND vedanta_synth >= regex_format
  - DISPROVEN: generic_synth ≈ vedanta_synth (any synthesis helps, Vedanta isn't special)
  - PARTIALLY PROVEN: vedanta > generic but < regex (deep synthesis worse than formatting)

Cost: ~$0.50
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.formatter import clean_format
from darshana.vedanta_synth import synthesize_response, generic_synthesize
from experiments.utils import (
    RESULTS_DIR, MODELS, DATA_DIR,
    get_client, load_jsonl, append_jsonl, load_existing_keys, mean,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp5_vedanta"
PHASE7_DATA = DATA_DIR / "phase7_outputs" / "pipeline_results.jsonl"
SYNTHESIS_MODEL = "sonnet"

CONFIGS = ["raw", "regex_format", "vedanta_synth", "generic_synth"]


def load_pipeline_records():
    """Load Phase 7 full_pipeline records (the 53% raw version)."""
    all_records = load_jsonl(PHASE7_DATA)
    pipeline = [r for r in all_records if r["config"] == "full_pipeline"]
    if not pipeline:
        print("ERROR: No full_pipeline records found in Phase 7 data.")
        sys.exit(1)
    return pipeline


# -- Processing ----------------------------------------------------------------

def process_records(pipeline_records, limit=None):
    """Apply four formatting/synthesis strategies to pipeline records."""
    client = get_client()
    model = MODELS[SYNTHESIS_MODEL]

    results_dir = RESULTS_DIR / EXPERIMENT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"
    existing = load_existing_keys(results_path)

    if limit:
        pipeline_records = pipeline_records[:limit]

    print(f"  Pipeline records: {len(pipeline_records)}")
    print(f"  Configs: {CONFIGS}")
    print(f"  Total: {len(pipeline_records) * len(CONFIGS)}")
    print(f"  Existing: {len(existing)} (will skip)")

    t0 = time.time()
    new_count = 0

    for rec in pipeline_records:
        raw_response = rec["response"]
        query = rec["query"]

        for config in CONFIGS:
            key = f"{config}|{query[:60]}"
            if key in existing:
                continue

            if config == "raw":
                response = raw_response
            elif config == "regex_format":
                response = clean_format(client, model, raw_response, query)
            elif config == "vedanta_synth":
                stages = rec.get("stages", [])
                response = synthesize_response(
                    client, model, raw_response, query, stages=stages
                )
            elif config == "generic_synth":
                stages = rec.get("stages", [])
                response = generic_synthesize(
                    client, model, raw_response, query, stages=stages
                )

            result = {
                "key": key,
                "query": query,
                "target_op": rec.get("target_op", ""),
                "domain": rec.get("domain", ""),
                "config": config,
                "response": response,
                "word_count": len(response.split()),
                "source_config": "full_pipeline",
            }

            append_jsonl(results_path, result)
            existing.add(key)
            new_count += 1

            if new_count % 10 == 0:
                elapsed = time.time() - t0
                print(f"    [{new_count} done] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Processing complete: {new_count} new in {elapsed:.0f}s")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 5: Vedanta Deep Synthesis")
    parser.add_argument("--limit", type=int, default=None, help="Limit records")
    parser.add_argument("--judge", action="store_true", help="Run judging")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    args = parser.parse_args()

    pipeline_records = load_pipeline_records()
    print(f"Loaded {len(pipeline_records)} full_pipeline records from Phase 7")

    process_records(pipeline_records, limit=args.limit)

    # Summary
    results_dir = RESULTS_DIR / EXPERIMENT_NAME
    all_results = load_jsonl(results_dir / "results.jsonl")

    by_config = defaultdict(list)
    for r in all_results:
        by_config[r["config"]].append(r)

    print(f"\n{'='*60}")
    print("WORD COUNT COMPARISON")
    print(f"{'='*60}")
    for cfg in CONFIGS:
        recs = by_config.get(cfg, [])
        if recs:
            avg_words = mean([r["word_count"] for r in recs])
            print(f"  {cfg:<22} | avg words: {avg_words:.0f} | n={len(recs)}")

    if args.judge:
        # Judge all vs raw baseline
        run_pairwise_judging(
            EXPERIMENT_NAME, "raw",
            ["regex_format", "vedanta_synth", "generic_synth"],
            judge_model=args.judge_model,
        )
        # Head-to-head: vedanta vs generic synthesis
        print("\n  --- Vedanta vs Generic Synthesis (head-to-head) ---")
        run_pairwise_judging(
            EXPERIMENT_NAME, "generic_synth", ["vedanta_synth"],
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
