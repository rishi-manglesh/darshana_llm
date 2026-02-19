#!/usr/bin/env python3
"""Exp 3: Mimamsa — 6 Lingas Prompt Rewriting vs Generic Prompt Engineering

LAYER: Prompt design (offline preprocessing, not runtime)
RESEARCH FRAMEWORK: Nyaya Pancha-avayava (5-Step Syllogism)

PRATIJNA (Thesis):
  Rewriting questions using Mimamsa's 6 Lingas produces higher-quality LLM
  responses than standard prompt engineering techniques.

HETU (Reason):
  The 6 Lingas are a systematic INTERPRETATION framework from 2000+ years of
  textual exegesis. They capture dimensions generic prompt engineering doesn't:
  novelty (Apurvata), purpose (Phala), context vs instruction (Arthavada).

UDAHARANA (Prior Evidence):
  - Phase 6: Mimamsa as system prompt: 0% win rate — total failure
  - Phase 7: full_pipeline (includes Mimamsa preprocessing): 53%
  - Phase 7: pipeline_no_mimamsa: 40%
  - Key insight: Mimamsa FAILED as runtime prompt but never tested as OFFLINE rewriter
  - MIMAMSA_REWRITE_SYSTEM exists but never tested

UPANAYA (Experiment Design):
  3 configs x 30 questions = 90 generations (API — Claude Sonnet)
  Step 1: Rewrite all 30 questions three ways (Mimamsa, Generic, Original)
  Step 2: Send all 90 questions to Sonnet for answers
  Step 3: Pairwise judge all combinations

NIGAMANA (Success Criteria):
  - PROVEN: mimamsa_rewritten > generic_rewritten by >10% win rate
  - DISPROVEN: generic_rewritten >= mimamsa_rewritten
  - PARTIALLY PROVEN: both > original but mimamsa ≈ generic

Cost: ~$0.60
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.mimamsa import rewrite_with_lingas, generic_rewrite
from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, MODELS,
    get_client, call_api, run_experiment,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp3_mimamsa"
GENERATION_MODEL = "sonnet"
REWRITE_MODEL = "sonnet"

CONFIGS = ["original", "mimamsa_rewritten", "generic_rewritten"]


# -- Rewriting -----------------------------------------------------------------

_rewrite_cache = {}
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


def get_rewritten_query(client, config, original_query):
    """Get the (possibly rewritten) query for a config."""
    cache_key = f"{config}|{original_query[:60]}"
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key]

    model = MODELS[REWRITE_MODEL]

    if config == "original":
        result = original_query
    elif config == "mimamsa_rewritten":
        result = rewrite_with_lingas(client, model, original_query)
    elif config == "generic_rewritten":
        result = generic_rewrite(client, model, original_query)
    else:
        result = original_query

    _rewrite_cache[cache_key] = result
    return result


# -- Generation ----------------------------------------------------------------

def generate_fn(config, question):
    """Generate a response through Sonnet with the (possibly rewritten) query."""
    client = _get_client()
    model = MODELS[GENERATION_MODEL]
    query = get_rewritten_query(client, config, question["query"])

    response = call_api(client, model, "", query, max_tokens=1024)
    if response is None:
        response = "[ERROR: API call failed]"

    return {
        "response": response,
        "rewritten_query": query,
        "word_count": len(response.split()),
    }


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 3: Mimamsa 6 Lingas Rewriting")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument("--judge", action="store_true", help="Run judging")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    args = parser.parse_args()

    results = run_experiment(
        name=EXPERIMENT_NAME,
        configs=CONFIGS,
        generate_fn=generate_fn,
        limit=args.limit,
    )

    # Show rewrite examples
    print(f"\n{'='*60}")
    print("REWRITE EXAMPLES (first 3 questions)")
    print(f"{'='*60}")
    for q in TRANSFER_QUESTIONS[:3]:
        original = q["query"]
        print(f"\nOriginal:  {original}")
        mimamsa = _rewrite_cache.get(f"mimamsa_rewritten|{original[:60]}", "N/A")
        generic = _rewrite_cache.get(f"generic_rewritten|{original[:60]}", "N/A")
        print(f"Mimamsa:   {mimamsa}")
        print(f"Generic:   {generic}")

    if args.judge:
        # Judge all vs original baseline
        run_pairwise_judging(
            EXPERIMENT_NAME, "original", ["mimamsa_rewritten", "generic_rewritten"],
            judge_model=args.judge_model,
        )
        # Head-to-head: mimamsa vs generic
        print("\n  --- Mimamsa vs Generic (head-to-head) ---")
        run_pairwise_judging(
            EXPERIMENT_NAME, "generic_rewritten", ["mimamsa_rewritten"],
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
