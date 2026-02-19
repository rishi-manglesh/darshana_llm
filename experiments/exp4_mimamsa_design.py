#!/usr/bin/env python3
"""Exp 4: Mimamsa — Prompt Design Methodology

LAYER: Prompt design (OFFLINE, not runtime)
VALIDATION: 6 Lingas as a prompt REWRITING methodology vs generic prompt engineering.

Method:
  1. Rewrite 30 questions using Mimamsa 6 Lingas (via Claude)
  2. Rewrite using generic prompt engineering (via Claude)
  3. Run all 3 versions through Sonnet, compare outputs

Configs (3 x 30 = 90 generations):
  - original: Original 30 questions -> Sonnet
  - mimamsa_rewritten: 6 Lingas rewritten -> Sonnet
  - generic_rewritten: Generic prompt eng rewritten -> Sonnet

Cost: ~$0.60 | Success: mimamsa > generic on response quality
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.mimamsa import rewrite_with_lingas
from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, MODELS,
    get_client, call_api, run_experiment,
    load_jsonl, append_jsonl,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp4_mimamsa"
GENERATION_MODEL = "sonnet"
REWRITE_MODEL = "sonnet"

CONFIGS = ["original", "mimamsa_rewritten", "generic_rewritten"]

GENERIC_REWRITE_SYSTEM = """You are a prompt engineer. Rewrite the following question to get a better response from an AI assistant.

Apply these standard prompt engineering techniques:
1. Be specific about what you want
2. Provide context
3. State the desired format
4. Ask for reasoning, not just facts
5. Remove ambiguity

Respond with ONLY the rewritten question. No explanation — just the improved question text."""


# -- Rewriting -----------------------------------------------------------------

_rewrite_cache = {}


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
        result = call_api(client, model, GENERIC_REWRITE_SYSTEM, original_query, max_tokens=300)
        if result is None:
            result = original_query
    else:
        result = original_query

    _rewrite_cache[cache_key] = result
    return result


# -- Generation ----------------------------------------------------------------

_client = None


def generate_fn(config, question):
    """Generate a response through Sonnet with the (possibly rewritten) query."""
    global _client
    if _client is None:
        _client = get_client()

    model = MODELS[GENERATION_MODEL]
    query = get_rewritten_query(_client, config, question["query"])

    response = call_api(_client, model, "", query, max_tokens=1024)
    if response is None:
        response = "[ERROR: API call failed]"

    return {
        "response": response,
        "rewritten_query": query,
        "word_count": len(response.split()),
    }


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 4: Mimamsa Prompt Design")
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
        run_pairwise_judging(
            EXPERIMENT_NAME, "original", ["mimamsa_rewritten", "generic_rewritten"],
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
