#!/usr/bin/env python3
"""Exp 5: Nyaya — Pramana-Based Tool Routing

LAYER: Tool use routing
VALIDATION: Pramana classification reduces search redundancy (from 57%)
while maintaining accuracy.

Configs (4 x 30 = 120 generations, API):
  - always_search: Force search on every question
  - never_search: No tools
  - pramana_routed: Classify -> search only for Shabda + uncertain Anumana
  - model_decides: Model chooses (baseline tool use)

Cost: ~$0.80 | Success: <30% redundant searches (vs 57%), same accuracy
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.nyaya import generate_with_tools, NYAYA_SYSTEM, NYAYA_TOOLS
from darshana.nyaya_router import route_query, classify_pramana
from darshana.vaisheshika import extract_structure
from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, MODELS,
    get_client, call_api, run_experiment,
    load_jsonl, append_jsonl,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp5_nyaya"
MODEL = "sonnet"

CONFIGS = ["always_search", "never_search", "pramana_routed", "model_decides"]

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


# -- Generation ----------------------------------------------------------------

def generate_fn(config, question):
    """Generate a response with the specified tool routing strategy."""
    client = _get_client()
    model = MODELS[MODEL]
    query = question["query"]

    if config == "always_search":
        # Force search via Nyaya pipeline
        result = generate_with_tools(client, model, query)
        return {
            "response": result["response"],
            "tool_calls": result["tool_calls"],
            "api_calls": result["total_api_calls"],
            "routing": {"strategy": "always_search"},
            "word_count": len(result["response"].split()),
            "metrics": extract_structure(result["response"], result["tool_calls"]),
        }

    elif config == "never_search":
        # Direct generation, no tools
        response = call_api(client, model, "", query, max_tokens=1024)
        if response is None:
            response = "[ERROR]"
        return {
            "response": response,
            "tool_calls": [],
            "api_calls": 1,
            "routing": {"strategy": "never_search"},
            "word_count": len(response.split()),
            "metrics": extract_structure(response),
        }

    elif config == "pramana_routed":
        # Classify first, then decide
        routing = route_query(client, model, query)
        if routing["use_search"]:
            result = generate_with_tools(client, model, query)
            return {
                "response": result["response"],
                "tool_calls": result["tool_calls"],
                "api_calls": result["total_api_calls"] + 1,  # +1 for classification
                "routing": routing,
                "word_count": len(result["response"].split()),
                "metrics": extract_structure(result["response"], result["tool_calls"]),
            }
        else:
            # No search needed per pramana classification
            response = call_api(client, model, NYAYA_SYSTEM, query, max_tokens=1024)
            if response is None:
                response = "[ERROR]"
            return {
                "response": response,
                "tool_calls": [],
                "api_calls": 2,  # classification + generation
                "routing": routing,
                "word_count": len(response.split()),
                "metrics": extract_structure(response),
            }

    elif config == "model_decides":
        # Give tools but let model decide (standard Nyaya behavior)
        result = generate_with_tools(client, model, query)
        return {
            "response": result["response"],
            "tool_calls": result["tool_calls"],
            "api_calls": result["total_api_calls"],
            "routing": {"strategy": "model_decides"},
            "word_count": len(result["response"].split()),
            "metrics": extract_structure(result["response"], result["tool_calls"]),
        }


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 5: Nyaya Pramana Routing")
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

    # Routing analysis
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    print(f"\n{'='*60}")
    print("ROUTING ANALYSIS")
    print(f"{'='*60}")
    for cfg in CONFIGS:
        recs = by_config.get(cfg, [])
        if not recs:
            continue
        total_searches = sum(len(r.get("tool_calls", [])) for r in recs)
        searched = sum(1 for r in recs if len(r.get("tool_calls", [])) > 0)
        avg_api = sum(r.get("api_calls", 1) for r in recs) / len(recs)
        print(f"  {cfg:<20} | searched: {searched}/{len(recs)} | "
              f"total searches: {total_searches} | avg API calls: {avg_api:.1f}")

    # Pramana distribution for routed config
    routed = by_config.get("pramana_routed", [])
    if routed:
        from collections import Counter
        pramanas = Counter()
        for r in routed:
            p = r.get("routing", {}).get("pramana", {}).get("pramana", "UNKNOWN")
            pramanas[p] += 1
        print(f"\n  Pramana distribution:")
        for p, count in pramanas.most_common():
            print(f"    {p}: {count}")

    if args.judge:
        run_pairwise_judging(
            EXPERIMENT_NAME, "never_search",
            ["always_search", "pramana_routed", "model_decides"],
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
