#!/usr/bin/env python3
"""Exp 2: Nyaya — Pramana Routing vs Heuristic Routing

LAYER: Tool use / search routing
RESEARCH FRAMEWORK: Nyaya Pancha-avayava (5-Step Syllogism)

PRATIJNA (Thesis):
  Classifying queries by Nyaya's 4 pramanas (Pratyaksha, Anumana, Upamana, Shabda)
  produces smarter search routing than simple heuristic rules, reducing the 57%
  search redundancy while maintaining accuracy.

HETU (Reason):
  Pramana classification captures the EPISTEMOLOGICAL reason search is needed.
  "This needs authoritative testimony" (Shabda) vs "This can be derived from logic"
  (Anumana) should produce different routing decisions than keyword-matching.

UDAHARANA (Prior Evidence):
  - Phase 7: 57% of all searches REDUNDANT, 26% helpful, 17% critical
  - Phase 7: All configs had 100% search rate — no selective routing
  - Nyaya tools overall: 70% win rate but -0.40 factual accuracy vs baseline
  - Pramana classifier built (nyaya_router.py) but never tested

UPANAYA (Experiment Design):
  5 configs x 30 questions = 150 generations (API — Claude Sonnet)
  - always_search: Force search every question (the Phase 7 approach)
  - never_search: No external search at all
  - pramana_routed: Classify by 4 pramanas -> search only for Pratyaksha + Shabda
  - heuristic_routed: Simple rules: search if dates/numbers/names/"latest"/"current"
  - model_decides: Ask the model "Do you need to search for this?"

NIGAMANA (Success Criteria):
  - PROVEN: pramana_routed has <30% redundancy AND higher factual accuracy
    than always_search AND beats heuristic_routed on at least one metric
  - DISPROVEN: heuristic_routed matches or beats pramana_routed on all metrics
  - INCONCLUSIVE: pramana < always_search on accuracy (safety > efficiency)

Cost: ~$1.00
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.nyaya import generate_with_tools, NYAYA_SYSTEM, NYAYA_TOOLS
from darshana.nyaya_router import (
    route_query, classify_pramana, heuristic_route, model_decides_route,
)
from darshana.vaisheshika import extract_structure
from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, MODELS,
    get_client, call_api, run_experiment,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp2_nyaya"
MODEL = "sonnet"

CONFIGS = ["always_search", "never_search", "pramana_routed", "heuristic_routed", "model_decides"]

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
        routing = route_query(client, model, query)
        if routing["use_search"]:
            result = generate_with_tools(client, model, query)
            return {
                "response": result["response"],
                "tool_calls": result["tool_calls"],
                "api_calls": result["total_api_calls"] + 1,
                "routing": routing,
                "word_count": len(result["response"].split()),
                "metrics": extract_structure(result["response"], result["tool_calls"]),
            }
        else:
            response = call_api(client, model, NYAYA_SYSTEM, query, max_tokens=1024)
            if response is None:
                response = "[ERROR]"
            return {
                "response": response,
                "tool_calls": [],
                "api_calls": 2,
                "routing": routing,
                "word_count": len(response.split()),
                "metrics": extract_structure(response),
            }

    elif config == "heuristic_routed":
        routing = heuristic_route(query)
        if routing["use_search"]:
            result = generate_with_tools(client, model, query)
            return {
                "response": result["response"],
                "tool_calls": result["tool_calls"],
                "api_calls": result["total_api_calls"],
                "routing": routing,
                "word_count": len(result["response"].split()),
                "metrics": extract_structure(result["response"], result["tool_calls"]),
            }
        else:
            response = call_api(client, model, NYAYA_SYSTEM, query, max_tokens=1024)
            if response is None:
                response = "[ERROR]"
            return {
                "response": response,
                "tool_calls": [],
                "api_calls": 1,
                "routing": routing,
                "word_count": len(response.split()),
                "metrics": extract_structure(response),
            }

    elif config == "model_decides":
        routing = model_decides_route(client, model, query)
        if routing["use_search"]:
            result = generate_with_tools(client, model, query)
            return {
                "response": result["response"],
                "tool_calls": result["tool_calls"],
                "api_calls": result["total_api_calls"] + 1,
                "routing": routing,
                "word_count": len(result["response"].split()),
                "metrics": extract_structure(result["response"], result["tool_calls"]),
            }
        else:
            response = call_api(client, model, "", query, max_tokens=1024)
            if response is None:
                response = "[ERROR]"
            return {
                "response": response,
                "tool_calls": [],
                "api_calls": 2,
                "routing": routing,
                "word_count": len(response.split()),
                "metrics": extract_structure(response),
            }


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 2: Nyaya Pramana Routing vs Heuristic")
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
        search_rate = searched / len(recs) * 100 if recs else 0
        print(f"  {cfg:<20} | searched: {searched}/{len(recs)} ({search_rate:.0f}%) | "
              f"total searches: {total_searches} | avg API calls: {avg_api:.1f}")

    # Pramana distribution for routed config
    routed = by_config.get("pramana_routed", [])
    if routed:
        pramanas = Counter()
        for r in routed:
            p = r.get("routing", {}).get("pramana", {}).get("pramana", "UNKNOWN")
            pramanas[p] += 1
        print(f"\n  Pramana distribution (pramana_routed):")
        for p, count in pramanas.most_common():
            print(f"    {p}: {count}")

    # Heuristic distribution
    heuristic = by_config.get("heuristic_routed", [])
    if heuristic:
        searched = sum(1 for r in heuristic if r.get("routing", {}).get("use_search", False))
        print(f"\n  Heuristic: {searched}/{len(heuristic)} routed to search")

    if args.judge:
        run_pairwise_judging(
            EXPERIMENT_NAME, "never_search",
            ["always_search", "pramana_routed", "heuristic_routed", "model_decides"],
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
