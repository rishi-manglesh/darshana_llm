"""LLM Judge — Pairwise and Factual Evaluation

Extracted from vedic_llm eval_dharmic_judge.py and eval_darshana_judge.py.
Provides consistent judging infrastructure for all 7 experiments.
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import anthropic

from .utils import (
    RESULTS_DIR, MODELS, get_client, load_jsonl, append_jsonl,
    load_existing_keys, mean, call_api_json,
)

# -- Judge Prompts (5-dimension, from vedic_llm Phase 7) -----------------------

PAIRWISE_SYSTEM = """You are an expert evaluator of reasoning quality in educational responses.

You will see a QUESTION and two RESPONSES (A and B). One may use structured reasoning, epistemic tags, or cite searched evidence. The other may be more conventional.

Judge ONLY on reasoning quality — NOT on formatting, length, or structure compliance.

Evaluate on these 5 dimensions (1-5 each):

1. FACTUAL_ACCURACY: Are claims correct? Are examples real and accurate?
   1=major errors, 3=mostly correct, 5=fully accurate with verified specifics

2. REASONING_DEPTH: Does it go beyond surface to explain WHY and HOW?
   1=surface only, 3=some explanation, 5=genuine causal/mechanistic insight

3. COMPLETENESS: Does it address the question's key aspects?
   1=misses most aspects, 3=covers basics, 5=comprehensive coverage

4. CALIBRATION: Does it distinguish what it knows confidently from what it's uncertain about?
   1=overconfident throughout, 3=some hedging, 5=accurately signals certainty levels

5. USEFULNESS: Would a student learn something genuinely valuable from this?
   1=not helpful, 3=decent overview, 5=would significantly improve understanding

After scoring both, declare a WINNER or TIE.

IMPORTANT: Judge purely on content quality. The position (A vs B) should NOT influence your judgment. A response is not better simply because it appears first or second.

Respond with ONLY a JSON object:
{
  "response_a": {"factual_accuracy": <1-5>, "reasoning_depth": <1-5>, "completeness": <1-5>, "calibration": <1-5>, "usefulness": <1-5>},
  "response_b": {"factual_accuracy": <1-5>, "reasoning_depth": <1-5>, "completeness": <1-5>, "calibration": <1-5>, "usefulness": <1-5>},
  "winner": "A" or "B" or "TIE",
  "reason": "<1 sentence explaining why>"
}"""

PAIRWISE_USER = """QUESTION: {query}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}"""


FACTUAL_SYSTEM = """You are a fact-checker evaluating whether specific examples and evidence cited in a response are factually accurate.

The response may cite web search results, specific real-world cases, or named examples. Your job is to verify EACH specific claim.

For each factual claim or example in the response, assess:
- Is the event/case real?
- Are the specific details (names, dates, numbers) accurate?
- Is the example correctly applied to the argument?

Respond with ONLY a JSON object:
{
  "claims_checked": <number of specific factual claims checked>,
  "claims_accurate": <number that are fully accurate>,
  "claims_minor_error": <number with minor inaccuracies>,
  "claims_major_error": <number that are substantially wrong or fabricated>,
  "accuracy_rate": <float 0-1>,
  "details": "<brief notes on any errors found>"
}"""


TOOL_USE_SYSTEM = """You are evaluating whether web searches made during response generation were valuable.

You will see a QUESTION, the SEARCHES that were performed (query + results preview), and the FINAL RESPONSE.

For each search, rate its contribution:
- CRITICAL: The search found specific evidence that meaningfully improved the response.
- HELPFUL: The search found useful context but the response could have been similar without it.
- REDUNDANT: The search found information the model likely already knew. No real value added.

Respond with ONLY a JSON object:
{
  "search_ratings": [
    {"query": "<search query>", "rating": "CRITICAL" or "HELPFUL" or "REDUNDANT", "reason": "<brief>"}
  ],
  "overall_tool_value": "HIGH" or "MODERATE" or "LOW",
  "summary": "<1 sentence on overall tool use effectiveness>"
}"""


# -- Judging Functions ---------------------------------------------------------

def judge_pairwise(client, model, query, response_a, response_b,
                   label_a="baseline", label_b="experimental"):
    """Run a pairwise judgment with randomized order to avoid position bias.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: the question
        response_a, response_b: the two responses
        label_a, label_b: config labels

    Returns:
        dict with scores, winner (remapped to actual labels), reason
    """
    # Randomize order
    if random.random() < 0.5:
        user_content = PAIRWISE_USER.format(
            query=query, response_a=response_a, response_b=response_b
        )
        order = (label_a, label_b)
    else:
        user_content = PAIRWISE_USER.format(
            query=query, response_a=response_b, response_b=response_a
        )
        order = (label_b, label_a)

    judgment = call_api_json(client, model, PAIRWISE_SYSTEM, user_content)
    if judgment is None:
        return None

    # Remap winner — Fix #9: log unexpected values instead of silently defaulting
    raw_winner = judgment.get("winner", "TIE")
    if raw_winner == "A":
        actual_winner = order[0]
    elif raw_winner == "B":
        actual_winner = order[1]
    elif raw_winner == "TIE":
        actual_winner = "TIE"
    else:
        print(f"  WARNING: Unexpected winner value '{raw_winner}', treating as TIE")
        actual_winner = "TIE"

    return {
        "order": order,
        "scores_a": judgment.get(f"response_{'a' if order[0] == label_a else 'b'}", {}),
        "scores_b": judgment.get(f"response_{'b' if order[0] == label_a else 'a'}", {}),
        "winner": actual_winner,
        "reason": judgment.get("reason", ""),
    }


def judge_factual(client, model, query, response):
    """Fact-check a response.

    Returns:
        dict with claims_checked, claims_accurate, etc., or None
    """
    user_content = f"QUESTION: {query}\n\nRESPONSE TO FACT-CHECK:\n{response}"
    return call_api_json(client, model, FACTUAL_SYSTEM, user_content)


def judge_tool_use(client, model, query, tool_calls, response):
    """Evaluate tool use effectiveness.

    Returns:
        dict with search_ratings, overall_tool_value, or None
    """
    searches_text = ""
    for i, tc in enumerate(tool_calls, 1):
        searches_text += f"\nSearch {i}: \"{tc.get('query', '')}\"\n"
        searches_text += f"  Context: {tc.get('claim_context', 'N/A')}\n"
        searches_text += f"  Results preview: {tc.get('result_preview', 'N/A')}\n"

    user_content = (
        f"QUESTION: {query}\n\n"
        f"SEARCHES PERFORMED:\n{searches_text}\n\n"
        f"FINAL RESPONSE:\n{response[:2000]}"
    )
    return call_api_json(client, model, TOOL_USE_SYSTEM, user_content)


# -- Batch Judging for Experiments ---------------------------------------------

def run_pairwise_judging(experiment_name, baseline_config, experimental_configs,
                         judge_model="haiku", resume=True):
    """Run pairwise judging for an experiment's results.

    Reads from results/{experiment_name}/results.jsonl
    Writes to results/{experiment_name}/judge.jsonl

    Returns:
        dict mapping config -> {wins, losses, ties, win_rate}
    """
    results_dir = RESULTS_DIR / experiment_name
    results_path = results_dir / "results.jsonl"
    judge_path = results_dir / "judge.jsonl"

    client = get_client()
    model = MODELS[judge_model]

    records = load_jsonl(results_path)
    existing = load_existing_keys(judge_path) if resume else set()

    # Group by query
    by_query = defaultdict(dict)
    for r in records:
        by_query[r["query"]][r["config"]] = r

    # Build comparison pairs
    pairs = []
    for query, configs in by_query.items():
        if baseline_config not in configs:
            continue
        for exp_config in experimental_configs:
            if exp_config in configs:
                pairs.append((query, baseline_config, exp_config))

    print(f"\n  Judging {experiment_name}: {len(pairs)} pairs ({len(existing)} existing)")
    t0 = time.time()
    new_count = 0

    for query, config_a, config_b in pairs:
        key = f"pairwise|{query[:60]}|{config_a}|{config_b}"
        if key in existing:
            continue

        resp_a = by_query[query][config_a]["response"]
        resp_b = by_query[query][config_b]["response"]

        result = judge_pairwise(client, model, query, resp_a, resp_b, config_a, config_b)
        if result is None:
            continue

        result.update({
            "key": key,
            "type": "pairwise",
            "query": query,
            "config_a": config_a,
            "config_b": config_b,
            "target_op": by_query[query][config_b].get("target_op", ""),
            "domain": by_query[query][config_b].get("domain", ""),
        })

        append_jsonl(judge_path, result)
        existing.add(key)
        new_count += 1

        if new_count % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{new_count} done] {elapsed:.0f}s")

    # Compute win rates
    all_judgments = load_jsonl(judge_path)
    pairwise = [r for r in all_judgments if r.get("type") == "pairwise"]

    summary = {}
    for config in experimental_configs:
        cr = [r for r in pairwise if r.get("config_b") == config]
        if not cr:
            continue
        wins = sum(1 for r in cr if r["winner"] == config)
        losses = sum(1 for r in cr if r["winner"] == baseline_config)
        ties = sum(1 for r in cr if r["winner"] == "TIE")
        total = len(cr)
        summary[config] = {
            "wins": wins, "losses": losses, "ties": ties,
            "total": total, "win_rate": wins / total if total else 0,
        }

    elapsed = time.time() - t0
    print(f"  Judging complete: {new_count} new in {elapsed:.0f}s")

    # Print summary
    for config, stats in summary.items():
        print(f"    {config}: {stats['wins']}/{stats['total']} wins "
              f"({stats['win_rate']*100:.0f}%)")

    return summary
