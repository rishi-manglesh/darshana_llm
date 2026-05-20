"""Experiment 9 — Structure-Only Placebo + Cross-Judge Validation

Two critical ablations suggested by peer review:

1. STRUCTURE-ONLY PLACEBO: Replace Sanskrit/meaningful Vritti labels with
   neutral labels (TYPE-A through TYPE-E) while keeping identical structure.
   Tests whether the TAXONOMY matters or just any structured tagging.

2. CROSS-JUDGE VALIDATION: Re-judge 20% of Exp 1-5 results using OpenAI
   GPT-4o instead of Claude Sonnet, to rule out same-model judge bias.

PRATIJNA: The Darshana taxonomy provides value beyond generic structure.
HETU: Sanskrit-derived categories map to genuine epistemic distinctions.
NIGAMANA: If placebo matches Vritti, value is structure not taxonomy.
          If Vritti beats placebo, the specific taxonomy matters.
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, MODELS, get_client,
    load_jsonl, append_jsonl, load_existing_keys, mean,
    count_vritti_tags, count_hedging, count_udaharana,
    call_api, call_api_json, run_experiment,
)
from darshana.vritti import VRITTI_INLINE_PROMPT, GENERIC_CONFIDENCE_PROMPT

# ============================================================================
# PART 1: STRUCTURE-ONLY PLACEBO
# ============================================================================

PLACEBO_STRUCTURED_PROMPT = """For each claim or statement in your response, tag it with one of these categories:

[TYPE-A] — You are confident this is factually correct, based on well-established evidence or direct logical inference from verified premises.
[TYPE-B] — You are reproducing well-known information from established sources. Common textbook knowledge.
[TYPE-C] — You are reasoning from premises to a conclusion. The premises are established but the conclusion is your inference.
[TYPE-D] — You are providing context, framing, or explanation that helps understanding but is not itself a factual claim.
[TYPE-E] — You are not confident in this claim. It might be wrong. Flag it honestly.

Place the tag at the START of each paragraph or major claim. Be honest — tagging uncertain claims as TYPE-A is worse than tagging them as TYPE-E."""


def generate_placebo(config, question, client, model):
    """Generate a response with one of three prompts."""
    prompts = {
        "vritti_5mode": VRITTI_INLINE_PROMPT,
        "placebo_structured": PLACEBO_STRUCTURED_PROMPT,
        "generic_confidence": GENERIC_CONFIDENCE_PROMPT,
    }

    system = prompts[config]
    response = call_api(client, model, system, question["query"], max_tokens=1024)
    if response is None:
        return {"response": "", "word_count": 0, "metrics": {}}

    return {
        "response": response,
        "word_count": len(response.split()),
        "metrics": {
            "vritti_tags": count_vritti_tags(response),
            "hedging_count": count_hedging(response),
            "udaharana_count": count_udaharana(response),
        },
    }


def run_placebo_experiment(limit=None):
    """Run the structure-only placebo experiment."""
    client = get_client()
    model = MODELS["sonnet"]
    configs = ["vritti_5mode", "placebo_structured", "generic_confidence"]

    results_dir = RESULTS_DIR / "exp9_placebo"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"
    existing = load_existing_keys(results_path)

    questions = TRANSFER_QUESTIONS[:limit] if limit else TRANSFER_QUESTIONS
    total = len(questions) * len(configs)
    done = len(existing)
    t0 = time.time()

    print(f"\n{'='*60}")
    print("EXPERIMENT 9a: Structure-Only Placebo")
    print(f"  Questions: {len(questions)}, Configs: {len(configs)}")
    print(f"  Total generations: {total}, Existing: {done}")
    print(f"{'='*60}\n")

    for qi, q in enumerate(questions):
        for config in configs:
            key = f"{config}|{q['query'][:60]}"
            if key in existing:
                continue

            done += 1
            elapsed = time.time() - t0
            rate = elapsed / max(done - len(load_existing_keys(results_path)), 1)
            print(f"  [{done}/{total}] {config:<25} | {q['domain']:<10} | "
                  f"{elapsed:.0f}s", flush=True)

            result = generate_placebo(config, q, client, model)
            result.update({
                "key": key,
                "query": q["query"],
                "target_op": q["target_op"],
                "domain": q["domain"],
                "config": config,
            })
            append_jsonl(results_path, result)
            existing.add(key)

    elapsed = time.time() - t0
    print(f"\n  Generation complete in {elapsed:.0f}s")
    return results_path


def judge_placebo(judge_model="sonnet"):
    """Judge placebo experiment: vritti vs placebo, vritti vs generic, placebo vs generic."""
    results_dir = RESULTS_DIR / "exp9_placebo"
    results_path = results_dir / "results.jsonl"
    judge_path = results_dir / "judge.jsonl"

    client = get_client()
    model = MODELS[judge_model]
    records = load_jsonl(results_path)
    existing = load_existing_keys(judge_path)

    # Group by query
    by_query = defaultdict(dict)
    for r in records:
        by_query[r["query"]][r["config"]] = r

    # Three comparison pairs
    comparisons = [
        ("generic_confidence", "vritti_5mode", "Vritti vs Generic"),
        ("generic_confidence", "placebo_structured", "Placebo vs Generic"),
        ("placebo_structured", "vritti_5mode", "Vritti vs Placebo"),
    ]

    from experiments.judge import judge_pairwise, PAIRWISE_SYSTEM

    for baseline, experimental, label in comparisons:
        pairs = []
        for query, configs in by_query.items():
            if baseline in configs and experimental in configs:
                pairs.append(query)

        print(f"\n  Judging: {label} ({len(pairs)} pairs)")
        wins, losses, ties = 0, 0, 0

        for query in pairs:
            key = f"pairwise|{query[:60]}|{baseline}|{experimental}"
            if key in existing:
                # Count existing
                for rec in load_jsonl(judge_path):
                    if rec.get("key") == key:
                        if rec["winner"] == experimental:
                            wins += 1
                        elif rec["winner"] == baseline:
                            losses += 1
                        else:
                            ties += 1
                continue

            resp_a = by_query[query][baseline]["response"]
            resp_b = by_query[query][experimental]["response"]

            result = judge_pairwise(client, model, query, resp_a, resp_b,
                                    baseline, experimental)
            if result is None:
                continue

            result.update({
                "key": key,
                "type": "pairwise",
                "query": query,
                "config_a": baseline,
                "config_b": experimental,
            })
            append_jsonl(judge_path, result)
            existing.add(key)

            if result["winner"] == experimental:
                wins += 1
            elif result["winner"] == baseline:
                losses += 1
            else:
                ties += 1

        total = wins + losses + ties
        if total > 0:
            win_rate = wins / total
            print(f"    {experimental}: {wins}/{total} wins ({win_rate*100:.0f}%)")
            print(f"    {baseline}: {losses}/{total} wins")
            print(f"    Ties: {ties}")


# ============================================================================
# PART 2: CROSS-JUDGE VALIDATION (GPT-4o re-judges subset of Exp 1-5)
# ============================================================================

def get_openai_client():
    """Get OpenAI client."""
    try:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                for line in open(env_path):
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
            if not api_key:
                print("ERROR: OPENAI_API_KEY not found in environment or .env")
                return None
        return openai.OpenAI(api_key=api_key)
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        return None


# Use the SAME judge prompt as Claude for fair comparison
CROSSJUDGE_SYSTEM = """You are an expert evaluator of reasoning quality in educational responses.

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


def gpt4o_judge_pairwise(openai_client, query, response_a, response_b,
                          label_a, label_b):
    """Judge a pair using GPT-4o with randomized order."""
    if random.random() < 0.5:
        user_content = f"QUESTION: {query}\n\nRESPONSE A:\n{response_a}\n\nRESPONSE B:\n{response_b}"
        order = (label_a, label_b)
    else:
        user_content = f"QUESTION: {query}\n\nRESPONSE A:\n{response_b}\n\nRESPONSE B:\n{response_a}"
        order = (label_b, label_a)

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CROSSJUDGE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=500,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        judgment = json.loads(text)
    except Exception as e:
        print(f"  [GPT-4o ERROR] {e}")
        return None

    raw_winner = judgment.get("winner", "TIE")
    if raw_winner == "A":
        actual_winner = order[0]
    elif raw_winner == "B":
        actual_winner = order[1]
    else:
        actual_winner = "TIE"

    return {
        "order": order,
        "scores_a": judgment.get(f"response_{'a' if order[0] == label_a else 'b'}", {}),
        "scores_b": judgment.get(f"response_{'b' if order[0] == label_a else 'a'}", {}),
        "winner": actual_winner,
        "reason": judgment.get("reason", ""),
    }


def run_crossjudge_validation(sample_pct=1.0):
    """Re-judge a random sample of Exp 1-5 results using GPT-4o."""
    openai_client = get_openai_client()
    if openai_client is None:
        return

    crossjudge_dir = RESULTS_DIR / "exp9_crossjudge"
    crossjudge_dir.mkdir(parents=True, exist_ok=True)
    crossjudge_path = crossjudge_dir / "crossjudge.jsonl"
    existing = load_existing_keys(crossjudge_path)

    experiments = {
        "exp1_vritti": ("generic_confidence", ["vritti_5mode"]),
        "exp3_mimamsa": ("generic_rewritten", ["mimamsa_rewritten"]),
        "exp5_vedanta": ("generic_synth", ["vedanta_synth"]),
    }

    agreement_stats = defaultdict(lambda: {"agree": 0, "disagree": 0, "total": 0})

    for exp_name, (baseline, experimentals) in experiments.items():
        results_path = RESULTS_DIR / exp_name / "results.jsonl"
        sonnet_judge_path = RESULTS_DIR / exp_name / "judge.jsonl"

        if not results_path.exists():
            print(f"  Skipping {exp_name}: no results file")
            continue

        records = load_jsonl(results_path)
        sonnet_judgments = load_jsonl(sonnet_judge_path) if sonnet_judge_path.exists() else []

        # Group results by query
        by_query = defaultdict(dict)
        for r in records:
            by_query[r["query"]][r["config"]] = r

        # Get Sonnet judgments for comparison
        sonnet_by_key = {}
        for j in sonnet_judgments:
            if j.get("type") == "pairwise":
                sonnet_by_key[j.get("key", "")] = j

        # Build pairs and sample
        all_pairs = []
        for query, configs in by_query.items():
            if baseline not in configs:
                continue
            for exp_config in experimentals:
                if exp_config in configs:
                    all_pairs.append((query, baseline, exp_config))

        sample_size = max(1, int(len(all_pairs) * sample_pct))
        random.seed(42)  # Reproducible sample
        sample = random.sample(all_pairs, min(sample_size, len(all_pairs)))

        print(f"\n  Cross-judging {exp_name}: {len(sample)} pairs (of {len(all_pairs)})")

        for query, config_a, config_b in sample:
            key = f"gpt4o|{query[:60]}|{config_a}|{config_b}"
            if key in existing:
                continue

            resp_a = by_query[query][config_a]["response"]
            resp_b = by_query[query][config_b]["response"]

            result = gpt4o_judge_pairwise(openai_client, query, resp_a, resp_b,
                                           config_a, config_b)
            if result is None:
                continue

            # Find matching Sonnet judgment
            sonnet_key = f"pairwise|{query[:60]}|{config_a}|{config_b}"
            sonnet_result = sonnet_by_key.get(sonnet_key, {})
            sonnet_winner = sonnet_result.get("winner", "UNKNOWN")

            result.update({
                "key": key,
                "experiment": exp_name,
                "query": query,
                "config_a": config_a,
                "config_b": config_b,
                "judge": "gpt-4o",
                "sonnet_winner": sonnet_winner,
                "judges_agree": result["winner"] == sonnet_winner,
            })

            append_jsonl(crossjudge_path, result)
            existing.add(key)

            # Track agreement
            agreement_stats[exp_name]["total"] += 1
            if result["winner"] == sonnet_winner:
                agreement_stats[exp_name]["agree"] += 1
            else:
                agreement_stats[exp_name]["disagree"] += 1

            time.sleep(0.5)  # Rate limiting for OpenAI

    # Print agreement summary
    print(f"\n{'='*60}")
    print("CROSS-JUDGE AGREEMENT (Claude Sonnet vs GPT-4o)")
    print(f"{'='*60}")

    total_agree, total_all = 0, 0
    for exp_name, stats in agreement_stats.items():
        if stats["total"] > 0:
            rate = stats["agree"] / stats["total"]
            print(f"  {exp_name}: {stats['agree']}/{stats['total']} agree ({rate*100:.0f}%)")
            total_agree += stats["agree"]
            total_all += stats["total"]

    if total_all > 0:
        print(f"\n  OVERALL: {total_agree}/{total_all} agree ({total_agree/total_all*100:.0f}%)")

    # Also print GPT-4o win rates for the experimental configs
    all_crossjudge = load_jsonl(crossjudge_path)
    print(f"\n  GPT-4o Win Rates:")
    for exp_name in experiments:
        exp_results = [r for r in all_crossjudge if r.get("experiment") == exp_name]
        if not exp_results:
            continue
        exp_config = experiments[exp_name][1][0]
        wins = sum(1 for r in exp_results if r["winner"] == exp_config)
        losses = sum(1 for r in exp_results if r["winner"] == experiments[exp_name][0])
        ties = sum(1 for r in exp_results if r["winner"] == "TIE")
        total = len(exp_results)
        if total > 0:
            print(f"    {exp_name} ({exp_config}): {wins}/{total} wins ({wins/total*100:.0f}%)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 9: Placebo + Cross-Judge")
    parser.add_argument("--placebo", action="store_true", help="Run placebo generation")
    parser.add_argument("--judge-placebo", action="store_true", help="Judge placebo results")
    parser.add_argument("--crossjudge", action="store_true", help="Run cross-judge validation")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions (smoke test)")
    parser.add_argument("--all", action="store_true", help="Run everything")
    args = parser.parse_args()

    if args.all or args.placebo:
        print("\n" + "="*60)
        print("PART 1: Structure-Only Placebo — Generation")
        print("="*60)
        run_placebo_experiment(limit=args.limit)

    if args.all or args.judge_placebo:
        print("\n" + "="*60)
        print("PART 1: Structure-Only Placebo — Judging")
        print("="*60)
        judge_placebo()

    if args.all or args.crossjudge:
        print("\n" + "="*60)
        print("PART 2: Cross-Judge Validation (GPT-4o)")
        print("="*60)
        run_crossjudge_validation()

    if not any([args.placebo, args.judge_placebo, args.crossjudge, args.all]):
        print("Usage:")
        print("  python exp9_placebo_and_crossjudge.py --placebo          # Generate placebo responses")
        print("  python exp9_placebo_and_crossjudge.py --judge-placebo    # Judge placebo results")
        print("  python exp9_placebo_and_crossjudge.py --crossjudge       # GPT-4o cross-judge")
        print("  python exp9_placebo_and_crossjudge.py --all              # Run everything")
        print("  python exp9_placebo_and_crossjudge.py --placebo --limit 3  # Smoke test")
