#!/usr/bin/env python3
"""Exp 1: Vritti — Epistemic Self-Classification vs Generic Confidence Tagging

LAYER: System prompt / inline prompt
RESEARCH FRAMEWORK: Nyaya Pancha-avayava (5-Step Syllogism)

PRATIJNA (Thesis):
  Vritti's 5-mode epistemic taxonomy (PRAMANA, SMRITI, ANUMANA, VIKALPA, UNCERTAIN)
  produces better-calibrated LLM responses than a generic confidence-tagging prompt
  of equal complexity.

HETU (Reason):
  Vritti's categories map to DISTINCT knowledge types, not just confidence levels.
  "I'm inferring from premises" (ANUMANA) is qualitatively different from
  "I'm recalling textbook facts" (SMRITI), even if both have high confidence.

UDAHARANA (Prior Evidence):
  - Phase 6a: 63% win rate on Qwen3-4B (vs bare baseline)
  - Phase 7: 90% win rate on Sonnet (vs bare baseline)
  - Phase 7: pipeline_clean 100% (confounded with formatting)
  - Gap: Never tested vs generic equivalent

UPANAYA (Experiment Design):
  5 configs x 30 questions = 150 generations (API — Claude Sonnet)
  - bare_baseline: No system prompt
  - vritti_5mode: Vritti inline prompt (Sanskrit: PRAMANA/SMRITI/ANUMANA/VIKALPA/UNCERTAIN)
  - vritti_contemporary: Same 5-mode taxonomy with English labels (ESTABLISHED/TEXTBOOK/INFERRED/FRAMING/UNCERTAIN)
  - generic_confidence: "Tag confidence: CERTAIN/LIKELY/UNCERTAIN/SPECULATIVE/UNKNOWN"
  - generic_cot: "Note reasoning basis and confidence before each claim"

  vritti_contemporary is the KEY config — it isolates whether the 5-mode TAXONOMY
  adds value (vs generic_confidence's single confidence axis) independently of
  whether Sanskrit labels help or hurt. If vritti_contemporary > generic_confidence,
  the framework is proven even before pretraining teaches the model Sanskrit.

NIGAMANA (Success Criteria):
  - PROVEN: vritti_5mode > generic_confidence by >10% win rate AND qualitative
    audit shows tags drive content differentiation (not just labels)
  - DISPROVEN: generic_confidence matches or beats vritti_5mode
  - INCONCLUSIVE: <10% win rate difference

Cost: ~$0.50
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.vritti import (
    VRITTI_INLINE_PROMPT, VRITTI_CONTEMPORARY_PROMPT,
    GENERIC_CONFIDENCE_PROMPT, GENERIC_COT_PROMPT,
)
from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, MODELS,
    get_client, call_api, run_experiment,
    count_vritti_tags, count_hedging, count_udaharana,
)
from experiments.judge import run_pairwise_judging


# -- Config --------------------------------------------------------------------

EXPERIMENT_NAME = "exp1_vritti"
MODEL = "sonnet"

SYSTEM_PROMPTS = {
    "bare_baseline": None,
    "vritti_5mode": VRITTI_INLINE_PROMPT,
    "vritti_contemporary": VRITTI_CONTEMPORARY_PROMPT,
    "generic_confidence": GENERIC_CONFIDENCE_PROMPT,
    "generic_cot": GENERIC_COT_PROMPT,
}

CONFIGS = list(SYSTEM_PROMPTS.keys())

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


# -- Generation ----------------------------------------------------------------

def generate_fn(config, question):
    """Generate a response via Claude Sonnet with the specified prompt config."""
    client = _get_client()
    model = MODELS[MODEL]
    system = SYSTEM_PROMPTS[config] or ""
    query = question["query"]

    response = call_api(client, model, system, query, max_tokens=1024)
    if response is None:
        response = "[ERROR: API call failed]"

    return {
        "response": response,
        "word_count": len(response.split()),
        "metrics": {
            "vritti_tags": count_vritti_tags(response),
            "hedging_count": count_hedging(response),
            "udaharana_count": count_udaharana(response),
        },
    }


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 1: Vritti 5-Mode vs Generic Confidence Tagging"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit questions (smoke test)")
    parser.add_argument("--judge", action="store_true", help="Run pairwise judging")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    args = parser.parse_args()

    results = run_experiment(
        name=EXPERIMENT_NAME,
        configs=CONFIGS,
        generate_fn=generate_fn,
        limit=args.limit,
    )

    # Metrics summary
    by_config = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print(f"{'='*60}")
    for cfg in CONFIGS:
        recs = by_config.get(cfg, [])
        if not recs:
            continue
        n = len(recs)
        avg_tags = sum(r.get("metrics", {}).get("vritti_tags", {}).get("total", 0) for r in recs) / n
        avg_hedge = sum(r.get("metrics", {}).get("hedging_count", 0) for r in recs) / n
        avg_words = sum(r.get("word_count", 0) for r in recs) / n
        print(f"  {cfg:<25} | tags: {avg_tags:.1f} | hedges: {avg_hedge:.1f} | words: {avg_words:.0f}")

    if args.judge:
        # Judge all configs against bare_baseline
        experimental = [c for c in CONFIGS if c != "bare_baseline"]
        run_pairwise_judging(
            EXPERIMENT_NAME, "bare_baseline", experimental,
            judge_model=args.judge_model,
        )
        # Also judge vritti_5mode directly vs generic_confidence
        print("\n  --- Vritti vs Generic Confidence (head-to-head) ---")
        run_pairwise_judging(
            EXPERIMENT_NAME, "generic_confidence", ["vritti_5mode"],
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
