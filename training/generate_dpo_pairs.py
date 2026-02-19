#!/usr/bin/env python3
"""Generate DPO Preference Pairs Using Yoga Curriculum

Generates 150 DPO pairs: 30 questions x 5 Yoga stages.
Uses Claude to create preferred/rejected pairs for each quality dimension.

Also generates 150 "generic" DPO pairs for comparison (standard_dpo).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.yoga_dpo import YOGA_STAGES, format_pair_prompt
from experiments.utils import (
    TRANSFER_QUESTIONS, MODELS, DATA_DIR,
    get_client, call_api_json,
)


# -- Config --------------------------------------------------------------------

DPO_DIR = DATA_DIR / "dpo_yoga_pairs"

GENERIC_DPO_SYSTEM = """Generate a DPO preference pair for the following question.

Create two responses:
- PREFERRED: A well-structured, accurate, insightful response
- REJECTED: A plausible but inferior response (shallow, slightly inaccurate, or poorly organized)

Both should attempt to answer the question. The rejected response is NOT garbage — it's just worse.
Keep each response 150-300 words.

Respond with ONLY a JSON object:
{
  "preferred": "the better response text",
  "rejected": "the worse response text",
  "quality_dimension": "general"
}"""


def generate_yoga_pairs(client, model, limit=None):
    """Generate Yoga-curriculum DPO pairs (30 questions x 5 stages = 150 pairs)."""
    output_path = DPO_DIR / "yoga_pairs.jsonl"
    DPO_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing for resume
    existing = set()
    if output_path.exists():
        for line in open(output_path):
            rec = json.loads(line)
            existing.add(rec.get("key", ""))

    questions = TRANSFER_QUESTIONS[:limit] if limit else TRANSFER_QUESTIONS
    total = len(questions) * 5  # 5 stages

    print(f"Generating Yoga DPO pairs: {total} total ({len(existing)} existing)")
    t0 = time.time()
    new_count = 0

    for qi, q in enumerate(questions):
        for stage_num in range(1, 6):
            key = f"yoga|{q['query'][:60]}|stage{stage_num}"
            if key in existing:
                continue

            prompt = format_pair_prompt(q["query"], stage_num)
            result = call_api_json(client, model, "", prompt, max_tokens=800)

            if result is None:
                print(f"  [WARN] Failed: stage {stage_num}, q{qi}")
                continue

            record = {
                "key": key,
                "query": q["query"],
                "target_op": q["target_op"],
                "domain": q["domain"],
                "stage": stage_num,
                "stage_name": YOGA_STAGES[stage_num - 1]["name"],
                "quality_dimension": YOGA_STAGES[stage_num - 1]["dpo_criterion"],
                "preferred": result.get("preferred", ""),
                "rejected": result.get("rejected", ""),
            }

            with open(output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing.add(key)
            new_count += 1

            if new_count % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{new_count}/{total - len(existing) + new_count}] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Yoga pairs complete: {new_count} new in {elapsed:.0f}s")
    print(f"  Saved to: {output_path}")


def generate_generic_pairs(client, model, limit=None):
    """Generate generic DPO pairs (no Yoga framework) for comparison."""
    output_path = DPO_DIR / "generic_pairs.jsonl"
    DPO_DIR.mkdir(parents=True, exist_ok=True)

    existing = set()
    if output_path.exists():
        for line in open(output_path):
            rec = json.loads(line)
            existing.add(rec.get("key", ""))

    questions = TRANSFER_QUESTIONS[:limit] if limit else TRANSFER_QUESTIONS
    # Generate 5 pairs per question (to match yoga count)
    total = len(questions) * 5

    print(f"Generating generic DPO pairs: {total} total ({len(existing)} existing)")
    t0 = time.time()
    new_count = 0

    for qi, q in enumerate(questions):
        for variant in range(5):
            key = f"generic|{q['query'][:60]}|v{variant}"
            if key in existing:
                continue

            result = call_api_json(
                client, model, GENERIC_DPO_SYSTEM, q["query"], max_tokens=800
            )

            if result is None:
                continue

            record = {
                "key": key,
                "query": q["query"],
                "target_op": q["target_op"],
                "domain": q["domain"],
                "variant": variant,
                "quality_dimension": "general",
                "preferred": result.get("preferred", ""),
                "rejected": result.get("rejected", ""),
            }

            with open(output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing.add(key)
            new_count += 1

            if new_count % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{new_count}] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Generic pairs complete: {new_count} new in {elapsed:.0f}s")
    print(f"  Saved to: {output_path}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate DPO Pairs")
    parser.add_argument("--mode", choices=["yoga", "generic", "both"], default="both")
    parser.add_argument("--model", choices=["haiku", "sonnet"], default="sonnet")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    args = parser.parse_args()

    client = get_client()
    model = MODELS[args.model]

    if args.mode in ("yoga", "both"):
        generate_yoga_pairs(client, model, limit=args.limit)

    if args.mode in ("generic", "both"):
        generate_generic_pairs(client, model, limit=args.limit)

    print("\nDone. Next: run training/train_dpo.py")


if __name__ == "__main__":
    main()
