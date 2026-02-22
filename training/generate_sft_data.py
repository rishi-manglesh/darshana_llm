#!/usr/bin/env python3
"""Generate SFT Training Data Using Yoga Curriculum

Generates ideal responses for each Yoga stage using Claude.
Each question gets 5 ideal responses (one per stage), each emphasizing
a different quality dimension grounded in Yoga Sutra text.

Modes:
  small: 10 questions × 5 stages = 50 examples (~$1, ~5 min)
  full:  75 questions × 5 stages = 375 examples (~$3.50, ~30 min)

Output format (SFT chat):
  {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}], "stage": N}

Usage:
  python training/generate_sft_data.py --mode small --model sonnet
  python training/generate_sft_data.py --mode full --model sonnet
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.yoga_sft import YOGA_SFT_STAGES, format_sft_prompt
from experiments.utils import (
    TRANSFER_QUESTIONS, DATA_DIR,
    get_client, call_api_json, get_all_questions,
)


# -- Config --------------------------------------------------------------------

SFT_DIR = DATA_DIR / "sft_yoga"

# Small test: 10 questions stratified across target_ops
SMALL_N = 10
# Full: all 75 questions
FULL_N = 75

# Held-out questions for evaluation (indices into TRANSFER_QUESTIONS)
# Stratified: pick questions NOT used in small training set
SMALL_TRAIN_SEED = 42


def select_small_questions(questions, n=SMALL_N, seed=SMALL_TRAIN_SEED):
    """Select N questions stratified across target_op types.

    Returns:
        (train_questions, holdout_questions) — train for SFT, holdout for eval
    """
    rng = random.Random(seed)

    by_op = {}
    for q in questions:
        by_op.setdefault(q["target_op"], []).append(q)

    # Take equal from each op type
    selected = []
    per_op = n // len(by_op)
    for op, qs in sorted(by_op.items()):
        rng.shuffle(qs)
        selected.extend(qs[:per_op])

    # Fill remainder
    remaining = [q for q in questions if q not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:n - len(selected)])

    holdout = [q for q in questions if q not in selected]
    return selected[:n], holdout


def generate_sft_examples(client, model, questions, output_path, stage_nums=None):
    """Generate SFT examples: one ideal response per question per stage.

    Args:
        client: Anthropic client
        model: model ID
        questions: list of question dicts
        output_path: path to write JSONL
        stage_nums: which stages to generate (default: all 5)
    """
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    stages = stage_nums or list(range(1, 6))
    total = len(questions) * len(stages)

    # Load existing for resume
    existing = set()
    if output_path.exists():
        for line in open(output_path):
            rec = json.loads(line)
            existing.add(rec.get("key", ""))

    print(f"Generating SFT examples: {total} total ({len(existing)} existing)")
    t0 = time.time()
    new_count = 0

    for qi, q in enumerate(questions):
        for stage_num in stages:
            key = f"sft|{q['query'][:60]}|stage{stage_num}"
            if key in existing:
                continue

            system, user_content = format_sft_prompt(q["query"], stage_num)
            result = call_api_json(client, model, system, user_content, max_tokens=800)

            if result is None:
                print(f"  [WARN] Failed: stage {stage_num}, q{qi}")
                continue

            response_text = result.get("response", "")
            if not response_text or len(response_text) < 50:
                print(f"  [WARN] Too short: stage {stage_num}, q{qi}")
                continue

            record = {
                "key": key,
                "messages": [
                    {"role": "user", "content": q["query"]},
                    {"role": "assistant", "content": response_text},
                ],
                "stage": stage_num,
                "stage_name": YOGA_SFT_STAGES[stage_num - 1]["name"],
                "quality_dimension": YOGA_SFT_STAGES[stage_num - 1]["sft_criterion"],
                "query": q["query"],
                "target_op": q["target_op"],
                "domain": q["domain"],
            }

            with open(output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing.add(key)
            new_count += 1

            if new_count % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{new_count}/{total - len(existing) + new_count}] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  SFT generation complete: {new_count} new in {elapsed:.0f}s")
    print(f"  Saved to: {output_path}")
    return new_count


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SFT Training Data")
    parser.add_argument("--mode", choices=["small", "full"], default="small",
                        help="small=50 examples, full=375 examples")
    parser.add_argument("--model", choices=["haiku", "sonnet"], default="sonnet",
                        help="Model for generation (default: sonnet)")
    args = parser.parse_args()

    from experiments.utils import MODELS
    client = get_client()
    model = MODELS[args.model]

    if args.mode == "small":
        train_qs, holdout_qs = select_small_questions(TRANSFER_QUESTIONS)
        output_path = SFT_DIR / "small_test.jsonl"

        print(f"\nSmall test mode:")
        print(f"  Training questions: {len(train_qs)}")
        print(f"  Holdout questions: {len(holdout_qs)}")
        print(f"  Target ops in train: {set(q['target_op'] for q in train_qs)}")

        # Save holdout for eval
        SFT_DIR.mkdir(parents=True, exist_ok=True)
        holdout_path = SFT_DIR / "small_holdout.json"
        with open(holdout_path, "w") as f:
            json.dump(holdout_qs, f, indent=2)
        print(f"  Holdout saved to: {holdout_path}")

        generate_sft_examples(client, model, train_qs, output_path)

    elif args.mode == "full":
        all_qs = get_all_questions()
        if len(all_qs) < FULL_N:
            print(f"  [WARN] Only {len(all_qs)} questions available (expected {FULL_N})")

        # Hold out 15 questions for eval
        rng = random.Random(SMALL_TRAIN_SEED)
        rng.shuffle(all_qs)
        holdout_qs = all_qs[:15]
        train_qs = all_qs[15:]

        output_path = SFT_DIR / "full_train.jsonl"

        print(f"\nFull mode:")
        print(f"  Training questions: {len(train_qs)}")
        print(f"  Holdout questions: {len(holdout_qs)}")

        SFT_DIR.mkdir(parents=True, exist_ok=True)
        holdout_path = SFT_DIR / "full_holdout.json"
        with open(holdout_path, "w") as f:
            json.dump(holdout_qs, f, indent=2)
        print(f"  Holdout saved to: {holdout_path}")

        generate_sft_examples(client, model, train_qs, output_path)

    print("\nDone. Next: python training/train_sft.py --mode yoga --phase small")


if __name__ == "__main__":
    main()
