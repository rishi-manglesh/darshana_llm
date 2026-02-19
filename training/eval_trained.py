#!/usr/bin/env python3
"""Evaluate Trained Models

Quick evaluation of pre/post-trained models on a few questions.
Useful for sanity checking before running full experiments.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import TRANSFER_QUESTIONS


def evaluate_model(model_path, questions, max_tokens=256):
    """Generate responses from a model and display them."""
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        print("ERROR: Requires mlx-lm package.")
        sys.exit(1)

    print(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=0.7, top_p=0.9)

    for i, q in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"Q{i+1}: {q['query']}")
        print(f"{'='*60}")

        messages = [{"role": "user", "content": q["query"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        result = generate(
            model, tokenizer, prompt=text,
            max_tokens=max_tokens, sampler=sampler, verbose=False
        )
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        print(result[:500])
        print(f"\n  [{len(result.split())} words]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Models")
    parser.add_argument("model_path", help="Path to model or HuggingFace model ID")
    parser.add_argument("--limit", type=int, default=5, help="Number of questions")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    questions = TRANSFER_QUESTIONS[:args.limit]
    evaluate_model(args.model_path, questions, args.max_tokens)


if __name__ == "__main__":
    main()
