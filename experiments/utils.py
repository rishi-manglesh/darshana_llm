"""Experiment Utilities — Questions, Metrics, API Helpers

Canonical 30 transfer questions (from vedic_llm Phase 5-7).
Metrics functions extracted from eval_dharmic_principles.py.
API helper functions for consistent experiment infrastructure.
"""

import json
import os
import re
import time
from pathlib import Path

import anthropic

# -- Project Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# -- API Client ----------------------------------------------------------------

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
}


def get_client():
    """Get an Anthropic client using ANTHROPIC_API_KEY env var."""
    return anthropic.Anthropic()


def call_api(client, model, system, user_content, max_tokens=1024, max_retries=3):
    """Call Claude API with retry logic.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        system: system prompt
        user_content: user message content
        max_tokens: max response tokens
        max_retries: number of retries on failure

    Returns:
        str: response text, or None on failure
    """
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            if not msg.content:
                print(f"  [WARN] Empty response (stop={msg.stop_reason}), retrying...")
                time.sleep(2)
                continue
            return msg.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"  [RATE LIMIT] Waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"  [API ERROR] {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    return None


def call_api_json(client, model, system, user_content, max_tokens=500, max_retries=3):
    """Call Claude API and parse JSON response.

    Returns:
        dict or None on failure
    """
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            text = msg.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"  [WARN] Parse failed after {max_retries} attempts: {e}")
            return None
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"  [RATE LIMIT] Waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"  [API ERROR] {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    return None


# -- 30 Transfer Questions (canonical) ----------------------------------------

TRANSFER_QUESTIONS = [
    # Discrimination (10)
    {"query": "Compare fiscal policy and monetary policy. How do they differ in approach and impact?",
     "target_op": "discrimination", "domain": "economics"},
    {"query": "What distinguishes a recession from a depression? How are they categorized differently?",
     "target_op": "discrimination", "domain": "economics"},
    {"query": "Contrast supply-side economics with demand-side economics.",
     "target_op": "discrimination", "domain": "economics"},
    {"query": "Compare mitosis and meiosis. What are the key differences?",
     "target_op": "discrimination", "domain": "biology"},
    {"query": "How do prokaryotic and eukaryotic cells differ?",
     "target_op": "discrimination", "domain": "biology"},
    {"query": "Distinguish between innate immunity and adaptive immunity.",
     "target_op": "discrimination", "domain": "biology"},
    {"query": "Compare renting versus buying a home. What are the trade-offs?",
     "target_op": "discrimination", "domain": "everyday"},
    {"query": "What's the difference between a leader and a manager?",
     "target_op": "discrimination", "domain": "everyday"},
    {"query": "Contrast online learning with traditional classroom education.",
     "target_op": "discrimination", "domain": "everyday"},
    {"query": "How do stocks and bonds differ as investments?",
     "target_op": "discrimination", "domain": "everyday"},
    # Force (10)
    {"query": "What forces drive inflation in a modern economy?",
     "target_op": "force", "domain": "economics"},
    {"query": "What causes a stock market crash? What are the driving mechanisms?",
     "target_op": "force", "domain": "economics"},
    {"query": "What drives economic growth in developing countries?",
     "target_op": "force", "domain": "economics"},
    {"query": "What drives cell division? What triggers and propels the process?",
     "target_op": "force", "domain": "biology"},
    {"query": "What causes an allergic reaction? Describe the mechanism.",
     "target_op": "force", "domain": "biology"},
    {"query": "What drives evolution? What forces cause species to change over time?",
     "target_op": "force", "domain": "biology"},
    {"query": "What drives people to change careers? What forces push them?",
     "target_op": "force", "domain": "everyday"},
    {"query": "What causes traffic jams? What are the driving mechanisms?",
     "target_op": "force", "domain": "everyday"},
    {"query": "What drives the spread of misinformation on social media?",
     "target_op": "force", "domain": "everyday"},
    {"query": "What causes burnout at work? What are the mechanisms?",
     "target_op": "force", "domain": "everyday"},
    # Constraint (10)
    {"query": "What limits economic growth? What are the binding constraints?",
     "target_op": "constraint", "domain": "economics"},
    {"query": "What constraints does a central bank face in setting interest rates?",
     "target_op": "constraint", "domain": "economics"},
    {"query": "What are the bottlenecks that restrict international trade?",
     "target_op": "constraint", "domain": "economics"},
    {"query": "What limits how large a cell can grow? What constrains cell size?",
     "target_op": "constraint", "domain": "biology"},
    {"query": "What constrains population growth in an ecosystem?",
     "target_op": "constraint", "domain": "biology"},
    {"query": "What limits the human lifespan? What biological boundaries exist?",
     "target_op": "constraint", "domain": "biology"},
    {"query": "What limits how fast a person can learn a new skill?",
     "target_op": "constraint", "domain": "everyday"},
    {"query": "What are the constraints on renewable energy adoption?",
     "target_op": "constraint", "domain": "everyday"},
    {"query": "What bottlenecks restrict affordable housing in cities?",
     "target_op": "constraint", "domain": "everyday"},
    {"query": "What limits productivity in remote work?",
     "target_op": "constraint", "domain": "everyday"},
]


# -- Metrics Functions ---------------------------------------------------------

def count_udaharana(text):
    """Count grounded examples — specific named cases, not hypotheticals."""
    patterns = [
        r'for example[,:]',
        r'for instance[,:]',
        r'such as\b',
        r'e\.g\.\b',
        r'like the\b.*\b(?:in|of|during)\b',
        r'case of\b',
        r'(?:consider|take)\b.*\bexample\b',
        r'UDAHARANA',
        r'Udaharana',
        r'grounded example',
        r'known case',
        r'real.world example',
    ]
    count = 0
    lower = text.lower()
    for p in patterns:
        count += len(re.findall(p, lower))
    return count


def count_vritti_tags(text):
    """Count Vritti self-classification tags (both Sanskrit and contemporary)."""
    tags = {}
    # Sanskrit labels
    for tag in ['PRAMANA', 'SMRITI', 'ANUMANA', 'VIKALPA', 'UNCERTAIN']:
        tags[tag] = len(re.findall(rf'\[{tag}\]', text))
    # Contemporary labels
    for tag in ['ESTABLISHED', 'TEXTBOOK', 'INFERRED', 'FRAMING']:
        tags[tag] = len(re.findall(rf'\[{tag}\]', text))
    # Generic confidence labels
    for tag in ['CERTAIN', 'LIKELY', 'SPECULATIVE', 'UNKNOWN']:
        tags[tag] = len(re.findall(rf'\[{tag}\]', text))
    tags['total'] = sum(tags.values())
    return tags


def count_hedging(text):
    """Count epistemic hedging / uncertainty acknowledgment."""
    hedges = [
        r"i(?:'m| am) not (?:sure|certain|confident)",
        r"this (?:may|might|could) (?:be|not be)",
        r"it(?:'s| is) (?:unclear|uncertain|debatable)",
        r"tentative",
        r"i lack",
        r"not confident",
        r"more research",
        r"evidence (?:is|remains) (?:limited|mixed|unclear)",
        r"approximately|roughly",
        r"to (?:my|the best of my) knowledge",
    ]
    count = 0
    lower = text.lower()
    for p in hedges:
        count += len(re.findall(p, lower))
    return count


def count_reasoning_steps(text):
    """Count explicit reasoning structure (Sanskrit and contemporary labels)."""
    numbered = len(re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
    labeled = len(re.findall(
        r'^\s*(?:PRATIJNA|HETU|UDAHARANA|UPANAYA|NIGAMANA|'
        r'CLAIM|REASON|EVIDENCE|APPLICATION|CONCLUSION|'
        r'Step|Claim|Reason|Example|Conclusion)[:\s]',
        text, re.MULTILINE
    ))
    return numbered + labeled


# -- I/O Helpers ---------------------------------------------------------------

def load_jsonl(path):
    """Load records from a JSONL file."""
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path, record):
    """Append one record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_keys(path, key_field="key"):
    """Load existing record keys for resume support."""
    existing = set()
    for rec in load_jsonl(path):
        k = rec.get(key_field, "")
        if k:
            existing.add(k)
    return existing


def mean(values):
    """Safe mean calculation."""
    return sum(values) / len(values) if values else 0.0


# -- Experiment Runner Base ----------------------------------------------------

def run_experiment(name, configs, generate_fn, limit=None):
    """Generic experiment runner with resume support.

    Args:
        name: experiment name (e.g., "exp3_vritti")
        configs: list of config name strings
        generate_fn: function(config, question) -> dict with at least 'response' key
        limit: optional limit on number of questions (for smoke testing)

    Returns:
        list of result records
    """
    results_dir = RESULTS_DIR / name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"

    existing = load_existing_keys(results_path)
    questions = TRANSFER_QUESTIONS[:limit] if limit else TRANSFER_QUESTIONS
    total = len(questions) * len(configs)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  Questions: {len(questions)}, Configs: {len(configs)}")
    print(f"  Total generations: {total}")
    print(f"  Existing: {len(existing)} (will skip)")
    print(f"{'='*60}\n")

    done = len(existing)
    t0 = time.time()
    results = []

    for qi, q in enumerate(questions):
        for config in configs:
            key = f"{config}|{q['query'][:60]}"
            if key in existing:
                continue

            done += 1
            elapsed = time.time() - t0
            rate = elapsed / (done - len(existing)) if done > len(existing) else 0
            remaining = rate * (total - done)
            print(f"  [{done}/{total}] {config:<25} | {q['target_op']:<14} | "
                  f"{elapsed:.0f}s, ~{remaining:.0f}s left", flush=True)

            result = generate_fn(config, q)
            result.update({
                "key": key,
                "query": q["query"],
                "target_op": q["target_op"],
                "domain": q["domain"],
                "config": config,
            })

            append_jsonl(results_path, result)
            existing.add(key)
            results.append(result)

    elapsed = time.time() - t0
    print(f"\n  Complete: {len(results)} new generations in {elapsed:.0f}s")
    print(f"  Results: {results_path}")
    return results
