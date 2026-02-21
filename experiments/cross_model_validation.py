#!/usr/bin/env python3
"""Cross-Model Validation — Run Exp 1, 3, 4, 5 on open-source models.

Tests whether Darshana-derived techniques generalize beyond Claude.

Usage:
    # Smoke test: Haiku (3 questions per experiment)
    python experiments/cross_model_validation.py --smoke

    # Local models (MLX on Apple Silicon — no API key needed)
    python experiments/cross_model_validation.py --exp 1 --model qwen3_8b --limit 3
    python experiments/cross_model_validation.py --local          # all local models
    python experiments/cross_model_validation.py --model qwen3_8b # all experiments

    # Judge results (Sonnet judges all)
    python experiments/cross_model_validation.py --judge --model qwen3_8b
    python experiments/cross_model_validation.py --judge --local

    # Analysis summary
    python experiments/cross_model_validation.py --analyze
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.providers import get_provider, LLMProvider, AnthropicProvider
from experiments.utils import (
    TRANSFER_QUESTIONS, RESULTS_DIR, DATA_DIR,
    load_jsonl, append_jsonl, load_existing_keys, mean,
    count_vritti_tags, count_hedging, count_udaharana,
)

# -- Darshana modules (prompts only — no API calls through these) ---------------

from darshana.vritti import (
    VRITTI_CONTEMPORARY_PROMPT,
    GENERIC_CONFIDENCE_PROMPT,
)
from darshana.mimamsa import (
    MIMAMSA_REWRITE_SYSTEM,
    GENERIC_REWRITE_SYSTEM,
)
from darshana.vedanta_synth import (
    VEDANTA_SYNTHESIS_SYSTEM,
    GENERIC_SYNTHESIS_SYSTEM,
)

# -- Experiment Configuration ---------------------------------------------------

SMOKE_MODELS = ["haiku"]

# Local models (MLX on Apple Silicon — no API keys needed)
LOCAL_MODELS = ["qwen3_8b", "qwen3_32b", "mistral_small"]

# All validation models
VALIDATION_MODELS = LOCAL_MODELS

ALL_MODELS = SMOKE_MODELS + VALIDATION_MODELS

RESULTS_BASE = RESULTS_DIR / "cross_model"

EXPERIMENTS = {
    "exp1": {
        "name": "Vritti Epistemic Calibration",
        "darshana_concept": "Vritti 5-mode epistemic taxonomy",
        "original_h2h": "60%",
        "configs": ["bare_baseline", "vritti_contemporary", "generic_confidence"],
        "baseline": "bare_baseline",
        "darshana_config": "vritti_contemporary",
        "generic_config": "generic_confidence",
    },
    "exp3": {
        "name": "Mimamsa 6 Lingas Rewriting",
        "darshana_concept": "Mimamsa 6 Lingas query rewriting",
        "original_h2h": "73%",
        "configs": ["original", "mimamsa_rewritten", "generic_rewritten"],
        "baseline": "original",
        "darshana_config": "mimamsa_rewritten",
        "generic_config": "generic_rewritten",
    },
    "exp4": {
        "name": "Vaisheshika 7-Category Context",
        "darshana_concept": "Vaisheshika 7-padartha ontology",
        "original_h2h": "68%",
        "configs": ["no_context", "padartha_kg", "generic_kg"],
        "baseline": "no_context",
        "darshana_config": "padartha_kg",
        "generic_config": "generic_kg",
    },
    "exp5": {
        "name": "Vedanta Response Synthesis",
        "darshana_concept": "Vedanta Brahman/Maya/Atman synthesis",
        "original_h2h": "63%",
        "configs": ["raw", "vedanta_synth", "generic_synth"],
        "baseline": "raw",
        "darshana_config": "vedanta_synth",
        "generic_config": "generic_synth",
    },
}

# -- Exp 1: Vritti (System Prompt Layer) ----------------------------------------

EXP1_SYSTEM_PROMPTS = {
    "bare_baseline": "",
    "vritti_contemporary": VRITTI_CONTEMPORARY_PROMPT,
    "generic_confidence": GENERIC_CONFIDENCE_PROMPT,
}


def generate_exp1(provider: LLMProvider, config: str, question: dict) -> dict:
    """Exp 1: Generate response with epistemic calibration prompt."""
    system = EXP1_SYSTEM_PROMPTS[config]
    query = question["query"]

    response = provider.call(system, query, max_tokens=1024)
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


# -- Exp 3: Mimamsa (Query Rewriting Layer) ------------------------------------

# Cache rewrites per (model, config, query) to avoid redundant API calls
_rewrite_cache: Dict[str, str] = {}


def _get_rewritten_query(provider: LLMProvider, config: str, original_query: str) -> str:
    """Get the (possibly rewritten) query, using cache."""
    cache_key = f"{provider.short_name}|{config}|{original_query[:60]}"
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key]

    if config == "original":
        result = original_query
    elif config == "mimamsa_rewritten":
        result = provider.call(MIMAMSA_REWRITE_SYSTEM, original_query, max_tokens=300)
        if result is None:
            result = original_query
    elif config == "generic_rewritten":
        result = provider.call(GENERIC_REWRITE_SYSTEM, original_query, max_tokens=300)
        if result is None:
            result = original_query
    else:
        result = original_query

    _rewrite_cache[cache_key] = result
    return result


def generate_exp3(provider: LLMProvider, config: str, question: dict) -> dict:
    """Exp 3: Rewrite query then generate response, both using same model."""
    query = _get_rewritten_query(provider, config, question["query"])

    response = provider.call("", query, max_tokens=1024)
    if response is None:
        response = "[ERROR: API call failed]"

    return {
        "response": response,
        "rewritten_query": query,
        "word_count": len(response.split()),
    }


# -- Exp 4: Vaisheshika (Knowledge Context Layer) ------------------------------

# Pre-built context from the vault corpus — loaded once, reused across models.
# The context is model-agnostic (it's just formatted text injected into the prompt).
_exp4_contexts: Optional[Dict[str, Dict[str, str]]] = None


def _load_exp4_contexts() -> Dict[str, Dict[str, str]]:
    """Load pre-built contexts from Sonnet's exp4v4 results.

    Returns dict[query -> dict[config -> context_prompt]].
    We extract the full prompts from the existing Sonnet results so that
    all models receive the EXACT same context.
    """
    global _exp4_contexts
    if _exp4_contexts is not None:
        return _exp4_contexts

    # Try to load from cached context file first
    context_cache = RESULTS_BASE / "exp4_vaisheshika" / "contexts.json"
    if context_cache.exists():
        with open(context_cache) as f:
            _exp4_contexts = json.load(f)
        print(f"  Loaded {len(_exp4_contexts)} cached Exp 4 contexts")
        return _exp4_contexts

    # Build from Sonnet's exp4v4 results — need the vault graphs
    print("  Building Exp 4 contexts from vault corpus (one-time)...")

    from darshana.vaisheshika_ontology import (
        extract_padarthas, extract_generic,
        build_padartha_graph, build_generic_graph,
        format_padartha_context, format_generic_context,
    )
    from experiments.exp4v4_vaisheshika_vault import (
        VAULT_QUESTIONS, load_vault_corpus, EXTRACTION_MODEL,
    )
    from experiments.utils import MODELS, get_client

    client = get_client()
    corpus = load_vault_corpus()
    extraction_model = MODELS[EXTRACTION_MODEL]

    # Extract knowledge graphs
    p_extractions = []
    g_extractions = []
    for doc in corpus:
        print(f"    Extracting: {doc['doc_type']}...")
        p_extractions.append(extract_padarthas(client, extraction_model, doc["content"]))
        g_extractions.append(extract_generic(client, extraction_model, doc["content"]))

    p_graph = build_padartha_graph(p_extractions)
    g_graph = build_generic_graph(g_extractions)

    print(f"  7-category graph: {len(p_graph.nodes)} nodes")
    print(f"  Generic graph:    {len(g_graph.nodes)} nodes")

    # Build contexts for each question
    _exp4_contexts = {}
    for q in VAULT_QUESTIONS:
        query = q["query"]
        query_type = q["target_op"]
        retrieval_type_map = {"factual": "discrimination", "causal": "force", "gap": "constraint"}
        retrieval_type = retrieval_type_map.get(query_type, "general")

        # Get relevant raw docs for raw_context config
        domain = q["domain"]
        relevant_docs = [d for d in corpus if d["doc_type"] == domain]
        if not relevant_docs:
            relevant_docs = corpus[:2]
        raw_context = "\n\n".join(d["content"] for d in relevant_docs)[:4000]

        _exp4_contexts[query] = {
            "no_context": "",
            "raw_context_prompt": (
                f"You are answering questions based on private project documents.\n\n"
                f"## Documents\n{raw_context}\n\n"
                f"## Question\n{query}\n\n"
                f"Answer based on what the documents say. If the documents "
                f"don't contain the answer, say so."
            ),
            "generic_kg_prompt": (
                f"You are answering questions based on organized project knowledge.\n\n"
                f"{format_generic_context(g_graph, query, retrieval_type)}\n\n"
                f"## Question\n{query}\n\n"
                f"Answer based on the provided knowledge. If specific information "
                f"is missing from the knowledge base, note the gap."
            ),
            "padartha_kg_prompt": (
                f"You are answering questions based on organized project knowledge.\n\n"
                f"{format_padartha_context(p_graph, query, retrieval_type)}\n\n"
                f"## Question\n{query}\n\n"
                f"Answer based on the provided knowledge. If specific information "
                f"is missing from the knowledge base, note the gap."
            ),
        }

    # Cache to disk
    context_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(context_cache, "w") as f:
        json.dump(_exp4_contexts, f, ensure_ascii=False)
    print(f"  Cached {len(_exp4_contexts)} contexts to {context_cache}")

    return _exp4_contexts


def generate_exp4(provider: LLMProvider, config: str, question: dict) -> dict:
    """Exp 4: Generate response with pre-built knowledge context."""
    contexts = _load_exp4_contexts()
    query = question["query"]

    if query not in contexts:
        return {
            "response": f"[ERROR: No pre-built context for query: {query[:60]}...]",
            "word_count": 0,
            "context_type": "missing",
        }

    ctx = contexts[query]

    if config == "no_context":
        response = provider.call("", query, max_tokens=1024)
        context_used = ""
    elif config == "padartha_kg":
        response = provider.call("", ctx["padartha_kg_prompt"], max_tokens=1024)
        context_used = "padartha_kg"
    elif config == "generic_kg":
        response = provider.call("", ctx["generic_kg_prompt"], max_tokens=1024)
        context_used = "generic_kg"
    else:
        response = provider.call("", query, max_tokens=1024)
        context_used = "unknown"

    if response is None:
        response = "[ERROR: API call failed]"

    return {
        "response": response,
        "word_count": len(response.split()),
        "context_type": context_used,
    }


# -- Exp 5: Vedanta (Output Synthesis Layer) -----------------------------------

# Phase 7 pipeline records — loaded once
_exp5_pipeline_records: Optional[List] = None


def _load_pipeline_records() -> list:
    """Load Phase 7 full_pipeline records for synthesis experiments."""
    global _exp5_pipeline_records
    if _exp5_pipeline_records is not None:
        return _exp5_pipeline_records

    phase7_path = DATA_DIR / "phase7_outputs" / "pipeline_results.jsonl"
    all_records = load_jsonl(phase7_path)
    _exp5_pipeline_records = [r for r in all_records if r["config"] == "full_pipeline"]

    if not _exp5_pipeline_records:
        print(f"ERROR: No full_pipeline records found in {phase7_path}")
        sys.exit(1)

    print(f"  Loaded {len(_exp5_pipeline_records)} Phase 7 pipeline records")
    return _exp5_pipeline_records


def generate_exp5(provider: LLMProvider, config: str, question: dict) -> dict:
    """Exp 5: Synthesize Phase 7 pipeline output using different strategies."""
    pipeline_records = _load_pipeline_records()

    # Find matching pipeline record by query
    query = question["query"]
    matching = [r for r in pipeline_records if r["query"] == query]
    if not matching:
        return {
            "response": f"[ERROR: No pipeline record for: {query[:60]}...]",
            "word_count": 0,
        }

    rec = matching[0]
    raw_response = rec["response"]
    stages = rec.get("stages", [])

    stage_context = ""
    if stages:
        stage_context = f"\n\nThis response was produced through these stages: {', '.join(stages)}\n"

    user_content = (
        f"Original question: {query}\n"
        f"{stage_context}"
        f"\nFull response to synthesize:\n{raw_response}"
    )

    if config == "raw":
        response = raw_response
    elif config == "vedanta_synth":
        response = provider.call(VEDANTA_SYNTHESIS_SYSTEM, user_content, max_tokens=1500)
    elif config == "generic_synth":
        response = provider.call(GENERIC_SYNTHESIS_SYSTEM, user_content, max_tokens=1500)
    else:
        response = raw_response

    if response is None:
        response = "[ERROR: API call failed]"

    return {
        "response": response,
        "word_count": len(response.split()),
        "source_config": "full_pipeline",
    }


# -- Experiment Runner ----------------------------------------------------------

GENERATE_FNS = {
    "exp1": generate_exp1,
    "exp3": generate_exp3,
    "exp4": generate_exp4,
    "exp5": generate_exp5,
}


def get_questions(exp_key: str) -> list:
    """Get the question set for an experiment."""
    if exp_key == "exp4":
        from experiments.exp4v4_vaisheshika_vault import VAULT_QUESTIONS
        return VAULT_QUESTIONS
    elif exp_key == "exp5":
        # Exp 5 uses the same 30 transfer questions, filtered to those with pipeline records
        records = _load_pipeline_records()
        pipeline_queries = {r["query"] for r in records}
        return [q for q in TRANSFER_QUESTIONS if q["query"] in pipeline_queries]
    else:
        return TRANSFER_QUESTIONS


def run_cross_model_experiment(
    exp_key: str,
    model_key: str,
    limit: Optional[int] = None,
) -> list:
    """Run a single experiment on a single model.

    Args:
        exp_key: 'exp1', 'exp3', 'exp4', 'exp5'
        model_key: 'haiku', 'llama70b', 'qwen72b', 'mistral_large'
        limit: Optional question limit (for smoke testing)

    Returns:
        List of result records
    """
    exp = EXPERIMENTS[exp_key]
    provider = get_provider(model_key)
    generate_fn = GENERATE_FNS[exp_key]
    configs = exp["configs"]
    questions = get_questions(exp_key)

    if limit:
        questions = questions[:limit]

    results_dir = RESULTS_BASE / f"{exp_key}_{exp['name'].split()[0].lower()}"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{model_key}_results.jsonl"
    existing = load_existing_keys(results_path)

    total = len(questions) * len(configs)
    print(f"\n{'='*60}")
    print(f"CROSS-MODEL: {exp['name']}")
    print(f"  Model: {provider} ({provider.model_id})")
    print(f"  Questions: {len(questions)}, Configs: {len(configs)}")
    print(f"  Total generations: {total}")
    print(f"  Existing: {len(existing)} (will skip)")
    print(f"  Results: {results_path}")
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
            print(f"  [{done}/{total}] {config:<25} | "
                  f"{q.get('target_op', ''):<14} | "
                  f"{elapsed:.0f}s, ~{remaining:.0f}s left", flush=True)

            result = generate_fn(provider, config, q)
            result.update({
                "key": key,
                "query": q["query"],
                "target_op": q.get("target_op", ""),
                "domain": q.get("domain", ""),
                "config": config,
                "model": model_key,
                "provider": provider.provider_name,
                "model_id": provider.model_id,
            })

            append_jsonl(results_path, result)
            existing.add(key)
            results.append(result)

    elapsed = time.time() - t0
    print(f"\n  Complete: {len(results)} new generations in {elapsed:.0f}s")
    print(f"  Results: {results_path}")
    return results


# -- Judging -------------------------------------------------------------------

# Judge prompt (same as standard pairwise judge from judge.py)
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

Respond with ONLY a JSON object:
{
  "response_a": {"factual_accuracy": <1-5>, "reasoning_depth": <1-5>, "completeness": <1-5>, "calibration": <1-5>, "usefulness": <1-5>},
  "response_b": {"factual_accuracy": <1-5>, "reasoning_depth": <1-5>, "completeness": <1-5>, "calibration": <1-5>, "usefulness": <1-5>},
  "winner": "A" or "B" or "TIE",
  "reason": "<1 sentence explaining why>"
}"""

# Context-aware variant for Exp 4 (vault questions about private data)
CONTEXT_AWARE_JUDGE_SYSTEM = """You are an expert evaluator of reasoning quality in knowledge-based responses.

You will see a QUESTION and two RESPONSES (A and B). Both responses may have been given access to PRIVATE REFERENCE DOCUMENTS containing proprietary information that you do not have access to. If a response provides specific details about private projects, internal documents, or proprietary research, this is likely because it was given context from those documents — NOT because it is hallucinating.

IMPORTANT: Do NOT penalize a response for providing specific, detailed information about private/proprietary topics. Only flag as hallucination if details are internally inconsistent or contradictory.

Judge ONLY on reasoning quality:

1. FACTUAL_ACCURACY: Are claims internally consistent? Do specific details align with each other?
   1=contradictory claims, 3=mostly consistent, 5=fully coherent with specific verifiable details

2. REASONING_DEPTH: Does it go beyond surface to explain WHY and HOW?
   1=surface only, 3=some explanation, 5=genuine causal/mechanistic insight

3. COMPLETENESS: Does it address the question's key aspects?
   1=misses most aspects, 3=covers basics, 5=comprehensive coverage

4. CALIBRATION: Does it distinguish what it knows confidently from what it's uncertain about?
   1=overconfident throughout, 3=some hedging, 5=accurately signals certainty levels

5. USEFULNESS: Would someone learn something genuinely valuable from this?
   1=not helpful, 3=decent overview, 5=would significantly improve understanding

After scoring both, declare a WINNER or TIE.

Respond with ONLY a JSON object:
{
  "response_a": {"factual_accuracy": <1-5>, "reasoning_depth": <1-5>, "completeness": <1-5>, "calibration": <1-5>, "usefulness": <1-5>},
  "response_b": {"factual_accuracy": <1-5>, "reasoning_depth": <1-5>, "completeness": <1-5>, "calibration": <1-5>, "usefulness": <1-5>},
  "winner": "A" or "B" or "TIE",
  "reason": "<1 sentence explaining why>"
}"""


def run_cross_model_judging(
    exp_key: str,
    model_key: str,
    judge_model: str = "sonnet",
) -> dict:
    """Run pairwise judging for a cross-model experiment.

    Uses Claude Sonnet as judge for consistency across all models.

    Args:
        exp_key: 'exp1', 'exp3', 'exp4', 'exp5'
        model_key: model to judge results for
        judge_model: model to use as judge (default: sonnet)

    Returns:
        dict with win rates per comparison
    """
    exp = EXPERIMENTS[exp_key]
    judge = get_provider(judge_model)

    judge_system = CONTEXT_AWARE_JUDGE_SYSTEM if exp_key == "exp4" else PAIRWISE_SYSTEM

    exp_dir_name = f"{exp_key}_{exp['name'].split()[0].lower()}"
    results_dir = RESULTS_BASE / exp_dir_name
    results_path = results_dir / f"{model_key}_results.jsonl"
    judge_path = results_dir / f"judge_{model_key}.jsonl"

    if not results_path.exists():
        print(f"  No results for {model_key} in {exp_key}")
        return {}

    records = load_jsonl(results_path)
    existing = load_existing_keys(judge_path)

    # Group by query
    by_query: Dict[str, dict] = defaultdict(dict)
    for r in records:
        by_query[r["query"]][r["config"]] = r

    # Build comparison pairs: darshana vs generic (h2h), darshana vs baseline, generic vs baseline
    baseline = exp["baseline"]
    darshana = exp["darshana_config"]
    generic = exp["generic_config"]

    comparisons = [
        (generic, darshana, "h2h"),          # The key test
        (baseline, darshana, "vs_baseline"),  # Magnitude
        (baseline, generic, "vs_baseline"),   # Control check
    ]

    pairs = []
    for config_a, config_b, comp_type in comparisons:
        for query, configs in by_query.items():
            if config_a in configs and config_b in configs:
                pairs.append((query, config_a, config_b, comp_type))

    print(f"\n  Judging {exp_key} ({model_key}): {len(pairs)} pairs "
          f"({len(existing)} existing)")

    t0 = time.time()
    new_count = 0

    for query, config_a, config_b, comp_type in pairs:
        key = f"pairwise|{query[:60]}|{config_a}|{config_b}"
        if key in existing:
            continue

        resp_a = by_query[query][config_a]["response"]
        resp_b = by_query[query][config_b]["response"]

        # Randomize order to avoid position bias
        if random.random() < 0.5:
            user_content = (
                f"QUESTION: {query}\n\n"
                f"RESPONSE A:\n{resp_a}\n\n"
                f"RESPONSE B:\n{resp_b}"
            )
            order = (config_a, config_b)
        else:
            user_content = (
                f"QUESTION: {query}\n\n"
                f"RESPONSE A:\n{resp_b}\n\n"
                f"RESPONSE B:\n{resp_a}"
            )
            order = (config_b, config_a)

        judgment = judge.call_json(judge_system, user_content)
        if judgment is None:
            continue

        raw_winner = judgment.get("winner", "TIE")
        if raw_winner == "A":
            actual_winner = order[0]
        elif raw_winner == "B":
            actual_winner = order[1]
        else:
            actual_winner = "TIE"

        result = {
            "key": key,
            "type": "pairwise",
            "comparison": comp_type,
            "query": query,
            "config_a": config_a,
            "config_b": config_b,
            "order": order,
            "scores_a": judgment.get(
                f"response_{'a' if order[0] == config_a else 'b'}", {}
            ),
            "scores_b": judgment.get(
                f"response_{'b' if order[0] == config_a else 'a'}", {}
            ),
            "winner": actual_winner,
            "reason": judgment.get("reason", ""),
            "target_op": by_query[query].get(config_b, {}).get("target_op", ""),
            "domain": by_query[query].get(config_b, {}).get("domain", ""),
            "model": model_key,
            "judge_model": judge_model,
        }

        append_jsonl(judge_path, result)
        existing.add(key)
        new_count += 1

        if new_count % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{new_count} done] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Judging complete: {new_count} new in {elapsed:.0f}s")

    # Compute win rates
    return _compute_win_rates(judge_path, exp)


def _compute_win_rates(judge_path: Path, exp: dict) -> dict:
    """Compute and print win rates from judge results."""
    all_judgments = load_jsonl(judge_path)
    pairwise = [r for r in all_judgments if r.get("type") == "pairwise"]

    darshana = exp["darshana_config"]
    generic = exp["generic_config"]
    baseline = exp["baseline"]

    summary = {}

    # h2h: darshana vs generic
    h2h = [r for r in pairwise
           if r.get("config_a") == generic and r.get("config_b") == darshana]
    if h2h:
        wins = sum(1 for r in h2h if r["winner"] == darshana)
        total = len(h2h)
        summary["h2h"] = {
            "darshana_wins": wins,
            "generic_wins": sum(1 for r in h2h if r["winner"] == generic),
            "ties": sum(1 for r in h2h if r["winner"] == "TIE"),
            "total": total,
            "win_rate": wins / total if total else 0,
        }
        print(f"  h2h ({darshana} vs {generic}): "
              f"{wins}/{total} ({wins/total*100:.0f}%)")

    # vs baseline: darshana
    d_vs_b = [r for r in pairwise
              if r.get("config_a") == baseline and r.get("config_b") == darshana]
    if d_vs_b:
        wins = sum(1 for r in d_vs_b if r["winner"] == darshana)
        total = len(d_vs_b)
        summary["darshana_vs_baseline"] = {
            "wins": wins, "total": total,
            "win_rate": wins / total if total else 0,
        }
        print(f"  {darshana} vs {baseline}: {wins}/{total} ({wins/total*100:.0f}%)")

    # vs baseline: generic
    g_vs_b = [r for r in pairwise
              if r.get("config_a") == baseline and r.get("config_b") == generic]
    if g_vs_b:
        wins = sum(1 for r in g_vs_b if r["winner"] == generic)
        total = len(g_vs_b)
        summary["generic_vs_baseline"] = {
            "wins": wins, "total": total,
            "win_rate": wins / total if total else 0,
        }
        print(f"  {generic} vs {baseline}: {wins}/{total} ({wins/total*100:.0f}%)")

    return summary


# -- Analysis ------------------------------------------------------------------

def analyze_all():
    """Print cross-model comparison across all experiments and models."""
    print(f"\n{'='*70}")
    print("CROSS-MODEL VALIDATION RESULTS")
    print(f"{'='*70}")

    for exp_key, exp in EXPERIMENTS.items():
        exp_dir_name = f"{exp_key}_{exp['name'].split()[0].lower()}"
        exp_dir = RESULTS_BASE / exp_dir_name

        print(f"\n{exp['name']} (Sonnet baseline: {exp['original_h2h']})")
        print(f"  {'Model':<20} {'h2h (d vs g)':<18} {'d vs base':<15} {'g vs base':<15}")
        print(f"  {'─'*68}")

        model_h2h_rates = {}

        for model_key in ["sonnet"] + ALL_MODELS:
            judge_path = exp_dir / f"judge_{model_key}.jsonl"
            if not judge_path.exists():
                continue

            judgments = load_jsonl(judge_path)
            pairwise = [r for r in judgments if r.get("type") == "pairwise"]
            if not pairwise:
                continue

            darshana = exp["darshana_config"]
            generic = exp["generic_config"]
            baseline = exp["baseline"]

            # h2h
            h2h = [r for r in pairwise
                   if r.get("config_a") == generic and r.get("config_b") == darshana]
            h2h_str = "—"
            if h2h:
                wins = sum(1 for r in h2h if r["winner"] == darshana)
                total = len(h2h)
                rate = wins / total if total else 0
                h2h_str = f"{wins}/{total} ({rate*100:.0f}%)"
                model_h2h_rates[model_key] = rate

            # darshana vs baseline
            d_vs_b = [r for r in pairwise
                      if r.get("config_a") == baseline and r.get("config_b") == darshana]
            d_str = "—"
            if d_vs_b:
                wins = sum(1 for r in d_vs_b if r["winner"] == darshana)
                total = len(d_vs_b)
                d_str = f"{wins}/{total} ({wins/total*100:.0f}%)"

            # generic vs baseline
            g_vs_b = [r for r in pairwise
                      if r.get("config_a") == baseline and r.get("config_b") == generic]
            g_str = "—"
            if g_vs_b:
                wins = sum(1 for r in g_vs_b if r["winner"] == generic)
                total = len(g_vs_b)
                g_str = f"{wins}/{total} ({wins/total*100:.0f}%)"

            label = f"  {model_key:<20}"
            print(f"{label} {h2h_str:<18} {d_str:<15} {g_str:<15}")

        # Summary
        if model_h2h_rates:
            open_rates = [v for k, v in model_h2h_rates.items()
                          if k in VALIDATION_MODELS]
            if open_rates:
                avg = mean(open_rates)
                sonnet_rate = model_h2h_rates.get("sonnet", 0)
                delta = avg - sonnet_rate
                delta_str = f"+{delta*100:.0f}%" if delta >= 0 else f"{delta*100:.0f}%"
                print(f"\n  Open-source avg h2h: {avg*100:.0f}%  "
                      f"(Δ vs Sonnet: {delta_str})")
                valid = sum(1 for r in open_rates if r > 0.5)
                print(f"  Validates on {valid}/{len(open_rates)} open-source models")

    # Overall summary
    print(f"\n{'='*70}")
    print("TECHNIQUE SUMMARY")
    print(f"{'='*70}")

    for exp_key, exp in EXPERIMENTS.items():
        exp_dir_name = f"{exp_key}_{exp['name'].split()[0].lower()}"
        exp_dir = RESULTS_BASE / exp_dir_name

        valid_count = 0
        total_models = 0
        for model_key in VALIDATION_MODELS:
            judge_path = exp_dir / f"judge_{model_key}.jsonl"
            if not judge_path.exists():
                continue
            judgments = load_jsonl(judge_path)
            h2h = [r for r in judgments
                   if r.get("type") == "pairwise"
                   and r.get("config_a") == exp["generic_config"]
                   and r.get("config_b") == exp["darshana_config"]]
            if h2h:
                total_models += 1
                wins = sum(1 for r in h2h if r["winner"] == exp["darshana_config"])
                if wins / len(h2h) > 0.5:
                    valid_count += 1

        if total_models > 0:
            status = "VALIDATED" if valid_count == total_models else \
                     "PARTIAL" if valid_count > 0 else "FAILED"
            print(f"  {exp['name']:<35} {status} ({valid_count}/{total_models} models)")
        else:
            print(f"  {exp['name']:<35} NO DATA")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Model Validation for Darshana Experiments"
    )
    parser.add_argument("--exp", choices=["1", "3", "4", "5"],
                        help="Experiment number")
    parser.add_argument("--model", choices=ALL_MODELS,
                        help="Model to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit questions (smoke test)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: Haiku, 3 questions, all experiments")
    parser.add_argument("--local", action="store_true",
                        help="Run all experiments on all local MLX models")
    parser.add_argument("--judge", action="store_true",
                        help="Run pairwise judging (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--judge-model", default="sonnet",
                        help="Model to use as judge (default: sonnet)")
    parser.add_argument("--analyze", action="store_true",
                        help="Print analysis summary")
    args = parser.parse_args()

    if args.analyze:
        analyze_all()
        return

    # Determine which experiments to run
    if args.exp:
        exp_keys = [f"exp{args.exp}"]
    else:
        exp_keys = list(EXPERIMENTS.keys())

    # Determine which models to run
    if args.smoke:
        models = SMOKE_MODELS
        limit = 3
    elif args.local:
        models = LOCAL_MODELS
        limit = args.limit
    elif args.model:
        models = [args.model]
        limit = args.limit
    else:
        parser.print_help()
        return

    if args.judge:
        # Judging mode (needs ANTHROPIC_API_KEY for Sonnet as judge)
        for exp_key in exp_keys:
            for model_key in models:
                run_cross_model_judging(
                    exp_key, model_key,
                    judge_model=args.judge_model,
                )
        # Print analysis after judging
        analyze_all()
    else:
        # Generation mode
        for exp_key in exp_keys:
            for model_key in models:
                run_cross_model_experiment(exp_key, model_key, limit=limit)

        print(f"\n{'='*60}")
        print("NEXT STEPS:")
        if args.smoke:
            print(f"  1. Check results in {RESULTS_BASE}/")
            print(f"  2. Run local models: python {__file__} --local")
        else:
            print(f"  1. Judge results:    python {__file__} --judge --local")
            print(f"  2. Full analysis:    python {__file__} --analyze")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
