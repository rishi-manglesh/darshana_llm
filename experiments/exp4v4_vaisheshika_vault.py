"""Experiment 4 v4: Vaisheshika as Knowledge Graph on Real Vault Corpus

PRATIJNA (Thesis):
    7-category ontology (entity, properties, actions, categories,
    differentiators, relationships, absences) organizes proprietary
    knowledge better than generic entity-relation schema for RAG.

HETU (Reason):
    v3 proved 63% h2h on synthetic NovaTech corpus, but:
    - Sanskrit labels leaked into responses (judge penalized randomly)
    - Synthetic corpus was artificial
    - Judge treated context-based answers as "hallucinations"

    v4 fixes ALL THREE: English-only labels, real Obsidian vault corpus,
    context-aware judge prompt.

UDAHARANA (Evidence):
    - v2: 63% h2h on Wikipedia (wrong corpus — model already knows)
    - v3: 63% h2h on synthetic org (right idea, artificial data)
    - v3 analysis: factual 70%, causal 70%, gap 50% (Sanskrit labels)

UPANAYA (Application):
    Phase 1: Read real vault files as corpus (no generation needed)
    Phase 2: Extract with both ontologies (7-category vs generic)
    Phase 3: Generate answers (Sonnet, 4 configs x 30 questions)
    Phase 4: Judge with CONTEXT-AWARE prompt (fixes hallucination bias)

NIGAMANA (Success Criteria):
    - PROVEN if 7-category > generic by >10% h2h
    - Gap questions should improve now that Sanskrit labels are removed
    - Baseline judging should show KG configs >> no_context
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.vaisheshika_ontology import (
    extract_padarthas, extract_generic,
    build_padartha_graph, build_generic_graph,
    format_padartha_context, format_generic_context,
)
from experiments.utils import (
    MODELS, RESULTS_DIR,
    get_client, call_api, call_api_json, load_jsonl, append_jsonl,
    load_existing_keys, mean,
)

# -- Config -------------------------------------------------------------------

EXPERIMENT_NAME = "exp4v4_vaisheshika_vault"
GENERATION_MODEL = "sonnet"
EXTRACTION_MODEL = "haiku"

CONFIGS = [
    "no_context",
    "raw_context",
    "generic_kg",
    "padartha_kg",
]

# -- Vault Corpus (real files, not synthetic) ---------------------------------

VAULT_BASE = Path("/Users/rishimanglesh/Library/Mobile Documents/com~apple~CloudDocs/Personal/Projects")

# 8 files across 4 domains — genuinely proprietary content
VAULT_FILES = {
    "research_overview": VAULT_BASE / "Knowledge_AI/README.md",
    "darshana_evidence": VAULT_BASE / "Knowledge_AI/Research/Darshana_AI_Evidence.md",
    "honest_assessment": VAULT_BASE / "Knowledge_AI/Research/Experimental_Honest_Assessment.md",
    "dversi_thesis": VAULT_BASE / "Evidence_Tech/DVERSI/Category_Thesis.md",
    "competitive_analysis": VAULT_BASE / "Evidence_Tech/DVERSI/Competitive_Deep_Dive.md",
    "risk_register": VAULT_BASE / "Evidence_Tech/Constitution/Risk_Register.md",
    "book_outline": VAULT_BASE / "Thought_Leadership/Ideas/Darshana_for_AI_Book.md",
    "prediction_tracker": VAULT_BASE / "Jyotish/Validation/Prediction_Tracker.md",
}

# 30 questions about the vault — things Sonnet genuinely cannot know
VAULT_QUESTIONS = [
    # Factual (10) — need entities + properties
    {"query": "What win rates did the darshana_llm experiments achieve for each Darshana?",
     "target_op": "factual", "domain": "research_overview"},
    {"query": "What is the current evidence verdict for each of the 6 Darshanas in the evidence tracker?",
     "target_op": "factual", "domain": "darshana_evidence"},
    {"query": "What experiments did vedic_llm run across its 8 phases?",
     "target_op": "factual", "domain": "honest_assessment"},
    {"query": "What is DVERSI's core category thesis and philosophy?",
     "target_op": "factual", "domain": "dversi_thesis"},
    {"query": "Who are DVERSI's main competitors and what are their strengths?",
     "target_op": "factual", "domain": "competitive_analysis"},
    {"query": "What are the top risks in the tech evidence risk register?",
     "target_op": "factual", "domain": "risk_register"},
    {"query": "What chapters and structure are planned for the Darshana for AI book?",
     "target_op": "factual", "domain": "book_outline"},
    {"query": "What predictions have been made and what are their current validation statuses?",
     "target_op": "factual", "domain": "prediction_tracker"},
    {"query": "What model architectures were tested in the Knowledge AI research project?",
     "target_op": "factual", "domain": "research_overview"},
    {"query": "What scoring model does the tech evidence constitution use for beliefs?",
     "target_op": "factual", "domain": "risk_register"},

    # Causal (10) — need actions + relationships
    {"query": "Why did Mimamsa fail as a runtime stage but succeed as a rewriter?",
     "target_op": "causal", "domain": "darshana_evidence"},
    {"query": "Why was Vaisheshika redesigned from a judge framework to a knowledge graph?",
     "target_op": "causal", "domain": "honest_assessment"},
    {"query": "Why does DVERSI use a 'Solve First, Automate Later' approach instead of full automation?",
     "target_op": "causal", "domain": "dversi_thesis"},
    {"query": "What competitive dynamics are driving DVERSI's positioning strategy?",
     "target_op": "causal", "domain": "competitive_analysis"},
    {"query": "Why did the Vritti experiment scale better on frontier models than small models?",
     "target_op": "causal", "domain": "research_overview"},
    {"query": "What factors determine whether a risk in the register is high vs medium severity?",
     "target_op": "causal", "domain": "risk_register"},
    {"query": "Why is the Darshana-AI book positioned as the first of its kind?",
     "target_op": "causal", "domain": "book_outline"},
    {"query": "What caused the prediction accuracy rates to vary across different prediction types?",
     "target_op": "causal", "domain": "prediction_tracker"},
    {"query": "Why did darshana_llm succeed where vedic_llm failed in proving the Darshana thesis?",
     "target_op": "causal", "domain": "honest_assessment"},
    {"query": "What drove the decision to use Nyaya pramana routing instead of letting the model decide tool use?",
     "target_op": "causal", "domain": "darshana_evidence"},

    # Gap (10) — need absences/constraints
    {"query": "What Darshana experiments are still untested or have inconclusive results?",
     "target_op": "gap", "domain": "darshana_evidence"},
    {"query": "What evidence is still missing before the Darshana for AI book can be written?",
     "target_op": "gap", "domain": "book_outline"},
    {"query": "What risks in the register have no mitigation strategy yet?",
     "target_op": "gap", "domain": "risk_register"},
    {"query": "What competitive intelligence gaps exist in DVERSI's market analysis?",
     "target_op": "gap", "domain": "competitive_analysis"},
    {"query": "What predictions have not yet been validated or have unknown outcomes?",
     "target_op": "gap", "domain": "prediction_tracker"},
    {"query": "What methodological weaknesses does the honest assessment identify in the research?",
     "target_op": "gap", "domain": "honest_assessment"},
    {"query": "What areas of the tech thesis lack sufficient evidence?",
     "target_op": "gap", "domain": "risk_register"},
    {"query": "What DVERSI product capabilities are planned but not yet built?",
     "target_op": "gap", "domain": "dversi_thesis"},
    {"query": "What knowledge gaps exist in the research project's understanding of Samkhya and Yoga?",
     "target_op": "gap", "domain": "research_overview"},
    {"query": "What data or experiments would change the current evidence verdicts?",
     "target_op": "gap", "domain": "darshana_evidence"},
]


# -- Context-Aware Judge Prompt -----------------------------------------------

CONTEXT_AWARE_JUDGE_SYSTEM = """You are an expert evaluator of reasoning quality in knowledge-based responses.

You will see a QUESTION and two RESPONSES (A and B). Both responses may have been given access to PRIVATE REFERENCE DOCUMENTS containing proprietary information that you do not have access to. If a response provides specific details about private projects, internal documents, or proprietary research, this is likely because it was given context from those documents — NOT because it is hallucinating.

IMPORTANT: Do NOT penalize a response for providing specific, detailed information about private/proprietary topics. The question is about PRIVATE organizational knowledge — specific answers are expected and correct when based on provided context. Only flag as hallucination if details are internally inconsistent or contradictory.

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


# -- Corpus Management -------------------------------------------------------

def load_vault_corpus():
    """Read vault files as corpus documents.

    Returns:
        list of dicts with 'doc_type', 'content', 'word_count'
    """
    docs = []
    for doc_type, path in VAULT_FILES.items():
        if path.exists():
            content = path.read_text(encoding="utf-8")
            docs.append({
                "doc_type": doc_type,
                "content": content,
                "word_count": len(content.split()),
            })
        else:
            print(f"  [WARN] Missing vault file: {path}")
    return docs


# -- Graph Building -----------------------------------------------------------

def build_vault_graphs(client, corpus):
    """Extract and build both 7-category and generic graphs from vault corpus.

    Returns:
        (seven_cat_graph, generic_graph, raw_corpus_text)
    """
    extraction_model = MODELS[EXTRACTION_MODEL]

    # Build raw corpus text
    raw_text = "\n\n---\n\n".join(
        f"## {d['doc_type'].upper()}\n\n{d['content']}" for d in corpus
    )

    print(f"\n  Extracting knowledge from {len(corpus)} documents...")
    seven_cat_extractions = []
    generic_extractions = []

    for doc in corpus:
        print(f"    Extracting: {doc['doc_type']}...")
        p_ext = extract_padarthas(client, extraction_model, doc["content"])
        g_ext = extract_generic(client, extraction_model, doc["content"])
        seven_cat_extractions.append(p_ext)
        generic_extractions.append(g_ext)

    p_graph = build_padartha_graph(seven_cat_extractions)
    g_graph = build_generic_graph(generic_extractions)

    print(f"  7-category graph: {len(p_graph.nodes)} nodes, {len(p_graph.abhava_index)} absences")
    print(f"  Generic graph:    {len(g_graph.nodes)} nodes, {len(g_graph.abhava_index)} gaps")

    return p_graph, g_graph, raw_text


# -- Generation ---------------------------------------------------------------

_client = None
_p_graph = None
_g_graph = None
_raw_text = None
_corpus = None


def ensure_setup():
    """Lazy initialization of client, corpus, and graphs."""
    global _client, _p_graph, _g_graph, _raw_text, _corpus

    if _client is None:
        _client = get_client()

    if _corpus is None:
        _corpus = load_vault_corpus()
        print(f"  Vault corpus: {len(_corpus)} docs, {sum(d['word_count'] for d in _corpus)} words")

    if _p_graph is None:
        _p_graph, _g_graph, _raw_text = build_vault_graphs(_client, _corpus)


def generate_fn(config, question):
    """Generate a response for a given config and vault question."""
    ensure_setup()

    model = MODELS[GENERATION_MODEL]
    query = question["query"]
    query_type = question["target_op"]

    retrieval_type_map = {
        "factual": "discrimination",
        "causal": "force",
        "gap": "constraint",
    }
    retrieval_type = retrieval_type_map.get(query_type, "general")

    if config == "no_context":
        response = call_api(_client, model, "", query)
        context_used = ""

    elif config == "raw_context":
        domain = question["domain"]
        relevant_docs = [d for d in _corpus if d["doc_type"] == domain]
        if not relevant_docs:
            relevant_docs = _corpus[:2]
        context = "\n\n".join(d["content"] for d in relevant_docs)[:4000]
        prompt = (
            f"You are answering questions based on private project documents.\n\n"
            f"## Documents\n{context}\n\n"
            f"## Question\n{query}\n\n"
            f"Answer based on what the documents say. If the documents "
            f"don't contain the answer, say so."
        )
        response = call_api(_client, model, "", prompt, max_tokens=1024)
        context_used = f"raw ({len(context)} chars)"

    elif config == "generic_kg":
        context = format_generic_context(_g_graph, query, retrieval_type)
        prompt = (
            f"You are answering questions based on organized project knowledge.\n\n"
            f"{context}\n\n"
            f"## Question\n{query}\n\n"
            f"Answer based on the provided knowledge. If specific information "
            f"is missing from the knowledge base, note the gap."
        )
        response = call_api(_client, model, "", prompt, max_tokens=1024)
        context_used = f"generic_kg ({len(_g_graph.nodes)} nodes)"

    elif config == "padartha_kg":
        context = format_padartha_context(_p_graph, query, retrieval_type)
        prompt = (
            f"You are answering questions based on organized project knowledge.\n\n"
            f"{context}\n\n"
            f"## Question\n{query}\n\n"
            f"Answer based on the provided knowledge. If specific information "
            f"is missing from the knowledge base, note the gap."
        )
        response = call_api(_client, model, "", prompt, max_tokens=1024)
        context_used = f"seven_cat_kg ({len(_p_graph.nodes)} nodes, {len(_p_graph.abhava_index)} absences)"

    else:
        response = f"Unknown config: {config}"
        context_used = ""

    if response is None:
        response = "[API ERROR]"

    return {
        "response": response,
        "word_count": len(response.split()),
        "context_type": context_used,
        "metrics": {
            "config": config,
            "query_type": query_type,
            "domain": question["domain"],
        },
    }


# -- Context-Aware Judging ---------------------------------------------------

def judge_pairwise_context_aware(client, model, query, response_a, response_b,
                                  label_a="baseline", label_b="experimental"):
    """Pairwise judgment with context-aware prompt (doesn't penalize for using context)."""
    import random

    user_content = f"QUESTION: {{query}}\n\nRESPONSE A:\n{{response_a}}\n\nRESPONSE B:\n{{response_b}}"

    if random.random() < 0.5:
        user_content = user_content.format(
            query=query, response_a=response_a, response_b=response_b
        )
        order = (label_a, label_b)
    else:
        user_content = user_content.format(
            query=query, response_a=response_b, response_b=response_a
        )
        order = (label_b, label_a)

    judgment = call_api_json(client, model, CONTEXT_AWARE_JUDGE_SYSTEM, user_content)
    if judgment is None:
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


def run_vault_judging(baseline, experimental_configs, judge_model="haiku"):
    """Run context-aware pairwise judging."""
    results_dir = RESULTS_DIR / EXPERIMENT_NAME
    results_path = results_dir / "results.jsonl"
    judge_path = results_dir / "judge.jsonl"

    client = get_client()
    model = MODELS[judge_model]

    records = load_jsonl(results_path)
    existing = load_existing_keys(judge_path)

    by_query = defaultdict(dict)
    for r in records:
        by_query[r["query"]][r["config"]] = r

    pairs = []
    for query, configs in by_query.items():
        if baseline not in configs:
            continue
        for exp_config in experimental_configs:
            if exp_config in configs:
                pairs.append((query, baseline, exp_config))

    print(f"\n  Judging {EXPERIMENT_NAME}: {len(pairs)} pairs ({len(existing)} existing)")
    t0 = time.time()
    new_count = 0

    for query, config_a, config_b in pairs:
        key = f"pairwise|{query[:60]}|{config_a}|{config_b}"
        if key in existing:
            continue

        resp_a = by_query[query][config_a]["response"]
        resp_b = by_query[query][config_b]["response"]

        result = judge_pairwise_context_aware(
            client, model, query, resp_a, resp_b, config_a, config_b
        )
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
        cr = [r for r in pairwise if r.get("config_b") == config
              and r.get("config_a") == baseline]
        if not cr:
            continue
        wins = sum(1 for r in cr if r["winner"] == config)
        total = len(cr)
        summary[config] = {"wins": wins, "total": total}

    elapsed = time.time() - t0
    print(f"  Judging complete: {new_count} new in {elapsed:.0f}s")
    for config, stats in summary.items():
        print(f"    {config}: {stats['wins']}/{stats['total']} wins "
              f"({stats['wins']/stats['total']*100:.0f}%)")

    return summary


# -- Experiment Runner --------------------------------------------------------

def run_vault_experiment(limit=None):
    """Run the experiment using VAULT_QUESTIONS."""
    results_dir = RESULTS_DIR / EXPERIMENT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"

    existing = load_existing_keys(results_path)
    questions = VAULT_QUESTIONS[:limit] if limit else VAULT_QUESTIONS
    total = len(questions) * len(CONFIGS)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"  Questions: {len(questions)}, Configs: {len(CONFIGS)}")
    print(f"  Total generations: {total}")
    print(f"  Existing: {len(existing)} (will skip)")
    print(f"{'='*60}\n")

    done = len(existing)
    t0 = time.time()
    results = []

    for qi, q in enumerate(questions):
        for config in CONFIGS:
            key = f"{config}|{q['query'][:60]}"
            if key in existing:
                continue

            done += 1
            elapsed = time.time() - t0
            rate = elapsed / (done - len(existing)) if done > len(existing) else 0
            remaining = rate * (total - done)
            print(f"  [{done}/{total}] {config:<25} | {q['target_op']:<10} | "
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
    return results


# -- Analysis ------------------------------------------------------------------

def analyze_results():
    """Print analysis of experiment results."""
    results_path = RESULTS_DIR / EXPERIMENT_NAME / "results.jsonl"
    records = load_jsonl(results_path)
    if not records:
        print("No results found.")
        return

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {EXPERIMENT_NAME}")
    print(f"  Total records: {len(records)}")
    print(f"{'='*60}\n")

    by_config = defaultdict(list)
    for r in records:
        by_config[r["config"]].append(r)

    print("Word counts by config:")
    for config in CONFIGS:
        recs = by_config.get(config, [])
        if recs:
            wc = [r["word_count"] for r in recs]
            print(f"  {config:<20}: mean={mean(wc):.0f}, min={min(wc)}, max={max(wc)}")

    print("\nWord counts by query type x config:")
    for qt in ["factual", "causal", "gap"]:
        print(f"\n  {qt.upper()}:")
        for config in CONFIGS:
            recs = [r for r in by_config.get(config, []) if r.get("target_op") == qt]
            if recs:
                wc = [r["word_count"] for r in recs]
                print(f"    {config:<20}: mean={mean(wc):.0f} ({len(recs)} questions)")


def analyze_judging():
    """Analyze judging results with query type breakdown."""
    judge_path = RESULTS_DIR / EXPERIMENT_NAME / "judge.jsonl"
    judgments = load_jsonl(judge_path)
    pairwise = [r for r in judgments if r.get("type") == "pairwise"]
    if not pairwise:
        print("No judging results found.")
        return

    print(f"\n{'='*60}")
    print(f"JUDGING ANALYSIS: {EXPERIMENT_NAME}")
    print(f"  Total judgments: {len(pairwise)}")
    print(f"{'='*60}\n")

    # Vs baseline
    for config_b in ["raw_context", "generic_kg", "padartha_kg"]:
        cr = [r for r in pairwise if r.get("config_b") == config_b
              and r.get("config_a") == "no_context"]
        if not cr:
            continue
        wins = sum(1 for r in cr if r["winner"] == config_b)
        total = len(cr)
        print(f"  {config_b} vs no_context: {wins}/{total} wins ({wins/total*100:.0f}%)")

    # Head-to-head
    h2h = [r for r in pairwise
           if (r.get("config_a") == "generic_kg" and r.get("config_b") == "padartha_kg")
           or (r.get("config_a") == "padartha_kg" and r.get("config_b") == "generic_kg")]

    if h2h:
        print("\n--- HEAD-TO-HEAD: 7-category vs generic ---")
        p_wins = sum(1 for r in h2h if r["winner"] == "padartha_kg")
        g_wins = sum(1 for r in h2h if r["winner"] == "generic_kg")
        ties = sum(1 for r in h2h if r["winner"] == "TIE")
        total = len(h2h)
        print(f"  7-category:  {p_wins}/{total} wins ({p_wins/total*100:.0f}%)")
        print(f"  generic:     {g_wins}/{total} wins ({g_wins/total*100:.0f}%)")
        print(f"  ties:        {ties}/{total}")

        print("\n  By query type:")
        for qt in ["factual", "causal", "gap"]:
            qt_recs = [r for r in h2h if r.get("target_op") == qt]
            if not qt_recs:
                continue
            p_w = sum(1 for r in qt_recs if r["winner"] == "padartha_kg")
            g_w = sum(1 for r in qt_recs if r["winner"] == "generic_kg")
            t = sum(1 for r in qt_recs if r["winner"] == "TIE")
            total_qt = len(qt_recs)
            print(f"    {qt:<15}: 7-cat {p_w}/{total_qt} ({p_w/total_qt*100:.0f}%) | "
                  f"generic {g_w}/{total_qt} ({g_w/total_qt*100:.0f}%) | ties {t}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 v4: 7-category ontology on real vault corpus"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit questions (for smoke testing)")
    parser.add_argument("--judge", action="store_true",
                        help="Run context-aware judging (vs no_context baseline)")
    parser.add_argument("--judge-h2h", action="store_true",
                        help="Run head-to-head: 7-category vs generic")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
        analyze_judging()
        return

    if args.judge:
        print("\n  Phase 1: Context-aware judging vs no_context baseline...")
        run_vault_judging(
            baseline="no_context",
            experimental_configs=["raw_context", "generic_kg", "padartha_kg"],
            judge_model=args.judge_model,
        )
        analyze_judging()
        return

    if args.judge_h2h:
        print("\n  Head-to-head: 7-category vs generic...")
        run_vault_judging(
            baseline="generic_kg",
            experimental_configs=["padartha_kg"],
            judge_model=args.judge_model,
        )
        analyze_judging()
        return

    # Generation phase
    results = run_vault_experiment(limit=args.limit)
    analyze_results()

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"  1. Judge vs baseline:  python {__file__} --judge")
    print(f"  2. Head-to-head:       python {__file__} --judge-h2h")
    print(f"  3. Full analysis:      python {__file__} --analyze")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
