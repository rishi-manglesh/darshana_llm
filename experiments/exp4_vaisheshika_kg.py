"""Experiment 4 (Redesigned): Vaisheshika Knowledge Graph Ontology

PRATIJNA (Thesis):
    Vaisheshika's 7 Padarthas, used as a knowledge graph ontology, produce
    better-organized context for RAG than a generic entity-relation ontology.

HETU (Reason):
    Padarthas are an ontology of WHAT EXISTS — substance, quality, action,
    universality, particularity, inherence, and ABSENCE. Standard KG schemas
    lack explicit absence tracking (Abhava), which is critical for constraint
    questions ("what limits X?").

UDAHARANA (Evidence):
    - Exp 4 v1 (judge framework) was INCONCLUSIVE — padarthas are ontology,
      not evaluation criteria.
    - Vaisheshika's Abhava (4 types of non-existence) has no equivalent in
      standard entity-relation schemas.
    - RAG systems that organize context well produce better answers than raw
      text injection (well-established in RAG literature).

UPANAYA (Application):
    4 configs × 30 questions = 120 generations:
    - no_context: Sonnet answers with no external context (baseline)
    - raw_context: Sonnet answers with raw Wikipedia text
    - generic_kg: Sonnet answers with generic-ontology-organized context
    - padartha_kg: Sonnet answers with padartha-organized context

    Both KG configs use the SAME extraction model (Haiku) and SAME source text.
    The only difference is the ontology schema used for extraction and retrieval.

NIGAMANA (Success Criteria):
    - PROVEN if padartha_kg > generic_kg by >10% win rate in pairwise judging
    - Bonus: constraint questions (where Abhava shines) should show largest
      advantage for padartha_kg
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Setup imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.vaisheshika_ontology import (
    extract_padarthas, extract_generic,
    build_padartha_graph, build_generic_graph,
    format_padartha_context, format_generic_context,
)
from darshana.search import search_wikipedia
from experiments.utils import (
    MODELS, RESULTS_DIR, TRANSFER_QUESTIONS,
    get_client, call_api, run_experiment, load_jsonl,
    append_jsonl, load_existing_keys, mean,
)
from experiments.judge import run_pairwise_judging

# -- Config -------------------------------------------------------------------

EXPERIMENT_NAME = "exp4_vaisheshika_kg"
GENERATION_MODEL = "sonnet"
EXTRACTION_MODEL = "haiku"

CONFIGS = [
    "no_context",
    "raw_context",
    "generic_kg",
    "padartha_kg",
]

# Domain-to-search-queries mapping for corpus building
DOMAIN_CORPUS = {
    "economics": [
        "macroeconomics fiscal monetary policy",
        "economic growth development factors",
        "market economics supply demand",
    ],
    "biology": [
        "cell biology mitosis meiosis",
        "evolution natural selection",
        "human immune system",
    ],
    "everyday": [
        "personal finance investing",
        "education learning methods",
        "urban planning housing",
    ],
}


# -- Corpus & Graph Building ---------------------------------------------------

def fetch_corpus(domain):
    """Fetch Wikipedia articles for a domain.

    Returns:
        list of text strings (article summaries)
    """
    queries = DOMAIN_CORPUS.get(domain, DOMAIN_CORPUS["everyday"])
    texts = []
    for q in queries:
        result = search_wikipedia(q, max_results=2, sentences=10)
        if result and "failed" not in result.lower():
            texts.append(result)
    return texts


def build_graphs_for_domain(client, domain, corpus_cache, graph_cache):
    """Build both Padartha and Generic graphs for a domain.

    Uses caching to avoid re-extracting for the same domain.

    Returns:
        (padartha_graph, generic_graph, raw_corpus_text)
    """
    if domain in graph_cache:
        return graph_cache[domain]

    # Fetch corpus
    if domain not in corpus_cache:
        corpus_cache[domain] = fetch_corpus(domain)
    corpus = corpus_cache[domain]
    raw_text = "\n\n---\n\n".join(corpus)

    extraction_model = MODELS[EXTRACTION_MODEL]

    # Extract with both schemas
    padartha_extractions = []
    generic_extractions = []
    for text in corpus:
        p_ext = extract_padarthas(client, extraction_model, text)
        g_ext = extract_generic(client, extraction_model, text)
        padartha_extractions.append(p_ext)
        generic_extractions.append(g_ext)

    # Build graphs
    p_graph = build_padartha_graph(padartha_extractions)
    g_graph = build_generic_graph(generic_extractions)

    result = (p_graph, g_graph, raw_text)
    graph_cache[domain] = result
    return result


# -- Generation ---------------------------------------------------------------

# Shared state for caching across generate_fn calls
_client = None
_corpus_cache = {}
_graph_cache = {}


def generate_fn(config, question):
    """Generate a response for a given config and question.

    Args:
        config: one of CONFIGS
        question: dict with 'query', 'target_op', 'domain'

    Returns:
        dict with 'response', 'word_count', 'context_type', 'metrics'
    """
    global _client, _corpus_cache, _graph_cache

    if _client is None:
        _client = get_client()

    client = _client
    model = MODELS[GENERATION_MODEL]
    query = question["query"]
    domain = question["domain"]
    query_type = question["target_op"]

    if config == "no_context":
        response = call_api(client, model, "", query)
        context_used = ""

    elif config == "raw_context":
        p_graph, g_graph, raw_text = build_graphs_for_domain(
            client, domain, _corpus_cache, _graph_cache
        )
        # Inject raw text as context (truncated to fit)
        context = raw_text[:3000]
        prompt = (
            f"Use the following reference material to help answer the question.\n\n"
            f"## Reference Material\n{context}\n\n"
            f"## Question\n{query}"
        )
        response = call_api(client, model, "", prompt, max_tokens=1024)
        context_used = f"raw ({len(context)} chars)"

    elif config == "generic_kg":
        p_graph, g_graph, raw_text = build_graphs_for_domain(
            client, domain, _corpus_cache, _graph_cache
        )
        context = format_generic_context(g_graph, query, query_type)
        prompt = f"{context}\n\n## Question\n{query}"
        response = call_api(client, model, "", prompt, max_tokens=1024)
        context_used = f"generic_kg ({len(g_graph.nodes)} nodes)"

    elif config == "padartha_kg":
        p_graph, g_graph, raw_text = build_graphs_for_domain(
            client, domain, _corpus_cache, _graph_cache
        )
        context = format_padartha_context(p_graph, query, query_type)
        prompt = f"{context}\n\n## Question\n{query}"
        response = call_api(client, model, "", prompt, max_tokens=1024)
        context_used = f"padartha_kg ({len(p_graph.nodes)} nodes, {len(p_graph.abhava_index)} absences)"

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
            "domain": domain,
        },
    }


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

    # Word count by config
    by_config = defaultdict(list)
    for r in records:
        by_config[r["config"]].append(r)

    print("Word counts by config:")
    for config in CONFIGS:
        recs = by_config.get(config, [])
        if recs:
            wc = [r["word_count"] for r in recs]
            print(f"  {config:<20}: mean={mean(wc):.0f}, min={min(wc)}, max={max(wc)}")

    # Word count by query type × config
    print("\nWord counts by query type:")
    for qt in ["discrimination", "force", "constraint"]:
        print(f"\n  {qt.upper()}:")
        for config in CONFIGS:
            recs = [r for r in by_config.get(config, []) if r.get("target_op") == qt]
            if recs:
                wc = [r["word_count"] for r in recs]
                print(f"    {config:<20}: mean={mean(wc):.0f} ({len(recs)} questions)")

    # Context info
    print("\nContext types used:")
    for config in CONFIGS:
        recs = by_config.get(config, [])
        if recs:
            ctx = recs[0].get("context_type", "none")
            print(f"  {config:<20}: {ctx}")


def analyze_judging():
    """Analyze judging results with focus on query type breakdown."""
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

    # Overall win rates for each experimental config vs baseline
    for config_b in ["raw_context", "generic_kg", "padartha_kg"]:
        cr = [r for r in pairwise if r.get("config_b") == config_b]
        if not cr:
            continue
        wins = sum(1 for r in cr if r["winner"] == config_b)
        losses = sum(1 for r in cr if r["winner"] != config_b and r["winner"] != "TIE")
        ties = sum(1 for r in cr if r["winner"] == "TIE")
        total = len(cr)
        print(f"  {config_b} vs no_context: {wins}/{total} wins ({wins/total*100:.0f}%)")

    # Head-to-head: padartha_kg vs generic_kg
    print("\n--- HEAD-TO-HEAD: padartha_kg vs generic_kg ---")
    h2h = [r for r in pairwise
           if (r.get("config_a") == "generic_kg" and r.get("config_b") == "padartha_kg")
           or (r.get("config_a") == "padartha_kg" and r.get("config_b") == "generic_kg")]

    if not h2h:
        # May need to look at specific pairing in judge results
        print("  No direct head-to-head found. Run --judge-h2h for direct comparison.")
    else:
        p_wins = sum(1 for r in h2h if r["winner"] == "padartha_kg")
        g_wins = sum(1 for r in h2h if r["winner"] == "generic_kg")
        ties = sum(1 for r in h2h if r["winner"] == "TIE")
        total = len(h2h)
        print(f"  padartha_kg: {p_wins}/{total} wins ({p_wins/total*100:.0f}%)")
        print(f"  generic_kg:  {g_wins}/{total} wins ({g_wins/total*100:.0f}%)")
        print(f"  ties:        {ties}/{total}")

        # Breakdown by query type
        print("\n  By query type:")
        for qt in ["discrimination", "force", "constraint"]:
            qt_recs = [r for r in h2h if r.get("target_op") == qt]
            if not qt_recs:
                continue
            p_w = sum(1 for r in qt_recs if r["winner"] == "padartha_kg")
            g_w = sum(1 for r in qt_recs if r["winner"] == "generic_kg")
            t = sum(1 for r in qt_recs if r["winner"] == "TIE")
            total_qt = len(qt_recs)
            print(f"    {qt:<15}: padartha {p_w}/{total_qt} ({p_w/total_qt*100:.0f}%) | "
                  f"generic {g_w}/{total_qt} ({g_w/total_qt*100:.0f}%) | ties {t}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 (Redesigned): Vaisheshika Knowledge Graph Ontology"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit questions (for smoke testing)")
    parser.add_argument("--judge", action="store_true",
                        help="Run pairwise judging (vs no_context baseline)")
    parser.add_argument("--judge-h2h", action="store_true",
                        help="Run head-to-head: padartha_kg vs generic_kg")
    parser.add_argument("--judge-model", choices=["haiku", "sonnet"], default="haiku",
                        help="Model for judging")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
        analyze_judging()
        return

    if args.judge:
        print("\n  Phase 1: Judging vs no_context baseline...")
        run_pairwise_judging(
            EXPERIMENT_NAME,
            baseline_config="no_context",
            experimental_configs=["raw_context", "generic_kg", "padartha_kg"],
            judge_model=args.judge_model,
        )
        analyze_judging()
        return

    if args.judge_h2h:
        print("\n  Head-to-head: padartha_kg vs generic_kg...")
        run_pairwise_judging(
            EXPERIMENT_NAME,
            baseline_config="generic_kg",
            experimental_configs=["padartha_kg"],
            judge_model=args.judge_model,
        )
        analyze_judging()
        return

    # Generation phase
    results = run_experiment(
        name=EXPERIMENT_NAME,
        configs=CONFIGS,
        generate_fn=generate_fn,
        limit=args.limit,
    )

    analyze_results()

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"  1. Judge vs baseline:  python {__file__} --judge")
    print(f"  2. Head-to-head:       python {__file__} --judge-h2h")
    print(f"  3. Full analysis:      python {__file__} --analyze")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
