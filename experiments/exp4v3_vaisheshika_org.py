"""Experiment 4 v3: Vaisheshika as Organizational Knowledge Graph

PRATIJNA (Thesis):
    Vaisheshika's 7 Padarthas, used as a knowledge graph ontology for
    PROPRIETARY organizational knowledge, produce better RAG answers than
    a generic entity-relation ontology — especially for gap/constraint
    queries where Abhava (absence tracking) has no generic equivalent.

HETU (Reason):
    v2 tested on Wikipedia (model already knows) — RAG was pointless.
    This v3 uses a SYNTHETIC PROPRIETARY CORPUS: internal docs, meeting
    notes, and org-specific knowledge that Sonnet genuinely lacks.
    The model MUST use the retrieved context to answer correctly.

    Abhava value: when the org's knowledge has gaps, the model should
    surface them ("we have no data on X") rather than hallucinating.

UDAHARANA (Evidence):
    - v1 (judge framework): INCONCLUSIVE — padarthas are ontology, not eval
    - v2 (Wikipedia KG): PARTIAL — 63% h2h, but wrong corpus. Abhava
      extracted source-text coverage gaps, not real constraints.
    - RAG literature: organized context > raw text (well-established)

UPANAYA (Application):
    Phase 1: Generate synthetic org corpus (Haiku creates "internal docs")
    Phase 2: Extract with both ontologies (Padartha vs Generic)
    Phase 3: Generate answers (Sonnet, 4 configs × 30 questions)
    Phase 4: Judge (pairwise, including h2h)

    4 configs:
    - no_context: Sonnet answers with no org context (baseline — should FAIL)
    - raw_context: Sonnet answers with raw org document text
    - generic_kg: Sonnet answers with generic-entity-organized org context
    - padartha_kg: Sonnet answers with padartha-organized org context

    3 question types (10 each):
    - factual: "What is TechCorp's policy on X?" (needs Dravya/Guna)
    - causal: "Why did TechCorp choose approach X?" (needs Karma/Samavaya)
    - gap: "What doesn't TechCorp know about X?" (needs Abhava — the killer)

NIGAMANA (Success Criteria):
    - PROVEN if padartha_kg > generic_kg by >10% h2h win rate
    - no_context should lose badly (model doesn't have org knowledge)
    - Gap questions should show largest Abhava advantage
    - Bonus: padartha_kg surfaces genuine knowledge gaps that generic_kg misses
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
    get_client, call_api, call_api_json, run_experiment,
    load_jsonl, append_jsonl, load_existing_keys, mean,
)
from experiments.judge import run_pairwise_judging

# -- Config -------------------------------------------------------------------

EXPERIMENT_NAME = "exp4v3_vaisheshika_org"
GENERATION_MODEL = "sonnet"
EXTRACTION_MODEL = "haiku"
CORPUS_MODEL = "haiku"

CONFIGS = [
    "no_context",
    "raw_context",
    "generic_kg",
    "padartha_kg",
]

# -- Synthetic Org Corpus Generation ------------------------------------------

CORPUS_GENERATION_SYSTEM = """You are generating realistic INTERNAL COMPANY DOCUMENTS for a fictional company called "NovaTech Solutions" — a mid-size B2B SaaS company (500 employees, $80M ARR) that sells an AI-powered supply chain optimization platform.

Generate a realistic internal document of the specified type. Include:
- Specific details (names, dates, metrics, decisions, trade-offs)
- Internal jargon and references to internal systems/teams
- Information that would NOT be available on the internet
- Deliberate GAPS in knowledge — things the company doesn't know yet, hasn't decided, or is investigating

The document should feel like something you'd find on an internal wiki or in meeting notes — not a polished external document.

IMPORTANT: Include realistic gaps, unknowns, and pending decisions. Real org knowledge is always incomplete."""

CORPUS_PROMPTS = {
    "architecture": """Generate an internal architecture decision record (ADR) for NovaTech's "Project Meridian" — migrating their monolithic supply chain optimizer to a microservices architecture. Include:
- The decision and rationale
- Services identified and their responsibilities
- What the team chose NOT to do and why
- Known risks and unknowns
- Performance targets they haven't validated yet
- Integration points with legacy systems that are unclear
~800 words.""",

    "customer_analysis": """Generate an internal quarterly customer analysis document for NovaTech's enterprise segment. Include:
- Customer retention metrics and churn analysis
- Specific customer accounts (fictional names) with details
- Feature requests and their business impact estimates
- Competitive losses and wins with analysis
- Gaps in customer understanding they're trying to close
- Pricing experiments they're considering but haven't run
~800 words.""",

    "ml_pipeline": """Generate internal documentation for NovaTech's ML pipeline that powers their demand forecasting engine. Include:
- Model architecture and training process
- Data sources and their reliability ratings
- Known model weaknesses and failure modes
- A/B test results from recent model updates
- Data quality issues they haven't resolved
- Accuracy metrics they're unsure how to measure
~800 words.""",

    "incident_review": """Generate a post-incident review (PIR) for a major outage at NovaTech. Include:
- Timeline of the incident
- Root cause analysis with contributing factors
- What the team still doesn't understand about why it happened
- Remediation steps (completed and pending)
- Process gaps identified
- Monitoring blind spots they discovered
~800 words.""",

    "strategy": """Generate an internal strategy memo for NovaTech's expansion into the European market. Include:
- Market sizing estimates with confidence levels
- Regulatory requirements (GDPR, data residency) and gaps in compliance
- Competitive landscape assessment with unknowns
- Go-to-market options being debated
- Resource requirements and trade-offs
- Risks they've identified but haven't mitigated
~800 words.""",

    "hiring": """Generate an internal hiring plan document for NovaTech's engineering org. Include:
- Current team structure and skill gaps
- Roles they need to fill with priority levels
- Compensation benchmarking data and uncertainties
- Interview process issues they've identified
- Diversity goals and where they're falling short
- Remote vs. office decisions still being debated
~800 words.""",
}

# 30 questions about NovaTech — things the model CAN'T know without the corpus
ORG_QUESTIONS = [
    # Factual (10) — need Dravya (entities) + Guna (attributes)
    {"query": "What microservices did NovaTech identify for Project Meridian?",
     "target_op": "factual", "domain": "architecture"},
    {"query": "What is NovaTech's current customer retention rate for enterprise accounts?",
     "target_op": "factual", "domain": "customer"},
    {"query": "What data sources does NovaTech's demand forecasting model use?",
     "target_op": "factual", "domain": "ml_pipeline"},
    {"query": "What was the root cause of NovaTech's recent major outage?",
     "target_op": "factual", "domain": "incident"},
    {"query": "What is NovaTech's estimated TAM for the European market?",
     "target_op": "factual", "domain": "strategy"},
    {"query": "What engineering roles is NovaTech currently hiring for?",
     "target_op": "factual", "domain": "hiring"},
    {"query": "What model architecture does NovaTech use for demand forecasting?",
     "target_op": "factual", "domain": "ml_pipeline"},
    {"query": "Which enterprise customers has NovaTech lost recently and to whom?",
     "target_op": "factual", "domain": "customer"},
    {"query": "What are the performance targets for Project Meridian's microservices?",
     "target_op": "factual", "domain": "architecture"},
    {"query": "What regulatory requirements has NovaTech identified for EU expansion?",
     "target_op": "factual", "domain": "strategy"},

    # Causal (10) — need Karma (actions) + Samavaya (inherent relationships)
    {"query": "Why did NovaTech choose microservices over a modular monolith for Project Meridian?",
     "target_op": "causal", "domain": "architecture"},
    {"query": "What factors are driving enterprise customer churn at NovaTech?",
     "target_op": "causal", "domain": "customer"},
    {"query": "Why does NovaTech's forecasting model perform worse on seasonal products?",
     "target_op": "causal", "domain": "ml_pipeline"},
    {"query": "What contributing factors led to NovaTech's outage beyond the root cause?",
     "target_op": "causal", "domain": "incident"},
    {"query": "Why is NovaTech prioritizing the UK market for European expansion?",
     "target_op": "causal", "domain": "strategy"},
    {"query": "What's causing NovaTech's engineering hiring to take longer than planned?",
     "target_op": "causal", "domain": "hiring"},
    {"query": "Why did NovaTech's recent A/B test show mixed results for the new model?",
     "target_op": "causal", "domain": "ml_pipeline"},
    {"query": "What drove the decision to keep NovaTech's legacy batch processing system?",
     "target_op": "causal", "domain": "architecture"},
    {"query": "Why are NovaTech's enterprise customers requesting real-time dashboards?",
     "target_op": "causal", "domain": "customer"},
    {"query": "What process gaps at NovaTech contributed to the incident response delays?",
     "target_op": "causal", "domain": "incident"},

    # Gap (10) — need Abhava (what's absent/unknown/missing)
    {"query": "What doesn't NovaTech know about their microservices migration risks?",
     "target_op": "gap", "domain": "architecture"},
    {"query": "What gaps exist in NovaTech's understanding of enterprise customer needs?",
     "target_op": "gap", "domain": "customer"},
    {"query": "What data quality issues has NovaTech not yet resolved in their ML pipeline?",
     "target_op": "gap", "domain": "ml_pipeline"},
    {"query": "What monitoring blind spots did NovaTech discover after the outage?",
     "target_op": "gap", "domain": "incident"},
    {"query": "What unknowns remain in NovaTech's European market assessment?",
     "target_op": "gap", "domain": "strategy"},
    {"query": "Where is NovaTech falling short on engineering diversity goals?",
     "target_op": "gap", "domain": "hiring"},
    {"query": "What accuracy metrics is NovaTech unsure how to measure for their forecasting model?",
     "target_op": "gap", "domain": "ml_pipeline"},
    {"query": "What pricing experiments has NovaTech considered but not yet run?",
     "target_op": "gap", "domain": "customer"},
    {"query": "What integration points between Meridian and legacy systems are still unclear?",
     "target_op": "gap", "domain": "architecture"},
    {"query": "What competitive intelligence is NovaTech missing about the EU market?",
     "target_op": "gap", "domain": "strategy"},
]


# -- Corpus Management --------------------------------------------------------

def get_corpus_path():
    """Path to the synthetic org corpus."""
    return RESULTS_DIR / EXPERIMENT_NAME / "corpus.jsonl"


def generate_corpus(client):
    """Generate the synthetic NovaTech corpus using Haiku.

    Creates 6 internal documents (~800 words each) covering architecture,
    customers, ML pipeline, incidents, strategy, and hiring.

    Returns:
        list of dicts with 'doc_type', 'content'
    """
    corpus_path = get_corpus_path()
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing corpus
    existing = load_jsonl(corpus_path)
    if len(existing) >= len(CORPUS_PROMPTS):
        print(f"  Corpus already exists ({len(existing)} docs). Skipping generation.")
        return existing

    print(f"\n  Generating synthetic NovaTech corpus ({len(CORPUS_PROMPTS)} documents)...")
    model = MODELS[CORPUS_MODEL]
    docs = list(existing)  # resume from existing
    existing_types = {d["doc_type"] for d in existing}

    for doc_type, prompt in CORPUS_PROMPTS.items():
        if doc_type in existing_types:
            continue
        print(f"    Generating: {doc_type}...")
        content = call_api(client, model, CORPUS_GENERATION_SYSTEM, prompt, max_tokens=2000)
        if content:
            doc = {"doc_type": doc_type, "content": content, "word_count": len(content.split())}
            append_jsonl(corpus_path, doc)
            docs.append(doc)
        else:
            print(f"    [WARN] Failed to generate {doc_type}")

    print(f"  Corpus complete: {len(docs)} docs, {sum(d['word_count'] for d in docs)} words")
    return docs


# -- Graph Building -----------------------------------------------------------

def build_org_graphs(client, corpus):
    """Extract and build both Padartha and Generic graphs from org corpus.

    Returns:
        (padartha_graph, generic_graph, raw_corpus_text)
    """
    extraction_model = MODELS[EXTRACTION_MODEL]

    # Build raw corpus text
    raw_text = "\n\n---\n\n".join(
        f"## {d['doc_type'].upper()}\n\n{d['content']}" for d in corpus
    )

    print(f"\n  Extracting knowledge from {len(corpus)} documents...")
    padartha_extractions = []
    generic_extractions = []

    for doc in corpus:
        print(f"    Extracting: {doc['doc_type']}...")
        p_ext = extract_padarthas(client, extraction_model, doc["content"])
        g_ext = extract_generic(client, extraction_model, doc["content"])
        padartha_extractions.append(p_ext)
        generic_extractions.append(g_ext)

    p_graph = build_padartha_graph(padartha_extractions)
    g_graph = build_generic_graph(generic_extractions)

    print(f"  Padartha graph: {len(p_graph.nodes)} nodes, {len(p_graph.abhava_index)} absences")
    print(f"  Generic graph:  {len(g_graph.nodes)} nodes, {len(g_graph.abhava_index)} gaps")

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
        _corpus = generate_corpus(_client)

    if _p_graph is None:
        _p_graph, _g_graph, _raw_text = build_org_graphs(_client, _corpus)


def generate_fn(config, question):
    """Generate a response for a given config and org question.

    Args:
        config: one of CONFIGS
        question: dict with 'query', 'target_op', 'domain'

    Returns:
        dict with 'response', 'word_count', 'context_type', 'metrics'
    """
    ensure_setup()

    model = MODELS[GENERATION_MODEL]
    query = question["query"]
    query_type = question["target_op"]

    # Map query types to retrieval modes
    retrieval_type_map = {
        "factual": "discrimination",    # needs entities + attributes
        "causal": "force",              # needs actions + relationships
        "gap": "constraint",            # needs absences + limitations
    }
    retrieval_type = retrieval_type_map.get(query_type, "general")

    if config == "no_context":
        response = call_api(_client, model, "", query)
        context_used = ""

    elif config == "raw_context":
        # Truncate to ~3000 chars — pick docs relevant to query domain
        domain = question["domain"]
        relevant_docs = [d for d in _corpus if d["doc_type"] == domain]
        if not relevant_docs:
            relevant_docs = _corpus[:2]
        context = "\n\n".join(d["content"] for d in relevant_docs)[:3000]
        prompt = (
            f"You are answering questions about NovaTech Solutions based on "
            f"their internal documents.\n\n"
            f"## Internal Documents\n{context}\n\n"
            f"## Question\n{query}\n\n"
            f"Answer ONLY based on what the documents say. If the documents "
            f"don't contain the answer, say so."
        )
        response = call_api(_client, model, "", prompt, max_tokens=1024)
        context_used = f"raw ({len(context)} chars)"

    elif config == "generic_kg":
        context = format_generic_context(_g_graph, query, retrieval_type)
        prompt = (
            f"You are answering questions about NovaTech Solutions based on "
            f"their organized internal knowledge.\n\n"
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
            f"You are answering questions about NovaTech Solutions based on "
            f"their organized internal knowledge.\n\n"
            f"{context}\n\n"
            f"## Question\n{query}\n\n"
            f"Answer based on the provided knowledge. If specific information "
            f"is missing from the knowledge base, note the gap."
        )
        response = call_api(_client, model, "", prompt, max_tokens=1024)
        context_used = f"padartha_kg ({len(_p_graph.nodes)} nodes, {len(_p_graph.abhava_index)} absences)"

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

    print("\nWord counts by query type × config:")
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
        print("\n--- HEAD-TO-HEAD: padartha_kg vs generic_kg ---")
        p_wins = sum(1 for r in h2h if r["winner"] == "padartha_kg")
        g_wins = sum(1 for r in h2h if r["winner"] == "generic_kg")
        ties = sum(1 for r in h2h if r["winner"] == "TIE")
        total = len(h2h)
        print(f"  padartha_kg: {p_wins}/{total} wins ({p_wins/total*100:.0f}%)")
        print(f"  generic_kg:  {g_wins}/{total} wins ({g_wins/total*100:.0f}%)")
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
            print(f"    {qt:<15}: padartha {p_w}/{total_qt} ({p_w/total_qt*100:.0f}%) | "
                  f"generic {g_w}/{total_qt} ({g_w/total_qt*100:.0f}%) | ties {t}")


# -- Custom run_experiment override -------------------------------------------

def run_org_experiment(limit=None):
    """Run the experiment using ORG_QUESTIONS instead of TRANSFER_QUESTIONS.

    We can't use the standard run_experiment() because it uses TRANSFER_QUESTIONS.
    This reimplements the same logic with ORG_QUESTIONS.
    """
    results_dir = RESULTS_DIR / EXPERIMENT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"

    existing = load_existing_keys(results_path)
    questions = ORG_QUESTIONS[:limit] if limit else ORG_QUESTIONS
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
    print(f"  Results: {results_path}")
    return results


def run_org_judging(baseline, experimental_configs, judge_model="haiku"):
    """Run pairwise judging adapted for org questions.

    Uses the standard run_pairwise_judging but the results file has
    org-specific target_ops (factual/causal/gap).
    """
    return run_pairwise_judging(
        EXPERIMENT_NAME,
        baseline_config=baseline,
        experimental_configs=experimental_configs,
        judge_model=judge_model,
    )


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 v3: Vaisheshika as Organizational Knowledge Graph"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit questions (for smoke testing)")
    parser.add_argument("--corpus-only", action="store_true",
                        help="Only generate the synthetic corpus")
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

    if args.corpus_only:
        client = get_client()
        corpus = generate_corpus(client)
        print(f"\n  Corpus: {len(corpus)} docs")
        for doc in corpus:
            print(f"    {doc['doc_type']}: {doc['word_count']} words")
        return

    if args.judge:
        print("\n  Phase 1: Judging vs no_context baseline...")
        run_org_judging(
            baseline="no_context",
            experimental_configs=["raw_context", "generic_kg", "padartha_kg"],
            judge_model=args.judge_model,
        )
        analyze_judging()
        return

    if args.judge_h2h:
        print("\n  Head-to-head: padartha_kg vs generic_kg...")
        run_org_judging(
            baseline="generic_kg",
            experimental_configs=["padartha_kg"],
            judge_model=args.judge_model,
        )
        analyze_judging()
        return

    # Generation phase
    results = run_org_experiment(limit=args.limit)
    analyze_results()

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"  1. Judge vs baseline:  python {__file__} --judge")
    print(f"  2. Head-to-head:       python {__file__} --judge-h2h")
    print(f"  3. Full analysis:      python {__file__} --analyze")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
