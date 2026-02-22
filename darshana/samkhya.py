"""Samkhya — Pretraining Data Organization

Samkhya's 25-tattva categories as a framework for organizing training data.
Hypothesis: data organized by epistemic category trains better than random order.

The 25 tattvas map to training data types:
  Purusha     → Meta-knowledge (how to reason, methodology)
  Prakriti    → Raw factual data (unprocessed)
  Sattva      → Analytical texts (clear reasoning, discrimination)
  Rajas       → Causal/mechanism texts (forces, dynamics)
  Tamas       → Constraint/limitation texts (boundaries, resistance)
  5 Tanmatras → Evidence types (empirical, statistical, case study, expert, historical)
"""

# -- Samkhya Category Definitions ----------------------------------------------

SAMKHYA_CATEGORIES = {
    # Purusha — the observer/witness
    "purusha": {
        "description": "Meta-knowledge: methodology, epistemology, how-to-reason texts",
        "keywords": [
            "methodology", "epistemology", "reasoning", "logic", "framework",
            "analysis method", "how to think", "critical thinking", "scientific method",
        ],
    },
    # Prakriti — primordial matter/nature
    "prakriti": {
        "description": "Raw facts: definitions, data tables, reference material",
        "keywords": [
            "definition", "defined as", "refers to", "is a type of",
            "classification", "taxonomy", "list of", "table of",
        ],
    },
    # Three Gunas
    "sattva": {
        "description": "Analytical texts: comparisons, distinctions, clear reasoning",
        "keywords": [
            "difference between", "compare", "contrast", "distinguish",
            "whereas", "on the other hand", "unlike", "classification",
            "categorize", "differentiate",
        ],
    },
    "rajas": {
        "description": "Causal/mechanism texts: forces, dynamics, how things work",
        "keywords": [
            "cause", "mechanism", "drives", "leads to", "results in",
            "because", "therefore", "consequently", "force", "dynamic",
            "process", "how it works", "trigger",
        ],
    },
    "tamas": {
        "description": "Constraint texts: limits, boundaries, what prevents change",
        "keywords": [
            "limit", "constraint", "boundary", "cannot", "restricted",
            "bottleneck", "barrier", "threshold", "maximum", "minimum",
            "prevent", "resist", "slow down",
        ],
    },
    # 5 Tanmatras (subtle elements) → Evidence types
    "tanmatra_empirical": {
        "description": "Empirical evidence: experiments, observations, measurements",
        "keywords": [
            "experiment", "observed", "measured", "data shows",
            "study found", "laboratory", "field observation",
        ],
    },
    "tanmatra_statistical": {
        "description": "Statistical evidence: numbers, percentages, surveys",
        "keywords": [
            "percent", "statistics", "survey", "correlation",
            "average", "median", "standard deviation", "p-value",
        ],
    },
    "tanmatra_case": {
        "description": "Case study evidence: specific examples, historical cases",
        "keywords": [
            "case study", "for example", "instance", "specific case",
            "in the case of", "anecdotal", "real-world example",
        ],
    },
    "tanmatra_expert": {
        "description": "Expert testimony: authorities, established knowledge",
        "keywords": [
            "according to", "expert", "authority", "established",
            "consensus", "widely accepted", "mainstream view",
        ],
    },
    "tanmatra_historical": {
        "description": "Historical evidence: precedents, trends, patterns over time",
        "keywords": [
            "historically", "precedent", "trend", "over time",
            "evolution of", "development of", "history shows",
        ],
    },
}


def categorize_text(text, threshold=2):
    """Classify a text passage into Samkhya categories.

    Fix #5: Uses word boundary matching (regex \\b) instead of substring
    matching to prevent false positives like "cause" matching "because".

    Args:
        text: the text to classify
        threshold: minimum keyword matches to assign a category

    Returns:
        list of (category_name, match_count) tuples, sorted by relevance
    """
    import re
    lower = text.lower()
    scores = []
    for cat_name, cat_info in SAMKHYA_CATEGORIES.items():
        count = 0
        for kw in cat_info["keywords"]:
            # Use word boundary matching for single words,
            # substring matching for multi-word phrases (they're specific enough)
            if " " in kw:
                if kw.lower() in lower:
                    count += 1
            else:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', lower):
                    count += 1
        if count >= threshold:
            scores.append((cat_name, count))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def organize_corpus(documents):
    """Organize a list of documents into Samkhya categories.

    Args:
        documents: list of dicts with at least 'text' field

    Returns:
        dict mapping category_name -> list of documents
    """
    organized = {cat: [] for cat in SAMKHYA_CATEGORIES}
    organized["unclassified"] = []

    for doc in documents:
        cats = categorize_text(doc.get("text", ""))
        if cats:
            organized[cats[0][0]].append(doc)
        else:
            organized["unclassified"].append(doc)

    return organized


def bloom_training_order():
    """Return a Bloom's Taxonomy-based training order (Western curriculum control).

    Bloom's Taxonomy: Remember → Understand → Apply → Analyze → Evaluate → Create.
    Maps to Samkhya categories by cognitive complexity.

    Returns:
        list of category names in Bloom's order
    """
    return [
        # Remember: raw facts, definitions
        "prakriti",
        # Understand: expert testimony, established knowledge
        "tanmatra_expert",
        # Apply: case studies, real-world examples
        "tanmatra_case",
        # Analyze: discrimination, comparison, causal analysis
        "sattva",
        "rajas",
        "tamas",
        # Evaluate: empirical and statistical evidence
        "tanmatra_empirical",
        "tanmatra_statistical",
        # Create: meta-knowledge, methodology, synthesis
        "purusha",
        "tanmatra_historical",
    ]


def bloom_ordered_corpus(organized):
    """Order an organized corpus by Bloom's Taxonomy instead of Samkhya.

    Args:
        organized: dict mapping category_name -> list of documents (from organize_corpus)

    Returns:
        list of documents in Bloom's order
    """
    ordered = []
    for cat in bloom_training_order():
        ordered.extend(organized.get(cat, []))
    ordered.extend(organized.get("unclassified", []))
    return ordered


def samkhya_training_order():
    """Return the Samkhya-principled training order.

    Samkhya's ontological order: Purusha first (observer establishes framework),
    then Prakriti (raw material), then Gunas (modes of processing),
    then Tanmatras (evidence types — subtle to gross).

    Returns:
        list of category names in training order
    """
    return [
        "purusha",           # First: meta-knowledge (how to reason)
        "prakriti",          # Second: raw facts (what to reason about)
        "sattva",            # Third: analytical mode (discrimination)
        "rajas",             # Fourth: causal mode (force/mechanism)
        "tamas",             # Fifth: constraint mode (limits)
        "tanmatra_empirical",    # Evidence: most direct
        "tanmatra_statistical",  # Evidence: quantitative
        "tanmatra_case",         # Evidence: exemplary
        "tanmatra_expert",       # Evidence: authoritative
        "tanmatra_historical",   # Evidence: temporal patterns
    ]
