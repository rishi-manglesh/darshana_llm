"""Nyaya Router — Pramana-Based Tool Routing

Uses Nyaya's 4 pramanas (valid means of knowledge) to classify queries
and decide whether external search is needed.

4 Pramanas:
  Pratyaksha (perception)  → Direct observation: needs real-time/current data → SEARCH
  Anumana (inference)      → Logical deduction from known premises → NO SEARCH (usually)
  Upamana (comparison)     → Analogy to known cases → NO SEARCH
  Shabda (testimony)       → Authoritative source needed → SEARCH

Exp 5 validates: does pramana routing reduce the 57% search redundancy
while maintaining accuracy?
"""

import json

PRAMANA_CLASSIFIER_SYSTEM = """Classify this question by the type of knowledge needed to answer it well.

4 PRAMANA (valid means of knowledge):

1. PRATYAKSHA (Direct Perception): The answer requires current/real-time data, specific facts, or verifiable specifics that should be looked up rather than recalled.
   Examples: "What is the current inflation rate?", "Who won the 2024 election?"

2. ANUMANA (Inference): The answer can be derived through logical reasoning from well-established premises. No external lookup needed.
   Examples: "Why does inflation reduce purchasing power?", "Compare mitosis and meiosis."

3. UPAMANA (Comparison/Analogy): The answer involves drawing analogies or comparisons between known concepts. Internal knowledge sufficient.
   Examples: "Is fiscal policy like a thermostat?", "How is evolution like an arms race?"

4. SHABDA (Authoritative Testimony): The answer requires citing specific sources, studies, statistics, or expert opinions to be credible.
   Examples: "What does the evidence say about minimum wage effects?", "What are the latest findings on antibiotic resistance?"

Respond with ONLY a JSON object:
{
  "pramana": "PRATYAKSHA" or "ANUMANA" or "UPAMANA" or "SHABDA",
  "confidence": <0.0-1.0>,
  "reasoning": "<1 sentence explaining why>",
  "needs_search": true or false
}

SEARCH RULE: PRATYAKSHA and SHABDA usually need search. ANUMANA and UPAMANA usually do not.
If confidence < 0.6, recommend search as a safety measure."""


def classify_pramana(client, model, query):
    """Classify a query by its pramana type.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: the user query

    Returns:
        dict with 'pramana', 'confidence', 'reasoning', 'needs_search'
    """
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            system=PRAMANA_CLASSIFIER_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        # Validate
        result.setdefault("pramana", "ANUMANA")
        result.setdefault("confidence", 0.5)
        result.setdefault("needs_search", result["pramana"] in ("PRATYAKSHA", "SHABDA"))
        return result
    except (json.JSONDecodeError, KeyError, IndexError):
        return {
            "pramana": "SHABDA",
            "confidence": 0.0,
            "reasoning": "classification_failed",
            "needs_search": True,  # Default to search on failure
        }


def heuristic_route(query):
    """Route a query using simple keyword heuristics (no LLM call).

    Searches if query contains date/time references, numbers, names,
    or currency keywords suggesting current/specific data is needed.

    Args:
        query: the user query

    Returns:
        dict with 'use_search' (bool), 'routing_reason'
    """
    import re
    lower = query.lower()

    search_triggers = [
        # Temporal: current/recent data
        r"\b(?:latest|current|recent|today|now|202\d|this year|this month)\b",
        # Specific data lookups
        r"\b(?:how much|how many|what percent|what rate|statistics|data)\b",
        # Named entities suggesting factual lookup
        r"\b(?:who (?:is|was|are)|when (?:did|was|is)|where (?:is|was))\b",
        # Currency/markets
        r"\b(?:price|cost|stock|market|gdp|inflation rate)\b",
    ]

    needs_search = any(re.search(p, lower) for p in search_triggers)

    return {
        "use_search": needs_search,
        "pramana": {"pramana": "HEURISTIC", "confidence": 1.0},
        "routing_reason": f"heuristic_{'search' if needs_search else 'no_search'}",
    }


def model_decides_route(client, model, query):
    """Let the model itself decide whether search is needed (no framework).

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: the user query

    Returns:
        dict with 'use_search' (bool), 'routing_reason'
    """
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=100,
            system="You decide whether an external web search would help answer this question. Respond with ONLY a JSON object: {\"needs_search\": true/false, \"reason\": \"<1 sentence>\"}",
            messages=[{"role": "user", "content": query}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return {
            "use_search": result.get("needs_search", True),
            "pramana": {"pramana": "MODEL_DECIDES", "confidence": 1.0},
            "routing_reason": f"model_decides_{'search' if result.get('needs_search', True) else 'no_search'}",
        }
    except Exception:
        return {
            "use_search": True,
            "pramana": {"pramana": "MODEL_DECIDES", "confidence": 0.0},
            "routing_reason": "model_decides_fallback_search",
        }


def route_query(client, model, query, force_search=None):
    """Route a query to the appropriate tool configuration.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: the user query
        force_search: override - True (always search), False (never search), None (pramana decides)

    Returns:
        dict with 'use_search' (bool), 'pramana' classification, 'routing_reason'
    """
    if force_search is not None:
        return {
            "use_search": force_search,
            "pramana": {"pramana": "FORCED", "confidence": 1.0},
            "routing_reason": f"forced_{'search' if force_search else 'no_search'}",
        }

    classification = classify_pramana(client, model, query)
    use_search = classification["needs_search"]

    # Safety: if confidence is low, search
    if classification["confidence"] < 0.6:
        use_search = True

    return {
        "use_search": use_search,
        "pramana": classification,
        "routing_reason": f"pramana_{classification['pramana'].lower()}",
    }
