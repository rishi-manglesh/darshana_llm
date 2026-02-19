"""Mimamsa — Hidden Query Preprocessing

Applies the 6 Lingas (interpretation principles) to parse user intent.
Output is HIDDEN from user — only used to refine the query for downstream stages.

Also usable as a prompt REWRITING methodology (Exp 4): rewrite questions
using the 6 Lingas as a prompt engineering framework.
"""

import json

MIMAMSA_SYSTEM = """You are a query preprocessing module. Your job is to analyze the user's question and produce a refined, clearer version that captures the true intent.

Apply these 6 interpretation principles internally (do NOT show them to the user):

1. UPAKRAMA-UPASAMHARA: What is stated at the beginning and end — these are the primary focus.
2. ABHYASA: What is repeated or emphasized — this signals priority.
3. APURVATA: What is novel or unusual about this question — give this attention.
4. PHALA: What OUTCOME does the questioner want? Interpret by purpose, not literal words.
5. ARTHAVADA: Distinguish CONTEXT (background) from INSTRUCTION (what to address).
6. UPAPATTI: If ambiguous, use logic to determine the most reasonable interpretation.

Respond with ONLY a JSON object:
{
  "refined_query": "the refined, clarified version of the question",
  "core_intent": "1 sentence: what the questioner actually wants to understand",
  "key_focus": ["list", "of", "key", "aspects", "to", "address"],
  "ambiguities_resolved": "any ambiguities you resolved and how (or 'none')"
}"""


MIMAMSA_REWRITE_SYSTEM = """You are a prompt engineer applying 6 interpretation principles to REWRITE a question for maximum clarity and LLM response quality.

Apply each principle explicitly:

1. UPAKRAMA-UPASAMHARA (Opening/Closing): Frame the question with the key concept at both start and end.
2. ABHYASA (Repetition): Emphasize the most important aspect by restating it.
3. APURVATA (Novelty): Highlight what makes this question non-trivial or interesting.
4. PHALA (Purpose): State the desired outcome explicitly ("I want to understand...").
5. ARTHAVADA (Context vs Instruction): Separate background context from the actual question.
6. UPAPATTI (Logic): Remove ambiguity through precise language.

Respond with ONLY the rewritten question. No explanation, no JSON — just the improved question text."""


GENERIC_REWRITE_SYSTEM = """You are a prompt engineer. Rewrite the following question to get a better response from an AI assistant.

Apply these standard prompt engineering techniques:
1. Be specific about what you want to know
2. Provide relevant context for the question
3. State the desired format or depth of answer
4. Ask for reasoning and evidence, not just facts
5. Remove ambiguity through precise language
6. Use chain-of-thought framing where appropriate

Respond with ONLY the rewritten question. No explanation — just the improved question text."""


def generic_rewrite(client, model, query):
    """Rewrite a question using standard prompt engineering (no Mimamsa framework).

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: original question

    Returns:
        str: rewritten question
    """
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            system=GENERIC_REWRITE_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return query


def preprocess_query(client, model, query):
    """Parse user intent via Mimamsa 6 Lingas. Returns refined query + metadata.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: raw user query

    Returns:
        dict with 'refined_query', 'core_intent', 'key_focus', 'ambiguities_resolved'
        Falls back to original query on parse failure.
    """
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            system=MIMAMSA_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        text = msg.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        result.setdefault("refined_query", query)
        return result
    except (json.JSONDecodeError, KeyError, IndexError):
        return {
            "refined_query": query,
            "core_intent": "",
            "key_focus": [],
            "ambiguities_resolved": "parse_failure",
        }


def rewrite_with_lingas(client, model, query):
    """Rewrite a question using Mimamsa 6 Lingas as prompt engineering methodology.

    Used in Exp 4 to compare Mimamsa-rewritten prompts vs generic prompt engineering.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: original question

    Returns:
        str: rewritten question
    """
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            system=MIMAMSA_REWRITE_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return query
