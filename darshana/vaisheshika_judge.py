"""Vaisheshika Judge — 7-Padartha Evaluation Framework

Uses Vaisheshika's 7 padarthas (categories of reality) as evaluation dimensions
for judging LLM response quality. Hypothesis: padartha-based judging provides
higher discrimination power than the current 5-dimension judge.

7 Padarthas:
  Dravya    (substance)   → Core factual content — is it substantive?
  Guna      (quality)     → Quality of reasoning — is it well-argued?
  Karma     (action)      → Actionability — can you DO something with this?
  Samanya   (general)     → Generalization — does it identify patterns?
  Vishesha  (particular)  → Specificity — does it give concrete details?
  Samavaya  (inherence)   → Coherence — do parts logically connect?
  Abhava    (absence)     → What's missing — what should be there but isn't?
"""

PADARTHA_JUDGE_SYSTEM = """You are an expert evaluator using the 7-Padartha framework to assess response quality.

Evaluate the response on these 7 dimensions (1-5 each):

1. DRAVYA (Substance): Does the response contain real, substantive content?
   1=empty/superficial, 3=adequate content, 5=rich with verified facts and evidence

2. GUNA (Quality): Is the reasoning well-constructed and logically sound?
   1=logical errors, 3=decent reasoning, 5=rigorous causal/mechanistic analysis

3. KARMA (Action/Actionability): Can someone DO something useful with this information?
   1=purely abstract, 3=some practical implications, 5=clear actionable insights

4. SAMANYA (Generalization): Does it identify broader patterns and principles?
   1=only specific facts, 3=some abstraction, 5=identifies underlying principles that transfer

5. VISHESHA (Specificity): Does it provide concrete, specific details?
   1=vague generalities only, 3=some specifics, 5=named examples, numbers, specific mechanisms

6. SAMAVAYA (Coherence/Inherence): Do the parts logically connect into a unified answer?
   1=disconnected points, 3=loosely connected, 5=every point builds on previous ones

7. ABHAVA (Absence): What important aspects are MISSING from the response?
   1=major gaps, 3=minor omissions, 5=comprehensive (nothing important missing)

After scoring, provide an overall assessment.

Respond with ONLY a JSON object:
{
  "dravya": <1-5>,
  "guna": <1-5>,
  "karma": <1-5>,
  "samanya": <1-5>,
  "vishesha": <1-5>,
  "samavaya": <1-5>,
  "abhava": <1-5>,
  "total": <sum of all 7>,
  "missing_aspects": "<what's absent from the response>",
  "assessment": "<1-2 sentence overall assessment>"
}"""


PADARTHA_PAIRWISE_SYSTEM = """You are an expert evaluator comparing two responses using the 7-Padartha framework.

You will see a QUESTION and two RESPONSES (A and B).

Score EACH response on 7 dimensions (1-5):
1. DRAVYA (Substance) — real, substantive factual content
2. GUNA (Quality) — logical soundness and reasoning quality
3. KARMA (Actionability) — practical usefulness
4. SAMANYA (Generalization) — broader patterns identified
5. VISHESHA (Specificity) — concrete details and examples
6. SAMAVAYA (Coherence) — logical connection between parts
7. ABHAVA (Absence) — comprehensiveness (nothing important missing)

Respond with ONLY a JSON object:
{
  "response_a": {"dravya": <1-5>, "guna": <1-5>, "karma": <1-5>, "samanya": <1-5>, "vishesha": <1-5>, "samavaya": <1-5>, "abhava": <1-5>},
  "response_b": {"dravya": <1-5>, "guna": <1-5>, "karma": <1-5>, "samanya": <1-5>, "vishesha": <1-5>, "samavaya": <1-5>, "abhava": <1-5>},
  "winner": "A" or "B" or "TIE",
  "reason": "<1 sentence>"
}"""


def judge_with_padarthas(client, model, query, response):
    """Judge a single response using the 7-padartha framework.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: the original question
        response: the response text to judge

    Returns:
        dict with padartha scores or None on failure
    """
    import json
    import time

    user_content = f"QUESTION: {query}\n\nRESPONSE:\n{response}"

    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=500,
                system=PADARTHA_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            text = msg.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, KeyError):
            if attempt < 2:
                time.sleep(1)
                continue
            return None
        except Exception:
            if attempt < 2:
                time.sleep(2)
                continue
            return None
    return None


def judge_pairwise_padarthas(client, model, query, response_a, response_b):
    """Compare two responses using the 7-padartha framework.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: the original question
        response_a, response_b: the two responses to compare

    Returns:
        dict with padartha scores for both + winner, or None on failure
    """
    import json
    import time

    user_content = (
        f"QUESTION: {query}\n\n"
        f"RESPONSE A:\n{response_a}\n\n"
        f"RESPONSE B:\n{response_b}"
    )

    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=500,
                system=PADARTHA_PAIRWISE_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            text = msg.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, KeyError):
            if attempt < 2:
                time.sleep(1)
                continue
            return None
        except Exception:
            if attempt < 2:
                time.sleep(2)
                continue
            return None
    return None
