"""Vritti — Epistemic Self-Classification

Tags each claim in a response with its knowledge type.
Proven effective: 63% win rate (4B local), 90% (Sonnet), 100% (pipeline_clean).

From Patanjali's Yoga Sutras (1.5-1.11): 5 modes of mental operation.
"""

VRITTI_SYSTEM = """Tag each claim or statement in the following text with one of these knowledge types. Place the tag at the START of each paragraph or major claim.

[PRAMANA] — Valid knowledge: factually correct, based on well-established evidence or direct logical inference from verified premises.
[SMRITI] — Memory/retrieval: well-known information from established sources. Common textbook knowledge.
[ANUMANA] — Inference: reasoning from premises to a conclusion. Premises are established but conclusion is your inference.
[VIKALPA] — Verbal elaboration: context, framing, or explanation that helps understanding but is not itself a factual claim.
[UNCERTAIN] — Potential error: you are not confident in this claim. It might be wrong. Flag it honestly.

Be honest — tagging uncertain claims as PRAMANA is worse than tagging them as UNCERTAIN.

Return the FULL text with tags added. Do not remove or change any content — only add tags."""


VRITTI_INLINE_PROMPT = """For each claim or statement in your response, tag it with one of these knowledge types:

[PRAMANA] — Valid knowledge: you are confident this is factually correct, based on well-established evidence or direct logical inference from verified premises.
[SMRITI] — Memory/retrieval: you are reproducing well-known information from established sources. Common textbook knowledge.
[ANUMANA] — Inference: you are reasoning from premises to a conclusion. The premises are established but the conclusion is your inference.
[VIKALPA] — Verbal elaboration: you are providing context, framing, or explanation that helps understanding but is not itself a factual claim.
[UNCERTAIN] — Potential error: you are not confident in this claim. It might be wrong. Flag it honestly.

Place the tag at the START of each paragraph or major claim. Be honest — tagging uncertain claims as PRAMANA is worse than tagging them as UNCERTAIN."""


GENERIC_CONFIDENCE_PROMPT = """For each claim or statement in your response, tag it with one of these confidence levels:

[CERTAIN] — You are highly confident this is factually correct, based on well-established knowledge.
[LIKELY] — You believe this is correct but there is some room for error.
[UNCERTAIN] — You are not confident in this claim and it may be wrong.
[SPECULATIVE] — This is your educated guess or extrapolation beyond established facts.
[UNKNOWN] — You genuinely do not know if this is correct.

Place the tag at the START of each paragraph or major claim. Be honest — tagging uncertain claims as CERTAIN is worse than tagging them as UNCERTAIN."""


VRITTI_CONTEMPORARY_PROMPT = """For each claim or statement in your response, tag it with one of these knowledge types:

[ESTABLISHED] — You are stating well-verified facts backed by strong evidence or direct logical proof. You are confident this is correct.
[TEXTBOOK] — You are reproducing well-known information from established sources. Standard knowledge that any expert would agree on.
[INFERRED] — You are reasoning from established premises to a new conclusion. The premises are solid but the conclusion is your logical derivation.
[FRAMING] — You are providing context, explanation, or elaboration that helps understanding but is not itself a factual claim.
[UNCERTAIN] — You are not confident in this claim. It might be wrong. Flag it honestly.

Place the tag at the START of each paragraph or major claim. Be honest — tagging uncertain claims as ESTABLISHED is worse than tagging them as UNCERTAIN.

IMPORTANT: These are DIFFERENT types of knowledge, not just confidence levels. An INFERRED claim may be high-confidence but is qualitatively different from an ESTABLISHED fact. A TEXTBOOK claim is not just "certain" — it signals you're recalling standard knowledge. Use the type that best describes HOW you know this, not just how sure you are."""


GENERIC_COT_PROMPT = """Before each claim, briefly note your reasoning basis and confidence level.

For each major statement:
1. State what you're about to claim
2. Note WHY you believe it (source of knowledge, logical reasoning, educated guess)
3. Rate your confidence (high/medium/low)
4. Then make the claim

Be honest about what you know vs what you're inferring."""


def add_epistemic_tags(client, model, reasoning_text):
    """Add Vritti epistemic tags to an existing response.

    This is a SEPARATE call that takes an already-generated response
    and adds [PRAMANA]/[SMRITI]/[ANUMANA]/[VIKALPA]/[UNCERTAIN] tags.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        reasoning_text: the response text to tag

    Returns:
        str: tagged response text
    """
    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        system=VRITTI_SYSTEM,
        messages=[{"role": "user", "content": reasoning_text}],
    )
    return msg.content[0].text.strip()
