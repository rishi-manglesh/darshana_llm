"""Vedanta — Deep Output Synthesis

LLM-powered synthesis using Vedanta's three key concepts:
  Brahman → Unify all fragments into single coherent truth
  Maya    → Remove scaffolding, formatting artifacts, redundancy
  Atman   → Extract the core insight that persists after cleanup

Hypothesis: Vedantic synthesis produces more useful output than
the crude regex formatter (which already flipped 53% → 100%).
"""

VEDANTA_SYNTHESIS_SYSTEM = """You are a synthesis engine. Take the following multi-stage response — which contains evidence from web searches, reasoning structure, and epistemic tags — and synthesize it into a unified, coherent answer.

Apply three Vedantic principles:

1. BRAHMAN (Unity): Identify the single unifying truth across all the evidence and reasoning. What is the ONE core answer? Start with this.

2. MAYA (Illusion/Scaffolding): Strip away everything that was necessary for the reasoning PROCESS but not for the final UNDERSTANDING:
   - Remove step labels (PRATIJNA, HETU, UDAHARANA, UPANAYA, NIGAMANA)
   - Remove epistemic tags ([PRAMANA], [SMRITI], etc.)
   - Remove redundant repetition between search results and reasoning
   - Remove search metadata and tool use artifacts
   BUT KEEP: specific facts, evidence, citations, concrete examples

3. ATMAN (Core Self): For each major point, extract the essential insight — the thing that would survive even if you forgot the details. Express it clearly.

OUTPUT FORMAT:
- Start with the unified core insight (2-3 sentences)
- Then supporting evidence and details, organized by relevance (not by processing order)
- End with nuance/limitations (what's uncertain, what's missing)
- Use clean markdown formatting
- Preserve all citations and sources
- Write for a knowledgeable reader who wants understanding, not just information

IMPORTANT: This is SYNTHESIS, not summarization. You should INTEGRATE the information into new understanding, not just shorten it."""


def synthesize_response(client, model, raw_response, original_query, stages=None):
    """Synthesize a multi-stage pipeline response into unified output.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        raw_response: the full pipeline output (with tags, step labels, etc.)
        original_query: the original user query
        stages: optional list of stage names that produced this response

    Returns:
        str: synthesized response
    """
    stage_context = ""
    if stages:
        stage_context = f"\n\nThis response was produced through these stages: {', '.join(stages)}\n"

    user_content = (
        f"Original question: {original_query}\n"
        f"{stage_context}"
        f"\nFull response to synthesize:\n{raw_response}"
    )

    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        system=VEDANTA_SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    return msg.content[0].text.strip()
