"""Formatter — Clean Output Stage

Takes an evidence-grounded, epistemically-tagged response and rewrites it
as a clean, readable answer. Preserves all factual claims and citations.
Removes structural scaffolding (Nyaya step labels, raw Vritti tags).
Converts epistemic signals to natural language.

This is the "crude Vedanta" stage that flipped pipeline from 53% to 100%.
"""

FORMATTER_SYSTEM = """You are a response formatter. Take the following response — which contains evidence from web searches, reasoning structure labels, and epistemic tags — and rewrite it as a clean, reader-friendly answer.

Rules:
1. KEEP all factual claims, specific examples, citations, and data points. Do not remove evidence.
2. REMOVE structural labels like PRATIJNA, HETU, UDAHARANA, UPANAYA, NIGAMANA — these were reasoning scaffolding.
3. CONVERT epistemic tags to natural language:
   - [PRAMANA] — state the claim directly with confidence (no marker needed)
   - [SMRITI] — "According to established knowledge..." or just state it directly
   - [ANUMANA] — use "This suggests...", "This implies...", "Based on this reasoning..."
   - [VIKALPA] — weave naturally into the text as context/framing
   - [UNCERTAIN] — use "It's worth noting that this is uncertain..." or "Evidence is limited here..."
4. FORMAT with clean markdown: headers, bullet points, bold for emphasis.
5. PRESERVE source citations (keep "Source: ..." references).
6. Write in a helpful, educational tone — as if explaining to a smart student.
7. Do NOT add new information. Only restructure what's already there.
8. Keep it comprehensive — do not shorten or summarize. The goal is readability, not brevity."""


def clean_format(client, model, tagged_response, original_query):
    """Reformat a tagged pipeline response into clean, readable output.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        tagged_response: the Vritti-tagged response text
        original_query: the original user query (for context)

    Returns:
        str: clean formatted response
    """
    user_content = (
        f"Original question: {original_query}\n\n"
        f"Response to reformat:\n{tagged_response}"
    )

    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        system=FORMATTER_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    return msg.content[0].text.strip()
