"""Nyaya — 5-Step Syllogism with Tool Use

Generates responses using Nyaya's 5-step reasoning structure.
Claude decides WHEN to search for evidence (Udaharana) — not forced.
Web search via duckduckgo-search (no API key needed).
"""

import json

from duckduckgo_search import DDGS

# -- Tool Definition -----------------------------------------------------------

NYAYA_TOOLS = [
    {
        "name": "search_evidence",
        "description": (
            "Search the web for real-world evidence to support or verify a claim "
            "(Udaharana step in Nyaya reasoning). Use this to find specific, real "
            "examples rather than relying on memory. Search when you need to ground "
            "a claim in verifiable evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find evidence for the claim",
                },
                "claim_context": {
                    "type": "string",
                    "description": "Brief description of the claim being supported",
                },
            },
            "required": ["query"],
        },
    }
]

NYAYA_SYSTEM = """For every major claim you make, follow this 5-step reasoning process:

1. PRATIJNA (Claim): State what you believe to be true.
2. HETU (Reason): State WHY you believe it.
3. UDAHARANA (Grounded Example): Provide a SPECIFIC, REAL case where this reasoning holds. Use the search_evidence tool to find real examples instead of relying on memory. Cite the source.
4. UPANAYA (Application): Show how the example's pattern applies here.
5. NIGAMANA (Conclusion): State the conclusion, grounded in both reason and evidence.

IMPORTANT:
- Use the search tool to find real evidence for your Udaharana steps.
- Do NOT fabricate examples from memory — search for them.
- If search returns no useful results, say "I could not find a grounding example — treat as tentative."
- You may make up to 3 searches total. Prioritize the most important claims."""

MAX_TOOL_CALLS = 3


# -- Web Search ----------------------------------------------------------------

def web_search(query, max_results=3):
    """Search the web via DuckDuckGo. Returns formatted results string."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No results found."
            formatted = []
            for r in results:
                formatted.append(f"Source: {r.get('title', 'Unknown')}\n{r.get('body', '')}")
            return "\n\n".join(formatted)[:2000]
    except Exception as e:
        return f"Search failed: {e}"


# -- Tool Call Loop ------------------------------------------------------------

def generate_with_tools(client, model, query, system_prompt=None, tools=None):
    """Generate a response with Nyaya reasoning and optional tool use.

    Handles the tool call loop: Claude generates -> requests tool -> we execute ->
    Claude continues. Up to MAX_TOOL_CALLS iterations.

    Args:
        client: anthropic.Anthropic instance
        model: model ID string
        query: user query (possibly refined by Mimamsa)
        system_prompt: override system prompt (defaults to NYAYA_SYSTEM)
        tools: override tool definitions (defaults to NYAYA_TOOLS)

    Returns:
        dict with 'response' (final text), 'tool_calls' (list of calls made),
        'total_api_calls' (number of API round-trips)
    """
    if system_prompt is None:
        system_prompt = NYAYA_SYSTEM
    if tools is None:
        tools = NYAYA_TOOLS

    messages = [{"role": "user", "content": query}]
    tool_calls_made = []
    api_calls = 0

    for _ in range(MAX_TOOL_CALLS + 1):  # +1 for final response after tools
        api_calls += 1
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            tools=tools,
        )

        # Check if model wants to use tools
        if response.stop_reason == "tool_use":
            assistant_content = response.content
            tool_results = []

            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    if tool_name == "search_evidence" and len(tool_calls_made) < MAX_TOOL_CALLS:
                        search_query = tool_input.get("query", "")
                        claim_context = tool_input.get("claim_context", "")
                        result = web_search(search_query)

                        tool_calls_made.append({
                            "query": search_query,
                            "claim_context": claim_context,
                            "result_preview": result[:200],
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result,
                        })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "Tool call limit reached. Please proceed without additional searches.",
                            "is_error": True,
                        })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            final_text = "\n".join(text_parts)

            return {
                "response": final_text,
                "tool_calls": tool_calls_made,
                "total_api_calls": api_calls,
            }

    # Safety: if we somehow exhaust the loop
    return {
        "response": "[ERROR: Tool call loop exhausted]",
        "tool_calls": tool_calls_made,
        "total_api_calls": api_calls,
    }
