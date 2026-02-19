"""Reliable search module — Wikipedia API primary, DuckDuckGo fallback.

Wikipedia is free, no API key, reliable, and perfect for factual questions
about science, economics, biology, everyday topics.
"""

import json
import re
import urllib.request
import urllib.parse


def search_wikipedia(query, max_results=3, sentences=5):
    """Search Wikipedia and return article summaries.

    Uses the Wikipedia REST API (no key needed).
    Returns formatted text with source attribution.
    """
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(max_results),
                "format": "json",
            })
        )
        req = urllib.request.Request(search_url, headers={"User-Agent": "darshana-llm/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = data.get("query", {}).get("search", [])
        if not results:
            return "No Wikipedia results found."

        titles = [r["title"] for r in results[:max_results]]
        summaries = []

        for title in titles:
            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + urllib.parse.quote(title)
            )
            req = urllib.request.Request(summary_url, headers={"User-Agent": "darshana-llm/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    sdata = json.loads(resp.read().decode())
                    extract = sdata.get("extract", "")
                    if extract:
                        sents = re.split(r"(?<=[.!?])\s+", extract)
                        trimmed = " ".join(sents[:sentences])
                        summaries.append(
                            f"Source: Wikipedia — {title}\n{trimmed}"
                        )
            except Exception:
                continue

        if not summaries:
            return "Wikipedia search matched titles but could not retrieve summaries."

        return "\n\n".join(summaries)

    except Exception as e:
        return f"Wikipedia search failed: {e}"


def search_ddg(query, max_results=3):
    """Fallback: DuckDuckGo search."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return None
            formatted = []
            for r in results:
                formatted.append(f"Source: {r.get('title', 'Unknown')}\n{r.get('body', '')}")
            return "\n\n".join(formatted)[:2000]
    except Exception:
        return None


def search(query, max_results=3):
    """Search with Wikipedia primary, DuckDuckGo fallback.

    Returns:
        tuple: (result_text, source) where source is 'wikipedia' or 'ddg' or 'failed'
    """
    result = search_wikipedia(query, max_results=max_results)
    if result and "failed" not in result.lower() and "no wikipedia" not in result.lower():
        return result, "wikipedia"

    ddg_result = search_ddg(query, max_results=max_results)
    if ddg_result:
        return ddg_result, "ddg"

    return "No search results found from any source.", "failed"
