"""Vaisheshika — Structured Output Extraction

Local regex parsing (no API call). Extracts structured metrics from responses:
- Vritti tag counts
- Claim count
- Nyaya step detection
- Tool use metrics
"""

import re
from collections import Counter


def extract_structure(response_text, tool_calls=None):
    """Extract structured metrics from a Darshana pipeline response.

    Args:
        response_text: the final response text
        tool_calls: list of tool call dicts from Nyaya stage (optional)

    Returns:
        dict with structured metrics
    """
    # -- Vritti tag counts -----------------------------------------------------
    vritti_tags = {}
    for tag in ["PRAMANA", "SMRITI", "ANUMANA", "VIKALPA", "UNCERTAIN"]:
        vritti_tags[tag] = len(re.findall(rf"\[{tag}\]", response_text))
    vritti_tags["total"] = sum(vritti_tags.values())

    # -- Claim counting --------------------------------------------------------
    sentences = re.split(r"[.!?]+", response_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    claim_count = len(sentences)

    # -- Nyaya structure detection ---------------------------------------------
    nyaya_steps = {}
    for step in ["PRATIJNA", "HETU", "UDAHARANA", "UPANAYA", "NIGAMANA"]:
        nyaya_steps[step] = len(re.findall(
            rf"(?:^|\n)\s*(?:\d+\.?\s*)?(?:\*\*)?{step}", response_text, re.IGNORECASE
        ))
    nyaya_steps["complete_syllogisms"] = min(nyaya_steps.values()) if nyaya_steps.values() else 0

    # -- Example grounding -----------------------------------------------------
    grounding_patterns = [
        r"for example[,:]",
        r"for instance[,:]",
        r"such as\b",
        r"e\.g\.\b",
        r"case of\b",
        r"Source:",
        r"According to",
        r"research (?:shows|found|suggests)",
        r"study (?:by|from|in)",
    ]
    grounded_examples = 0
    lower = response_text.lower()
    for p in grounding_patterns:
        grounded_examples += len(re.findall(p, lower))

    # -- Hedging / uncertainty -------------------------------------------------
    hedge_patterns = [
        r"i(?:'m| am) not (?:sure|certain|confident)",
        r"this (?:may|might|could) (?:be|not be)",
        r"it(?:'s| is) (?:unclear|uncertain|debatable)",
        r"tentative",
        r"not confident",
        r"treat as tentative",
        r"could not find.*example",
    ]
    hedging_count = 0
    for p in hedge_patterns:
        hedging_count += len(re.findall(p, lower))

    # -- Tool use metrics ------------------------------------------------------
    tool_metrics = {
        "total_searches": 0,
        "search_queries": [],
    }
    if tool_calls:
        tool_metrics["total_searches"] = len(tool_calls)
        tool_metrics["search_queries"] = [tc.get("query", "") for tc in tool_calls]

    # -- Word count ------------------------------------------------------------
    word_count = len(response_text.split())

    return {
        "vritti_tags": vritti_tags,
        "claim_count": claim_count,
        "nyaya_steps": nyaya_steps,
        "grounded_examples": grounded_examples,
        "hedging_count": hedging_count,
        "tool_metrics": tool_metrics,
        "word_count": word_count,
    }
