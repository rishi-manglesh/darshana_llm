"""Vaisheshika Ontology — 7 Padarthas as Knowledge Graph Schema

Vaisheshika's 7 Padarthas are an ONTOLOGY (what exists), not evaluation criteria.
They map to knowledge graph organization:

    Dravya (Substance)      → Entities / nodes
    Guna (Quality)          → Node attributes / properties
    Karma (Action)          → Capabilities / operations
    Samanya (Universality)  → Taxonomy / categories
    Vishesha (Particularity)→ Differentiators
    Samavaya (Inherence)    → Structural relationships
    Abhava (Absence)        → Gap tracking — what's MISSING

The killer feature is Abhava: explicitly tracking what's NOT in the knowledge
lets us answer constraint questions ("what limits X?") with superior precision.

Generic control: standard entity/relation/attribute extraction.
"""

import json
import re
from dataclasses import dataclass, field


def _extract_json(raw_text):
    """Robustly extract JSON from LLM output that may include prose wrapping.

    Tries in order:
    1. Direct JSON parse
    2. Extract from ```json ... ``` or ``` ... ``` code blocks
    3. Find first { ... } substring (greedy)
    """
    text = raw_text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Code block extraction
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find outermost { ... }
    start = text.find('{')
    if start != -1:
        # Find matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break

    return None

# -- Dataclasses ---------------------------------------------------------------

@dataclass
class PadarthaNode:
    """A knowledge node organized by 7 Padarthas."""
    name: str
    dravya: str = ""              # What is it? (substance/entity)
    guna: list = field(default_factory=list)       # Properties/attributes
    karma: list = field(default_factory=list)      # Actions/capabilities
    samanya: list = field(default_factory=list)     # Categories/taxonomy
    vishesha: list = field(default_factory=list)    # Differentiators
    samavaya: list = field(default_factory=list)    # Inherent relationships
    abhava: list = field(default_factory=list)      # What's missing/absent


@dataclass
class GenericNode:
    """A knowledge node using standard entity-relation schema."""
    name: str
    entity_type: str = ""
    attributes: list = field(default_factory=list)
    capabilities: list = field(default_factory=list)
    categories: list = field(default_factory=list)
    differentiators: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    gaps: list = field(default_factory=list)


@dataclass
class KnowledgeGraph:
    """In-memory knowledge graph with indexes."""
    nodes: dict = field(default_factory=dict)       # name -> node
    samanya_index: dict = field(default_factory=dict)  # category -> [node names]
    abhava_index: list = field(default_factory=list)   # list of absence records


# -- Extraction Prompts --------------------------------------------------------

PADARTHA_EXTRACTION_SYSTEM = """You are a knowledge extractor using Vaisheshika ontology (7 Padarthas).

Extract structured knowledge from the text using these 7 categories:

1. DRAVYA (Substance): The core entities, concepts, or things discussed.
2. GUNA (Quality): Properties, attributes, or characteristics of each entity.
3. KARMA (Action): What each entity DOES — capabilities, operations, mechanisms.
4. SAMANYA (Universality): Categories, taxonomies — what class does each entity belong to?
5. VISHESHA (Particularity): What makes each entity UNIQUE — differentiators, distinguishing features.
6. SAMAVAYA (Inherence): Structural, non-incidental relationships — what is inherently connected?
7. ABHAVA (Absence): What is MISSING, NOT discussed, limited, or constrained. Gaps in the knowledge.

For ABSENCE, identify 4 types:
- PRIOR GAP: Things that don't exist YET (planned but not built)
- CEASED: Things that CEASED to exist or were deprecated
- IMPOSSIBLE: Things that are IMPOSSIBLE in this context (fundamental constraints)
- MUTUAL EXCLUSION: Things that are mutually exclusive (choosing A means not-B)

Extract the 5-8 MOST IMPORTANT entities. Be concise — use short phrases, not sentences.

Respond with ONLY a JSON object:
{
  "entities": [
    {
      "name": "<entity name>",
      "dravya": "<what it is>",
      "guna": ["<property1>", "<property2>"],
      "karma": ["<action1>", "<action2>"],
      "samanya": ["<category1>"],
      "vishesha": ["<differentiator1>"],
      "samavaya": ["<inherent_relation: target>"],
      "abhava": ["<type: what is absent>"]
    }
  ],
  "global_abhava": ["<gaps/absences spanning multiple entities>"]
}"""

GENERIC_EXTRACTION_SYSTEM = """You are a knowledge extractor using standard entity-relation schema.

Extract structured knowledge from the text:

1. ENTITIES: The core concepts, things, or ideas discussed.
2. ATTRIBUTES: Properties, characteristics of each entity.
3. CAPABILITIES: What each entity does — actions, functions, mechanisms.
4. CATEGORIES: What class or type does each entity belong to?
5. DIFFERENTIATORS: What makes each entity unique compared to similar things?
6. RELATIONSHIPS: How entities connect — causes, enables, depends-on, part-of.
7. GAPS: What is not covered, missing, limited, or constrained in the knowledge.

Extract the 5-8 MOST IMPORTANT entities. Be concise — use short phrases, not sentences.

Respond with ONLY a JSON object:
{
  "entities": [
    {
      "name": "<entity name>",
      "entity_type": "<what it is>",
      "attributes": ["<attr1>", "<attr2>"],
      "capabilities": ["<cap1>", "<cap2>"],
      "categories": ["<cat1>"],
      "differentiators": ["<diff1>"],
      "relationships": ["<rel_type: target>"],
      "gaps": ["<gap1>"]
    }
  ],
  "global_gaps": ["<gaps spanning multiple entities>"]
}"""


# -- Extraction Functions ------------------------------------------------------

def extract_padarthas(client, model, text):
    """Extract 7 Padarthas from text using LLM.

    Returns:
        dict with 'entities' list and 'global_abhava', or None on failure.
    """
    import time
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=4096,
                system=PADARTHA_EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": f"Extract knowledge from:\n\n{text[:4000]}"}],
            )
            raw = msg.content[0].text.strip()
            parsed = _extract_json(raw)
            if parsed and "entities" in parsed:
                return parsed
            if attempt < 2:
                time.sleep(1)
                continue
            print(f"    [WARN] Padartha extraction failed. Stop reason: {msg.stop_reason}, Raw[:300]: {raw[:300]}")
            return None
        except Exception as e:
            if "rate" in str(e).lower() or "overloaded" in str(e).lower():
                time.sleep(2 ** (attempt + 1))
                continue
            print(f"    [WARN] Padartha extraction error: {e}")
            if attempt < 2:
                time.sleep(1)
                continue
            return None
    return None


def extract_generic(client, model, text):
    """Extract standard entity-relation schema from text using LLM.

    Returns:
        dict with 'entities' list and 'global_gaps', or None on failure.
    """
    import time
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=4096,
                system=GENERIC_EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": f"Extract knowledge from:\n\n{text[:4000]}"}],
            )
            raw = msg.content[0].text.strip()
            parsed = _extract_json(raw)
            if parsed and "entities" in parsed:
                return parsed
            if attempt < 2:
                time.sleep(1)
                continue
            print(f"    [WARN] Generic extraction failed. Stop reason: {msg.stop_reason}, Raw[:300]: {raw[:300]}")
            return None
        except Exception as e:
            if "rate" in str(e).lower() or "overloaded" in str(e).lower():
                time.sleep(2 ** (attempt + 1))
                continue
            print(f"    [WARN] Generic extraction error: {e}")
            if attempt < 2:
                time.sleep(1)
                continue
            return None
    return None


# -- Graph Building -----------------------------------------------------------

def build_padartha_graph(extractions):
    """Build a KnowledgeGraph from Padartha extractions.

    Args:
        extractions: list of dicts from extract_padarthas()

    Returns:
        KnowledgeGraph with nodes, samanya_index, and abhava_index
    """
    graph = KnowledgeGraph()

    for ext in extractions:
        if ext is None:
            continue
        for entity in ext.get("entities", []):
            name = entity.get("name", "unknown")
            node = PadarthaNode(
                name=name,
                dravya=entity.get("dravya", ""),
                guna=entity.get("guna", []),
                karma=entity.get("karma", []),
                samanya=entity.get("samanya", []),
                vishesha=entity.get("vishesha", []),
                samavaya=entity.get("samavaya", []),
                abhava=entity.get("abhava", []),
            )
            graph.nodes[name] = node

            # Build samanya (taxonomy) index
            for cat in node.samanya:
                graph.samanya_index.setdefault(cat, []).append(name)

            # Build abhava (absence) index
            for absence in node.abhava:
                graph.abhava_index.append({"entity": name, "absence": absence})

        # Global absences
        for gap in ext.get("global_abhava", []):
            graph.abhava_index.append({"entity": "_global", "absence": gap})

    return graph


def build_generic_graph(extractions):
    """Build a KnowledgeGraph from generic extractions.

    Args:
        extractions: list of dicts from extract_generic()

    Returns:
        KnowledgeGraph with nodes, samanya_index (as categories), and abhava_index (as gaps)
    """
    graph = KnowledgeGraph()

    for ext in extractions:
        if ext is None:
            continue
        for entity in ext.get("entities", []):
            name = entity.get("name", "unknown")
            node = GenericNode(
                name=name,
                entity_type=entity.get("entity_type", ""),
                attributes=entity.get("attributes", []),
                capabilities=entity.get("capabilities", []),
                categories=entity.get("categories", []),
                differentiators=entity.get("differentiators", []),
                relationships=entity.get("relationships", []),
                gaps=entity.get("gaps", []),
            )
            graph.nodes[name] = node

            for cat in node.categories:
                graph.samanya_index.setdefault(cat, []).append(name)

            for gap in node.gaps:
                graph.abhava_index.append({"entity": name, "absence": gap})

        for gap in ext.get("global_gaps", []):
            graph.abhava_index.append({"entity": "_global", "absence": gap})

    return graph


# -- Retrieval Functions -------------------------------------------------------

def retrieve_padartha(graph, query, query_type="general"):
    """Retrieve context from a Padartha graph, adapted to query type.

    Query-type-aware retrieval:
    - discrimination → emphasize Vishesha (differentiators)
    - force → emphasize Karma (actions) + Guna (mechanisms)
    - constraint → emphasize Abhava (what's missing/limiting)

    Args:
        graph: KnowledgeGraph with PadarthaNode entries
        query: the user's question
        query_type: 'discrimination', 'force', or 'constraint'

    Returns:
        str: formatted context for the LLM
    """
    if not graph.nodes:
        return "No knowledge extracted."

    sections = []

    # Always include entities overview
    entity_lines = []
    for name, node in graph.nodes.items():
        if not isinstance(node, PadarthaNode):
            continue
        entity_lines.append(f"- **{name}** ({node.dravya})")
    if entity_lines:
        sections.append("## Entities\n" + "\n".join(entity_lines))

    if query_type == "discrimination":
        # Emphasize Vishesha (differentiators) and Samanya (taxonomy)
        vis_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, PadarthaNode):
                continue
            if node.vishesha:
                vis_lines.append(f"- **{name}**: {'; '.join(node.vishesha)}")
        if vis_lines:
            sections.append("## Key Differentiators\n" + "\n".join(vis_lines))

        # Taxonomy for grouping
        if graph.samanya_index:
            tax_lines = []
            for cat, members in graph.samanya_index.items():
                tax_lines.append(f"- **{cat}**: {', '.join(members)}")
            sections.append("## Categories\n" + "\n".join(tax_lines))

    elif query_type == "force":
        # Emphasize Karma (actions) and Guna (mechanisms/properties)
        karma_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, PadarthaNode):
                continue
            if node.karma:
                karma_lines.append(f"- **{name}**: {'; '.join(node.karma)}")
        if karma_lines:
            sections.append("## Actions & Mechanisms\n" + "\n".join(karma_lines))

        guna_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, PadarthaNode):
                continue
            if node.guna:
                guna_lines.append(f"- **{name}**: {'; '.join(node.guna)}")
        if guna_lines:
            sections.append("## Properties\n" + "\n".join(guna_lines))

        # Inherent relationships
        rel_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, PadarthaNode):
                continue
            if node.samavaya:
                rel_lines.append(f"- **{name}**: {'; '.join(node.samavaya)}")
        if rel_lines:
            sections.append("## Structural Relationships\n" + "\n".join(rel_lines))

    elif query_type == "constraint":
        # Emphasize Abhava (absences/gaps/limits)
        if graph.abhava_index:
            ab_lines = []
            for rec in graph.abhava_index:
                entity = rec["entity"]
                prefix = "" if entity == "_global" else f"[{entity}] "
                ab_lines.append(f"- {prefix}{rec['absence']}")
            sections.append("## Absences & Constraints\n" + "\n".join(ab_lines))

        # Also include properties that define limits
        guna_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, PadarthaNode):
                continue
            if node.guna:
                guna_lines.append(f"- **{name}**: {'; '.join(node.guna)}")
        if guna_lines:
            sections.append("## Properties\n" + "\n".join(guna_lines))

    else:
        # General: balanced overview of all padarthas
        for label, attr in [
            ("Properties (Guna)", "guna"),
            ("Actions (Karma)", "karma"),
            ("Differentiators (Vishesha)", "vishesha"),
            ("Relationships (Samavaya)", "samavaya"),
        ]:
            lines = []
            for name, node in graph.nodes.items():
                if not isinstance(node, PadarthaNode):
                    continue
                vals = getattr(node, attr, [])
                if vals:
                    lines.append(f"- **{name}**: {'; '.join(vals)}")
            if lines:
                sections.append(f"## {label}\n" + "\n".join(lines))

        if graph.abhava_index:
            ab_lines = [f"- {r['absence']}" for r in graph.abhava_index[:5]]
            sections.append("## Notable Absences\n" + "\n".join(ab_lines))

    return "\n\n".join(sections) if sections else "No relevant knowledge found."


def retrieve_generic(graph, query, query_type="general"):
    """Retrieve context from a generic graph, adapted to query type.

    Same query-type adaptation as retrieve_padartha but using generic labels.

    Args:
        graph: KnowledgeGraph with GenericNode entries
        query: the user's question
        query_type: 'discrimination', 'force', or 'constraint'

    Returns:
        str: formatted context for the LLM
    """
    if not graph.nodes:
        return "No knowledge extracted."

    sections = []

    # Always include entities
    entity_lines = []
    for name, node in graph.nodes.items():
        if not isinstance(node, GenericNode):
            continue
        entity_lines.append(f"- **{name}** ({node.entity_type})")
    if entity_lines:
        sections.append("## Entities\n" + "\n".join(entity_lines))

    if query_type == "discrimination":
        diff_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, GenericNode):
                continue
            if node.differentiators:
                diff_lines.append(f"- **{name}**: {'; '.join(node.differentiators)}")
        if diff_lines:
            sections.append("## Differentiators\n" + "\n".join(diff_lines))

        if graph.samanya_index:
            cat_lines = [f"- **{c}**: {', '.join(m)}" for c, m in graph.samanya_index.items()]
            sections.append("## Categories\n" + "\n".join(cat_lines))

    elif query_type == "force":
        cap_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, GenericNode):
                continue
            if node.capabilities:
                cap_lines.append(f"- **{name}**: {'; '.join(node.capabilities)}")
        if cap_lines:
            sections.append("## Capabilities & Mechanisms\n" + "\n".join(cap_lines))

        attr_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, GenericNode):
                continue
            if node.attributes:
                attr_lines.append(f"- **{name}**: {'; '.join(node.attributes)}")
        if attr_lines:
            sections.append("## Attributes\n" + "\n".join(attr_lines))

        rel_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, GenericNode):
                continue
            if node.relationships:
                rel_lines.append(f"- **{name}**: {'; '.join(node.relationships)}")
        if rel_lines:
            sections.append("## Relationships\n" + "\n".join(rel_lines))

    elif query_type == "constraint":
        if graph.abhava_index:
            gap_lines = []
            for rec in graph.abhava_index:
                entity = rec["entity"]
                prefix = "" if entity == "_global" else f"[{entity}] "
                gap_lines.append(f"- {prefix}{rec['absence']}")
            sections.append("## Gaps & Constraints\n" + "\n".join(gap_lines))

        attr_lines = []
        for name, node in graph.nodes.items():
            if not isinstance(node, GenericNode):
                continue
            if node.attributes:
                attr_lines.append(f"- **{name}**: {'; '.join(node.attributes)}")
        if attr_lines:
            sections.append("## Attributes\n" + "\n".join(attr_lines))

    else:
        for label, attr in [
            ("Attributes", "attributes"),
            ("Capabilities", "capabilities"),
            ("Differentiators", "differentiators"),
            ("Relationships", "relationships"),
        ]:
            lines = []
            for name, node in graph.nodes.items():
                if not isinstance(node, GenericNode):
                    continue
                vals = getattr(node, attr, [])
                if vals:
                    lines.append(f"- **{name}**: {'; '.join(vals)}")
            if lines:
                sections.append(f"## {label}\n" + "\n".join(lines))

        if graph.abhava_index:
            gap_lines = [f"- {r['absence']}" for r in graph.abhava_index[:5]]
            sections.append("## Notable Gaps\n" + "\n".join(gap_lines))

    return "\n\n".join(sections) if sections else "No relevant knowledge found."


# -- Context Formatting --------------------------------------------------------

def format_padartha_context(graph, query, query_type="general"):
    """Format Padartha graph context for RAG injection.

    Returns a string suitable for inclusion in an LLM prompt as reference context.
    """
    retrieved = retrieve_padartha(graph, query, query_type)
    return (
        "## Reference Knowledge (7-category organized)\n\n"
        f"The following knowledge was extracted and organized using a 7-category ontology "
        f"(entity, properties, actions, categories, differentiators, relationships, absences). "
        f"Use it to inform your answer.\n\n"
        f"{retrieved}"
    )


def format_generic_context(graph, query, query_type="general"):
    """Format generic graph context for RAG injection.

    Returns a string suitable for inclusion in an LLM prompt as reference context.
    """
    retrieved = retrieve_generic(graph, query, query_type)
    return (
        "## Reference Knowledge (Entity-organized)\n\n"
        f"The following knowledge was extracted and organized from source documents. "
        f"Use it to inform your answer.\n\n"
        f"{retrieved}"
    )
