"""Darshana LLM — 7 Darshanas as LLM Components

Each Darshana maps to a specific layer of the LLM stack:

  Samkhya    → Pretraining data organization (25 tattvas as categories)
  Yoga       → Post-training curriculum (8 limbs as DPO stages)
  Vritti     → System prompt epistemic self-classification
  Mimamsa    → Prompt design methodology (6 Lingas)
  Nyaya      → Tool use routing (4 pramanas)
  Vaisheshika→ Evaluation framework (7 padarthas)
  Vedanta    → Output synthesis (Brahman/Maya/Atman)

Each module also includes a GENERIC equivalent for controlled comparison.
"""

# Copied from vedic_llm (proven in Phase 7)
from .vritti import add_epistemic_tags, VRITTI_SYSTEM, VRITTI_INLINE_PROMPT
from .vritti import GENERIC_CONFIDENCE_PROMPT, GENERIC_COT_PROMPT, VRITTI_CONTEMPORARY_PROMPT
from .nyaya import generate_with_tools, NYAYA_TOOLS, NYAYA_SYSTEM
from .mimamsa import preprocess_query, MIMAMSA_SYSTEM
from .mimamsa import GENERIC_REWRITE_SYSTEM, generic_rewrite
from .vaisheshika import extract_structure
from .formatter import clean_format
from .search import search

# New for darshana_llm
from .samkhya import SAMKHYA_CATEGORIES, categorize_text, bloom_ordered_corpus
from .yoga_dpo import YOGA_STAGES, generate_stage_criteria
from .yoga_dpo import reverse_stage_order, generic_curriculum_order, format_generic_pair_prompt
from .nyaya_router import classify_pramana, route_query, heuristic_route, model_decides_route
from .vaisheshika_judge import PADARTHA_JUDGE_SYSTEM, judge_with_padarthas
from .vaisheshika_judge import GENERIC_7DIM_JUDGE_SYSTEM, judge_with_generic_7dim
from .vedanta_synth import synthesize_response, VEDANTA_SYNTHESIS_SYSTEM
from .vedanta_synth import GENERIC_SYNTHESIS_SYSTEM, generic_synthesize
from .vaisheshika_ontology import (
    extract_padarthas, extract_generic,
    build_padartha_graph, build_generic_graph,
    format_padartha_context, format_generic_context,
    PADARTHA_EXTRACTION_SYSTEM, GENERIC_EXTRACTION_SYSTEM,
)
