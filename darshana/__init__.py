"""Darshana LLM — 7 Darshanas as LLM Components

Each Darshana maps to a specific layer of the LLM stack:

  Samkhya    → Pretraining data organization (25 tattvas as categories)
  Yoga       → Post-training curriculum (8 limbs as DPO stages)
  Vritti     → System prompt epistemic self-classification
  Mimamsa    → Prompt design methodology (6 Lingas)
  Nyaya      → Tool use routing (4 pramanas)
  Vaisheshika→ Evaluation framework (7 padarthas)
  Vedanta    → Output synthesis (Brahman/Maya/Atman)
"""

# Copied from vedic_llm (proven in Phase 7)
from .vritti import add_epistemic_tags, VRITTI_SYSTEM, VRITTI_INLINE_PROMPT
from .nyaya import generate_with_tools, NYAYA_TOOLS, NYAYA_SYSTEM
from .mimamsa import preprocess_query, MIMAMSA_SYSTEM
from .vaisheshika import extract_structure
from .formatter import clean_format
from .search import search

# New for darshana_llm
from .samkhya import SAMKHYA_CATEGORIES, categorize_text
from .yoga_dpo import YOGA_STAGES, generate_stage_criteria
from .nyaya_router import classify_pramana, route_query
from .vaisheshika_judge import PADARTHA_JUDGE_SYSTEM, judge_with_padarthas
from .vedanta_synth import synthesize_response
