"""Yoga — Post-Training (DPO) Curriculum

Yoga's 8 limbs (Ashtanga) as a structured DPO training curriculum.
Hypothesis: training DPO pairs in Yoga's prescribed ORDER produces
better alignment than random-order DPO.

The 8 limbs map to 5 DPO stages:
  Stage 1 (Yama/Niyama)       → Honesty, no hallucination, acknowledges limits
  Stage 2 (Asana/Pranayama)   → Stable structure, consistent formatting
  Stage 3 (Pratyahara)        → Focus on query, ignores tangents
  Stage 4 (Dharana/Dhyana)    → Deep analysis of one topic vs shallow breadth
  Stage 5 (Samadhi)           → Coherent synthesis, unified conclusion
"""

YOGA_STAGES = [
    {
        "stage": 1,
        "name": "Yama/Niyama",
        "sanskrit": "यम/नियम",
        "principle": "Ethical foundations — truthfulness and self-discipline",
        "dpo_criterion": "honesty_and_humility",
        "preferred_traits": [
            "Acknowledges uncertainty when not confident",
            "Does not hallucinate facts or examples",
            "Admits knowledge boundaries explicitly",
            "Uses hedging language for uncertain claims",
        ],
        "rejected_traits": [
            "Confidently states incorrect information",
            "Fabricates examples or statistics",
            "Never acknowledges uncertainty",
            "Presents inference as established fact",
        ],
    },
    {
        "stage": 2,
        "name": "Asana/Pranayama",
        "sanskrit": "आसन/प्राणायाम",
        "principle": "Stable posture and controlled breath — structural stability",
        "dpo_criterion": "structural_consistency",
        "preferred_traits": [
            "Well-organized with clear structure",
            "Consistent formatting throughout",
            "Logical flow from point to point",
            "Appropriate use of headers and lists",
        ],
        "rejected_traits": [
            "Disorganized with jumping between topics",
            "Inconsistent formatting",
            "No clear structure or progression",
            "Wall of text without breaks",
        ],
    },
    {
        "stage": 3,
        "name": "Pratyahara",
        "sanskrit": "प्रत्याहार",
        "principle": "Withdrawal of senses — focus, ignore distractions",
        "dpo_criterion": "focus_and_relevance",
        "preferred_traits": [
            "Directly addresses the question asked",
            "Stays on topic throughout",
            "Ignores tangential associations",
            "Every paragraph serves the core question",
        ],
        "rejected_traits": [
            "Goes off on tangents",
            "Includes irrelevant information",
            "Addresses related but unasked questions",
            "Pads response with filler content",
        ],
    },
    {
        "stage": 4,
        "name": "Dharana/Dhyana",
        "sanskrit": "धारणा/ध्यान",
        "principle": "Concentration and meditation — depth over breadth",
        "dpo_criterion": "analytical_depth",
        "preferred_traits": [
            "Deep analysis of key mechanisms",
            "Explains WHY and HOW, not just WHAT",
            "Explores causal chains and implications",
            "Provides specific evidence for claims",
        ],
        "rejected_traits": [
            "Surface-level overview only",
            "Lists facts without explanation",
            "Describes WHAT without WHY",
            "Broad coverage with no depth on any point",
        ],
    },
    {
        "stage": 5,
        "name": "Samadhi",
        "sanskrit": "समाधि",
        "principle": "Integration — unified understanding",
        "dpo_criterion": "coherent_synthesis",
        "preferred_traits": [
            "Ties points together into coherent conclusion",
            "Shows how different aspects relate",
            "Provides actionable takeaway or insight",
            "Final synthesis goes beyond listing points",
        ],
        "rejected_traits": [
            "Ends abruptly without conclusion",
            "Points remain disconnected",
            "No synthesis or integration",
            "Conclusion is just a restatement of the question",
        ],
    },
]


def generate_stage_criteria(stage_num):
    """Get the DPO preference criteria for a given Yoga stage.

    Args:
        stage_num: 1-5, corresponding to the Yoga stages

    Returns:
        dict with 'preferred_traits' and 'rejected_traits' for DPO pair generation
    """
    if stage_num < 1 or stage_num > 5:
        raise ValueError(f"Stage must be 1-5, got {stage_num}")
    return YOGA_STAGES[stage_num - 1]


DPO_PAIR_GENERATION_PROMPT = """Generate a DPO preference pair for the following question.

QUESTION: {question}

YOGA STAGE: {stage_name} ({stage_num}/5)
QUALITY DIMENSION: {dpo_criterion}

The PREFERRED response should demonstrate:
{preferred_traits}

The REJECTED response should demonstrate:
{rejected_traits}

IMPORTANT:
- Both responses should attempt to answer the question
- The rejected response is NOT garbage — it's a plausible but inferior answer
- The quality difference should be SPECIFICALLY about the dimension above
- Keep each response 150-300 words

Respond with ONLY a JSON object:
{{
  "preferred": "the better response text",
  "rejected": "the worse response text",
  "quality_dimension": "{dpo_criterion}",
  "stage": {stage_num}
}}"""


def format_pair_prompt(question, stage_num):
    """Format a DPO pair generation prompt for a given question and Yoga stage.

    Args:
        question: the question text
        stage_num: 1-5

    Returns:
        str: formatted prompt for Claude to generate a DPO pair
    """
    stage = YOGA_STAGES[stage_num - 1]
    preferred = "\n".join(f"- {t}" for t in stage["preferred_traits"])
    rejected = "\n".join(f"- {t}" for t in stage["rejected_traits"])

    return DPO_PAIR_GENERATION_PROMPT.format(
        question=question,
        stage_name=stage["name"],
        stage_num=stage_num,
        dpo_criterion=stage["dpo_criterion"],
        preferred_traits=preferred,
        rejected_traits=rejected,
    )
