"""Yoga — SFT Curriculum (Sutra-Grounded)

Yoga's 8 limbs (Ashtanga) as a structured SFT training curriculum.
Each stage is grounded in actual Yoga Sutra text with Sanskrit key concepts.

Unlike the DPO approach (yoga_dpo.py) which requires contrastive pairs,
SFT teaches "respond like THIS" — each stage teaches a specific quality.

Hypothesis: SFT training in Ashtanga order (1→5) produces better models
than shuffled SFT, because progressive development of faculties matters.

Reference: Patanjali Yoga Sutras (YS), Sadhana Pada (Chapter 2) & Vibhuti Pada (Chapter 3)
"""

YOGA_SFT_STAGES = [
    {
        "stage": 1,
        "name": "Yama/Niyama",
        "sanskrit": "यम/नियम",
        "sutra_refs": ["YS 2.30", "YS 2.32"],
        "sutra_text": {
            "YS 2.30": "ahimsa-satya-asteya-brahmacharya-aparigraha yamah",
            "YS 2.32": "shaucha-santosha-tapah-svadhyaya-ishvarapranidhanani niyamah",
        },
        "sutra_concepts": {
            "satya": "truthfulness — the foundation of right speech and thought",
            "ahimsa": "non-harm — do not mislead or cause confusion",
            "tapas": "discipline — rigorous self-examination before asserting",
        },
        "sft_criterion": "epistemic_honesty",
        "ideal_traits": [
            "Acknowledges uncertainty explicitly when not confident",
            "Does not hallucinate facts, examples, or statistics",
            "Distinguishes established knowledge from inference",
            "Uses hedging language for uncertain claims",
            "Admits knowledge boundaries rather than fabricating",
        ],
    },
    {
        "stage": 2,
        "name": "Asana/Pranayama",
        "sanskrit": "आसन/प्राणायाम",
        "sutra_refs": ["YS 2.46", "YS 2.49"],
        "sutra_text": {
            "YS 2.46": "sthira-sukham asanam",
            "YS 2.49": "tasmin sati shvasa-prashvasayor gati-vicchedah pranayamah",
        },
        "sutra_concepts": {
            "sthira-sukham": "steady and comfortable — stable without rigidity",
            "gati-vicchedah": "regulated flow — controlled pacing of breath/output",
        },
        "sft_criterion": "structural_stability",
        "ideal_traits": [
            "Well-organized with clear, consistent structure",
            "Logical flow from introduction through body to conclusion",
            "Measured pacing — neither rushed nor padded",
            "Consistent formatting and paragraph structure",
            "Each section builds naturally on the previous one",
        ],
    },
    {
        "stage": 3,
        "name": "Pratyahara",
        "sanskrit": "प्रत्याहार",
        "sutra_refs": ["YS 2.54", "YS 2.55"],
        "sutra_text": {
            "YS 2.54": "sva-vishaya-asamprayoge chittasya svarupanukarah iva indriyanam pratyaharah",
            "YS 2.55": "tatah parama vashyata indriyanam",
        },
        "sutra_concepts": {
            "vishaya-asamprayoge": "disengagement from sense objects — ignoring distractions",
            "parama-vashyata": "supreme mastery — complete control over attention",
        },
        "sft_criterion": "focus",
        "ideal_traits": [
            "Directly addresses the specific question asked",
            "Ignores tangential associations and distractions",
            "No filler content or unnecessary preamble",
            "Every paragraph serves the core query",
            "Resists the pull of related-but-unasked topics",
        ],
    },
    {
        "stage": 4,
        "name": "Dharana/Dhyana",
        "sanskrit": "धारणा/ध्यान",
        "sutra_refs": ["YS 3.1", "YS 3.2"],
        "sutra_text": {
            "YS 3.1": "desha-bandha chittasya dharana",
            "YS 3.2": "tatra pratyaya-ekatanata dhyanam",
        },
        "sutra_concepts": {
            "desha-bandha": "fixing attention on a single point — deep concentration",
            "pratyaya-ekatanata": "unbroken flow of cognition — sustained analysis",
        },
        "sft_criterion": "analytical_depth",
        "ideal_traits": [
            "Explains WHY and HOW, not just WHAT",
            "Traces causal chains and mechanisms",
            "Provides specific evidence and concrete examples",
            "Goes beyond surface description to genuine insight",
            "Sustained analysis of key points rather than shallow breadth",
        ],
    },
    {
        "stage": 5,
        "name": "Samadhi",
        "sanskrit": "समाधि",
        "sutra_refs": ["YS 3.3"],
        "sutra_text": {
            "YS 3.3": "tad eva artha-matra-nirbhasam svarupa-shunyam iva samadhih",
        },
        "sutra_concepts": {
            "artha-matra-nirbhasam": "the object alone shines — pure understanding without noise",
            "trayam-ekatra": "the three (dharana, dhyana, samadhi) as one — unified cognition",
        },
        "sft_criterion": "synthesis",
        "ideal_traits": [
            "Ties all points into a unified, coherent insight",
            "Shows how different aspects connect and reinforce each other",
            "The whole is greater than the sum of its parts",
            "Provides an actionable or illuminating takeaway",
            "Final synthesis transcends the individual points made",
        ],
    },
]


# -- SFT Prompt Template -------------------------------------------------------

SFT_GENERATION_SYSTEM = """You are generating an ideal training response for a language model.

YOGA STAGE: {stage_name} (Stage {stage_num}/5)
SUTRA FOUNDATION: {sutra_refs_str}
KEY CONCEPT: {concept_summary}

QUALITY DIMENSION: {sft_criterion}

The response MUST demonstrate these qualities:
{ideal_traits_str}

INSTRUCTIONS:
- Write an ideal response to the question below (200-350 words)
- The response should EXEMPLIFY the quality dimension above
- Be factually accurate and substantive
- Write as the model would naturally respond (no meta-commentary)
- Do NOT mention the stage, sutra, or training framework in the response

Respond with ONLY a JSON object:
{{
  "response": "the ideal response text",
  "stage": {stage_num},
  "quality_dimension": "{sft_criterion}"
}}"""


def format_sft_prompt(question, stage_num):
    """Format an SFT generation prompt for a given question and Yoga stage.

    Args:
        question: the question text
        stage_num: 1-5

    Returns:
        tuple: (system_prompt, user_content) for Claude API call
    """
    if stage_num < 1 or stage_num > 5:
        raise ValueError(f"Stage must be 1-5, got {stage_num}")

    stage = YOGA_SFT_STAGES[stage_num - 1]
    ideal_traits = "\n".join(f"- {t}" for t in stage["ideal_traits"])
    sutra_refs = ", ".join(stage["sutra_refs"])
    concepts = "; ".join(f"{k}: {v}" for k, v in stage["sutra_concepts"].items())

    system = SFT_GENERATION_SYSTEM.format(
        stage_name=stage["name"],
        stage_num=stage_num,
        sutra_refs_str=sutra_refs,
        concept_summary=concepts,
        sft_criterion=stage["sft_criterion"],
        ideal_traits_str=ideal_traits,
    )

    return system, question


def get_stage(stage_num):
    """Get a single SFT stage definition.

    Args:
        stage_num: 1-5

    Returns:
        dict with all stage info
    """
    if stage_num < 1 or stage_num > 5:
        raise ValueError(f"Stage must be 1-5, got {stage_num}")
    return YOGA_SFT_STAGES[stage_num - 1]
