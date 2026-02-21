# Cross-Model Validation — Experiment Design

> **Status**: DESIGN
> **Last Updated**: 2026-02-20
> **Prerequisite**: Exp 1-5 PROVEN on Claude Sonnet (API)

## Problem

All 5 proven experiments ran on **one model family** (Anthropic Claude Sonnet + Haiku judge). This is a critical gap:

1. DVERSI supports BYOK (Bring Your Own Key) — techniques must work across providers
2. Win rates measured against Claude may not transfer to other models
3. Smaller/cheaper models may benefit MORE from structured techniques (the real value proposition)
4. If a technique only works on Claude, it's a Claude quirk, not a generalizable principle

## Research Question

> Do Darshana-derived techniques maintain their advantage over generic equivalents
> when the underlying model changes?

**Hypothesis:** Structured techniques (Vritti, Vaisheshika, Vedanta) will show LARGER
gains on open-source models because they compensate for reduced reasoning capacity.
Mimamsa (query rewriting) should be model-agnostic since it preprocesses the input.

## Experiments to Validate

| Exp | Technique | Layer | Original h2h | Why Validate |
|-----|-----------|-------|-------------|-------------|
| 1 | Vritti (Epistemic Calibration) | System prompt | 60% | Applied to ALL DVERSI agents — must work everywhere |
| 3 | Mimamsa (6 Lingas Rewriting) | Query preprocessing | 73% | Designed as separate LLM call — must justify cost on cheaper models |
| 4 | Vaisheshika (7-category Context) | Knowledge formatting | 68% | Core of DVERSI's context injection — used on every query |
| 5 | Vedanta (Response Synthesis) | Output post-processing | 63% | Applied to multi-source responses — must work across models |

**Skip Exp 2 (Nyaya routing):** Removed from DVERSI — tool call efficiency comes from
model training, not hardcoded regex classification.

## Model Selection

### Smoke Test: Anthropic Size Effect

Quick validation within the same family — isolates model size from provider differences.

| Model | Role | Why |
|-------|------|-----|
| Claude Sonnet 4.5 | **Existing baseline** | Already have full results |
| Claude Haiku 4.5 | Size-effect check | Same family, smaller — do techniques help MORE? |

Run Haiku on 3 questions per experiment (smoke test only). If results directionally
match Sonnet, model size within the same family isn't a concern. If Haiku shows
significantly different patterns, that's interesting data before running open-source.

### Primary: Open-Source Models (the real validation)

The real question: do techniques generalize beyond Claude?

| Model | Provider | Why |
|-------|----------|-----|
| Llama 3.3 70B | Meta (via Together/Groq) | Most popular open-source, different training methodology |
| Qwen 2.5 72B | Alibaba (via Together) | Strong multilingual, different architecture + training data |
| Mistral Large 2 | Mistral (via API) | European alternative, different design philosophy |

These three cover the major open-source families with fundamentally different
training approaches. If techniques work on all three + Claude, they're generalizable.

### Cost Estimate

| Run | Models | Questions | Cost |
|-----|--------|-----------|------|
| Smoke test | Haiku | 3 x 4 experiments | ~$0.05 |
| Full open-source | Llama, Qwen, Mistral | 30 x 4 experiments each | ~$1.00-3.00 |
| Judging (Sonnet) | — | ~360 pairwise comparisons | ~$2.00-3.00 |
| **Total** | | | **~$3-6** |

## Architecture Changes

### Provider Abstraction

Current `utils.py` is hardcoded to Anthropic. Need a minimal provider interface:

```
experiments/
├── utils.py                  # Existing — shared infrastructure
├── providers/                # NEW — model provider abstraction
│   ├── __init__.py
│   ├── base.py               # Abstract LLMProvider
│   ├── anthropic_provider.py # Existing Anthropic logic, extracted
│   └── together_provider.py  # Together AI (Llama, Qwen, Mistral — OpenAI-compatible)
├── cross_model_validation.py # NEW — experiment runner
└── cross_model_analysis.py   # NEW — analysis + visualization
```

Only two providers needed: Anthropic (existing) + Together (covers all three open-source models via OpenAI-compatible API).

### `providers/base.py`

```python
"""Abstract LLM Provider — unified interface for cross-model experiments.

Each provider wraps a specific API and translates to a common interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Unified response from any provider."""
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def call(
        self,
        system: str,
        user_content: str,
        max_tokens: int = 1024,
    ) -> LLMResponse | None:
        """Single-turn text generation."""
        ...

    @abstractmethod
    def call_json(
        self,
        system: str,
        user_content: str,
        max_tokens: int = 500,
    ) -> dict | None:
        """Generate and parse JSON response."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Full model identifier."""
        ...

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Short name for results keys (e.g., 'sonnet', 'llama70b')."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name (e.g., 'anthropic', 'together')."""
        ...
```

### `providers/anthropic_provider.py`

Extract existing `call_api()` and `call_api_json()` from `utils.py` into this class.
Keep the retry logic, rate limit handling. The existing `utils.py` functions
remain as thin wrappers for backward compatibility.

### `providers/together_provider.py`

```python
"""Together AI provider — open-source models via OpenAI-compatible API.

Covers: Llama 3.3 70B, Qwen 2.5 72B, Mistral Large 2
"""

import json
import os
import time
from openai import OpenAI

from .base import LLMProvider, LLMResponse

TOGETHER_MODELS = {
    "llama70b":       "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "qwen72b":        "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "mistral_large":  "mistralai/Mistral-Large-Instruct-2411",
}

class TogetherProvider(LLMProvider):
    def __init__(self, model_key: str = "llama70b"):
        self.client = OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )
        self._model_id = TOGETHER_MODELS[model_key]
        self._short_name = model_key

    def call(self, system, user_content, max_tokens=1024):
        # OpenAI-compatible: system as messages[0], user as messages[1]
        # Retry with exponential backoff on rate limits
        ...

    def call_json(self, system, user_content, max_tokens=500):
        # JSON mode less reliable on open-source — extra retry + parsing fallback
        # Strip markdown fences, attempt json.loads with recovery
        ...

    @property
    def model_id(self): return self._model_id
    @property
    def short_name(self): return self._short_name
    @property
    def provider_name(self): return "together"
```

## Experiment Runner

### `cross_model_validation.py`

```python
"""Cross-Model Validation — Run Exp 1, 3, 4, 5 on open-source models.

Tests whether Darshana-derived techniques generalize beyond Claude.

Usage:
    # Smoke test: Haiku vs Sonnet size effect (3 questions)
    python experiments/cross_model_validation.py --smoke

    # Single open-source model, single experiment
    python experiments/cross_model_validation.py --exp 1 --model llama70b

    # All experiments, all open-source models
    python experiments/cross_model_validation.py --all

    # Judge results
    python experiments/cross_model_validation.py --judge
"""

ALL_MODELS = {
    # Baseline (existing results)
    "sonnet":         ("anthropic", "sonnet"),
    # Smoke test
    "haiku":          ("anthropic", "haiku"),
    # Primary validation
    "llama70b":       ("together", "llama70b"),
    "qwen72b":        ("together", "qwen72b"),
    "mistral_large":  ("together", "mistral_large"),
}

SMOKE_MODELS = ["haiku"]
VALIDATION_MODELS = ["llama70b", "qwen72b", "mistral_large"]

EXPERIMENTS = {
    "exp1": {
        "name": "Vritti Epistemic Calibration",
        "configs": ["bare_baseline", "vritti_contemporary", "generic_confidence"],
        "baseline": "bare_baseline",
        "darshana": "vritti_contemporary",
        "generic": "generic_confidence",
    },
    "exp3": {
        "name": "Mimamsa 6 Lingas Rewriting",
        "configs": ["original", "mimamsa_rewritten", "generic_rewritten"],
        "baseline": "original",
        "darshana": "mimamsa_rewritten",
        "generic": "generic_rewritten",
    },
    "exp4": {
        "name": "Vaisheshika 7-Category Context",
        "configs": ["no_context", "padartha_context", "generic_context"],
        "baseline": "no_context",
        "darshana": "padartha_context",
        "generic": "generic_context",
    },
    "exp5": {
        "name": "Vedanta Response Synthesis",
        "configs": ["raw", "vedanta_synth", "generic_synth"],
        "baseline": "raw",
        "darshana": "vedanta_synth",
        "generic": "generic_synth",
    },
}
```

### Results Structure

```
results/
├── cross_model/
│   ├── exp1_vritti/
│   │   ├── sonnet_results.jsonl      # Copied from exp1_vritti/ (existing)
│   │   ├── haiku_results.jsonl       # Smoke test (3 questions)
│   │   ├── llama70b_results.jsonl    # Full (30 questions)
│   │   ├── qwen72b_results.jsonl
│   │   ├── mistral_large_results.jsonl
│   │   ├── judge_sonnet.jsonl        # Copied from exp1_vritti/
│   │   ├── judge_llama70b.jsonl
│   │   ├── judge_qwen72b.jsonl
│   │   └── judge_mistral_large.jsonl
│   ├── exp3_mimamsa/
│   │   └── ...
│   ├── exp4_vaisheshika/
│   │   └── ...
│   └── exp5_vedanta/
│       └── ...
```

## Judging Strategy

### Judge Model

Use **Claude Sonnet** as the judge for ALL models. Reasoning:

1. Judge must be CONSISTENT — same judge for all comparisons
2. Judge must be CAPABLE — can evaluate responses from all tiers
3. Using a frontier model as judge is standard practice (LMSYS, Chatbot Arena)
4. Using the same judge as the original experiments ensures comparable win rates

### What We Measure

For each (experiment, model) pair:

| Metric | How | What It Tells Us |
|--------|-----|-----------------|
| **h2h win rate** (darshana vs generic) | Pairwise judge | Does the technique help THIS model? |
| **h2h win rate** (darshana vs bare) | Pairwise judge | Magnitude of improvement |
| **Absolute quality** | 5-dimension scores | Quality floor across models |
| **Gain delta** | h2h(model) - h2h(sonnet) | Does technique help MORE on open-source? |

### Key Comparisons

```
WITHIN each model:
  darshana_config vs generic_config (h2h) → "Does it help THIS model?"
  darshana_config vs bare_baseline (h2h)  → "How much does it help?"

ACROSS models:
  sonnet_h2h vs haiku_h2h     → "Size effect within Claude family" (smoke test)
  sonnet_h2h vs llama_h2h     → "Does it transfer to Llama?"
  sonnet_h2h vs qwen_h2h      → "Does it transfer to Qwen?"
  sonnet_h2h vs mistral_h2h   → "Does it transfer to Mistral?"

THE KEY QUESTION:
  If h2h(open_source) > h2h(sonnet) → technique compensates for model weakness → HIGH VALUE
  If h2h(open_source) ≈ h2h(sonnet) → technique is model-agnostic → GOOD
  If h2h(open_source) < h2h(sonnet) → technique needs model capability → CONCERNING
```

## Analysis Output

### `cross_model_analysis.py`

Produces a summary table:

```
CROSS-MODEL VALIDATION RESULTS
═══════════════════════════════════════════════════════════════════════

Exp 1: Vritti Epistemic Calibration
                    darshana vs generic (h2h)    darshana vs bare (h2h)
  ─────────────────────────────────────────────────────────────────────
  BASELINE:
    sonnet              60%                          73%
  SMOKE TEST:
    haiku               ??% (3 questions)            ??%
  OPEN-SOURCE:
    llama70b            ??%                          ??%
    qwen72b             ??%                          ??%
    mistral_large       ??%                          ??%

  Open-source average: ??%
  Δ (open vs sonnet):  +??%  → technique helps MORE/LESS on open-source

[... repeat for Exp 3, 4, 5 ...]

SUMMARY — Which techniques generalize beyond Claude?
═════════════════════════════════════════════════════
  Vritti:       ✓/✗  (works on ??/3 open-source models)
  Mimamsa:      ✓/✗  (works on ??/3 open-source models)
  Vaisheshika:  ✓/✗  (works on ??/3 open-source models)
  Vedanta:      ✓/✗  (works on ??/3 open-source models)
```

## Execution Order

```
PHASE 1 — Provider Abstraction (~1 hr implementation):
  a. Create providers/base.py with LLMProvider ABC
  b. Extract Anthropic logic into providers/anthropic_provider.py
  c. Implement providers/together_provider.py (OpenAI-compatible)
  d. Verify backward compatibility (existing experiments still run)

PHASE 2 — Cross-Model Runner (~1 hr implementation):
  a. Create cross_model_validation.py
  b. Adapt each experiment's generate_fn to accept LLMProvider
  c. Handle model-specific quirks (JSON parsing fallbacks)

PHASE 3 — Smoke Test (~$0.05, ~10 min):
  a. Run Haiku on 3 questions x 4 experiments
  b. Sanity check: do results directionally match Sonnet?
  c. If wildly different → investigate before running open-source

PHASE 4 — Open-Source Validation (~$1-3, ~2-4 hrs wall time):
  a. Llama 3.3 70B — 30 questions x 4 experiments
  b. Qwen 2.5 72B — 30 questions x 4 experiments
  c. Mistral Large 2 — 30 questions x 4 experiments

PHASE 5 — Judging (~$2-3, ~1-2 hrs):
  a. Run pairwise judging with Sonnet as judge for all open-source results
  b. Generate analysis tables

PHASE 6 — Analysis & Conclusions:
  a. Which techniques generalize beyond Claude?
  b. Do techniques help MORE on open-source models?
  c. Update DVERSI's intelligence layer if any technique fails
  d. Decide whether to proceed to Samkhya/Yoga (Exp 6-7)
```

## Exp 3 (Mimamsa) Special Handling

Mimamsa uses a **separate rewriting LLM call** before the main generation. For cross-model:

- **Rewriting call**: Always use the SAME model being tested (not Sonnet)
  - This tests whether the model can follow the 6 Lingas instruction
  - If we used Sonnet for rewriting, we'd be measuring Sonnet's rewriting + model's answering
- **Main generation call**: Uses the model being tested
- This means Mimamsa has 2x the API calls per question per model

## Exp 4 (Vaisheshika) Special Handling

Vaisheshika injects structured knowledge context. The context itself is pre-built
(from the vault corpus) — we don't regenerate it per model. But:

- Different models may handle long context differently
- Token limits vary across models
- Mitigation: measure context tokens and ensure all models receive same context

## Success Criteria

A technique is **cross-model validated** if:

1. **h2h win rate > 50%** (darshana vs generic) on **all 3 open-source models**
2. **No model where generic wins > 60%** (no strong counter-evidence)

A technique **partially validates** if:

1. **h2h win rate > 50%** on **2/3 open-source models**
2. The failing model has a known quirk (e.g., poor JSON following for judge tasks)

A technique **fails cross-model validation** if:

1. **h2h win rate < 50%** on **2+ open-source models**
2. **Only works on Claude** (Claude-specific quirk, not generalizable)

## Dependencies

```bash
# New pip packages needed
pip install openai         # Together uses OpenAI-compatible API

# No google-generativeai needed — skipping frontier cross-provider
```

```bash
# Environment variables needed
ANTHROPIC_API_KEY=...     # Existing
TOGETHER_API_KEY=...      # New (for Llama, Qwen, Mistral)
```
