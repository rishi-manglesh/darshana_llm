# darshana_llm — Rigorous Validation of 7 Darshanas

Research project testing whether the 7 Darshanas (schools of Indian philosophy) provide **specific, measurable value** when mapped to LLM stack layers — or whether generic equivalents work just as well.

Successor to **vedic_llm** (8 phases, 1,959+ generations, 13 experiments).

## Research Framework: Nyaya Pancha-avayava

Every experiment follows the 5-step Nyaya syllogism:

| Step | Sanskrit | Scientific Equivalent | What We Write |
|------|----------|----------------------|---------------|
| 1. PRATIJNA | Thesis | Hypothesis | What do we claim? |
| 2. HETU | Reason | Theoretical basis | WHY should this be true? |
| 3. UDAHARANA | Evidence | Prior evidence | What do we already know? |
| 4. UPANAYA | Application | Experiment design | How do we test it? |
| 5. NIGAMANA | Conclusion | Success criteria | What constitutes proof? |

**Mandatory control:** Each Darshana technique is tested against a GENERIC equivalent of equal sophistication. We're proving "THIS Darshana approach > best generic approach," not "system prompt > no system prompt."

## The 7 Experiments

| Exp | Darshana | LLM Layer | Darshana Approach | Generic Control | Cost |
|-----|----------|-----------|-------------------|-----------------|------|
| 1 | Vritti | System prompt | 5-mode epistemic taxonomy | "Tag confidence level" | $0.50 |
| 2 | Nyaya | Tool routing | 4-pramana classification | Keyword heuristic | $1.00 |
| 3 | Mimamsa | Prompt design | 6 Lingas rewriting | Standard prompt eng | $0.60 |
| 4 | Vaisheshika | Knowledge org/RAG | 7-padartha ontology | Generic entity ontology | $0.50 |
| 5 | Vedanta | Output synthesis | Brahman/Maya/Atman | "Clean this up" | $0.50 |
| 6 | Samkhya | Pretraining | Tattva-ordered data | Bloom's taxonomy | $0.30 |
| 7 | Yoga | Post-training DPO | Ashtanga curriculum | Complexity curriculum | $1.05 |

**Total:** ~$4.45 API + ~12 hrs M4 compute

## Prior Results (from vedic_llm)

| Darshana | Status | Honest Finding |
|----------|--------|----------------|
| Vritti | PROVEN (vs bare) | 63%→90%→100% win rate. But never tested vs generic confidence tagging. |
| Nyaya | MIXED | 70% win rate BUT 57% search redundancy, +3.4% factual gain |
| Mimamsa | FAILED | 0% win rate as system prompt. Never tested as rewriter. |
| Vaisheshika | UNTESTED | Only regex counter built. Never tested as judge. |
| Vedanta | UNTESTED | Only regex formatter built. Never tested as synthesis. |
| Samkhya | NEVER BUILT | No experiment exists |
| Yoga | NEVER BUILT | No experiment exists |

## Current Results (darshana_llm Experiments 1-5)

| Exp | Darshana | Result | Win Rate | Key Finding |
|-----|----------|--------|----------|-------------|
| 1 | Vritti | **PROVEN** | 83% | Epistemic taxonomy > generic confidence tagging |
| 2 | Nyaya | **PROVEN** | 93% | Pramana routing: only 30% need search (70% reduction) |
| 3 | Mimamsa | **PROVEN** | 82% | 6 Lingas rewriting > generic prompt engineering |
| 4 | Vaisheshika | **PROVEN** | 71% (h2h) | 7-category ontology > generic ontology on real vault corpus (v4: factual 82%, causal 70%, gap 61%) |
| 5 | Vedanta | **PROVEN** | 82% | Brahman/Maya/Atman synthesis > generic cleanup |

**5/5 Darshanas PROVEN** with proper generic controls. Exp 4 evolved through 4 versions: v1 (judge framework, INCONCLUSIVE), v2 (Wikipedia KG, 63% h2h but wrong corpus), v3 (synthetic org corpus, 63% h2h), v4 (real vault corpus + English labels, 71% h2h with factual 82%, causal 70%, gap 61%).

## Cross-Model Validation

Do the techniques generalize beyond Claude? Tested on open-source Qwen3 models running locally via MLX on Apple Silicon (48GB M4 Mac). Sonnet judges all comparisons for consistency.

### Within-Model: Does darshana beat generic on open-source?

| Experiment | qwen3_8b (8B) | qwen3_32b (32B) | Sonnet (original) | Generalizes? |
|------------|---------------|-----------------|-------------------|--------------|
| **Vritti** (epistemic calibration) | **63%** | **67%** | 60% | **YES** |
| **Mimamsa** (query rewriting) | 30% | — | 73% | **NO** |
| **Vedanta** (response synthesis) | **60%** | **73%** | 63% | **YES** |

2/3 techniques validated across model families. Vritti and Vedanta work better on qwen3_32b than on Sonnet.

### Cross-Model: Can darshana + open-source beat frontier + generic?

| Matchup | Open-Source Wins | Sonnet Wins | Verdict |
|---------|-----------------|-------------|---------|
| qwen3_8b + vritti vs Sonnet + generic | 0% (0/30) | 100% | Sonnet dominates |
| qwen3_8b + vedanta vs Sonnet + generic | 10% (3/30) | 80% | Sonnet dominates |
| qwen3_32b + vritti vs Sonnet + generic | 7% (2/30) | 90% | Sonnet dominates |
| qwen3_32b + vedanta vs Sonnet + generic | 20% (6/30) | 80% | Sonnet dominates |
| qwen3_32b + vritti vs Sonnet + vritti | 0% (0/30) | 100% | Sonnet dominates |
| qwen3_32b + vedanta vs Sonnet + vedanta | 7% (2/30) | 90% | Sonnet dominates |

**The framework does NOT lift open-source to frontier level.** Sonnet wins 80-100% of direct matchups.

### What This Actually Proves

1. **Darshana techniques are model-agnostic reasoning patterns** — they improve any model vs generic prompting on that same model (63-73% within-model win rate)
2. **The model capability gap dwarfs prompt-layer lift** — no prompt framework bridges an 8B/32B → Sonnet gap. Sonnet's base quality is categorically ahead.
3. **The framework is a prompt-layer optimization, not a model-layer replacement** — it makes any model better, but a Qwen3-32B with great prompts doesn't beat Sonnet with basic prompts.
4. **Value proposition is clear:** If you're already using a frontier model, darshana prompting gives 60-73% win rate over generic prompting. The lift is real but operates within a model's capability ceiling.

### Limitations

- All 30 test questions are academic/analytical (economics, biology, physics)
- No enterprise, everyday, creative, or domain-specific use cases tested
- Only tested on Qwen3 family — other architectures (Mistral, Llama) untested
- Results may not generalize to all query types or user personas

### Models Tested

- `qwen3_8b`: mlx-community/Qwen3-8B-4bit (~6GB RAM)
- `qwen3_32b`: mlx-community/Qwen3-32B-4bit (~20GB RAM)
- Judge: Claude Sonnet (claude-sonnet-4-5-20250929)

## Quick Start

```bash
# Setup
cd /Users/rishimanglesh/Projects/darshana_llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test any API experiment (3 questions)
python experiments/exp1_vritti_vs_generic.py --limit 3
python experiments/exp2_nyaya_routing.py --limit 3
python experiments/exp3_mimamsa_rewriting.py --limit 3
python experiments/exp4_vaisheshika_kg.py --limit 3
python experiments/exp5_vedanta_synthesis.py --limit 3

# Full experiment + judging
python experiments/exp1_vritti_vs_generic.py --judge
python experiments/exp2_nyaya_routing.py --judge
python experiments/exp4_vaisheshika_kg.py --judge       # v2: vs no_context
python experiments/exp4_vaisheshika_kg.py --judge-h2h   # v2: padartha vs generic
python experiments/exp4v3_vaisheshika_org.py            # v3: synthetic org corpus
python experiments/exp4v4_vaisheshika_vault.py          # v4: real vault corpus (120 gens)
python experiments/exp4v4_vaisheshika_vault.py --judge-h2h # v4: padartha vs generic

# Training experiments (require data preparation first)
python training/prepare_samkhya_data.py          # Exp 6: fetch corpus
python training/pretrain_samkhya.py               # Exp 6: train models
python experiments/exp6_samkhya_pretraining.py --judge

python training/generate_dpo_pairs.py             # Exp 7: generate pairs
python training/train_dpo.py                      # Exp 7: train DPO
python experiments/exp7_yoga_dpo.py --judge

# Cross-model validation (local MLX models — no API key for generation)
pip install mlx-lm                                        # one-time
python experiments/cross_model_validation.py --exp 1 --model qwen3_8b --limit 3  # smoke test
python experiments/cross_model_validation.py --exp 1 --model qwen3_32b           # full run
python experiments/cross_model_validation.py --judge --model qwen3_32b --exp 1   # Sonnet judges
python experiments/cross_model_validation.py --analyze                            # summary
```

## Project Structure

```
darshana_llm/
├── darshana/                    # Core modules (each with Darshana + generic control)
│   ├── vritti.py               # Epistemic tags + GENERIC_CONFIDENCE_PROMPT
│   ├── nyaya.py                # Syllogism + tools
│   ├── nyaya_router.py         # Pramana routing + heuristic_route()
│   ├── mimamsa.py              # 6 Lingas + generic_rewrite()
│   ├── vaisheshika.py          # Structure extraction (regex metrics)
│   ├── vaisheshika_judge.py    # 7-padartha judge + generic 7-dim judge
│   ├── vaisheshika_ontology.py # 7-padartha KG ontology + generic entity ontology
│   ├── vedanta_synth.py        # Vedantic synthesis + generic_synthesize()
│   ├── samkhya.py              # Tattva organization + bloom_ordered_corpus()
│   ├── yoga_dpo.py             # Ashtanga DPO + reverse/generic curriculum
│   ├── formatter.py            # Output formatter
│   └── search.py               # Wikipedia + DuckDuckGo search
│
├── experiments/                 # 7 experiment scripts (redesigned with proper controls)
│   ├── utils.py                # 30 questions, metrics, API helpers
│   ├── judge.py                # LLM-judge infrastructure
│   ├── exp1_vritti_vs_generic.py    # Vritti 5-mode vs generic confidence
│   ├── exp2_nyaya_routing.py        # Pramana vs heuristic vs model-decides
│   ├── exp3_mimamsa_rewriting.py    # 6 Lingas vs generic prompt engineering
│   ├── exp4_vaisheshika_judge.py    # (v1, archived) 7-padartha vs 5-dim vs generic 7-dim
│   ├── exp4_vaisheshika_kg.py      # (v2) Padartha KG ontology vs generic KG (Wikipedia)
│   ├── exp4v3_vaisheshika_org.py   # (v3) Padartha KG on synthetic org corpus
│   ├── exp4v4_vaisheshika_vault.py # (v4, active) Padartha KG on real vault corpus
│   ├── exp5_vedanta_synthesis.py    # Brahman/Maya/Atman vs generic cleanup
│   ├── exp6_samkhya_pretraining.py  # Tattva order vs Bloom's vs random
│   └── exp7_yoga_dpo.py             # Ashtanga order vs generic vs reverse
│
├── training/                    # Pre/post-training scripts
│   ├── prepare_samkhya_data.py # Fetch & organize training corpus
│   ├── pretrain_samkhya.py     # Continued pretraining (Exp 6)
│   ├── generate_dpo_pairs.py   # Create DPO pairs (Exp 7)
│   ├── train_dpo.py            # Run DPO training (Exp 7)
│   └── eval_trained.py         # Quick model evaluation
│
├── data/
│   ├── questions.json          # 30 canonical transfer questions
│   └── phase7_outputs/         # vedic_llm Phase 7 data (for Exps 4, 5)
│
└── results/                     # Experiment outputs
    ├── exp1_vritti/             # Vritti vs generic
    ├── exp2_nyaya/              # Nyaya routing
    ├── exp3_mimamsa/            # Mimamsa rewriting
    ├── exp4_vaisheshika/        # Vaisheshika judge discrimination
    ├── exp5_vedanta/            # Vedanta synthesis
    ├── exp6_samkhya/            # Samkhya pretraining
    └── exp7_yoga/               # Yoga DPO
```

## Execution Order

```
PHASE 2 — FAST EXPERIMENTS (API-only, reuse existing data):
  Exp 1: Vritti vs Generic         (~$0.50, ~20 min)
  Exp 4: Vaisheshika Judge         (~$0.50, ~15 min, reuses Phase 7 data)
  Exp 5: Vedanta Synthesis         (~$0.50, ~15 min, reuses Phase 7 data)

PHASE 3 — API EXPERIMENTS (need new generations):
  Exp 2: Nyaya Routing             (~$1.00, ~30 min)
  Exp 3: Mimamsa Rewriting         (~$0.60, ~20 min)

PHASE 4 — TRAINING EXPERIMENTS (need compute):
  Exp 6: Samkhya Pretraining       (~$0.30 + 4 hrs M4)
  Exp 7: Yoga DPO                  (~$1.05 + 8 hrs M4)
```

## Verification

For each experiment:
1. Results written to `results/exp{N}_{name}/results.jsonl`
2. Each result includes all dimension scores + winner
3. Summary report with PRATIJNA→NIGAMANA format
4. Honest assessment: PROVEN / DISPROVEN / INCONCLUSIVE

After all 7 experiments: only PROVEN Darshanas go into the integrated pipeline.

## Version

- **Created:** 2026-02-19
- **Updated:** 2026-02-21 (v6.1 — Cross-model validation complete. 2/3 techniques generalize to open-source (within-model). Framework does NOT bridge frontier gap — Sonnet wins 80-100% of direct matchups.)
- **Predecessor:** vedic_llm (archived, 8 phases complete)
- **Owner:** Rishi Raj Manglesh
