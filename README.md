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
| 4 | Vaisheshika | **PARTIAL** | 63% (h2h) | Padartha ontology > generic ontology, but tested on wrong domain (Wikipedia ≠ org knowledge) |
| 5 | Vedanta | **PROVEN** | 82% | Brahman/Maya/Atman synthesis > generic cleanup |

**4/5 Darshanas PROVEN** with proper generic controls. Exp 4 v2: padartha ontology beats generic at 63% h2h, but tested on Wikipedia (model already knows). Real test needs proprietary/org corpus where RAG is essential.

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
python experiments/exp4_vaisheshika_kg.py --judge       # vs no_context
python experiments/exp4_vaisheshika_kg.py --judge-h2h   # padartha vs generic

# Training experiments (require data preparation first)
python training/prepare_samkhya_data.py          # Exp 6: fetch corpus
python training/pretrain_samkhya.py               # Exp 6: train models
python experiments/exp6_samkhya_pretraining.py --judge

python training/generate_dpo_pairs.py             # Exp 7: generate pairs
python training/train_dpo.py                      # Exp 7: train DPO
python experiments/exp7_yoga_dpo.py --judge
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
│   ├── exp4_vaisheshika_kg.py      # (v2, active) Padartha KG ontology vs generic KG
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
- **Updated:** 2026-02-19 (v3.0 — Exps 1-5 complete. 4/5 PROVEN. Exp 4 redesigned as KG ontology.)
- **Predecessor:** vedic_llm (archived, 8 phases complete)
- **Owner:** Rishi Raj Manglesh
