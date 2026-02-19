# darshana_llm — Validate All 7 Darshanas at Their Correct Layers

Research project validating that each of the 7 Darshanas (schools of Indian philosophy) maps to a specific layer of the LLM stack. Successor to **vedic_llm** (8 phases, 1,959+ generations, 13 experiments).

## The Mapping

```
DARSHANA              LLM LAYER              VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Samkhya (categories)  Pretraining data       Tattva-organized data vs random
Yoga (8 limbs)        Post-training (DPO)    Yoga curriculum order vs random DPO
Vritti (5 modes)      System prompt          Transfer to Qwen2.5-7B
Mimamsa (6 Lingas)    Prompt design          Rewrite prompts with 6 Lingas
Nyaya (4 pramanas)    Tool routing           Pramana classification reduces redundancy
Vaisheshika (7 padarthas) Evaluation         7-padartha judge vs 5-dimension judge
Vedanta (synthesis)   Output synthesis       LLM Vedantic synthesis vs regex
```

## Prior Results (from vedic_llm)

| Finding | Phase | Win Rate |
|---------|-------|----------|
| System prompts match LoRA adapters | 4 | 3/3 PASS |
| Vritti epistemic tagging | 6-7 | 63% (4B) → 90% (Sonnet) → 100% (pipeline_clean) |
| 6 Darshanas → 6 LLM components | 7 | Corrected mapping validated |
| 57% of tool searches redundant | 7 | Opportunity for Nyaya routing |

## 7 Experiments

| Exp | Darshana | Layer | Model | Compute | API Cost |
|-----|----------|-------|-------|---------|----------|
| 1 | Samkhya | Pretraining | Qwen2.5-0.5B | ~2 hrs M4 | $0.30 |
| 2 | Yoga | Post-training (DPO) | Qwen2.5-0.5B | ~6 hrs M4 | $1.05 |
| 3 | Vritti | System prompt | Qwen2.5-7B | ~2 hrs M4 | $0.30 |
| 4 | Mimamsa | Prompt design | Sonnet (API) | — | $0.60 |
| 5 | Nyaya | Tool routing | Sonnet (API) | — | $0.80 |
| 6 | Vaisheshika | Evaluation | Sonnet (API) | — | $0.30 |
| 7 | Vedanta | Synthesis | Sonnet (API) | — | $0.30 |
| **Total** | | | | **~10 hrs M4** | **~$3.65** |

## Quick Start

```bash
# Setup
cd /Users/rishimanglesh/Projects/darshana_llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test any experiment (3 questions)
python experiments/exp3_vritti_transfer.py --limit 3
python experiments/exp4_mimamsa_design.py --limit 3
python experiments/exp5_nyaya_routing.py --limit 3
python experiments/exp6_vaisheshika_eval.py --limit 3
python experiments/exp7_vedanta_synthesis.py --limit 3

# Full experiment + judging
python experiments/exp3_vritti_transfer.py --judge
python experiments/exp5_nyaya_routing.py --judge

# Training experiments (require data preparation first)
python training/prepare_samkhya_data.py          # Exp 1: fetch corpus
python training/pretrain_samkhya.py               # Exp 1: train models
python experiments/exp1_samkhya_pretraining.py --judge

python training/generate_dpo_pairs.py             # Exp 2: generate pairs
python training/train_dpo.py                      # Exp 2: train DPO
python experiments/exp2_yoga_posttraining.py --judge
```

## Project Structure

```
darshana_llm/
├── darshana/                    # Core modules (6 from vedic_llm + 5 new)
│   ├── vritti.py               # Epistemic tags (from vedic_llm)
│   ├── nyaya.py                # Syllogism + tools (from vedic_llm)
│   ├── mimamsa.py              # 6 Lingas (from vedic_llm + rewrite extension)
│   ├── vaisheshika.py          # Structure extraction (from vedic_llm)
│   ├── formatter.py            # Output formatter (from vedic_llm)
│   ├── search.py               # Web search (from vedic_llm)
│   ├── samkhya.py              # Taxonomy/data organization (NEW)
│   ├── yoga_dpo.py             # DPO curriculum templates (NEW)
│   ├── nyaya_router.py         # Pramana-based routing (NEW)
│   ├── vaisheshika_judge.py    # 7-padartha judge (NEW)
│   └── vedanta_synth.py        # Deep synthesis (NEW)
│
├── experiments/                 # 7 experiment scripts
│   ├── utils.py                # 30 questions, metrics, API helpers
│   ├── judge.py                # LLM-judge infrastructure
│   └── exp[1-7]_*.py           # Individual experiments
│
├── training/                    # Pre/post-training scripts
│   ├── prepare_samkhya_data.py # Fetch & organize training corpus
│   ├── pretrain_samkhya.py     # Continued pretraining (Exp 1)
│   ├── generate_dpo_pairs.py   # Create Yoga DPO pairs (Exp 2)
│   ├── train_dpo.py            # Run DPO training (Exp 2)
│   └── eval_trained.py         # Quick model evaluation
│
├── data/
│   ├── questions.json          # 30 canonical transfer questions
│   └── phase7_outputs/         # vedic_llm Phase 7 data (for Exps 6, 7)
│
└── results/                     # Experiment outputs (exp1-exp7)
```

## Execution Order

```
Day 1:  Exp 3 (Vritti transfer)     — local, quick
        Exp 6 (Vaisheshika judge)    — reuses existing data
        Exp 7 (Vedanta synthesis)    — reuses Phase 7 output

Day 2:  Exp 4 (Mimamsa design)      — API, straightforward
        Exp 5 (Nyaya routing)        — API, most interesting

Day 3:  Exp 1 (Samkhya pretraining) — needs corpus + training
        Exp 2 (Yoga DPO)            — needs pairs + 3 training runs
```

## Version

- **Created:** 2026-02-19
- **Predecessor:** vedic_llm (archived, 8 phases complete)
- **Owner:** Rishi Raj Manglesh
