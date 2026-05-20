# darshana_llm — Validation Phase

**Controlled validation of the Darshana Framework for LLMs.** This repo contains the validation-phase experiments (9 experiments, ~3,000 generations, 3 model architectures) that test whether the six classical Indian philosophical schools (*Shad Darshana*: Nyaya, Vaisheshika, Samkhya, Yoga, Mimamsa, Vedanta) provide measurable engineering value when mapped to specific LLM stack layers — each tested against equal-sophistication generic controls.

The exploration-phase experiments are in [vedic_llm](https://github.com/rishi-manglesh/vedic_llm). Both repos are part of the same research program; the paper draws on both.

## Paper

Manglesh, R.R. (2026). *Darshana: A Six-School Interpretation Framework for Large Language Model Orchestration and Training*. Preprint.

```bibtex
@article{manglesh2026darshana,
  title  = {Darshana: A Six-School Interpretation Framework for Large Language Model Orchestration and Training},
  author = {Manglesh, Rishi Raj},
  year   = {2026},
  note   = {Preprint}
}
```

## Headline results

All five orchestration-layer Darshanas outperformed equal-sophistication generic controls. Cross-judge validation (GPT-4o) confirmed the advantage at 60–67%, with same-model bias estimated at 15–20pp.

| Experiment | Darshana | LLM Layer | Win Rate (Sonnet / GPT-4o) |
|---|---|---|---|
| E1 | Vritti (from Yoga) | Epistemic self-classification (system prompt) | 83% / 60% |
| E2 | Nyaya | Tool routing via 4 Pramanas | 93% / — |
| E3 | Mimamsa | Query rewriting (6 Lingas) | 82% / 67% |
| E4 | Vaisheshika | Knowledge ontology (7 Padarthas + Abhava) | 71% h2h |
| E5 | Vedanta | Output synthesis (Brahman/Maya/Atman) | 82% / 63% |
| E8 | Yoga | Stage-gated SFT (Ashtanga progression) | 60–62% across Qwen 3B, LLaMA 3B, Phi 3.5 |

**Layer assignment is the critical variable:** Mimamsa scored 0% when applied at runtime (as system prompt) but 82% when applied as a query rewriter. Same technique, different layer. This is the central methodological finding.

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

## Current Results (darshana_llm Experiments 1-5, 9)

| Exp | Darshana | Result | Win Rate (Sonnet) | Win Rate (GPT-4o) | Key Finding |
|-----|----------|--------|-------------------|-------------------|-------------|
| 1 | Vritti | **Validated** | 83% | 60% | Epistemic taxonomy > generic confidence tagging |
| 2 | Nyaya | **Validated** | 93% | — | Pramana routing: only 30% need search (70% reduction) |
| 3 | Mimamsa | **Validated** | 82% | 67% | 6 Lingas rewriting > generic prompt engineering |
| 4 | Vaisheshika | **Validated** | 71% (h2h) | — | 7-category ontology > generic ontology on real vault corpus |
| 5 | Vedanta | **Validated** | 82% | 63% | Brahman/Maya/Atman synthesis > generic cleanup |
| 9a | Vritti (placebo) | **Taxonomy > structure** | 63% | — | Vritti labels vs neutral TYPE-A/B/C/D/E: taxonomy content matters |
| 9b | — (placebo) | Structure helps | 53% | — | Neutral labels vs generic: structure alone provides modest lift |

All 5 orchestration components outperformed equal-sophistication generic controls. Cross-judge validation (GPT-4o) confirms Darshana advantage at lower rates (60-67% vs 82-83%). Same-model bias estimated at 15-20pp. Structure-only placebo tested: neutral labels beat generic (53%) but Vritti taxonomy beats neutral labels (63%), confirming taxonomy content matters beyond structure alone.

Exp 4 evolved through 4 versions: v1 (judge framework, INCONCLUSIVE), v2 (Wikipedia KG, 63% h2h but wrong corpus), v3 (synthetic org corpus, 63% h2h), v4 (real vault corpus + English labels, 71% h2h with factual 82%, causal 70%, gap 61%).

**Training experiments (Exp 6-7):** DPO failed — base model won 70-75% against all trained variants. See vedic_llm Exp 8 (separate repo) for successful SFT curriculum results.

### Cross-Judge Validation (Exp 9)

GPT-4o re-judged all 90 pairs from Experiments 1, 3, and 5 to test for same-model bias (Sonnet judging Sonnet-generated outputs).

**Cross-judge results:**

| Experiment | Sonnet Win Rate | GPT-4o Win Rate | Delta | GPT-4o Confirms? |
|------------|----------------|-----------------|-------|------------------|
| Vritti (Exp 1) | 83% | 60% (18/30) | -23pp | **Yes** (>50%) |
| Mimamsa (Exp 3) | 82% | 67% (20/30) | -15pp | **Yes** (>50%) |
| Vedanta (Exp 5) | 82% | 63% (19/30) | -19pp | **Yes** (>50%) |

- **Inter-judge agreement:** 51% (judges agree on specific pairs about half the time)
- **Same-model bias:** Estimated 15-20pp. Sonnet's advantage appears in reasoning_depth and usefulness dimensions — Sonnet values epistemic structure, GPT-4o values information density.
- **Key finding:** GPT-4o independently confirms Darshana advantage (all >50%) but at lower rates. The effect is real; the magnitude depends on what the judge values.

**Structure-only placebo (Exp 9a):**

| Comparison | Win Rate (30 pairs) | Interpretation |
|------------|---------------------|----------------|
| Vritti labels vs Generic | 60% (18/30) | Full taxonomy > no structure |
| Neutral TYPE-A/B/C/D/E vs Generic | 53% (16/30) | Structure alone provides modest lift |
| Vritti labels vs Neutral labels | 63% (19/30) | **Taxonomy content matters beyond structure** |

The 10pp gap between Vritti (63%) and neutral placebo (53%) against each other's baselines — and the direct 63% h2h — confirms that the Sanskrit epistemic taxonomy contributes value beyond the mere act of structured labeling.

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
git clone https://github.com/rishi-manglesh/darshana_llm.git
cd darshana_llm
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
4. Honest assessment: VALIDATED / FAILED / INCONCLUSIVE

After all 7 experiments: only validated Darshanas go into the integrated pipeline. Training experiments (Exp 6-7) failed — DPO approach did not produce improvements over base model.

## Citation

If you use this work, please cite the paper:

```bibtex
@article{manglesh2026darshana,
  title  = {Darshana: A Six-School Interpretation Framework for Large Language Model Orchestration and Training},
  author = {Manglesh, Rishi Raj},
  year   = {2026},
  note   = {Preprint}
}
```

## Version

- **Created:** 2026-02-19
- **Updated:** 2026-05-20 (positioning aligned with published Darshana paper; companion repo: [vedic_llm](https://github.com/rishi-manglesh/vedic_llm) — exploration phase)
- **Author:** Rishi Raj Manglesh — rm@hundredsolutions.com
