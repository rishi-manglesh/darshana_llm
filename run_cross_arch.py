#!/usr/bin/env python3
"""Cross-Architecture Validation: Yoga SFT Curriculum on LLaMA 3.2 3B + Phi 3.5 Mini

Replicates the Exp 8 stage-gated vs reverse comparison on non-Qwen architectures
to validate that the finding isn't Qwen-specific.

For each architecture:
  1. Train gated (5-gate, 30 iters each = 150 total)
  2. Train reverse (single epoch, ~135 iters)
  3. Generate responses (base + gated + reverse × 200 questions = 600)
  4. Judge H2H: gated vs reverse (200 Sonnet judgments)

Usage:
  python run_cross_arch.py              # Run everything
  python run_cross_arch.py --skip-train # Skip training, just generate + judge
  python run_cross_arch.py --skip-gen   # Skip generation, just judge
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "sft_yoga"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))

# -- Architecture configs -------------------------------------------------------

ARCHITECTURES = {
    "llama3b": {
        "model_id": "mlx-community/Llama-3.2-3B-Instruct",
        "short": "llama3b",
        "num_layers": 16,
    },
    "phi3.5": {
        "model_id": "mlx-community/Phi-3.5-mini-instruct",
        "short": "phi35",
        "num_layers": 16,
    },
}

SEED = 42
ITERS_PER_GATE = 30
NUM_GATES = 5
BATCH_SIZE = 2
LR = "1e-5"


# -- Training -------------------------------------------------------------------

def train_gated(arch_key):
    """Train stage-gated model (5 gates, resume from previous)."""
    arch = ARCHITECTURES[arch_key]
    model_id = arch["model_id"]
    short = arch["short"]
    ordered_script = str(PROJECT_ROOT / "training" / "mlx_ordered_lora.py")

    # Check if already complete (adapter weights exist)
    final_adapter = MODELS_DIR / f"{short}-gated-sft-1ep" / "adapters" / "adapters.safetensors"
    if final_adapter.exists() and final_adapter.stat().st_size > 1000:
        print(f"\n  SKIP: Gated adapter already exists for {arch_key} ({final_adapter.stat().st_size} bytes)")
        return True

    # Load full training data
    sft_path = DATA_DIR / "full_train.jsonl"
    with open(sft_path) as f:
        all_records = [json.loads(l) for l in f if l.strip()]

    print(f"\n{'='*60}")
    print(f"STAGE-GATED TRAINING: {arch_key} ({model_id})")
    print(f"{'='*60}")

    prev_adapter = None
    for gate in range(1, NUM_GATES + 1):
        # Filter to stages <= gate, sort by stage
        gate_records = sorted(
            [r for r in all_records if r.get("stage", 0) <= gate],
            key=lambda r: r.get("stage", 0)
        )

        # Prepare data directory
        gate_data_dir = DATA_DIR / f"{short}_gated_stage{gate}_data"
        gate_data_dir.mkdir(parents=True, exist_ok=True)

        # 90/10 split
        n_val = max(1, len(gate_records) // 10)
        val_indices = set(range(0, len(gate_records), max(1, len(gate_records) // n_val)))
        train = [r for i, r in enumerate(gate_records) if i not in val_indices]
        valid = [r for i, r in enumerate(gate_records) if i in val_indices]

        with open(gate_data_dir / "train.jsonl", "w") as f:
            for r in train:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        with open(gate_data_dir / "valid.jsonl", "w") as f:
            for r in valid:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

        # Adapter directory
        adapter_dir = MODELS_DIR / f"{short}-gated-sft-1ep" / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Gate {gate}/5: stages 1-{gate}, {len(train)} train, {len(valid)} valid")

        cmd = [
            sys.executable, ordered_script,
            "--model", model_id,
            "--data", str(gate_data_dir),
            "--train",
            "--adapter-path", str(adapter_dir),
            "--iters", str(ITERS_PER_GATE),
            "--batch-size", str(BATCH_SIZE),
            "--learning-rate", LR,
            "--num-layers", str(arch["num_layers"]),
            "--seed", str(SEED),
        ]
        if prev_adapter is not None:
            cmd.extend(["--resume-adapter-file", str(prev_adapter)])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  ERROR: Gate {gate} training failed!")
            return False

        prev_adapter = adapter_dir / "adapters.safetensors"
        print(f"  Gate {gate} complete. Adapter: {prev_adapter}")

    print(f"\n  GATED TRAINING COMPLETE: {short}")
    return True


def train_reverse(arch_key):
    """Train reverse-ordered model (single epoch)."""
    arch = ARCHITECTURES[arch_key]
    model_id = arch["model_id"]
    short = arch["short"]
    ordered_script = str(PROJECT_ROOT / "training" / "mlx_ordered_lora.py")

    # Check if already complete (adapter weights exist)
    final_adapter = MODELS_DIR / f"{short}-reverse-sft-1ep" / "adapters" / "adapters.safetensors"
    if final_adapter.exists() and final_adapter.stat().st_size > 1000:
        print(f"\n  SKIP: Reverse adapter already exists for {arch_key} ({final_adapter.stat().st_size} bytes)")
        return True

    # Prepare reverse data
    sft_path = DATA_DIR / "full_train.jsonl"
    with open(sft_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    records.sort(key=lambda r: -r.get("stage", 0))

    data_dir = DATA_DIR / f"{short}_reverse_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n_val = max(1, len(records) // 10)
    val_indices = set(range(0, len(records), max(1, len(records) // n_val)))
    train = [r for i, r in enumerate(records) if i not in val_indices]
    valid = [r for i, r in enumerate(records) if i in val_indices]

    with open(data_dir / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    with open(data_dir / "valid.jsonl", "w") as f:
        for r in valid:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

    adapter_dir = MODELS_DIR / f"{short}-reverse-sft-1ep" / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Single epoch: ~len(train)/batch_size iters
    n_iters = len(train) // BATCH_SIZE
    print(f"\n{'='*60}")
    print(f"REVERSE TRAINING: {arch_key} ({model_id})")
    print(f"  {len(train)} train, {n_iters} iters (1 epoch)")
    print(f"{'='*60}")

    cmd = [
        sys.executable, ordered_script,
        "--model", model_id,
        "--data", str(data_dir),
        "--train",
        "--adapter-path", str(adapter_dir),
        "--iters", str(n_iters),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", LR,
        "--num-layers", str(arch["num_layers"]),
        "--seed", str(SEED),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: Reverse training failed!")
        return False

    print(f"\n  REVERSE TRAINING COMPLETE: {short}")
    return True


# -- Generation -----------------------------------------------------------------

def generate_responses(arch_key):
    """Generate responses for base + gated + reverse on 200 eval questions."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    arch = ARCHITECTURES[arch_key]
    model_id = arch["model_id"]
    short = arch["short"]

    # Load eval questions
    with open(DATA_DIR / "eval_200.json") as f:
        questions = json.load(f)

    results_dir = RESULTS_DIR / f"exp8_{short}_200q"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.jsonl"

    # Check existing
    existing = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                existing.add((r["config"], r["query"][:80]))

    configs = {
        f"base_{short}": (model_id, None),
        f"gated_{short}": (model_id, str(MODELS_DIR / f"{short}-gated-sft-1ep" / "adapters")),
        f"reverse_{short}": (model_id, str(MODELS_DIR / f"{short}-reverse-sft-1ep" / "adapters")),
    }

    # Verify adapters exist
    for cname, (base, adapter) in configs.items():
        if adapter and not Path(adapter).exists():
            print(f"  WARNING: Adapter not found for {cname}: {adapter}")
            print(f"  Skipping {cname}")
            del configs[cname]
            break

    sampler = make_sampler(temp=0.7, top_p=0.9)
    total = len(questions) * len(configs)
    done = len(existing)

    print(f"\n{'='*60}")
    print(f"GENERATING RESPONSES: {arch_key}")
    print(f"  {len(questions)} questions × {len(configs)} configs = {total} total")
    print(f"  Already done: {done}")
    print(f"{'='*60}")

    model_cache = {}

    for config_name, (base_model, adapter_path) in configs.items():
        cache_key = f"{base_model}|{adapter_path}"
        if cache_key not in model_cache:
            print(f"\n  Loading {config_name}...")
            if adapter_path:
                model_cache[cache_key] = load(base_model, adapter_path=adapter_path)
            else:
                model_cache[cache_key] = load(base_model)

        model, tokenizer = model_cache[cache_key]

        for i, q in enumerate(questions):
            if (config_name, q["query"][:80]) in existing:
                continue

            messages = [{"role": "user", "content": q["query"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = generate(
                model, tokenizer, prompt=text,
                max_tokens=512, sampler=sampler, verbose=False
            )
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

            result = {
                "config": config_name,
                "query": q["query"],
                "target_op": q.get("target_op", ""),
                "domain": q.get("domain", ""),
                "response": response,
                "word_count": len(response.split()),
            }

            with open(results_file, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            done += 1
            if done % 50 == 0:
                print(f"  Progress: {done}/{total}")

        # Free memory after each config
        del model_cache[cache_key]
        import gc
        import mlx.core as mx
        gc.collect()
        mx.clear_cache()
        model_cache = {}

    print(f"\n  GENERATION COMPLETE: {arch_key} ({done} total responses)")
    return True


# -- Judging --------------------------------------------------------------------

def judge_h2h(arch_key):
    """Run gated vs reverse H2H judging."""
    from experiments.utils import get_client, load_existing_keys, append_jsonl
    from experiments.judge import judge_pairwise

    arch = ARCHITECTURES[arch_key]
    short = arch["short"]

    results_dir = RESULTS_DIR / f"exp8_{short}_200q"
    results_file = results_dir / "results.jsonl"
    judge_file = results_dir / "judge_h2h.jsonl"

    if not results_file.exists():
        print(f"  ERROR: No results file at {results_file}")
        return False

    with open(results_file) as f:
        results = [json.loads(l) for l in f]

    # Index by (query, config)
    by_qc = {}
    for r in results:
        by_qc[(r["query"], r["config"])] = r["response"]

    # Get questions with both configs
    gated_key = f"gated_{short}"
    reverse_key = f"reverse_{short}"
    queries = sorted(set(
        r["query"] for r in results if r["config"] == gated_key
    ))

    # Load metadata
    with open(DATA_DIR / "eval_200.json") as f:
        eval_qs = json.load(f)
    q_meta = {q["query"]: q for q in eval_qs}

    existing = load_existing_keys(str(judge_file))
    client = get_client()
    model = "claude-sonnet-4-20250514"

    print(f"\n{'='*60}")
    print(f"H2H JUDGING: {arch_key} (gated vs reverse)")
    print(f"  {len(queries)} questions, {len(existing)} already judged")
    print(f"{'='*60}")

    done = 0
    errors = 0

    for q in queries:
        key = f"pairwise|{q[:50]}|{gated_key}|{reverse_key}"
        if key in existing:
            done += 1
            continue

        resp_g = by_qc.get((q, gated_key))
        resp_r = by_qc.get((q, reverse_key))
        if not resp_g or not resp_r:
            continue

        result = judge_pairwise(
            client, model, q, resp_g, resp_r,
            label_a=gated_key, label_b=reverse_key
        )
        if result is None:
            errors += 1
            continue

        meta = q_meta.get(q, {})
        result["key"] = key
        result["type"] = "pairwise"
        result["query"] = q
        result["config_a"] = gated_key
        result["config_b"] = reverse_key
        result["target_op"] = meta.get("target_op", "")
        result["domain"] = meta.get("domain", "")

        append_jsonl(str(judge_file), result)
        done += 1

        if done % 20 == 0:
            with open(judge_file) as f:
                jj = [json.loads(l) for l in f]
            gw = sum(1 for j in jj if j["winner"] == gated_key)
            rw = sum(1 for j in jj if j["winner"] == reverse_key)
            tw = sum(1 for j in jj if j["winner"] == "TIE")
            print(f"  Progress: {done}/200 | Gated {gw} Rev {rw} Tie {tw}")

    print(f"\n  JUDGING COMPLETE: {arch_key} ({done} judgments, {errors} errors)")
    return True


# -- Analysis -------------------------------------------------------------------

def analyze_results(arch_key):
    """Print summary analysis for an architecture."""
    import numpy as np

    arch = ARCHITECTURES[arch_key]
    short = arch["short"]
    gated_key = f"gated_{short}"
    reverse_key = f"reverse_{short}"

    judge_file = RESULTS_DIR / f"exp8_{short}_200q" / "judge_h2h.jsonl"
    if not judge_file.exists():
        print(f"  No judge results for {arch_key}")
        return

    with open(judge_file) as f:
        jj = [json.loads(l) for l in f]

    gw = sum(1 for j in jj if j["winner"] == gated_key)
    rw = sum(1 for j in jj if j["winner"] == reverse_key)
    tw = sum(1 for j in jj if j["winner"] == "TIE")

    # Bootstrap
    rng = np.random.default_rng(42)
    wins = [1 if j["winner"] == gated_key else 0 for j in jj]
    boots = [np.mean(rng.choice(wins, size=len(wins), replace=True)) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\n{'='*60}")
    print(f"  {arch_key.upper()} — GATED vs REVERSE H2H")
    print(f"{'='*60}")
    print(f"  Gated: {gw} ({gw/len(jj)*100:.1f}%)")
    print(f"  Reverse: {rw} ({rw/len(jj)*100:.1f}%)")
    print(f"  Tie: {tw} ({tw/len(jj)*100:.1f}%)")
    print(f"  CI: [{lo*100:.1f}-{hi*100:.1f}%]")
    print(f"  Significant: {'YES' if lo > 0.5 else 'NO'}")

    # Per-dimension
    dims = ["factual_accuracy", "reasoning_depth", "completeness", "calibration", "usefulness"]
    print(f"\n  Per-dimension (gated - reverse):")
    for dim in dims:
        diffs = []
        for j in jj:
            sa = j.get("scores_a")
            sb = j.get("scores_b")
            if not isinstance(sa, dict) or not isinstance(sb, dict):
                continue
            va = sa.get(dim)
            vb = sb.get(dim)
            if va is None or vb is None:
                continue
            order = j["order"]
            if order[0] == gated_key:
                diffs.append(va - vb)
            else:
                diffs.append(vb - va)
        if diffs:
            print(f"    {dim:<20} {np.mean(diffs):+.2f}")

    return {"gated_pct": gw / len(jj), "ci_lo": lo, "ci_hi": hi, "n": len(jj)}


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-Architecture Yoga SFT Validation")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judging")
    parser.add_argument("--arch", choices=list(ARCHITECTURES.keys()) + ["all"], default="all",
                        help="Architecture to test (default: all)")
    args = parser.parse_args()

    archs = list(ARCHITECTURES.keys()) if args.arch == "all" else [args.arch]
    start_time = time.time()

    for arch_key in archs:
        arch = ARCHITECTURES[arch_key]
        print(f"\n{'#'*60}")
        print(f"  ARCHITECTURE: {arch_key} ({arch['model_id']})")
        print(f"{'#'*60}")

        if not args.skip_train:
            if not train_gated(arch_key):
                print(f"  FAILED: gated training for {arch_key}")
                continue
            if not train_reverse(arch_key):
                print(f"  FAILED: reverse training for {arch_key}")
                continue

        if not args.skip_gen:
            if not generate_responses(arch_key):
                print(f"  FAILED: generation for {arch_key}")
                continue

        if not args.skip_judge:
            if not judge_h2h(arch_key):
                print(f"  FAILED: judging for {arch_key}")
                continue

        analyze_results(arch_key)

    # Final cross-architecture summary
    elapsed = time.time() - start_time
    print(f"\n{'#'*60}")
    print(f"  CROSS-ARCHITECTURE SUMMARY")
    print(f"  Total time: {elapsed/3600:.1f} hours")
    print(f"{'#'*60}")

    all_results = {}
    for arch_key in archs:
        r = analyze_results(arch_key)
        if r:
            all_results[arch_key] = r

    # Add Qwen result for comparison
    qwen_judge = RESULTS_DIR / "exp8_3b_200q" / "judge_h2h.jsonl"
    if qwen_judge.exists():
        import numpy as np
        with open(qwen_judge) as f:
            jj = [json.loads(l) for l in f]
        gw = sum(1 for j in jj if j["winner"] == "gated_3b")
        all_results["qwen3b"] = {"gated_pct": gw / len(jj), "n": len(jj)}

    print(f"\n  Architecture    Gated Win%   Significant?")
    print(f"  {'─'*45}")
    for arch, r in sorted(all_results.items()):
        sig = "YES" if r.get("ci_lo", 0) > 0.5 else "—" if "ci_lo" not in r else "NO"
        ci = f"[{r['ci_lo']*100:.0f}-{r['ci_hi']*100:.0f}%]" if "ci_lo" in r else ""
        print(f"  {arch:<16} {r['gated_pct']*100:.1f}%  {ci:>14}  {sig}")

    # Verdict
    proven_count = sum(1 for r in all_results.values() if r.get("ci_lo", 0) > 0.5)
    total_archs = len(all_results)
    print(f"\n  VERDICT: Gated > Reverse on {proven_count}/{total_archs} architectures")
    if proven_count >= 2:
        print(f"  CROSS-ARCHITECTURE VALIDATION: CONFIRMED")
    elif proven_count == 1:
        print(f"  CROSS-ARCHITECTURE VALIDATION: PARTIAL (need more architectures)")
    else:
        print(f"  CROSS-ARCHITECTURE VALIDATION: NOT CONFIRMED")


if __name__ == "__main__":
    main()
