#!/usr/bin/env python3
"""Prepare Samkhya-Organized Training Data

Curates a small corpus (~5MB) relevant to the 30 transfer questions,
then organizes it two ways:
  1. Samkhya: By tattva categories (purusha, prakriti, gunas, tanmatras)
  2. Random: Same data, shuffled

Uses Wikipedia API to gather relevant texts.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darshana.samkhya import categorize_text, organize_corpus, samkhya_training_order
from darshana.search import search_wikipedia

# -- Config --------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMKHYA_DIR = DATA_DIR / "samkhya_corpus"

# Topics to search Wikipedia for (covers our 30 questions' domains)
SEARCH_TOPICS = [
    # Economics
    "fiscal policy", "monetary policy", "recession economics", "economic depression",
    "supply-side economics", "demand-side economics", "inflation causes", "stock market crash",
    "economic growth developing countries", "central bank interest rate", "international trade barriers",
    "economic growth constraints",
    # Biology
    "mitosis cell division", "meiosis cell division", "prokaryotic cell", "eukaryotic cell",
    "innate immunity", "adaptive immunity", "cell division regulation", "allergic reaction mechanism",
    "evolution natural selection", "cell size limits", "population ecology carrying capacity",
    "human lifespan biology",
    # Methodology (purusha category)
    "scientific method", "critical thinking", "logical reasoning", "epistemology",
    "causal inference", "statistical reasoning",
    # Everyday
    "rent vs buy house", "leadership management", "online learning education",
    "stock market bonds", "career change psychology", "traffic congestion causes",
    "misinformation social media", "workplace burnout", "skill learning psychology",
    "renewable energy adoption barriers", "housing affordability", "remote work productivity",
]


def fetch_corpus():
    """Fetch Wikipedia articles for all topics."""
    print(f"Fetching corpus for {len(SEARCH_TOPICS)} topics...")
    documents = []

    for i, topic in enumerate(SEARCH_TOPICS):
        print(f"  [{i+1}/{len(SEARCH_TOPICS)}] {topic}", flush=True)
        result = search_wikipedia(topic, max_results=2, sentences=20)
        if result and "failed" not in result.lower():
            documents.append({
                "topic": topic,
                "text": result,
                "length": len(result),
            })
        time.sleep(0.5)  # Rate limit

    total_chars = sum(d["length"] for d in documents)
    print(f"\n  Fetched {len(documents)} articles, {total_chars:,} chars ({total_chars/1024/1024:.1f} MB)")
    return documents


def save_organized(documents):
    """Organize documents by Samkhya categories and save."""
    SAMKHYA_DIR.mkdir(parents=True, exist_ok=True)

    # Organize by Samkhya categories
    organized = organize_corpus(documents)

    # Save Samkhya-ordered version
    samkhya_order = samkhya_training_order()
    samkhya_docs = []
    for cat in samkhya_order:
        for doc in organized.get(cat, []):
            doc_with_cat = dict(doc)
            doc_with_cat["samkhya_category"] = cat
            samkhya_docs.append(doc_with_cat)
    # Add unclassified at the end
    for doc in organized.get("unclassified", []):
        doc_with_cat = dict(doc)
        doc_with_cat["samkhya_category"] = "unclassified"
        samkhya_docs.append(doc_with_cat)

    samkhya_path = SAMKHYA_DIR / "samkhya_ordered.jsonl"
    with open(samkhya_path, "w") as f:
        for doc in samkhya_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Save random-ordered version (same documents, shuffled)
    random_docs = list(samkhya_docs)
    random.shuffle(random_docs)
    random_path = SAMKHYA_DIR / "random_ordered.jsonl"
    with open(random_path, "w") as f:
        for doc in random_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Print distribution
    print(f"\nSamkhya category distribution:")
    for cat in samkhya_order + ["unclassified"]:
        count = len(organized.get(cat, []))
        if count:
            print(f"  {cat:<25}: {count} documents")

    print(f"\nSaved:")
    print(f"  Samkhya-ordered: {samkhya_path}")
    print(f"  Random-ordered:  {random_path}")

    # Also save raw text files for pretraining
    for name, docs in [("samkhya", samkhya_docs), ("random", random_docs)]:
        txt_path = SAMKHYA_DIR / f"{name}_corpus.txt"
        with open(txt_path, "w") as f:
            for doc in docs:
                f.write(doc["text"] + "\n\n")
        print(f"  Text corpus: {txt_path} ({txt_path.stat().st_size / 1024:.0f} KB)")


# -- Main ----------------------------------------------------------------------

def main():
    documents = fetch_corpus()
    if not documents:
        print("ERROR: No documents fetched. Check network.")
        sys.exit(1)
    save_organized(documents)
    print("\nDone. Next: run training/pretrain_samkhya.py")


if __name__ == "__main__":
    main()
