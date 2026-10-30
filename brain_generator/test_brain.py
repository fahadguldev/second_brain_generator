#!/usr/bin/env python3
"""
Test Brain - Standalone offline similarity retrieval CLI.

Usage:
    python test_brain.py "How do I get started with cloud computing?"
    python test_brain.py --top-k 10 "Should I prioritize degree or skills?"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from second_brain_generator.config import load_config
from second_brain_generator.evaluator import BrainEvaluator
from second_brain_generator.embedder.embedder import _deterministic_mock_vector


def main():
    parser = argparse.ArgumentParser(description="Query local Second Brain embeddings offline.")
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--config", type=str, default="brain_config.yaml", help="Path to config file")

    args = parser.parse_args()

    cfg_p = Path(args.config).resolve()
    if not cfg_p.exists():
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    cfg = load_config(cfg_p)
    emb_p = Path(cfg.paths.get("embeddings_path", "brain_data/gemini_embeddings.npz")).resolve()
    rag_p = Path(cfg.paths.get("rag_records_path", "brain_data/rag_records.jsonl")).resolve()

    if not emb_p.exists() or not rag_p.exists():
        print("Error: Embeddings or RAG records not found. Run `python run_generator.py --process --embed` first.")
        sys.exit(1)

    evaluator = BrainEvaluator(embeddings_path=emb_p, records_path=rag_p)
    q_vec = _deterministic_mock_vector(args.query, evaluator.embeddings.shape[1])
    results = evaluator.search(q_vec, top_k=args.top_k)

    print(f"\nQuery: \"{args.query}\"")
    print("-" * 75)
    print(f"{'Rank':<5} | {'Score':<6} | {'ID':<36} | {'Excerpt'}")
    print("-" * 75)
    for idx, r in enumerate(results, 1):
        excerpt = r["text"].replace("\n", " ")[:45] + "..." if len(r["text"]) > 45 else r["text"].replace("\n", " ")
        print(f"{idx:<5} | {r['score']:<6.4f} | {r['id']:<36} | {excerpt}")
    print("-" * 75 + "\n")


if __name__ == "__main__":
    main()
