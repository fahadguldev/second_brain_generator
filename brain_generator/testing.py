#!/usr/bin/env python3
"""
Ganjiswag AI Second Brain - Roman Urdu Testing Pipeline CLI.

Usage Examples:
    # 1. Ask a single question directly:
    python testing.py "Karachi mein street crime aur snatching se bachne ke kya tareeqay hain?"

    # 2. Interactive chat session:
    python testing.py --interactive

    # 3. Run automated demo benchmark questions:
    python testing.py --demo

    # 4. Save response to a file:
    python testing.py "CS degree vs skills mein kya behtar hai?" --output response.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from second_brain_generator.responder import (
    GenerationConfig,
    PipelineResult,
    TestingPipeline,
)

DEMO_QUESTIONS = [
    "Computer Science degree zaroori hai ya skills? Agar degree na ho toh market mein survive kaise karein?",
    "Karachi mein street crime aur mobile snatching ka kya masla hai? Iska practical solution kya hai?",
    "Pakistani society mein shadi ki sahi umar kya honi chahiye aur rishta culture ke kya maslay hain?",
    "Job vs Business: 20s mein middle class larke ko kya prioritize karna chahiye?",
]


def print_banner() -> None:
    banner = """
===========================================================================
      GANJISWAG AI SECOND BRAIN - ROMAN URDU TESTING PIPELINE
===========================================================================
Persona: Junaid Akram (Ask Ganjiswag)
Language: Roman Urdu (Natural Pakistani dialogue)
Database: Multi-lingual embeddings & Hindi/Devanagari transcripts
===========================================================================
"""
    print(banner.strip())
    print()


def display_result(result: PipelineResult, show_sources: bool = True) -> None:
    """Formats and prints the pipeline result with citations and metrics."""
    print("=" * 75)
    print(f"SAWAAL (Question): {result.question}")
    print("=" * 75)

    if show_sources and result.sources:
        print("\n--- RETRIEVED EPISODE SOURCES ---")
        for idx, src in enumerate(result.sources, 1):
            ts = f" [{src.start_time} - {src.end_time}]" if src.start_time else ""
            chap = f" | Chapter: {src.chapter}" if src.chapter else ""
            print(f"[{idx}] {src.video_title}{chap}{ts} (Score: {src.similarity_score:.4f})")
            if src.video_url:
                print(f"    Link: {src.video_url}")
        print("-" * 75)

    print("\n" + result.answer + "\n")
    print("-" * 75)
    m = result.metrics
    status = "OFFLINE" if result.is_offline else f"ONLINE ({m.model_used})"
    print(
        f"[Status: {status}] Embed: {m.embedding_time_sec}s | Search: {m.retrieval_time_sec}s "
        f"| Generation: {m.generation_time_sec}s | Total: {m.total_time_sec}s "
        f"| Vectors Indexed: {m.total_vectors_indexed}"
    )
    print("=" * 75 + "\n")


def save_to_file(filepath: Path, result: PipelineResult) -> None:
    """Saves the question, sources, and Roman Urdu answer to disk."""
    try:
        content = [
            f"# GanjiSwag AI Response: {result.question}\n",
            f"**Generated via**: `{result.metrics.model_used}` | **Total Latency**: `{result.metrics.total_time_sec}s`\n",
            "## Sources Consulted\n",
        ]
        for idx, s in enumerate(result.sources, 1):
            ts = f" ({s.start_time} - {s.end_time})" if s.start_time else ""
            content.append(f"- **[{idx}] {s.video_title}**{ts}: {s.video_url} *(Score: {s.similarity_score:.4f})*")

        content.append("\n---\n")
        content.append(result.answer)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        print(f"Response successfully saved to: {filepath.resolve()}")
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")


def run_interactive(pipeline: TestingPipeline, top_k: int, show_sources: bool) -> None:
    """Runs interactive command loop for continuous user testing."""
    print("Starting Interactive Mode. Type your question in Roman Urdu, English, or Urdu.")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        try:
            user_input = input("User Question > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("\nKhuda Hafiz! Exiting testing pipeline.")
                break

            print("\n[Second Brain thinking & retrieving context...]")
            res = pipeline.ask(user_input, top_k=top_k)
            display_result(res, show_sources=show_sources)
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Khuda Hafiz!")
            break


def run_demo(pipeline: TestingPipeline, top_k: int, show_sources: bool) -> None:
    """Runs automatic test on standard benchmark questions."""
    print(f"Running automated benchmark suite ({len(DEMO_QUESTIONS)} questions)...\n")
    for idx, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n>>> Running Benchmark Query [{idx}/{len(DEMO_QUESTIONS)}]: {q}")
        res = pipeline.ask(q, top_k=top_k)
        display_result(res, show_sources=show_sources)
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ganjiswag AI Second Brain - Roman Urdu Testing Pipeline CLI."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Question to ask GanjiSwag Second Brain (in Roman Urdu, English, or Urdu)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive questioning session in terminal",
    )
    parser.add_argument(
        "--demo",
        "-d",
        action="store_true",
        help="Run automated benchmark demo questions",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-flash-lite-latest",
        help="Gemini model for response generation (default: gemini-flash-lite-latest)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="brain_config.yaml",
        help="Path to brain_config.yaml",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline fallback mode without making remote API calls",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source citation details from output",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save response to Markdown or text file",
    )

    args = parser.parse_args()
    print_banner()

    gen_cfg = GenerationConfig(
        model=args.model,
        top_k=args.top_k,
        offline=args.offline,
    )

    pipeline = TestingPipeline(
        config_path=args.config if Path(args.config).exists() else None,
        generation_config=gen_cfg,
    )

    if args.demo:
        run_demo(pipeline, top_k=args.top_k, show_sources=not args.no_sources)
    elif args.interactive or args.query is None:
        run_interactive(pipeline, top_k=args.top_k, show_sources=not args.no_sources)
    else:
        print(f"[Processing query: \"{args.query}\"...]\n")
        res = pipeline.ask(args.query, top_k=args.top_k)
        display_result(res, show_sources=not args.no_sources)
        if args.output:
            save_to_file(Path(args.output), res)


if __name__ == "__main__":
    main()
