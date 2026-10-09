#!/usr/bin/env python3
"""
Second Brain Generator - Unified CLI Runner.

Supports executing the complete generation pipeline or individual stages:
    python run_generator.py              # Run complete generation (Stages 1-5, no Qdrant upload)
    python run_generator.py --all        # Run complete generation (Stages 1-5, no Qdrant upload)
    python run_generator.py --upload     # Separate command to push vectors to Qdrant
    python run_generator.py --qdrant     # Alias for --upload
    python uploader.py                   # Standalone command to push vectors to Qdrant
    python run_generator.py --transcribe # Run transcription stage only
    python run_generator.py --process    # Run compilation/processing stage only
    python run_generator.py --embed      # Run embedding generation stage only
    python run_generator.py --test       # Run retrieval evaluation stage only
    python run_generator.py --init       # Initialize directory structure only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import List, Optional

# Ensure second_brain_generator is on sys.path
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from second_brain_generator.config import BrainConfig, load_config, scaffold_directories
from second_brain_generator.utils.env import load_environment, get_gemini_keys, get_qdrant_credentials
from second_brain_generator.transcriber import MediaTranscriber
from second_brain_generator.compiler import KnowledgeCompiler
from second_brain_generator.embedder import GeminiEmbedder
from second_brain_generator.evaluator import BrainEvaluator
from second_brain_generator.uploader import QdrantUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_generator")


def print_banner(stage_num: int, title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f" STAGE {stage_num}: {title.upper()}")
    print(f"{'=' * 65}")


def step_init(config: BrainConfig, dry_run: bool = False) -> None:
    print_banner(1, "Scaffold Directory Hierarchy")
    if dry_run:
        print("[DRY RUN] Would create input directories (data/vids, audios, text, md, comments) and brain_data/text")
        return
    base_dir = Path(getattr(config, "base_dir", Path.cwd()))
    scaffold_directories(base_dir, config)
    print("✓ Directory structure initialized successfully.")


def step_transcribe(config: BrainConfig, force: bool = False, dry_run: bool = False) -> None:
    print_banner(2, "Speech-to-Text Media Transcription")
    base_dir = Path(getattr(config, "base_dir", Path.cwd()))
    vids_p = (base_dir / config.paths.get("vids_dir", "data/vids")).resolve()
    auds_p = (base_dir / config.paths.get("audios_dir", "data/audios")).resolve()

    media_count = 0
    for p in [vids_p, auds_p]:
        if p.exists():
            media_count += len([f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in [".mp4", ".mov", ".mkv", ".mp3", ".wav", ".m4a"]])

    if dry_run:
        print(f"[DRY RUN] Found {media_count} media files. Would run faster-whisper on CPU/GPU.")
        return

    transcriber = MediaTranscriber(config)
    stats = transcriber.transcribe_all(force=force)
    print(f"✓ Transcription complete: {stats['processed']} processed, {stats['skipped']} skipped, {stats['failed']} failed.")


def step_process(config: BrainConfig, dry_run: bool = False) -> None:
    print_banner(3, "Knowledge Compilation & Classification")
    base_dir = Path(getattr(config, "base_dir", Path.cwd()))
    rec_path = (base_dir / config.paths.get("records_path", "brain_data/records.jsonl")).resolve()
    rag_path = (base_dir / config.paths.get("rag_records_path", "brain_data/rag_records.jsonl")).resolve()

    if dry_run:
        print(f"[DRY RUN] Would compile notes, transcripts, and comments into:")
        print(f"          - {rec_path}")
        print(f"          - {rag_path}")
        return

    compiler = KnowledgeCompiler(config)
    counts = compiler.export_jsonl(rec_path, rag_path)
    print(f"✓ Compilation complete: {counts['records']} records, {counts['rag_records']} search-ready RAG records.")


def step_embed(config: BrainConfig, dry_run: bool = False) -> None:
    print_banner(4, "Vector Embedding Generation")
    base_dir = Path(getattr(config, "base_dir", Path.cwd()))
    rag_path = (base_dir / config.paths.get("rag_records_path", "brain_data/rag_records.jsonl")).resolve()
    emb_path = (base_dir / config.paths.get("embeddings_path", "brain_data/gemini_embeddings.npz")).resolve()
    ckpt_path = (base_dir / "brain_data/gemini_embeddings_checkpoint.npz").resolve()

    if dry_run:
        print(f"[DRY RUN] Would generate 3072-dim embeddings for {rag_path} -> {emb_path}")
        return

    if not rag_path.exists():
        print(f"❌ Error: RAG records file not found at {rag_path}. Run --process first.")
        sys.exit(1)

    texts: List[str] = []
    ids: List[str] = []
    with open(rag_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ids.append(item["id"])
                texts.append(item["text"])

    if not ids:
        print("ℹ️  No RAG records found to embed.")
        print("   Place your raw content in data/ (e.g. data/vids, data/text) and run --transcribe and --process first.")
        import numpy as np
        embedder = GeminiEmbedder(api_keys=[], batch_size=25)
        embedder.save_npz(emb_path, ids=[], embeddings=np.empty((0, 3072), dtype=np.float32), checkpoint_path=ckpt_path)
        return

    api_keys = get_gemini_keys()
    if not api_keys:
        logger.warning("No GEMINI_API_KEY found in environment. Using deterministic fallback embeddings for testing/offline mode.")

    embedder = GeminiEmbedder(api_keys=api_keys, batch_size=config.embedding.batch_size if hasattr(config, "embedding") else 25)
    embeddings = embedder.embed_texts(texts=texts, ids=ids, checkpoint_path=ckpt_path)
    embedder.save_npz(emb_path, ids=ids, embeddings=embeddings, checkpoint_path=ckpt_path)
    print(f"✓ Embedded {len(ids)} items with dimension {embeddings.shape[1]}. Saved to {emb_path.name}.")


def step_test(config: BrainConfig, dry_run: bool = False) -> None:
    print_banner(5, "Offline Retrieval Evaluation")
    base_dir = Path(getattr(config, "base_dir", Path.cwd()))
    rag_path = (base_dir / config.paths.get("rag_records_path", "brain_data/rag_records.jsonl")).resolve()
    emb_path = (base_dir / config.paths.get("embeddings_path", "brain_data/gemini_embeddings.npz")).resolve()

    if dry_run:
        print(f"[DRY RUN] Would calculate cosine similarity across benchmark queries using {emb_path}")
        return

    if not emb_path.exists() or not rag_path.exists():
        print("❌ Error: Missing embeddings or rag_records. Run --embed first.")
        return

    evaluator = BrainEvaluator(embeddings_path=emb_path, records_path=rag_path)
    if len(evaluator.embeddings) == 0:
        print("ℹ️  Embeddings database is empty. Add data to run retrieval tests.")
        return

    metrics = evaluator.evaluate_benchmarks()
    print(f"✓ Evaluation complete across {int(metrics['total_queries'])} benchmark queries.")
    print(f"  Mean Cosine Similarity: {metrics['mean_similarity_score']}")


def step_upload(config: BrainConfig, dry_run: bool = False) -> None:
    print_banner(6, "Qdrant Vector Indexing")
    base_dir = Path(getattr(config, "base_dir", Path.cwd()))
    rag_path = (base_dir / config.paths.get("rag_records_path", "brain_data/rag_records.jsonl")).resolve()
    emb_path = (base_dir / config.paths.get("embeddings_path", "brain_data/gemini_embeddings.npz")).resolve()

    coll_name = "second_brain"
    if hasattr(config, "qdrant") and getattr(config.qdrant, "collection_name", None):
        coll_name = config.qdrant.collection_name

    creds = get_qdrant_credentials()
    url = creds.url or getattr(config.qdrant, "url", None)
    api_key = creds.api_key or getattr(config.qdrant, "api_key", None)

    if dry_run:
        print(f"[DRY RUN] Would upload points from {emb_path} into collection '{coll_name}' at {url or ':memory:'}")
        return

    if not emb_path.exists() or not rag_path.exists():
        print("❌ Error: Missing embeddings or records. Run --embed first.")
        return

    import numpy as np
    with np.load(emb_path, allow_pickle=True) as data:
        if len(data["embeddings"]) == 0:
            print(f"ℹ️  No points to upload. Collection '{coll_name}' ready once data is embedded.")
            return

    uploader = QdrantUploader(url=url, api_key=api_key, collection_name=coll_name)
    uploaded_count = uploader.upload_from_disk(embeddings_path=emb_path, records_path=rag_path)
    print(f"✓ Successfully indexed {uploaded_count} points into Qdrant collection '{coll_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Second Brain Generator - Automated AI Second Brain Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    action_group = parser.add_argument_group("Pipeline Execution Flags")
    action_group.add_argument("--all", action="store_true", help="Execute complete generation pipeline (init -> transcribe -> process -> embed -> test). Does NOT push vectors to Qdrant.")
    action_group.add_argument("--init", action="store_true", help="Initialize and scaffold data directory structure")
    action_group.add_argument("--transcribe", action="store_true", help="Transcribe audio and video media files with Whisper")
    action_group.add_argument("--process", action="store_true", help="Compile notes, transcripts, comments and classify topics/languages")
    action_group.add_argument("--embed", action="store_true", help="Generate vector embeddings using Gemini API")
    action_group.add_argument("--test", action="store_true", help="Run offline retrieval evaluations and benchmark tests")
    action_group.add_argument("--upload", "--qdrant", dest="upload", action="store_true", help="Separate command to index and batch-upsert points into Qdrant vector database")

    opt_group = parser.add_argument_group("Options")
    opt_group.add_argument("--config", type=str, default="brain_config.yaml", help="Path to brain_config.yaml")
    opt_group.add_argument("--data-dir", type=str, help="Override root data directory path")
    opt_group.add_argument("--force", action="store_true", help="Force re-transcription / re-processing of existing files")
    opt_group.add_argument("--dry-run", action="store_true", help="Simulate pipeline actions without disk/network side effects")

    args = parser.parse_args()

    # Load environment variables (.env)
    load_environment()

    # Verify config path
    config_path = Path(args.config).resolve()
    if not config_path.exists() and (SCRIPT_DIR / args.config).exists():
        config_path = (SCRIPT_DIR / args.config).resolve()
    if not config_path.exists():
        print(f"❌ Error: Configuration file '{args.config}' not found.")
        sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    if args.data_dir:
        config.paths["data_dir"] = args.data_dir

    dry_run = args.dry_run

    start_time = time.time()
    print("\n🧠 SECOND BRAIN GENERATOR")
    print(f"Creator: {getattr(config, 'creator_name', 'Fahad Gul')} | Config: {config_path.name}")
    if dry_run:
        print("⚡ MODE: DRY-RUN (Simulating execution)")

    # Determine if full generation pipeline should run
    stage_flags_passed = [args.init, args.transcribe, args.process, args.embed, args.test, args.upload]
    run_whole_pipeline = args.all or not any(stage_flags_passed)

    if run_whole_pipeline:
        step_init(config, dry_run=dry_run)
        step_transcribe(config, force=args.force, dry_run=dry_run)
        step_process(config, dry_run=dry_run)
        step_embed(config, dry_run=dry_run)
        step_test(config, dry_run=dry_run)

        if not args.upload:
            print("\n" + "=" * 65)
            print(" GENERATION PIPELINE COMPLETE (Stages 1-5)")
            print("=" * 65)
            print("✓ Vectors generated and saved to disk.")
            print("ℹ️  Vectors were NOT pushed to Qdrant.")
            print("   To push vectors to Qdrant, run the separate command:")
            print("       python run_generator.py --upload")
            print("       (or python uploader.py)")
            print("=" * 65)
    else:
        if args.init:
            step_init(config, dry_run=dry_run)
        if args.transcribe:
            step_transcribe(config, force=args.force, dry_run=dry_run)
        if args.process:
            step_process(config, dry_run=dry_run)
        if args.embed:
            step_embed(config, dry_run=dry_run)
        if args.test:
            step_test(config, dry_run=dry_run)

    # Separate step for Qdrant indexing
    if args.upload:
        step_upload(config, dry_run=dry_run)

    duration = time.time() - start_time
    print(f"\n✨ Pipeline execution finished in {duration:.2f}s.\n")


if __name__ == "__main__":
    main()
