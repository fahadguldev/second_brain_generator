#!/usr/bin/env python3
"""
Standalone Qdrant Uploader CLI for Second Brain Generator.

Usage:
    python uploader.py [--config brain_config.yaml] [--batch-size 100]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from second_brain_generator.config import load_config
from second_brain_generator.utils.env import load_environment, get_qdrant_credentials
from second_brain_generator.uploader import QdrantUploader


def main():
    parser = argparse.ArgumentParser(description="Upload Second Brain records to Qdrant.")
    parser.add_argument("--config", type=str, default="brain_config.yaml", help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for upsert")
    args = parser.parse_args()

    load_environment()
    cfg_p = Path(args.config).resolve()
    if not cfg_p.exists():
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    cfg = load_config(cfg_p)
    emb_p = Path(cfg.paths.get("embeddings_path", "brain_data/gemini_embeddings.npz")).resolve()
    rag_p = Path(cfg.paths.get("rag_records_path", "brain_data/rag_records.jsonl")).resolve()

    creds = get_qdrant_credentials()
    url = creds.url or getattr(cfg.qdrant, "url", None)
    api_key = creds.api_key or getattr(cfg.qdrant, "api_key", None)
    coll_name = getattr(cfg.qdrant, "collection_name", "second_brain")

    uploader = QdrantUploader(url=url, api_key=api_key, collection_name=coll_name)
    count = uploader.upload_from_disk(emb_p, rag_p, batch_size=args.batch_size)
    print(f"Successfully uploaded {count} points to Qdrant collection '{coll_name}'.")


if __name__ == "__main__":
    main()
