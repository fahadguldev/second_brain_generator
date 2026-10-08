"""KnowledgeCompiler orchestrating data reading, classification, normalization, and JSONL export."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import uuid

from ..config import BrainConfig
from ..schemas import KnowledgeRecord, RagRecord, serialize_jsonl
from ..utils.uuids import UUID_NAMESPACE_SECOND_BRAIN, generate_record_uuid, normalize_text
from .classifier import detect_language, infer_booleans, is_deflection, is_style_only, make_topics
from .chunker import chunk_text
from .readers import read_comments_csv, read_comments_json, read_markdown_notes, read_text_notes, read_transcripts

logger = logging.getLogger(__name__)


class KnowledgeCompiler:
    """
    Ingests raw and transcribed knowledge assets, standardizes metadata,
    applies topic and language classification, and writes records.jsonl and rag_records.jsonl.
    """

    def __init__(self, config: BrainConfig) -> None:
        self.config = config
        self.base_dir = Path(getattr(config, "base_dir", Path.cwd())).resolve()
        self.paths = config.paths
        self.topics_lexicon = getattr(config, "topics", {}) or {}

        # Resolve creator handles
        handles = getattr(config, "creator_handles", []) or []
        if hasattr(config, "creator") and getattr(config.creator, "handles", None):
            handles.extend(config.creator.handles)
        self.creator_handles = list(set(handles))

        self.default_domain = "professional"
        if hasattr(config, "persona") and getattr(config.persona, "default_domain", None):
            self.default_domain = config.persona.default_domain

        self.chunk_size = getattr(config, "chunk_size", 1500)
        self.chunk_overlap = getattr(config, "chunk_overlap", 200)

    def process_all(self) -> Tuple[List[KnowledgeRecord], List[RagRecord]]:
        """
        Executes ingestion across all input directories, performs classification,
        and returns validated lists of KnowledgeRecord and RagRecord.
        """
        raw_items = []

        # 1. Transcripts from brain_data/text or transcripts
        trans_path = (self.base_dir / self.paths.get("transcripts_dir", "brain_data/text")).resolve()
        raw_items.extend(read_transcripts(trans_path))

        # 2. Text notes from data/text
        text_path = (self.base_dir / self.paths.get("text_dir", "data/text")).resolve()
        raw_items.extend(read_text_notes(text_path))

        # 3. Markdown notes from data/md
        md_path = (self.base_dir / self.paths.get("md_dir", "data/md")).resolve()
        raw_items.extend(read_markdown_notes(md_path))

        # 4. Comments from data/comments
        comm_path = (self.base_dir / self.paths.get("comments_dir", "data/comments")).resolve()
        raw_items.extend(read_comments_json(comm_path, self.creator_handles))
        raw_items.extend(read_comments_csv(comm_path))

        knowledge_records: List[KnowledgeRecord] = []
        rag_records: List[RagRecord] = []
        seen_keys = set()

        for item in raw_items:
            text = item.get("text", "")
            if not text or not str(text).strip():
                continue

            q = item.get("question")
            # Filter deflections
            if is_deflection(text):
                logger.debug("Filtering deflection response: %s", text)
                continue

            # Deduplication key
            norm_t = normalize_text(text)
            norm_q = normalize_text(q) if q else ""
            dedup_key = f"{item['source_type']}::{norm_q}::{norm_t}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Chunk the content using brain app's chunking algorithm
            chunks = chunk_text(text, size=self.chunk_size, overlap=self.chunk_overlap)
            if not chunks:
                continue

            total_chunks = len(chunks)
            base_ident = item.get("file_name") or item.get("file_path", "default")

            for position, chunk in enumerate(chunks):
                # Classifications
                lang = detect_language(chunk)
                q_lang = detect_language(q) if q else None
                topics = make_topics(f"{chunk} {q or ''}", self.topics_lexicon, filename=item.get("file_name", ""))
                exp, opi = infer_booleans(chunk)
                style_only = is_style_only(chunk, q)

                # Deterministic UUID
                chunk_ident = f"{base_ident}#chunk-{position}" if total_chunks > 1 else base_ident
                rec_id = generate_record_uuid(
                    source_type=item["source_type"],
                    identifier=chunk_ident,
                    text=chunk,
                    question=q,
                    namespace=UUID_NAMESPACE_SECOND_BRAIN,
                )

                extra = dict(item.get("extra", {}))
                if total_chunks > 1:
                    extra["chunk_index"] = position
                    extra["chunks_total"] = total_chunks

                # KnowledgeRecord payload
                source_payload = {
                    "type": item["source_type"],
                    "file": item.get("file_path", ""),
                    "file_name": item.get("file_name"),
                    "date": item.get("date"),
                    "extra": extra,
                }

                content_payload = {
                    "text": chunk,
                    "language": lang,
                    "question": q,
                    "question_language": q_lang,
                }

                class_payload = {
                    "domain": self.default_domain,
                    "topics": topics,
                    "knowledge": not style_only,
                    "experience": exp,
                    "opinion": opi,
                    "style": True,
                    "ai_assisted": False,
                }

                k_rec = KnowledgeRecord(
                    id=rec_id,
                    source=source_payload,
                    content=content_payload,
                    classification=class_payload,
                )
                knowledge_records.append(k_rec)

                # RagRecord payload
                search_text = f"{q}\n{chunk}" if q else chunk
                rag_rec = RagRecord(
                    id=rec_id,
                    text=search_text,
                    metadata={
                        "source": source_payload,
                        "classification": class_payload,
                        "language": lang,
                        "question": q,
                    },
                )
                rag_records.append(rag_rec)

        logger.info(
            "Compiled %d knowledge records and %d RAG records",
            len(knowledge_records), len(rag_records)
        )
        return knowledge_records, rag_records

    def export_jsonl(
        self,
        records_path: Optional[Union[str, Path]] = None,
        rag_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, int]:
        """
        Processes knowledge and writes records.jsonl and rag_records.jsonl atomically.
        """
        rec_p = Path(records_path or (self.base_dir / self.paths.get("records_path", "brain_data/records.jsonl"))).resolve()
        rag_p = Path(rag_path or (self.base_dir / self.paths.get("rag_records_path", "brain_data/rag_records.jsonl"))).resolve()

        records, rag_records = self.process_all()

        serialize_jsonl(records, rec_p)
        serialize_jsonl(rag_records, rag_p)

        return {"records": len(records), "rag_records": len(rag_records)}
