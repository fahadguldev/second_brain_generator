"""Core end-to-end testing pipeline connecting vector retrieval and Roman Urdu response generation."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..evaluator import BrainEvaluator
from ..config import BrainConfig, load_config
from .types import GenerationConfig, PipelineMetrics, PipelineResult, SourceCitation
from .prompts import GANJISWAG_SYSTEM_INSTRUCTION, build_user_prompt
from .llm import GeminiResponderClient

logger = logging.getLogger(__name__)


def parse_source_chunk(rec: Dict[str, Any], score: float) -> SourceCitation:
    """Parses video metadata and transcript text into a structured SourceCitation."""
    rec_id = str(rec.get("id", ""))
    full_text = rec.get("text") or rec.get("content", {}).get("text", "")

    # Extract metadata from header lines
    video_id = ""
    video_title = "Ask Ganjiswag Episode"
    chapter = ""
    start_time = ""
    end_time = ""
    video_url = ""

    lines = full_text.split("\n")
    transcript_lines: List[str] = []
    is_transcript = False

    for line in lines:
        line_clean = line.strip()
        if "--- TRANSCRIPT ---" in line_clean:
            is_transcript = True
            continue
        if is_transcript:
            transcript_lines.append(line)
        else:
            if line_clean.startswith("Video ID:"):
                video_id = line_clean.replace("Video ID:", "").strip()
            elif line_clean.startswith("Video Title:"):
                video_title = line_clean.replace("Video Title:", "").strip()
            elif "Question/Chapter:" in line_clean:
                parts = line_clean.split("Question/Chapter:", 1)
                chapter = parts[1].split("Chapter Number:")[0].strip()
            elif line_clean.startswith("Start Timestamp:"):
                start_time = line_clean.replace("Start Timestamp:", "").strip()
            elif line_clean.startswith("End Timestamp:"):
                end_time = line_clean.replace("End Timestamp:", "").strip()
            elif line_clean.startswith("Video URL:"):
                video_url = line_clean.replace("Video URL:", "").strip()

    raw_transcript = "\n".join(transcript_lines).strip()
    if not raw_transcript:
        raw_transcript = full_text[:1200]

    return SourceCitation(
        record_id=rec_id,
        video_id=video_id,
        video_title=video_title,
        chapter=chapter,
        start_time=start_time,
        end_time=end_time,
        video_url=video_url,
        similarity_score=score,
        raw_excerpt=raw_transcript[:2000],
    )


class TestingPipeline:
    """
    End-to-End Testing Pipeline for Ask Ganjiswag AI Second Brain.
    Performs cross-lingual semantic search over Hindi/Devanagari transcripts
    and generates comprehensive, maximum-length responses in Roman Urdu.
    """
    __test__ = False

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        embeddings_path: Optional[Union[str, Path]] = None,
        records_path: Optional[Union[str, Path]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.gen_config = generation_config or GenerationConfig()
        self.llm_client = GeminiResponderClient(self.gen_config)

        # Resolve paths relative to working directory or project root
        base_dir = Path.cwd()
        if config_path and Path(config_path).exists():
            cfg = load_config(config_path)
            base_dir = Path(getattr(cfg, "base_dir", base_dir))

        self.embeddings_path = self._resolve_embeddings_file(base_dir, embeddings_path)
        self.records_path = self._resolve_records_file(base_dir, records_path)

        # In-memory cached evaluator
        self._evaluator: Optional[BrainEvaluator] = None

    def _resolve_embeddings_file(
        self, base_dir: Path, explicit_path: Optional[Union[str, Path]]
    ) -> Path:
        """
        Discovers the most complete, valid embeddings NPZ file.
        Safely ignores incomplete temporary write files.
        """
        if explicit_path and Path(explicit_path).exists():
            return Path(explicit_path).resolve()

        # Gather potential directories
        search_dirs = [
            base_dir / "brain_data",
            base_dir / "second_brain_generator" / "brain_data",
        ]

        candidate_files: List[Path] = []
        for d in search_dirs:
            if d.exists() and d.is_dir():
                for p in d.glob("*.npz"):
                    if not p.name.endswith(".tmp") and ".tmp." not in p.name:
                        candidate_files.append(p)

        # Sort preference: files with valid arrays and largest number of items
        best_file: Optional[Path] = None
        max_items = -1

        for c in candidate_files:
            try:
                with np.load(c, allow_pickle=True) as data:
                    if "ids" in data and "embeddings" in data:
                        num_ids = len(data["ids"])
                        if num_ids > max_items:
                            max_items = num_ids
                            best_file = c.resolve()
            except Exception:
                continue

        if best_file:
            return best_file

        return (base_dir / "brain_data/gemini_embeddings.npz").resolve()

    def _resolve_records_file(
        self, base_dir: Path, explicit_path: Optional[Union[str, Path]]
    ) -> Path:
        """Locates the search-ready RAG records JSONL file."""
        if explicit_path and Path(explicit_path).exists():
            return Path(explicit_path).resolve()

        candidates = [
            base_dir / "brain_data/rag_records.jsonl",
            base_dir / "second_brain_generator/brain_data/rag_records.jsonl",
            base_dir / "brain_data/records.jsonl",
            base_dir / "second_brain_generator/brain_data/records.jsonl",
        ]

        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()

        return (base_dir / "brain_data/rag_records.jsonl").resolve()

    @property
    def evaluator(self) -> BrainEvaluator:
        """Caches and returns the BrainEvaluator instance."""
        if self._evaluator is None:
            if not self.embeddings_path.exists() or not self.records_path.exists():
                logger.warning(
                    "Embeddings (%s) or records (%s) missing on disk.",
                    self.embeddings_path,
                    self.records_path,
                )
            self._evaluator = BrainEvaluator(
                embeddings_path=self.embeddings_path,
                records_path=self.records_path,
            )
        return self._evaluator

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> PipelineResult:
        """
        Executes complete RAG pipeline for a question:
        1. Embeds question.
        2. Retrieves top_k relevant context chunks.
        3. Builds comprehensive Roman Urdu prompt.
        4. Invokes LLM to generate maximal Roman Urdu response.
        5. Returns structured PipelineResult with timings and citations.
        """
        k = top_k or self.gen_config.top_k
        start_pipeline = time.perf_counter()

        # Step 1: Embed question
        q_vec, embed_time = self.llm_client.embed_query(question)

        # Step 2: Semantic retrieval
        retrieval_start = time.perf_counter()
        sources: List[SourceCitation] = []
        indexed_count = 0

        try:
            eval_instance = self.evaluator
            indexed_count = len(eval_instance.embeddings)
            results = eval_instance.search(q_vec, top_k=k)
            for r in results:
                src = parse_source_chunk(r.get("record", {}), r.get("score", 0.0))
                sources.append(src)
        except Exception as err:
            logger.warning("Retrieval warning: %s. Continuing with direct synthesis.", err)

        retrieval_time = time.perf_counter() - retrieval_start

        # Step 3: Build Prompt
        prompt = build_user_prompt(question, sources)

        # Step 4: Generate Roman Urdu Answer
        answer_text, model_name, gen_time = self.llm_client.generate_response(
            prompt=prompt,
            system_instruction=GANJISWAG_SYSTEM_INSTRUCTION,
        )

        total_time = time.perf_counter() - start_pipeline

        metrics = PipelineMetrics(
            embedding_time_sec=round(embed_time, 3),
            retrieval_time_sec=round(retrieval_time, 3),
            generation_time_sec=round(gen_time, 3),
            total_time_sec=round(total_time, 3),
            model_used=model_name,
            retrieved_count=len(sources),
            total_vectors_indexed=indexed_count,
        )

        return PipelineResult(
            question=question,
            answer=answer_text,
            sources=sources,
            metrics=metrics,
            is_offline=self.gen_config.offline or "offline" in model_name,
            metadata={"embeddings_file": str(self.embeddings_path.name)},
        )
