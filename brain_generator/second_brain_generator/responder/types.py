"""Typed data structures and interfaces for the Roman Urdu testing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SourceCitation:
    """Represents a retrieved context source citation from Ask Ganjiswag transcripts."""
    record_id: str
    video_id: str
    video_title: str
    chapter: str
    start_time: str
    end_time: str
    video_url: str
    similarity_score: float
    raw_excerpt: str

    def format_citation(self, rank: int) -> str:
        """Returns human-readable citation line for terminal and markdown outputs."""
        timestamp_info = f" ({self.start_time} - {self.end_time})" if self.start_time else ""
        chapter_info = f" - Chapter: {self.chapter}" if self.chapter else ""
        url_info = f" | {self.video_url}" if self.video_url else ""
        return (
            f"[{rank}] {self.video_title}{chapter_info}{timestamp_info} "
            f"(Score: {self.similarity_score:.4f}){url_info}"
        )


@dataclass(frozen=True)
class PipelineMetrics:
    """Execution timing and resource metrics for a query response cycle."""
    embedding_time_sec: float
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    model_used: str
    retrieved_count: int
    total_vectors_indexed: int


@dataclass
class GenerationConfig:
    """Configuration options for Roman Urdu answer generation."""
    model: str = "gemini-flash-lite-latest"
    fallback_models: List[str] = field(
        default_factory=lambda: [
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview",
        ]
    )
    embedding_model: str = "models/gemini-embedding-2"
    embedding_dimension: int = 3072
    top_k: int = 5
    min_similarity_threshold: float = 0.25
    temperature: float = 0.7
    max_output_tokens: int = 4096
    offline: bool = False


@dataclass
class PipelineResult:
    """Complete structured response from the testing pipeline."""
    question: str
    answer: str
    sources: List[SourceCitation]
    metrics: PipelineMetrics
    is_offline: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
