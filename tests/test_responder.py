"""Unit tests for the Roman Urdu testing pipeline and responder module."""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from second_brain_generator.responder.types import (
    GenerationConfig,
    SourceCitation,
)
from second_brain_generator.responder.prompts import (
    GANJISWAG_SYSTEM_INSTRUCTION,
    build_user_prompt,
)
from second_brain_generator.responder.pipeline import (
    TestingPipeline,
    parse_source_chunk,
)
from second_brain_generator.responder.llm import GeminiResponderClient


def test_source_citation_format():
    citation = SourceCitation(
        record_id="rec-123",
        video_id="abcXYZ",
        video_title="Ask Ganjiswag #50",
        chapter="Karachi Street Crime",
        start_time="00:10:00",
        end_time="00:15:30",
        video_url="https://youtube.com/watch?v=abcXYZ",
        similarity_score=0.8842,
        raw_excerpt="Yeh transcript ka hissa hai.",
    )
    formatted = citation.format_citation(rank=1)
    assert "[1] Ask Ganjiswag #50" in formatted
    assert "Karachi Street Crime" in formatted
    assert "00:10:00 - 00:15:30" in formatted
    assert "0.8842" in formatted


def test_parse_source_chunk_metadata():
    sample_text = (
        "Video ID: Jso8aknlnMo\n"
        "Video Title: Minorities in Pakistan\n"
        "Question/Chapter: Karachi city of Crime. Chapter Number: 5\n"
        "Start Timestamp: 00:46:01\n"
        "End Timestamp: 01:00:17\n"
        "Video URL: https://www.youtube.com/watch?v=Jso8aknlnMo\n"
        "--- TRANSCRIPT ---\n"
        "Yahan par transcript ka data mojood hai."
    )
    rec = {"id": "test-id", "text": sample_text}
    citation = parse_source_chunk(rec, score=0.7654)

    assert citation.record_id == "test-id"
    assert citation.video_id == "Jso8aknlnMo"
    assert citation.video_title == "Minorities in Pakistan"
    assert "Karachi city of Crime." in citation.chapter
    assert citation.start_time == "00:46:01"
    assert citation.end_time == "01:00:17"
    assert citation.similarity_score == 0.7654
    assert "Yahan par transcript ka data mojood hai." in citation.raw_excerpt


def test_build_user_prompt_roman_urdu():
    sources = [
        SourceCitation(
            record_id="1",
            video_id="v1",
            video_title="Episode 1",
            chapter="Intro",
            start_time="00:00",
            end_time="01:00",
            video_url="http://yt.com/1",
            similarity_score=0.9,
            raw_excerpt="Kuch text context ka",
        )
    ]
    prompt = build_user_prompt("CS degree vs skills", sources)
    assert "USER KA SAWAAL:" in prompt
    assert "CS degree vs skills" in prompt
    assert "ROMAN URDU" in prompt
    assert "Episode 1" in prompt
    assert "ROMAN URDU" in GANJISWAG_SYSTEM_INSTRUCTION


def test_offline_pipeline_execution():
    gen_cfg = GenerationConfig(offline=True, top_k=2)
    pipeline = TestingPipeline(generation_config=gen_cfg)

    result = pipeline.ask("Karachi street crime par kya khayal hai?")
    assert result.question == "Karachi street crime par kya khayal hai?"
    assert result.is_offline is True
    assert "GanjiSwag" in result.answer
    assert "Seedhi Baat" in result.answer
    assert result.metrics.total_time_sec >= 0.0
