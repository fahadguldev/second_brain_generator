"""Tests for text chunking matching Second Brain App ingestion."""

from pathlib import Path
import pytest

from second_brain_generator.compiler.chunker import chunk_text
from second_brain_generator.compiler.compiler import KnowledgeCompiler
from second_brain_generator.config import BrainConfig


def test_chunk_text_empty_and_whitespace():
    """Empty or whitespace-only text produces no chunks."""
    assert chunk_text("") == []
    assert chunk_text("   \n\n  \t  ") == []


def test_chunk_text_short_text_single_chunk():
    """Text shorter than chunk size returns a single chunk without truncation."""
    short = "This is a short note on Python and FastAPI."
    chunks = chunk_text(short, size=1500, overlap=200)
    assert len(chunks) == 1
    assert chunks[0] == short


def test_chunk_text_clean_whitespace_lines():
    """Leading and trailing empty lines are cleaned."""
    text = "\n\nLine 1\n\n  Line 2  \n\n"
    chunks = chunk_text(text, size=1500, overlap=200)
    assert len(chunks) == 1
    assert chunks[0] == "Line 1\nLine 2"


def test_chunk_text_long_text_splits_with_overlap():
    """Long text splits into multiple chunks respecting max size and overlap."""
    words = ["word" + str(i) for i in range(500)]
    long_text = " ".join(words)
    chunks = chunk_text(long_text, size=300, overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 300
    # Overlap check: adjacent chunks share common text
    assert any(w in chunks[1] for w in chunks[0].split()[-3:])


def test_compiler_chunks_large_markdown_notes(tmp_path: Path):
    """Compiler splits large notes into multiple indexed KnowledgeRecord and RagRecord items."""
    md_dir = tmp_path / "data" / "md"
    md_dir.mkdir(parents=True)
    
    # Create a 4000-character markdown note
    paragraphs = [
        f"Paragraph {i}: Detailed explanation of architecture and deployment for project scaling."
        for i in range(50)
    ]
    long_content = "\n\n".join(paragraphs)
    (md_dir / "large_project.md").write_text(long_content, encoding="utf-8")

    cfg = BrainConfig(
        base_dir=tmp_path,
        paths={"md_dir": str(md_dir.relative_to(tmp_path))},
        chunk_size=500,
        chunk_overlap=50,
    )

    compiler = KnowledgeCompiler(cfg)
    k_recs, rag_recs = compiler.process_all()

    assert len(k_recs) > 1
    assert len(k_recs) == len(rag_recs)

    # Check chunk metadata in extra
    for i, rec in enumerate(k_recs):
        assert rec.source.extra.get("chunk_index") == i
        assert rec.source.extra.get("chunks_total") == len(k_recs)
        assert len(rec.content.text) <= 500
