"""Text chunking module matching the Second Brain App ingestion logic."""

from __future__ import annotations

from typing import List


def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Splits text into chunks of maximum `size` characters with `overlap`,
    breaking cleanly on whitespace boundaries when possible.
    Matches the chunking behavior of second_brain_app.
    """
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not clean:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(clean[start:end].strip())
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks
