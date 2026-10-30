"""Deterministic UUIDv5 generation and text normalization utilities for second_brain_generator."""

from __future__ import annotations

import hashlib
from pathlib import PurePath
import re
from typing import Optional, Union
import uuid

# Canonical namespace for Second Brain UUID generation
DEFAULT_NAMESPACE: uuid.UUID = uuid.NAMESPACE_DNS
UUID_NAMESPACE_SECOND_BRAIN: uuid.UUID = uuid.UUID("a2b4c6e8-1357-4920-b846-91e8f237a4c5")


def normalize_text(text: Optional[str]) -> str:
    """
    Normalizes text by stripping whitespace, collapsing multiple whitespace
    characters (spaces, tabs, newlines, CRLF) into a single space, and lowercasing.
    Returns an empty string if input is None or whitespace-only.
    """
    if not text:
        return ""
    # Strip null characters and collapse whitespace
    cleaned = str(text).replace("\x00", "").strip()
    return re.sub(r"\s+", " ", cleaned).lower()


def normalize_identifier(identifier: Optional[Union[str, PurePath]]) -> str:
    """
    Normalizes file paths or resource identifiers across operating systems:
    - Converts backslashes to forward slashes.
    - Strips leading and trailing whitespace.
    - Converts to lowercase.
    - Replaces empty identifier with 'default'.
    """
    if identifier is None:
        return "default"
    ident_str = str(identifier).replace("\\", "/").strip().lower()
    return ident_str or "default"


def compute_content_hash(text: str, question: Optional[str] = None) -> str:
    """
    Computes SHA-256 hex digest over normalized content string.
    If question is provided and non-empty, concatenates '{norm_q}||{norm_text}'.
    """
    norm_text = normalize_text(text)
    if question and question.strip():
        norm_q = normalize_text(question)
        payload = f"{norm_q}||{norm_text}"
    else:
        payload = norm_text

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_record_uuid(
    source_type: str,
    identifier: Union[str, PurePath],
    text: str,
    question: Optional[str] = None,
    namespace: uuid.UUID = DEFAULT_NAMESPACE,
) -> str:
    """
    Generates a deterministic RFC 4122 UUIDv5 for a knowledge or RAG record.

    Args:
        source_type: Classification of source ('video_transcript', 'text_note', etc.).
        identifier: Relative file path, comment ID, or item key.
        text: Verbatim body text.
        question: Optional question for Q&A pairs.
        namespace: UUID namespace (defaults to uuid.NAMESPACE_DNS).

    Returns:
        36-character canonical UUIDv5 string (e.g., 'dcfc463a-6859-5d3c-90b3-0e2c7c668f28').
    """
    norm_st = normalize_text(source_type)
    norm_id = normalize_identifier(identifier)
    chash = compute_content_hash(text, question)

    seed = f"{norm_st}:{norm_id}:{chash}"
    return str(uuid.uuid5(namespace, seed))


def is_valid_uuid(val: str, expected_version: Optional[int] = 5) -> bool:
    """
    Checks whether a string is a valid UUID, optionally verifying the version.
    """
    if not isinstance(val, str):
        return False
    try:
        u = uuid.UUID(val)
        if expected_version is not None and u.version != expected_version:
            return False
        return True
    except (ValueError, AttributeError):
        return False
