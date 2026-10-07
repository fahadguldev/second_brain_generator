"""Pydantic v2 schema definitions and JSONL serialization for second_brain_generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceMetadata(BaseModel):
    """Metadata describing original source location, type, and lineage."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str  # video_transcript, audio_transcript, text_note, markdown_note, social_comment_reply, faq_qa
    file: str  # Relative path or identifier (e.g., 'data/vids/aws_intro.mp4')
    file_name: Optional[str] = None
    date: Optional[str] = None  # ISO-8601 YYYY-MM-DD or None
    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def populate_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "file_name" not in data or data["file_name"] is None:
                file_val = data.get("file", "")
                if file_val:
                    data["file_name"] = Path(file_val).name
        return data

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class ContentPayload(BaseModel):
    """Verbatim text content and language attributes."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    text: str  # Verbatim body text or creator response
    language: Optional[str] = None  # english, hinglish, urdu, hindi, or None
    question: Optional[str] = None  # Question text for Q&A pairs, or None
    question_language: Optional[str] = None

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Content text cannot be empty or whitespace-only.")
        return v

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class ClassificationPayload(BaseModel):
    """Topic, domain, and heuristic classifications."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    domain: str = "professional"
    topics: List[str] = Field(default_factory=list)
    knowledge: bool = True
    experience: bool = False
    opinion: bool = False
    style: bool = True
    ai_assisted: bool = False

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class KnowledgeRecord(BaseModel):
    """
    Canonical knowledge record stored in brain_data/records.jsonl.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str  # RFC 4122 UUID string generated via UUIDv5
    source: SourceMetadata
    content: ContentPayload
    classification: ClassificationPayload = Field(default_factory=ClassificationPayload)

    @field_validator("id")
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError(f"Record id must be a string: '{v}'")
        try:
            uuid.UUID(v)
        except (ValueError, AttributeError) as err:
            raise ValueError(f"Record id must be a valid UUID string: '{v}'") from err
        return v

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_rag_record(self) -> RagRecord:
        """
        Converts KnowledgeRecord to searchable RagRecord.
        If a question is present, search text is formatted as '{question}\n{text}'.
        """
        q = self.content.question
        ans = self.content.text
        if q and q.strip():
            search_text = f"{q.strip()}\n{ans.strip()}"
        else:
            search_text = ans.strip()

        meta: Dict[str, Any] = {
            "source": self.source.model_dump(mode="json"),
            "source_type": self.source.type,
            "classification": self.classification.model_dump(mode="json"),
            "domain": self.classification.domain,
            "topics": list(self.classification.topics),
            "language": self.content.language,
        }
        if q and q.strip():
            meta["question"] = q.strip()
            meta["question_language"] = self.content.question_language

        return RagRecord(
            id=self.id,
            text=search_text,
            metadata=meta,
        )


class RagRecord(BaseModel):
    """
    Search-optimized record stored in brain_data/rag_records.jsonl.
    Directly embedded by Gemini in M4 and upserted to Qdrant in M5.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str  # Deterministic UUIDv5
    text: str  # Target text for embedding and retrieval context
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError(f"RagRecord id must be a string: '{v}'")
        try:
            uuid.UUID(v)
        except (ValueError, AttributeError) as err:
            raise ValueError(f"RagRecord id must be a valid UUID string: '{v}'") from err
        return v

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("RagRecord text cannot be empty or whitespace-only.")
        return v

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_point_payload(self) -> PointPayload:
        """Converts to Qdrant point payload struct."""
        return PointPayload.from_rag_record(self)


class PointPayload(BaseModel):
    """
    Verified payload attached to Qdrant Vector Points.
    Compatible with existing second_brain_app/backend/src/rag.py retrieval logic.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_type: Optional[str] = None
    domain: Optional[str] = None
    topics: List[str] = Field(default_factory=list)

    @classmethod
    def from_rag_record(cls, rag_record: RagRecord) -> PointPayload:
        meta = dict(rag_record.metadata)
        source_type = meta.get("source_type")
        if not source_type and isinstance(meta.get("source"), dict):
            source_type = meta["source"].get("type")

        domain = meta.get("domain")
        if not domain and isinstance(meta.get("classification"), dict):
            domain = meta["classification"].get("domain", "professional")
        domain = domain or "professional"

        topics = meta.get("topics")
        if topics is None and isinstance(meta.get("classification"), dict):
            topics = meta["classification"].get("topics", [])
        topics = list(topics) if isinstance(topics, (list, tuple)) else []

        return cls(
            id=rag_record.id,
            text=rag_record.text,
            metadata=meta,
            source_type=source_type,
            domain=domain,
            topics=topics,
        )

    def to_qdrant_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary for Qdrant payload."""
        return self.model_dump(mode="json")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


def serialize_jsonl(records: List[Union[KnowledgeRecord, RagRecord]], file_path: Union[Path, str]) -> int:
    """
    Writes a list of Pydantic records to a JSONL file with UTF-8 encoding.
    Guarantees that each record occupies exactly one physical line.
    """
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            count += 1
    return count


def deserialize_records(file_path: Union[Path, str]) -> List[KnowledgeRecord]:
    """
    Reads and validates KnowledgeRecord objects from records.jsonl.
    """
    path = Path(file_path).resolve()
    records: List[KnowledgeRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(KnowledgeRecord.model_validate_json(line))
    return records


def deserialize_rag_records(file_path: Union[Path, str]) -> List[RagRecord]:
    """
    Reads and validates RagRecord objects from rag_records.jsonl.
    """
    path = Path(file_path).resolve()
    records: List[RagRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(RagRecord.model_validate_json(line))
    return records
