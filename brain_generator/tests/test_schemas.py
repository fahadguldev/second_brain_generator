"""Unit and integration tests for Schemas, UUIDv5 Generator, and JSONL Serializer (F05-F06)."""

import json
from pathlib import Path, PureWindowsPath, PurePosixPath
import uuid
import pytest
from pydantic import ValidationError

from second_brain_generator.schemas import (
    KnowledgeRecord,
    RagRecord,
    PointPayload,
    SourceMetadata,
    ContentPayload,
    ClassificationPayload,
    serialize_jsonl,
    deserialize_records,
    deserialize_rag_records,
)
from second_brain_generator.utils.uuids import (
    generate_record_uuid,
    normalize_text,
    normalize_identifier,
    compute_content_hash,
    is_valid_uuid,
    DEFAULT_NAMESPACE,
    UUID_NAMESPACE_SECOND_BRAIN,
)


# ==============================================================================
# Category 1: Deterministic UUIDv5 Generation & Normalization
# ==============================================================================

def test_uuid5_deterministic_identity():
    """Identical inputs produce identical UUIDv5 across calls."""
    u1 = generate_record_uuid("video_transcript", "data/vids/aws.mp4", "Welcome to AWS EC2")
    u2 = generate_record_uuid("video_transcript", "data/vids/aws.mp4", "Welcome to AWS EC2")
    assert u1 == u2
    assert u1 == "dcfc463a-6859-5d3c-90b3-0e2c7c668f28"


def test_uuid5_whitespace_and_newline_invariance():
    """Whitespace, tabs, CRLF, and LF variations produce identical UUIDs."""
    t1 = "  Welcome   to \r\n  AWS   EC2  "
    t2 = "Welcome to AWS EC2"
    t3 = "welcome\t\tto aws ec2\n"

    u1 = generate_record_uuid("video_transcript", "data/vids/aws.mp4", t1)
    u2 = generate_record_uuid("video_transcript", "data/vids/aws.mp4", t2)
    u3 = generate_record_uuid("video_transcript", "data/vids/aws.mp4", t3)
    assert u1 == u2 == u3


def test_uuid5_cross_platform_path_invariance():
    """Windows backslash paths and POSIX paths yield identical UUIDs."""
    posix_path = "data/vids/aws.mp4"
    win_path = r"data\vids\aws.mp4"
    u1 = generate_record_uuid("video_transcript", posix_path, "Welcome to AWS EC2")
    u2 = generate_record_uuid("video_transcript", win_path, "Welcome to AWS EC2")
    assert u1 == u2

    # Also test PurePath objects
    u3 = generate_record_uuid("video_transcript", PurePosixPath(posix_path), "Welcome to AWS EC2")
    u4 = generate_record_uuid("video_transcript", PureWindowsPath(win_path), "Welcome to AWS EC2")
    assert u1 == u3 == u4


def test_uuid5_case_insensitivity():
    """Casing variations normalize to identical UUIDs."""
    u1 = generate_record_uuid("video_transcript", "data/vids/aws.mp4", "WELCOME TO AWS EC2")
    u2 = generate_record_uuid("video_transcript", "data/vids/aws.mp4", "welcome to aws ec2")
    assert u1 == u2


def test_uuid5_qa_pair_delimitation():
    """Q&A pairs with distinct questions produce distinct UUIDs."""
    u1 = generate_record_uuid(
        "social_comment_reply", "item1", "Yes, start with Python.", question="Should I learn Python?"
    )
    u2 = generate_record_uuid(
        "social_comment_reply", "item1", "Yes, start with Python.", question="Should I learn Java?"
    )
    assert u1 != u2


def test_uuid5_multilingual_utf8_stability():
    """Non-ASCII scripts (Urdu, Hindi) produce stable, valid UUIDv5 strings."""
    urdu_text = "آپ کا دن کیسا رہا؟"
    u1 = generate_record_uuid("text_note", "urdu_note.txt", urdu_text)
    u2 = generate_record_uuid("text_note", "urdu_note.txt", urdu_text)
    assert u1 == u2
    assert u1 == "141f9d26-7577-5cc3-ba6e-9d0005c65e2a"
    assert is_valid_uuid(u1, expected_version=5)


def test_uuid5_collision_resistance_bulk():
    """10,000 distinct items produce 10,000 distinct UUIDs."""
    generated = set()
    for i in range(10000):
        u = generate_record_uuid("test_batch", f"file_{i}.txt", f"Body content line variation {i}")
        generated.add(u)
    assert len(generated) == 10000


def test_uuid5_rfc4122_validation():
    """Verifies that generated string is a valid RFC 4122 version 5 UUID."""
    u_str = generate_record_uuid("type", "id", "text content")
    parsed = uuid.UUID(u_str)
    assert parsed.version == 5
    assert str(parsed) == u_str
    assert is_valid_uuid(u_str, expected_version=5)
    assert not is_valid_uuid("not-a-uuid", expected_version=5)
    assert not is_valid_uuid(12345, expected_version=5)


def test_uuid5_custom_namespace_support():
    """Supports custom namespace while maintaining determinism."""
    u1 = generate_record_uuid("type", "id", "text", namespace=UUID_NAMESPACE_SECOND_BRAIN)
    u2 = generate_record_uuid("type", "id", "text", namespace=UUID_NAMESPACE_SECOND_BRAIN)
    assert u1 == u2
    assert is_valid_uuid(u1, expected_version=5)


# ==============================================================================
# Category 2: KnowledgeRecord Schema Validation
# ==============================================================================

def test_knowledge_record_valid_instantiation():
    """Creates a valid KnowledgeRecord with complete submodels."""
    rec_id = generate_record_uuid("video_transcript", "data/vids/ep1.mp4", "AWS Lambda tutorial")
    rec = KnowledgeRecord(
        id=rec_id,
        source=SourceMetadata(type="video_transcript", file="data/vids/ep1.mp4", date="2026-09-24"),
        content=ContentPayload(text="AWS Lambda tutorial", language="english"),
        classification=ClassificationPayload(domain="professional", topics=["aws", "cloud"]),
    )
    assert rec.id == rec_id
    assert rec.source.type == "video_transcript"
    assert rec.content.text == "AWS Lambda tutorial"
    assert "aws" in rec.classification.topics
    assert rec.source.file_name == "ep1.mp4"


def test_knowledge_record_missing_required_fields_fails():
    """Omitting top-level required fields raises ValidationError."""
    with pytest.raises(ValidationError):
        KnowledgeRecord(id="some-id")


def test_knowledge_record_invalid_uuid_rejected():
    """Invalid UUID string fails validation."""
    with pytest.raises(ValidationError):
        KnowledgeRecord(
            id="not-a-valid-uuid",
            source=SourceMetadata(type="text_note", file="note.txt"),
            content=ContentPayload(text="Valid text body"),
            classification=ClassificationPayload(),
        )


def test_knowledge_record_empty_content_text_rejected():
    """Empty content text raises ValidationError."""
    rec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "test"))
    with pytest.raises(ValidationError):
        KnowledgeRecord(
            id=rec_id,
            source=SourceMetadata(type="text_note", file="note.txt"),
            content=ContentPayload(text="   "),
            classification=ClassificationPayload(),
        )


def test_knowledge_record_subscript_access():
    """Backwards compatibility subscripting on submodels."""
    rec_id = generate_record_uuid("text_note", "note.txt", "Some note text")
    rec = KnowledgeRecord(
        id=rec_id,
        source=SourceMetadata(type="text_note", file="note.txt"),
        content=ContentPayload(text="Some note text"),
        classification=ClassificationPayload(),
    )
    assert rec.source["type"] == "text_note"
    assert rec.content["text"] == "Some note text"
    assert rec.classification["domain"] == "professional"
    assert rec["id"] == rec_id


def test_knowledge_record_classification_defaults():
    """Default boolean values in classification."""
    payload = ClassificationPayload()
    assert payload.domain == "professional"
    assert payload.knowledge is True
    assert payload.ai_assisted is False
    assert payload.topics == []


# ==============================================================================
# Category 3: RagRecord Schema Validation & Transformation
# ==============================================================================

def test_rag_record_valid_instantiation():
    """Creates a valid RagRecord."""
    rec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "rag_test"))
    rag_rec = RagRecord(
        id=rec_id,
        text="Searchable content for vector embedding",
        metadata={"source_type": "text_note", "domain": "professional"},
    )
    assert rag_rec.id == rec_id
    assert rag_rec.text == "Searchable content for vector embedding"


def test_rag_record_from_knowledge_record_single_text():
    """Transforms a single-text KnowledgeRecord into a RagRecord."""
    rec_id = generate_record_uuid("video_transcript", "vids/ep1.mp4", "Transcript text line")
    k_rec = KnowledgeRecord(
        id=rec_id,
        source=SourceMetadata(type="video_transcript", file="vids/ep1.mp4"),
        content=ContentPayload(text="Transcript text line", language="english"),
        classification=ClassificationPayload(domain="professional", topics=["cloud"]),
    )
    rag_rec = k_rec.to_rag_record()
    assert rag_rec.id == rec_id
    assert rag_rec.text == "Transcript text line"
    assert rag_rec.metadata["source"]["type"] == "video_transcript"
    assert rag_rec.metadata["classification"]["domain"] == "professional"


def test_rag_record_from_knowledge_record_qa_pair():
    """Transforms a Q&A pair into a RagRecord with combined text."""
    rec_id = generate_record_uuid("social_comment_reply", "c1", "EC2 is a virtual server", question="What is EC2?")
    k_rec = KnowledgeRecord(
        id=rec_id,
        source=SourceMetadata(type="social_comment_reply", file="comments/tiktok.json"),
        content=ContentPayload(text="EC2 is a virtual server", question="What is EC2?", language="english"),
        classification=ClassificationPayload(domain="professional", topics=["aws", "cloud"]),
    )
    rag_rec = k_rec.to_rag_record()
    assert rag_rec.id == rec_id
    assert rag_rec.text == "What is EC2?\nEC2 is a virtual server"
    assert rag_rec.metadata["question"] == "What is EC2?"


def test_rag_record_empty_text_rejected():
    """Empty search text raises ValidationError."""
    rec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "test"))
    with pytest.raises(ValidationError):
        RagRecord(id=rec_id, text="", metadata={})


def test_rag_record_preserves_metadata_lineage():
    """Full lineage is preserved in metadata dictionary."""
    rec_id = generate_record_uuid("markdown_note", "notes/aws.md", "Notes on S3 storage tiers")
    k_rec = KnowledgeRecord(
        id=rec_id,
        source=SourceMetadata(type="markdown_note", file="notes/aws.md", date="2026-09-01"),
        content=ContentPayload(text="Notes on S3 storage tiers", language="english"),
        classification=ClassificationPayload(domain="professional", topics=["aws", "s3"]),
    )
    rag_rec = k_rec.to_rag_record()
    assert rag_rec.metadata["source"]["file"] == "notes/aws.md"
    assert rag_rec.metadata["source"]["date"] == "2026-09-01"


# ==============================================================================
# Category 4: PointPayload & Qdrant Formatting
# ==============================================================================

def test_point_payload_from_rag_record():
    """Converts RagRecord to PointPayload."""
    rec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "point_test"))
    rag_rec = RagRecord(id=rec_id, text="Body text", metadata={"topic": "ai", "domain": "professional"})
    point_payload = PointPayload.from_rag_record(rag_rec)
    assert point_payload.id == rec_id
    assert point_payload.text == "Body text"
    assert point_payload.metadata["topic"] == "ai"
    assert point_payload.domain == "professional"


def test_point_payload_to_qdrant_dict():
    """to_qdrant_dict returns pure JSON-serializable dictionary."""
    rec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "qdrant_test"))
    rag_rec = RagRecord(id=rec_id, text="Qdrant text", metadata={"score": 100})
    payload = rag_rec.to_point_payload().to_qdrant_dict()
    assert isinstance(payload, dict)
    assert payload["id"] == rec_id
    assert payload["text"] == "Qdrant text"
    # Verify JSON serializability
    serialized = json.dumps(payload)
    assert rec_id in serialized


def test_point_payload_id_preservation():
    """UUID format is preserved across transformations."""
    rec_id = generate_record_uuid("type", "id", "content")
    rag = RagRecord(id=rec_id, text="content")
    point = rag.to_point_payload()
    assert point.id == rec_id


def test_point_payload_search_text_preservation():
    """Search text matches between RagRecord and PointPayload."""
    rag = RagRecord(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "x")), text="unique text query")
    point = rag.to_point_payload()
    assert point.text == "unique text query"


# ==============================================================================
# Category 5: JSONL Serialization Roundtrip & Parity
# ==============================================================================

def test_knowledge_record_jsonl_serialization_roundtrip(tmp_path):
    """Serializes KnowledgeRecords to JSONL and deserializes without loss."""
    rec1_id = generate_record_uuid("text_note", "note1.txt", "First record content")
    rec2_id = generate_record_uuid("text_note", "note2.txt", "Second record content")

    rec1 = KnowledgeRecord(
        id=rec1_id,
        source=SourceMetadata(type="text_note", file="note1.txt"),
        content=ContentPayload(text="First record content"),
        classification=ClassificationPayload(),
    )
    rec2 = KnowledgeRecord(
        id=rec2_id,
        source=SourceMetadata(type="text_note", file="note2.txt"),
        content=ContentPayload(text="Second record content"),
        classification=ClassificationPayload(),
    )

    out_file = tmp_path / "records.jsonl"
    count = serialize_jsonl([rec1, rec2], out_file)
    assert count == 2
    assert out_file.exists()

    loaded = deserialize_records(out_file)
    assert len(loaded) == 2
    assert loaded[0].id == rec1_id
    assert loaded[0].content.text == "First record content"
    assert loaded[1].id == rec2_id
    assert loaded[1].content.text == "Second record content"


def test_rag_record_jsonl_serialization_roundtrip(tmp_path):
    """Serializes RagRecords to JSONL and deserializes without loss."""
    r1 = RagRecord(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "1")), text="RAG text 1", metadata={"a": 1})
    r2 = RagRecord(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "2")), text="RAG text 2", metadata={"b": 2})

    out_file = tmp_path / "rag_records.jsonl"
    count = serialize_jsonl([r1, r2], out_file)
    assert count == 2

    loaded = deserialize_rag_records(out_file)
    assert len(loaded) == 2
    assert loaded[0].text == "RAG text 1"
    assert loaded[1].metadata["b"] == 2


def test_jsonl_single_line_invariant(tmp_path):
    """Verifies that multi-line body text is safely escaped onto a single line."""
    multiline_text = "Line 1\nLine 2\nLine 3"
    rec_id = generate_record_uuid("text_note", "multi.txt", multiline_text)
    rec = KnowledgeRecord(
        id=rec_id,
        source=SourceMetadata(type="text_note", file="multi.txt"),
        content=ContentPayload(text=multiline_text),
        classification=ClassificationPayload(),
    )

    out_file = tmp_path / "single_line.jsonl"
    serialize_jsonl([rec], out_file)

    lines = [ln for ln in out_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1, "Record must occupy exactly one line in JSONL file"


def test_cross_schema_id_invariant():
    """Identical ID flows intact from raw input to KnowledgeRecord, RagRecord, and PointPayload."""
    raw_source = "video_transcript"
    raw_ident = "data/vids/lesson.mp4"
    raw_text = "Learn how to build AI Second Brains."
    raw_q = "What is this tutorial about?"

    gen_id = generate_record_uuid(raw_source, raw_ident, raw_text, question=raw_q)

    k_rec = KnowledgeRecord(
        id=gen_id,
        source=SourceMetadata(type=raw_source, file=raw_ident),
        content=ContentPayload(text=raw_text, question=raw_q),
        classification=ClassificationPayload(domain="professional", topics=["ai"]),
    )
    rag_rec = k_rec.to_rag_record()
    point_rec = rag_rec.to_point_payload()

    assert gen_id == k_rec.id == rag_rec.id == point_rec.id


def test_normalize_identifier_edge_cases():
    """Tests normalization of paths, empty, and None identifiers."""
    assert normalize_identifier(None) == "default"
    assert normalize_identifier("") == "default"
    assert normalize_identifier("  ") == "default"
    assert normalize_identifier(r"C:\Users\Name\file.txt") == "c:/users/name/file.txt"
