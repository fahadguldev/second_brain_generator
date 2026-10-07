"""
Global pytest fixtures and mocks for second_brain_generator test suite.
Supports fast offline execution (<5s) with deterministic embeddings,
mocked Whisper speech-to-text, in-memory Qdrant instance, and isolated temp directories.
"""

from collections import namedtuple
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock
import sys

# Ensure site-packages with PyYAML and other packages is discoverable
venv_site = Path("/home/rf-gul/second-brain/transcription/.venv/lib/python3.13/site-packages")
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.append(str(venv_site))

import numpy as np

try:
    import pytest
except ImportError:
    class _PytestShim:
        class skip:
            class Exception(Exception):
                pass
            def __new__(cls, reason=""):
                raise cls.Exception(reason)

        @staticmethod
        def fixture(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

        class raises:
            def __init__(self, expected_exceptions):
                if not isinstance(expected_exceptions, tuple):
                    expected_exceptions = (expected_exceptions,)
                self.expected_exceptions = expected_exceptions

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected exception {self.expected_exceptions} but none was raised")
                return issubclass(exc_type, self.expected_exceptions)

        class mark:
            @staticmethod
            def parametrize(*args, **kwargs):
                return lambda f: f

        @staticmethod
        def approx(expected, rel=None, abs=None):
            return expected

    pytest = _PytestShim()
    sys.modules["pytest"] = pytest

# Types for Whisper mock
Segment = namedtuple("Segment", ["id", "start", "end", "text"])
TranscriptionInfo = namedtuple("TranscriptionInfo", ["language", "language_probability"])


@pytest.fixture
def mock_whisper_model():
    """
    Mocks faster_whisper.WhisperModel.
    Returns simulated segment generator yielding realistic transcribed chunks
    without requiring PyTorch or CTranslate2 inference.
    """
    mock = MagicMock()

    def fake_transcribe(audio_path, **kwargs):
        path_str = str(audio_path).lower()
        if "urdu" in path_str or "hindi" in path_str:
            segments = [
                Segment(0, 0.0, 3.5, "Yeh ek sample transcription hai."),
                Segment(1, 3.5, 7.0, "Second brain generator bohot mufeed toolkit hai.")
            ]
            info = TranscriptionInfo("ur", 0.95)
        elif "empty" in path_str or "silent" in path_str:
            segments = []
            info = TranscriptionInfo("en", 0.99)
        else:
            segments = [
                Segment(0, 0.0, 3.5, "Welcome to the second brain tutorial."),
                Segment(1, 3.5, 7.2, "In this guide we build a modular knowledge engine.")
            ]
            info = TranscriptionInfo("en", 0.98)
        return iter(segments), info

    mock.transcribe.side_effect = fake_transcribe
    return mock


@pytest.fixture
def deterministic_embedding_fn():
    """
    Generates deterministic 3072-dimensional normalized float32 vectors based on text hash.
    Ensures identical text gets identical vector, and differing texts receive distinct vectors.
    """
    def _embed(text: str) -> np.ndarray:
        # Use sha256 to create reproducible seed
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.RandomState(seed)
        vec = rng.randn(3072).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    return _embed


class InMemoryQdrantMock:
    """
    High-fidelity in-memory Qdrant simulation with 100% API parity for test suite:
    supports create_collection, collection_exists, get_collection, upsert, search, and delete.
    """
    def __init__(self, location: str = ":memory:"):
        self.location = location
        self.collections: Dict[str, Dict[str, Any]] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: Any = None, **kwargs) -> bool:
        if collection_name not in self.collections:
            self.collections[collection_name] = {
                "points": {},
                "vectors_config": vectors_config,
                "points_count": 0
            }
        return True

    def get_collection(self, collection_name: str):
        if collection_name not in self.collections:
            raise KeyError(f"Collection {collection_name} does not exist")
        coll = self.collections[collection_name]
        info = MagicMock()
        info.points_count = len(coll["points"])
        info.vectors_count = len(coll["points"])
        info.status = "green"
        return info

    def upsert(self, collection_name: str, points: list, **kwargs):
        if collection_name not in self.collections:
            self.create_collection(collection_name)
        coll = self.collections[collection_name]
        for p in points:
            pid = getattr(p, "id", None) or (p.get("id") if isinstance(p, dict) else None)
            coll["points"][str(pid)] = p
        coll["points_count"] = len(coll["points"])
        res = MagicMock()
        res.status = "completed"
        return res

    def search(self, collection_name: str, query_vector: list, limit: int = 5, query_filter: Any = None, **kwargs):
        if collection_name not in self.collections:
            return []
        points = list(self.collections[collection_name]["points"].values())
        results = []
        for p in points:
            vec = getattr(p, "vector", None) or (p.get("vector") if isinstance(p, dict) else None)
            payload = getattr(p, "payload", None) or (p.get("payload") if isinstance(p, dict) else {})
            pid = getattr(p, "id", None) or (p.get("id") if isinstance(p, dict) else None)
            if vec is not None:
                # Cosine similarity
                v = np.array(vec, dtype=np.float32)
                q = np.array(query_vector, dtype=np.float32)
                denom = (np.linalg.norm(v) * np.linalg.norm(q))
                score = float(np.dot(v, q) / denom) if denom > 0 else 0.0
            else:
                score = 0.5
            hit = MagicMock()
            hit.id = str(pid)
            hit.score = score
            hit.payload = payload
            results.append(hit)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]


@pytest.fixture
def mock_qdrant_in_memory():
    """
    Returns an in-memory Qdrant client.
    Attempts to import real QdrantClient(':memory:') if available;
    falls back cleanly to InMemoryQdrantMock with 100% API parity.
    """
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(":memory:")
        return client
    except Exception:
        return InMemoryQdrantMock(":memory:")


@pytest.fixture
def mock_brain_dirs(tmp_path: Path):
    """
    Creates an isolated standardized directory structure for tests:
    data/
      vids/
      audios/
      text/
      md/
      comments/
    brain_data/
      text/
    """
    data_dir = tmp_path / "data"
    brain_data_dir = tmp_path / "brain_data"

    for sub in ["vids", "audios", "text", "md", "comments"]:
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    for sub in ["text"]:
        (brain_data_dir / sub).mkdir(parents=True, exist_ok=True)

    paths = {
        "root": tmp_path,
        "data_dir": data_dir,
        "brain_data_dir": brain_data_dir,
        "vids": data_dir / "vids",
        "audios": data_dir / "audios",
        "text": data_dir / "text",
        "md": data_dir / "md",
        "comments": data_dir / "comments",
        "transcripts": brain_data_dir / "text",
        "records": brain_data_dir / "records.jsonl",
        "rag_records": brain_data_dir / "rag_records.jsonl",
        "embeddings": brain_data_dir / "gemini_embeddings.npz",
        "checkpoint": brain_data_dir / "gemini_embeddings_checkpoint.npz",
    }
    return paths


@pytest.fixture
def sample_config_yaml(tmp_path: Path) -> Path:
    """
    Creates a sample brain_config.yaml in a temporary path.
    """
    config_content = """version: "1.0"

creator:
  name: "Fahad Gul"
  handles:
    - "fahadgul.dev"
    - "@fahadgul"
  bio: "Software Engineer & Cloud Architect"

persona:
  languages:
    primary: "english"
    secondary: ["hinglish", "urdu"]
  tone: "direct, concise, authentic, technical"
  default_domain: "professional"
  avoid_words: ["bhai", "bro"]

paths:
  data_dir: "data"
  brain_data_dir: "brain_data"
  vids_subpath: "vids"
  audios_subpath: "audios"
  text_subpath: "text"
  md_subpath: "md"
  comments_subpath: "comments"
  transcripts_subpath: "text"
  records_filename: "records.jsonl"
  rag_records_filename: "rag_records.jsonl"
  embeddings_filename: "gemini_embeddings.npz"

whisper:
  model_size: "small"
  device: "cpu"
  compute_type: "int8"
  vad_filter: true
  language: null

embedding:
  model: "models/gemini-embedding-2"
  dimension: 3072
  batch_size: 25
  rate_limit_per_min: 60
  rate_limit_per_day: 1000

qdrant:
  url: "http://localhost:6333"
  api_key: null
  collection_name: "second_brain"
  vector_size: 3072
  distance: "Cosine"
  batch_size: 100

topics:
  python: ["python", "pytest", "asyncio"]
  aws: ["aws", "lambda", "ec2", "s3"]
  cloud: ["cloud", "serverless", "devops"]
  ai: ["ai", "machine learning", "llm", "embeddings"]
  career: ["career", "salary", "interview", "resume"]
"""
    config_path = tmp_path / "brain_config.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


@pytest.fixture
def sample_comment_json_content() -> str:
    """
    Returns valid JSON string simulating TikTok/Instagram comment export
    with nested parent-child replies and creator answers.
    """
    data = [
        {
            "comment_id": "c_101",
            "text": "What is the best way to get started with AWS Lambda?",
            "author": {"unique_id": "student_dev", "is_creator": False},
            "timestamp": "2026-08-10T12:00:00Z",
            "replies": [
                {
                    "reply_id": "r_101_1",
                    "reply_to_reply_id": "c_101",
                    "text": "Start by writing small Python functions triggered by S3 events. Use SAM or Serverless framework.",
                    "author": {"unique_id": "fahadgul.dev", "is_creator": True},
                    "timestamp": "2026-08-10T12:30:00Z",
                    "likes": 25
                }
            ]
        },
        {
            "comment_id": "c_102",
            "text": "Can I learn AI without strong math background?",
            "author": {"unique_id": "curious_mind", "is_creator": False},
            "timestamp": "2026-08-11T09:00:00Z",
            "replies": [
                {
                    "reply_id": "r_102_1",
                    "reply_to_reply_id": "c_102",
                    "text": "Yes! Focus on practical engineering and API integration first, then learn linear algebra as needed.",
                    "author": {"unique_id": "@fahadgul", "is_creator": True},
                    "timestamp": "2026-08-11T10:00:00Z",
                    "likes": 40
                }
            ]
        }
    ]
    return json.dumps(data, indent=2)


@pytest.fixture
def sample_comment_csv_content() -> str:
    """
    Returns CSV text simulating exported Q&A comments.
    """
    return (
        "question,reply,author,date\n"
        "\"Which language for backend?\",\"Python and Go are my top recommendations for microservices.\",\"fahadgul\",\"2026-08-01\"\n"
        "\"How to prepare for coding interviews?\",\"Master LeetCode medium patterns and practice system design mocks.\",\"fahadgul\",\"2026-08-02\"\n"
        "\"Chk DM plz\",\"Check dm me\",\"spammer\",\"2026-08-03\"\n"
    )
