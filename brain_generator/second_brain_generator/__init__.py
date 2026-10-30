"""
Second Brain Generator (`second_brain_generator`).

A modular, standalone Python toolkit that enables anyone to create their own AI
Second Brain from scratch using raw media, documents, and notes.
"""

from .config import (
    BrainConfig,
    ConfigError,
    load_config,
    scaffold_directories,
    DEFAULT_PATHS,
    DEFAULT_TOPICS,
    DEFAULT_TONE_GUIDELINES,
)
from .schemas import (
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
from .utils.dirs import (
    initialize_directories,
    validate_directories,
    scan_category_files,
    DirectoryValidationReport,
    DATA_SUBDIRS,
    BRAIN_DATA_SUBDIRS,
    SUPPORTED_EXTENSIONS,
    OUTPUT_FILENAMES,
)
from .utils.uuids import (
    generate_record_uuid,
    compute_content_hash,
    normalize_text,
    normalize_identifier,
    is_valid_uuid,
    DEFAULT_NAMESPACE,
    UUID_NAMESPACE_SECOND_BRAIN,
)
from .utils.env import (
    load_environment,
    get_gemini_keys,
    get_qdrant_credentials,
    mask_secret,
    is_offline_mode,
    MissingSecretError,
    QdrantCredentials,
)

from .transcriber import MediaTranscriber
from .compiler import KnowledgeCompiler
from .embedder import GeminiEmbedder
from .evaluator import BrainEvaluator
from .uploader import QdrantUploader

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BrainConfig",
    "ConfigError",
    "load_config",
    "scaffold_directories",
    "DEFAULT_PATHS",
    "DEFAULT_TOPICS",
    "DEFAULT_TONE_GUIDELINES",
    "KnowledgeRecord",
    "RagRecord",
    "PointPayload",
    "SourceMetadata",
    "ContentPayload",
    "ClassificationPayload",
    "serialize_jsonl",
    "deserialize_records",
    "deserialize_rag_records",
    "initialize_directories",
    "validate_directories",
    "scan_category_files",
    "DirectoryValidationReport",
    "DATA_SUBDIRS",
    "BRAIN_DATA_SUBDIRS",
    "SUPPORTED_EXTENSIONS",
    "OUTPUT_FILENAMES",
    "generate_record_uuid",
    "compute_content_hash",
    "normalize_text",
    "normalize_identifier",
    "is_valid_uuid",
    "DEFAULT_NAMESPACE",
    "UUID_NAMESPACE_SECOND_BRAIN",
    "load_environment",
    "get_gemini_keys",
    "get_qdrant_credentials",
    "mask_secret",
    "is_offline_mode",
    "MissingSecretError",
    "QdrantCredentials",
    "MediaTranscriber",
    "KnowledgeCompiler",
    "GeminiEmbedder",
    "BrainEvaluator",
    "QdrantUploader",
]
