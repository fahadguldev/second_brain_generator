"""Utility modules for second_brain_generator."""

from .dirs import (
    initialize_directories,
    validate_directories,
    scan_category_files,
    DirectoryValidationReport,
    DATA_SUBDIRS,
    BRAIN_DATA_SUBDIRS,
    SUPPORTED_EXTENSIONS,
    OUTPUT_FILENAMES,
)
from .uuids import (
    generate_record_uuid,
    compute_content_hash,
    normalize_text,
    normalize_identifier,
    is_valid_uuid,
    DEFAULT_NAMESPACE,
    UUID_NAMESPACE_SECOND_BRAIN,
)
from .env import (
    load_environment,
    get_gemini_keys,
    get_qdrant_credentials,
    mask_secret,
    is_offline_mode,
    MissingSecretError,
    QdrantCredentials,
)

__all__ = [
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
]
