"""Directory hierarchy and file discovery utilities for second_brain_generator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Union

# Canonical subdirectories under data/ (R1)
DATA_SUBDIRS: Dict[str, str] = {
    "vids": "vids",
    "audios": "audios",
    "text": "text",
    "md": "md",
    "comments": "comments",
}

# Canonical subdirectories under brain_data/ (R1)
BRAIN_DATA_SUBDIRS: Dict[str, str] = {
    "text": "text",
}

# Permissible file extensions per input category (stored lowercase)
SUPPORTED_EXTENSIONS: Dict[str, Set[str]] = {
    "vids": {".mp4", ".mov", ".mkv"},
    "audios": {".mp3", ".wav", ".m4a"},
    "text": {".txt"},
    "md": {".md"},
    "comments": {".json", ".csv"},
}

# Canonical output artifact filenames under brain_data/
OUTPUT_FILENAMES: Dict[str, str] = {
    "records": "records.jsonl",
    "rag_records": "rag_records.jsonl",
    "embeddings": "gemini_embeddings.npz",
    "checkpoint": "gemini_embeddings_checkpoint.npz",
}


class DirectoryValidationReport(NamedTuple):
    is_valid: bool
    existing_dirs: List[Path]
    missing_dirs: List[Path]
    unwritable_dirs: List[Path]
    conflicts: List[Path]


def initialize_directories(
    base_dir: Optional[Union[Path, str]] = None,
    data_dirname: str = "data",
    brain_data_dirname: str = "brain_data",
    create_gitkeep: bool = True,
) -> Dict[str, Path]:
    """
    Idempotently initialize all required input and output directories.

    Args:
        base_dir: Project root directory (defaults to Path.cwd()).
        data_dirname: Relative path of raw data directory (default 'data').
        brain_data_dirname: Relative path of output directory (default 'brain_data').
        create_gitkeep: If True, creates .gitkeep in empty directories.

    Returns:
        Dictionary mapping path keys to resolved Path objects.

    Raises:
        NotADirectoryError: If an expected directory path exists as a regular file.
        PermissionError: If brain_data or subdirectories are not writable.
    """
    root = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()

    data_root = (root / data_dirname).resolve() if not Path(data_dirname).is_absolute() else Path(data_dirname).resolve()
    brain_data_root = (root / brain_data_dirname).resolve() if not Path(brain_data_dirname).is_absolute() else Path(brain_data_dirname).resolve()

    paths: Dict[str, Path] = {
        "root": root,
        "data": data_root,
        "brain_data": brain_data_root,
    }

    # 1. Build input data directories
    for key, sub in DATA_SUBDIRS.items():
        p = data_root / sub
        paths[f"data_{key}"] = p
        paths[key] = p  # Short alias

    # 2. Build output directories
    for key, sub in BRAIN_DATA_SUBDIRS.items():
        p = brain_data_root / sub
        paths[f"brain_data_{key}"] = p
        if key == "text":
            paths["transcripts"] = p

    # 3. Create directories idempotently and verify types
    for key, path in paths.items():
        if key == "root":
            continue
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"Target path exists but is not a directory: {path}")

        path.mkdir(parents=True, exist_ok=True)

        # Verify write permission on brain_data directories if parent exists
        if "brain_data" in key or key == "transcripts":
            if not os.access(path, os.W_OK):
                raise PermissionError(f"Output directory is not writable: {path}")

        # Create .gitkeep if requested
        if create_gitkeep:
            gitkeep_file = path / ".gitkeep"
            existing_files = [f for f in path.iterdir() if f.name != ".gitkeep"]
            if not existing_files and not gitkeep_file.exists():
                gitkeep_file.touch(exist_ok=True)

    return paths


def validate_directories(
    base_dir: Optional[Union[Path, str]] = None,
    data_dirname: str = "data",
    brain_data_dirname: str = "brain_data",
) -> DirectoryValidationReport:
    """
    Non-destructive health-check on the required directory structure.

    Args:
        base_dir: Base workspace path.
        data_dirname: Relative or absolute data root name.
        brain_data_dirname: Relative or absolute brain_data root name.

    Returns:
        DirectoryValidationReport with is_valid, existing, missing, unwritable, and conflict paths.
    """
    root = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    data_root = (root / data_dirname).resolve() if not Path(data_dirname).is_absolute() else Path(data_dirname).resolve()
    brain_data_root = (root / brain_data_dirname).resolve() if not Path(brain_data_dirname).is_absolute() else Path(brain_data_dirname).resolve()

    expected = [
        data_root / "vids",
        data_root / "audios",
        data_root / "text",
        data_root / "md",
        data_root / "comments",
        brain_data_root,
        brain_data_root / "text",
    ]

    existing: List[Path] = []
    missing: List[Path] = []
    unwritable: List[Path] = []
    conflicts: List[Path] = []

    for p in expected:
        try:
            exists = p.exists()
        except (PermissionError, OSError):
            if p not in unwritable:
                unwritable.append(p)
            continue

        if not exists:
            missing.append(p)
        else:
            try:
                is_dir = p.is_dir()
            except (PermissionError, OSError):
                if p not in unwritable:
                    unwritable.append(p)
                continue

            if not is_dir:
                conflicts.append(p)
            else:
                existing.append(p)
                # All expected directories require read access
                if not os.access(p, os.R_OK):
                    if p not in unwritable:
                        unwritable.append(p)
                # Output directories (brain_data_root and subdirectories) require write access
                is_output_dir = (p == brain_data_root) or p.is_relative_to(brain_data_root)
                if is_output_dir and not os.access(p, os.W_OK):
                    if p not in unwritable:
                        unwritable.append(p)

    is_valid = len(missing) == 0 and len(conflicts) == 0 and len(unwritable) == 0
    return DirectoryValidationReport(
        is_valid=is_valid,
        existing_dirs=existing,
        missing_dirs=missing,
        unwritable_dirs=unwritable,
        conflicts=conflicts,
    )


def scan_category_files(
    category: str,
    base_dir: Optional[Union[Path, str]] = None,
    data_dirname: str = "data",
    recursive: bool = True,
) -> List[Path]:
    """
    Discovers files for an input category using case-insensitive extension matching.
    Results are returned deterministically sorted by POSIX path.

    Args:
        category: Category name ('vids', 'audios', 'text', 'md', 'comments').
        base_dir: Base workspace path.
        data_dirname: Data directory path or name.
        recursive: Whether to scan recursively inside category directory.

    Returns:
        Sorted list of matching Path objects.
    """
    if category not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unknown category '{category}'. Permitted: {list(SUPPORTED_EXTENSIONS.keys())}")

    root = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    data_root = (root / data_dirname).resolve() if not Path(data_dirname).is_absolute() else Path(data_dirname).resolve()
    target_dir = data_root / category

    if not target_dir.exists() or not target_dir.is_dir():
        return []

    valid_exts = SUPPORTED_EXTENSIONS[category]
    iterator = target_dir.rglob("*") if recursive else target_dir.glob("*")

    matched: List[Path] = []
    for p in iterator:
        if p.is_file() and not p.name.startswith("."):
            if p.suffix.lower() in valid_exts:
                matched.append(p)

    return sorted(matched, key=lambda p: p.as_posix())
