"""Configuration engine for second_brain_generator."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .utils.dirs import initialize_directories

logger = logging.getLogger(__name__)

# Standard default paths conforming to ORIGINAL_REQUEST §R1
DEFAULT_PATHS: Dict[str, str] = {
    "data_dir": "data",
    "brain_data_dir": "brain_data",
    "vids_dir": "data/vids",
    "audios_dir": "data/audios",
    "text_dir": "data/text",
    "md_dir": "data/md",
    "comments_dir": "data/comments",
    "transcripts_dir": "brain_data/text",
    "records_path": "brain_data/records.jsonl",
    "rag_records_path": "brain_data/rag_records.jsonl",
    "embeddings_path": "brain_data/gemini_embeddings.npz",
}

DEFAULT_TOPICS: Dict[str, List[str]] = {
    "python": ["python", "fastapi", "pytest", "asyncio"],
    "cloud": ["cloud", "aws", "serverless", "ec2", "s3", "lambda"],
    "cybersecurity": ["cyber", "security", "hacking", "ethical"],
    "ai": ["artificial intelligence", "machine learning", "ml", "llm", "rag", "embeddings"],
    "datascience": ["data science", "datascience", "analytics", "pandas", "numpy"],
    "webdev": ["web dev", "webdev", "frontend", "backend", "html", "css", "react", "nextjs"],
    "devops": ["devops", "docker", "kubernetes", "ci/cd", "container"],
    "database": ["database", "sql", "mysql", "mongodb", "postgres", "qdrant"],
    "career": ["career", "salary", "job", "jobs", "scope", "market", "hiring"],
    "education": ["degree", "university", "semester", "cgpa", "gpa", "admission"],
    "software-engineering": ["software engineering", "software dev", "system design"],
}

DEFAULT_TONE_GUIDELINES: List[str] = [
    "Short and punchy; 1-2 sentences default",
    "English base with natural Romanized Hinglish code-switching",
    "Direct verdict first, followed by minimal needed explanation",
    "Authentic personal knowledge and opinions over generic AI answers",
    "Factual grounding strictly overrides communication style",
]


class ConfigError(Exception):
    """Exception raised for configuration parsing or validation failures."""
    pass


def _safe_is_file(path: Union[str, Path, None]) -> bool:
    """Safely check if path is an existing regular file without raising OSError or ValueError."""
    if path is None:
        return False
    try:
        p = path if isinstance(path, Path) else Path(path)
        return p.is_file()
    except (OSError, ValueError):
        return False


safe_is_file = _safe_is_file


class WhisperConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    vad_filter: bool = True
    language: Optional[str] = None


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "models/gemini-embedding-2"
    dimension: int = 3072
    batch_size: int = 25
    rate_limit_per_min: int = 60
    rate_limit_per_day: int = 1000


class QdrantConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "second_brain"
    vector_size: int = 3072
    distance: str = "Cosine"
    batch_size: int = 100


class CreatorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "Second Brain Creator"
    handles: List[str] = Field(default_factory=lambda: ["@secondbrain"])
    bio: Optional[str] = None


class PersonaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    languages: Any = Field(default_factory=lambda: ["english", "hinglish"])
    tone: Optional[Union[str, List[str]]] = None
    tone_guidelines: List[str] = Field(default_factory=list)
    default_domain: str = "professional"
    avoid_words: List[str] = Field(default_factory=list)


class BrainConfig(BaseModel):
    """Authoritative BrainConfig class adhering to PROJECT.md §1 and R6."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    version: str = "1.0"
    creator_name: str = "Second Brain Creator"
    creator_handles: List[str] = Field(default_factory=lambda: ["@secondbrain"])
    languages: List[str] = Field(default_factory=lambda: ["english", "hinglish"])
    tone_guidelines: List[str] = Field(default_factory=lambda: list(DEFAULT_TONE_GUIDELINES))
    topics: Dict[str, List[str]] = Field(default_factory=lambda: dict(DEFAULT_TOPICS))
    paths: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_PATHS))
    bio: Optional[str] = None

    # Subsystem and nested models
    creator: CreatorConfig = Field(default_factory=CreatorConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if isinstance(data, BrainConfig):
            return data
        if not isinstance(data, dict):
            raise ConfigError(
                f"YAML must contain a top-level mapping/dictionary, got {type(data).__name__}"
            )

        d = dict(data)

        # 1. Normalize nested 'creator' section
        creator_data = d.get("creator")
        if isinstance(creator_data, dict):
            if "creator_name" not in d and "name" in creator_data:
                d["creator_name"] = str(creator_data["name"])
            if "creator_handles" not in d and "handles" in creator_data:
                d["creator_handles"] = creator_data["handles"]
            if "bio" not in d and "bio" in creator_data:
                d["bio"] = str(creator_data["bio"])

        # 2. Normalize nested 'persona' section
        persona_data = d.get("persona")
        if isinstance(persona_data, dict):
            if "creator_name" not in d and "name" in persona_data:
                d["creator_name"] = str(persona_data["name"])
            if "languages" not in d and "languages" in persona_data:
                lang_val = persona_data["languages"]
                if isinstance(lang_val, dict):
                    # e.g., {'primary': 'english', 'secondary': ['hinglish', 'urdu']}
                    collected: List[str] = []
                    prim = lang_val.get("primary")
                    if prim:
                        collected.append(str(prim))
                    sec = lang_val.get("secondary", [])
                    if isinstance(sec, list):
                        collected.extend(str(s) for s in sec)
                    elif isinstance(sec, str):
                        collected.append(sec)
                    d["languages"] = collected
                elif isinstance(lang_val, (list, tuple)):
                    d["languages"] = list(lang_val)
                elif isinstance(lang_val, str):
                    d["languages"] = [lang_val]

            if "tone_guidelines" not in d:
                if "tone_guidelines" in persona_data:
                    d["tone_guidelines"] = persona_data["tone_guidelines"]
                elif "tone" in persona_data:
                    tone_val = persona_data["tone"]
                    if isinstance(tone_val, str):
                        d["tone_guidelines"] = [t.strip() for t in tone_val.split(",") if t.strip()]
                    elif isinstance(tone_val, (list, tuple)):
                        d["tone_guidelines"] = list(tone_val)

        # 3. Type coercions for list fields
        if "creator_handles" in d and isinstance(d["creator_handles"], str):
            d["creator_handles"] = [d["creator_handles"]]

        if "languages" in d and isinstance(d["languages"], str):
            d["languages"] = [d["languages"]]

        if "tone_guidelines" in d and isinstance(d["tone_guidelines"], str):
            d["tone_guidelines"] = [d["tone_guidelines"]]

        # 4. Sync creator and persona submodel data
        name = d.get("creator_name", "Second Brain Creator")
        handles = d.get("creator_handles", ["@secondbrain"])
        bio = d.get("bio")

        if not isinstance(creator_data, dict):
            d["creator"] = {"name": name, "handles": handles, "bio": bio}
        else:
            c_dict = dict(creator_data)
            if "name" not in c_dict:
                c_dict["name"] = name
            if "handles" not in c_dict:
                c_dict["handles"] = handles
            if "bio" not in c_dict and bio:
                c_dict["bio"] = bio
            d["creator"] = c_dict

        if not isinstance(persona_data, dict):
            d["persona"] = {
                "name": name,
                "languages": d.get("languages", ["english", "hinglish"]),
                "tone_guidelines": d.get("tone_guidelines", list(DEFAULT_TONE_GUIDELINES)),
            }
        else:
            p_dict = dict(persona_data)
            if "name" not in p_dict:
                p_dict["name"] = name
            d["persona"] = p_dict

        # 5. Paths normalization and merging
        paths_input = d.get("paths")
        if paths_input is not None and not isinstance(paths_input, dict):
            raise ConfigError(f"Configuration 'paths' must be a dictionary, got {type(paths_input).__name__}")

        merged_paths = dict(DEFAULT_PATHS)
        if isinstance(paths_input, dict):
            merged_paths.update(paths_input)

            # Support subpath keys from older or alternate configs
            data_dir = merged_paths.get("data_dir", "data")
            brain_dir = merged_paths.get("brain_data_dir", "brain_data")
            subpath_map = {
                "vids_subpath": ("vids_dir", data_dir),
                "audios_subpath": ("audios_dir", data_dir),
                "text_subpath": ("text_dir", data_dir),
                "md_subpath": ("md_dir", data_dir),
                "comments_subpath": ("comments_dir", data_dir),
                "transcripts_subpath": ("transcripts_dir", brain_dir),
            }
            for sub_key, (target_key, prefix) in subpath_map.items():
                if sub_key in paths_input and target_key not in paths_input:
                    merged_paths[target_key] = f"{prefix}/{paths_input[sub_key]}"

            if "records_filename" in paths_input and "records_path" not in paths_input:
                merged_paths["records_path"] = f"{brain_dir}/{paths_input['records_filename']}"
            if "rag_records_filename" in paths_input and "rag_records_path" not in paths_input:
                merged_paths["rag_records_path"] = f"{brain_dir}/{paths_input['rag_records_filename']}"
            if "embeddings_filename" in paths_input and "embeddings_path" not in paths_input:
                merged_paths["embeddings_path"] = f"{brain_dir}/{paths_input['embeddings_filename']}"

        d["paths"] = merged_paths

        # 6. Topics normalization and validation
        topics_input = d.get("topics")
        if topics_input is not None and not isinstance(topics_input, dict):
            raise ConfigError(f"Configuration 'topics' must be a dictionary, got {type(topics_input).__name__}")

        if not topics_input:
            d["topics"] = dict(DEFAULT_TOPICS)
        else:
            # Merge with default topics so standard categories are not lost
            merged_topics = dict(DEFAULT_TOPICS)
            merged_topics.update(topics_input)
            d["topics"] = merged_topics

        return d

    def get_path(self, key: str, base_dir: Optional[Union[Path, str]] = None) -> Path:
        """
        Resolves a configured path relative to base_dir or project root.
        If the resolved path is already absolute, it is returned directly.
        """
        raw_path = self.paths.get(key, DEFAULT_PATHS.get(key, key))
        path_obj = Path(raw_path)
        if path_obj.is_absolute():
            return path_obj
        base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        return (base / path_obj).resolve()

    @classmethod
    def from_yaml(
        cls,
        yaml_source: Union[str, Path],
        data_dir_override: Optional[Union[str, Path]] = None,
    ) -> BrainConfig:
        """
        Load and validate BrainConfig from a YAML file path or raw YAML string.

        Args:
            yaml_source: Path to a YAML file (str or Path) or raw YAML string content.
            data_dir_override: Optional CLI override for the root data directory.

        Returns:
            BrainConfig: Validated configuration object.

        Raises:
            ConfigError: If YAML syntax is invalid or root element is not a mapping/dictionary.
            FileNotFoundError: If a file path is provided but does not exist.
        """
        raw_data: Any = None
        candidate_path: Optional[Path] = None

        if isinstance(yaml_source, Path):
            candidate_path = yaml_source
            if not _safe_is_file(candidate_path):
                raise FileNotFoundError(f"Configuration file not found: {candidate_path}")
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigError(f"YAML syntax error in configuration file '{candidate_path}': {e}") from e
            except Exception as e:
                raise ConfigError(f"Failed to read configuration file '{candidate_path}': {e}") from e
        elif isinstance(yaml_source, str):
            if "\n" not in yaml_source:
                stripped = yaml_source.strip()
                has_mapping_syntax = bool(re.search(r":(\s|$)", yaml_source)) or (
                    stripped.startswith("{") and stripped.endswith("}")
                )

                if not has_mapping_syntax and _safe_is_file(yaml_source):
                    candidate_path = Path(yaml_source)
                    try:
                        with open(candidate_path, "r", encoding="utf-8") as f:
                            raw_data = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        raise ConfigError(f"YAML syntax error in configuration file '{candidate_path}': {e}") from e
                    except Exception as e:
                        raise ConfigError(f"Failed to read configuration file '{candidate_path}': {e}") from e
                else:
                    yaml_parsed = None
                    parse_error = None
                    try:
                        yaml_parsed = yaml.safe_load(yaml_source)
                    except yaml.YAMLError as e:
                        parse_error = e
                    except Exception as e:
                        parse_error = e

                    if isinstance(yaml_parsed, dict):
                        raw_data = yaml_parsed
                    elif not has_mapping_syntax and (
                        yaml_source.endswith((".yaml", ".yml")) or "/" in yaml_source or "\\" in yaml_source
                    ):
                        raise FileNotFoundError(f"Configuration file not found: {yaml_source}")
                    else:
                        if parse_error is not None:
                            raise ConfigError(f"YAML syntax error in configuration string: {parse_error}") from parse_error
                        raw_data = yaml_parsed
            else:
                try:
                    raw_data = yaml.safe_load(yaml_source)
                except yaml.YAMLError as e:
                    raise ConfigError(f"YAML syntax error in configuration string: {e}") from e
                except Exception as e:
                    raise ConfigError(f"Failed to parse YAML: {e}") from e
        else:
            raise ConfigError(f"Unsupported YAML source type: {type(yaml_source).__name__}")

        if raw_data is None:
            raw_data = {}
        elif not isinstance(raw_data, dict):
            raise ConfigError(
                f"YAML must contain a top-level mapping/dictionary, got {type(raw_data).__name__}"
            )

        try:
            config = cls.model_validate(raw_data)
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"Configuration validation failed: {e}") from e

        if data_dir_override:
            data_dir_str = str(data_dir_override)
            config.paths["data_dir"] = data_dir_str
            config.paths["vids_dir"] = str(Path(data_dir_str) / "vids")
            config.paths["audios_dir"] = str(Path(data_dir_str) / "audios")
            config.paths["text_dir"] = str(Path(data_dir_str) / "text")
            config.paths["md_dir"] = str(Path(data_dir_str) / "md")
            config.paths["comments_dir"] = str(Path(data_dir_str) / "comments")

        return config


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    data_dir_override: Optional[Union[str, Path]] = None,
    raise_if_missing: bool = False,
) -> BrainConfig:
    """
    Load configuration from a YAML file with robust fallback defaults.

    Args:
        config_path: Path to YAML file. If None, checks BRAIN_CONFIG_PATH or default locations.
        data_dir_override: Optional CLI override for the root data directory.
        raise_if_missing: If True, raises FileNotFoundError when explicit config_path doesn't exist.

    Returns:
        BrainConfig: Fully validated configuration object with fallbacks applied.

    Raises:
        FileNotFoundError: If raise_if_missing is True and config file cannot be found.
        ConfigError: If YAML file has syntax errors or schema validation fails.
    """
    candidate_path: Optional[Path] = None

    if config_path:
        candidate_path = Path(config_path)
    elif (env_cfg := os.getenv("BRAIN_CONFIG_PATH")):
        candidate_path = Path(env_cfg)
    else:
        # Standard search locations
        search_locations = [
            Path("brain_config.yaml"),
            Path.cwd() / "brain_config.yaml",
            Path(__file__).resolve().parent.parent / "brain_config.yaml",
        ]
        for loc in search_locations:
            if _safe_is_file(loc):
                candidate_path = loc
                break

    if candidate_path and _safe_is_file(candidate_path):
        try:
            with open(candidate_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
            logger.info("Loaded configuration from %s", candidate_path)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML syntax error in configuration file '{candidate_path}': {e}") from e
        except Exception as e:
            raise ConfigError(f"Failed to read configuration file '{candidate_path}': {e}") from e

        if raw_data is None:
            raw_data = {}
        elif not isinstance(raw_data, dict):
            raise ConfigError(
                f"YAML must contain a top-level mapping/dictionary, got {type(raw_data).__name__}"
            )
    else:
        if candidate_path and raise_if_missing:
            raise FileNotFoundError(f"Configuration file not found: {candidate_path}")
        logger.debug(
            "Configuration file not found or not specified (checked %s). Using default BrainConfig.",
            candidate_path,
        )
        raw_data = {}

    try:
        config = BrainConfig.model_validate(raw_data)
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e

    # Apply data_dir override from CLI if specified
    if data_dir_override:
        data_dir_str = str(data_dir_override)
        config.paths["data_dir"] = data_dir_str
        config.paths["vids_dir"] = str(Path(data_dir_str) / "vids")
        config.paths["audios_dir"] = str(Path(data_dir_str) / "audios")
        config.paths["text_dir"] = str(Path(data_dir_str) / "text")
        config.paths["md_dir"] = str(Path(data_dir_str) / "md")
        config.paths["comments_dir"] = str(Path(data_dir_str) / "comments")

    return config


def scaffold_directories(
    base_dir: Optional[Union[Path, str, BrainConfig]] = None,
    config: Optional[BrainConfig] = None,
) -> Dict[str, Path]:
    """
    Idempotently creates all required data and brain_data directories.
    Flexible argument handling supports (base_dir, config) or (config).
    """
    if isinstance(base_dir, BrainConfig):
        config = base_dir
        base_dir = None

    data_dirname = "data"
    brain_data_dirname = "brain_data"

    if config is not None:
        data_dirname = config.paths.get("data_dir", "data")
        brain_data_dirname = config.paths.get("brain_data_dir", "brain_data")

    return initialize_directories(
        base_dir=base_dir,
        data_dirname=data_dirname,
        brain_data_dirname=brain_data_dirname,
        create_gitkeep=True,
    )


# Provide module aliases so 'from second_brain_generator.config.settings import ...'
# and 'from second_brain_generator.config.models import ...' resolve without extra directories
current_module = sys.modules[__name__]
sys.modules["second_brain_generator.config.settings"] = current_module
sys.modules["second_brain_generator.config.models"] = current_module
