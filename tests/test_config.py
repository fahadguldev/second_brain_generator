"""Unit and integration tests for Configuration, Directories, and Environment (F01-F04)."""

import os
from pathlib import Path
import pytest
from pydantic import ValidationError

from second_brain_generator.config import (
    BrainConfig,
    load_config,
    scaffold_directories,
    ConfigError,
    DEFAULT_PATHS,
    DEFAULT_TOPICS,
)
from second_brain_generator.utils.dirs import (
    initialize_directories,
    validate_directories,
    scan_category_files,
    DATA_SUBDIRS,
    BRAIN_DATA_SUBDIRS,
    SUPPORTED_EXTENSIONS,
)
from second_brain_generator.utils.env import (
    load_environment,
    get_gemini_keys,
    get_qdrant_credentials,
    mask_secret,
    is_offline_mode,
    MissingSecretError,
)


# ==============================================================================
# Category 1: Default Config Loading & Fallbacks
# ==============================================================================

def test_default_config_instantiation():
    """Validates default instantiation with zero parameters."""
    config = BrainConfig()
    assert config.creator_name != ""
    assert isinstance(config.creator_handles, list)
    assert len(config.creator_handles) > 0
    assert "english" in config.languages
    assert len(config.tone_guidelines) > 0
    assert "python" in config.topics
    assert "cloud" in config.topics
    assert isinstance(config.paths, dict)
    assert "data_dir" in config.paths
    assert "brain_data_dir" in config.paths
    assert config.creator.name == config.creator_name
    assert config.whisper.vad_filter is True
    assert config.embedding.dimension == 3072
    assert config.qdrant.collection_name == "second_brain"


def test_load_config_when_file_missing_returns_default(tmp_path, monkeypatch):
    """Missing config file should gracefully fall back to defaults."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BRAIN_CONFIG_PATH", raising=False)
    config = load_config(config_path=None)
    assert isinstance(config, BrainConfig)
    assert config.creator_name != ""


def test_load_config_explicit_missing_file_raises_error(tmp_path):
    """Explicitly requested config file that does not exist must raise FileNotFoundError."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(config_path=missing, raise_if_missing=True)


def test_load_config_from_env_var(tmp_path, monkeypatch):
    """BRAIN_CONFIG_PATH environment variable should be honored."""
    cfg_file = tmp_path / "custom_env.yaml"
    cfg_file.write_text("creator_name: 'Env Creator'\n", encoding="utf-8")
    monkeypatch.setenv("BRAIN_CONFIG_PATH", str(cfg_file))

    config = load_config(config_path=None)
    assert config.creator_name == "Env Creator"


def test_default_paths_completeness():
    """All required paths from R1 must be present in default configuration."""
    config = BrainConfig()
    required_paths = [
        "data_dir", "brain_data_dir", "vids_dir", "audios_dir",
        "text_dir", "md_dir", "comments_dir", "transcripts_dir",
        "records_path", "rag_records_path", "embeddings_path"
    ]
    for p in required_paths:
        assert p in config.paths, f"Missing required path key: {p}"


# ==============================================================================
# Category 2: Custom Config Loading & Normalization
# ==============================================================================

def test_custom_config_loading_from_yaml(tmp_path):
    """Loads a fully specified custom YAML file."""
    yaml_content = """
creator_name: "Dr. Aris"
creator_handles:
  - "@aris_bio"
languages:
  - "english"
  - "french"
tone_guidelines:
  - "Precise and academic"
topics:
  genomics:
    - "crispr"
    - "cas9"
paths:
  data_dir: "my_data"
  brain_data_dir: "my_brain"
"""
    cfg_path = tmp_path / "custom.yaml"
    cfg_path.write_text(yaml_content, encoding="utf-8")

    config = load_config(config_path=cfg_path)
    assert config.creator_name == "Dr. Aris"
    assert config.creator_handles == ["@aris_bio"]
    assert config.languages == ["english", "french"]
    assert "genomics" in config.topics
    assert config.topics["genomics"] == ["crispr", "cas9"]
    assert config.paths["data_dir"] == "my_data"
    assert config.paths["brain_data_dir"] == "my_brain"


def test_cli_data_dir_override(tmp_path):
    """Overriding data_dir via CLI argument updates all relative paths."""
    override_path = tmp_path / "custom_data"
    config = load_config(data_dir_override=override_path)
    assert Path(config.paths["data_dir"]) == override_path
    vids_path = config.get_path("vids_dir")
    assert str(override_path) in str(vids_path)


def test_nested_yaml_structure_normalization(tmp_path):
    """Normalizes nested YAML schema into flat contract attributes and submodels."""
    nested_yaml = """
creator:
  name: "Nested Creator"
  handles: ["@nested"]
persona:
  languages: ["english", "urdu"]
  tone: ["Direct and encouraging"]
"""
    cfg_path = tmp_path / "nested.yaml"
    cfg_path.write_text(nested_yaml, encoding="utf-8")

    config = load_config(config_path=cfg_path)
    assert config.creator_name == "Nested Creator"
    assert config.creator_handles == ["@nested"]
    assert config.languages == ["english", "urdu"]
    assert config.tone_guidelines == ["Direct and encouraging"]
    assert config.creator.name == "Nested Creator"


def test_path_resolution_helper(tmp_path):
    """Resolves relative and absolute paths accurately."""
    config = BrainConfig()
    resolved = config.get_path("vids_dir", base_dir=tmp_path)
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "data/vids").resolve()

    # Absolute path pass-through
    abs_dir = (tmp_path / "absolute/path").resolve()
    config.paths["abs_key"] = str(abs_dir)
    assert config.get_path("abs_key") == abs_dir


def test_topics_custom_override_and_merging(tmp_path):
    """Adding custom topics does not erase base topics unless desired."""
    yaml_content = """
topics:
  quantum:
    - "qubit"
    - "superposition"
"""
    cfg_path = tmp_path / "topics.yaml"
    cfg_path.write_text(yaml_content, encoding="utf-8")
    config = load_config(config_path=cfg_path)
    assert "quantum" in config.topics
    assert "python" in config.topics  # Merged default topic preserved


# ==============================================================================
# Category 3: YAML Parsing Edge Cases & Defensive Validation
# ==============================================================================

def test_yaml_syntax_error_raises_config_error(tmp_path):
    """Malformed YAML raises ConfigError."""
    broken = tmp_path / "broken.yaml"
    broken.write_text("creator_name: [unclosed list\n  bad: {", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=broken)


def test_yaml_empty_file_loads_defaults(tmp_path):
    """Empty YAML file falls back to defaults without error."""
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    config = load_config(config_path=empty_file)
    assert config.creator_name != ""


def test_yaml_partial_config_missing_fields(tmp_path):
    """Partial YAML specification merges with defaults."""
    partial = tmp_path / "partial.yaml"
    partial.write_text("creator_name: 'Only Name'\n", encoding="utf-8")
    config = load_config(config_path=partial)
    assert config.creator_name == "Only Name"
    assert len(config.languages) > 0
    assert len(config.tone_guidelines) > 0


def test_yaml_extra_fields_tolerance(tmp_path):
    """Unknown fields in YAML do not cause unhandled crashes."""
    extra = tmp_path / "extra.yaml"
    extra.write_text("creator_name: 'Alice'\nextra_unknown_key: 12345\n", encoding="utf-8")
    config = load_config(config_path=extra)
    assert config.creator_name == "Alice"


def test_yaml_single_string_coercion(tmp_path):
    """Single strings for list fields are coerced into single-element lists."""
    coerced = tmp_path / "coerce.yaml"
    coerced.write_text(
        "creator_handles: '@single_handle'\n"
        "languages: 'english'\n"
        "tone_guidelines: 'Single tone rule'\n",
        encoding="utf-8"
    )
    config = load_config(config_path=coerced)
    assert config.creator_handles == ["@single_handle"]
    assert config.languages == ["english"]
    assert config.tone_guidelines == ["Single tone rule"]


def test_yaml_invalid_types_rejection(tmp_path):
    """Completely invalid types raise validation errors."""
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("paths: 12345\n", encoding="utf-8")
    with pytest.raises((ValidationError, ConfigError)):
        load_config(config_path=invalid)


# ==============================================================================
# Category 4: Directory Creation & Idempotency
# ==============================================================================

def test_scaffold_directories_creates_all_subdirs(tmp_path):
    """Scaffolding creates all required subdirectories."""
    config = BrainConfig()
    scaffold_directories(base_dir=tmp_path, config=config)

    assert (tmp_path / "data/vids").is_dir()
    assert (tmp_path / "data/audios").is_dir()
    assert (tmp_path / "data/text").is_dir()
    assert (tmp_path / "data/md").is_dir()
    assert (tmp_path / "data/comments").is_dir()
    assert (tmp_path / "brain_data/text").is_dir()


def test_scaffold_directories_idempotent(tmp_path):
    """Calling scaffolding multiple times does not alter existing files."""
    config = BrainConfig()
    scaffold_directories(base_dir=tmp_path, config=config)

    test_file = tmp_path / "data/vids/sample.mp4"
    test_file.write_bytes(b"dummy video content")
    mtime_before = test_file.stat().st_mtime_ns

    # Second execution
    scaffold_directories(base_dir=tmp_path, config=config)

    assert test_file.exists()
    assert test_file.read_bytes() == b"dummy video content"
    assert test_file.stat().st_mtime_ns == mtime_before


def test_scaffold_directories_creates_gitkeep(tmp_path):
    """Initializes .gitkeep in empty directories."""
    paths = initialize_directories(base_dir=tmp_path, create_gitkeep=True)
    gitkeep = tmp_path / "data/vids/.gitkeep"
    assert gitkeep.exists()


def test_scaffold_directories_file_conflict_raises_error(tmp_path):
    """If a file blocks directory creation, raise an explicit error."""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    conflict_file = tmp_path / "data/vids"
    conflict_file.write_text("blocking file", encoding="utf-8")

    with pytest.raises((NotADirectoryError, ConfigError)):
        initialize_directories(base_dir=tmp_path)


def test_validate_directories_health_check(tmp_path):
    """validate_directories reports missing vs present directories."""
    report1 = validate_directories(base_dir=tmp_path)
    assert not report1.is_valid
    assert len(report1.missing_dirs) > 0

    initialize_directories(base_dir=tmp_path)
    report2 = validate_directories(base_dir=tmp_path)
    assert report2.is_valid
    assert len(report2.missing_dirs) == 0


def test_scan_category_files(tmp_path):
    """Discovers matching files for category case-insensitively and deterministically."""
    initialize_directories(base_dir=tmp_path)
    vids_dir = tmp_path / "data" / "vids"
    (vids_dir / "vid1.mp4").write_text("vid1", encoding="utf-8")
    (vids_dir / "vid2.MOV").write_text("vid2", encoding="utf-8")
    (vids_dir / "ignore.txt").write_text("ignore", encoding="utf-8")

    found = scan_category_files("vids", base_dir=tmp_path)
    found_names = [f.name for f in found]
    assert "vid1.mp4" in found_names
    assert "vid2.MOV" in found_names
    assert "ignore.txt" not in found_names

    with pytest.raises(ValueError):
        scan_category_files("invalid_category", base_dir=tmp_path)


# ==============================================================================
# Category 5: Environment Loading & Multi-Key Retrieval
# ==============================================================================

def test_load_environment_explicit_path(tmp_path):
    """load_environment loads variables from a specified file."""
    env_file = tmp_path / ".env"
    env_file.write_text("SAMPLE_VAR_TEST=second_brain_value\n", encoding="utf-8")
    loaded = load_environment(dotenv_path=env_file, override=True)
    assert loaded is True
    assert os.getenv("SAMPLE_VAR_TEST") == "second_brain_value"


def test_load_environment_missing_file_safe(tmp_path):
    """Missing .env file returns False safely."""
    loaded = load_environment(dotenv_path=tmp_path / "absent.env")
    assert loaded is False


def test_get_gemini_keys_single_key(monkeypatch):
    """Discovers single GEMINI_API_KEY."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for i in range(1, 50):
        monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)

    monkeypatch.setenv("GEMINI_API_KEY", "key_single_secret")
    keys = get_gemini_keys(require=False)
    assert keys == ["key_single_secret"]


def test_get_gemini_keys_multi_key_numeric_sorting(monkeypatch):
    """Discovers multi-keys and sorts them numerically."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for i in range(1, 50):
        monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)

    monkeypatch.setenv("GEMINI_API_KEY_10", "key_10")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key_2")
    monkeypatch.setenv("GEMINI_API_KEY_1", "key_1")
    monkeypatch.setenv("GEMINI_API_KEY", "key_base")

    keys = get_gemini_keys(require=False)
    assert keys == ["key_base", "key_1", "key_2", "key_10"]


def test_get_gemini_keys_deduplication(monkeypatch):
    """Identical keys across variables are deduplicated preserving order."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for i in range(1, 50):
        monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)

    monkeypatch.setenv("GEMINI_API_KEY", "shared_key")
    monkeypatch.setenv("GEMINI_API_KEY_1", "shared_key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "unique_key")

    keys = get_gemini_keys(require=False)
    assert keys == ["shared_key", "unique_key"]


def test_get_gemini_keys_filters_placeholders_and_empty(monkeypatch):
    """Filters empty strings and known placeholder values."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for i in range(1, 50):
        monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)

    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GEMINI_API_KEY_1", "your_gemini_api_key_here")
    monkeypatch.setenv("GEMINI_API_KEY_2", "valid_secret_key")

    keys = get_gemini_keys(require=False)
    assert keys == ["valid_secret_key"]


def test_get_gemini_keys_require_flag(monkeypatch):
    """When require=True, raises MissingSecretError if no keys found."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for i in range(1, 50):
        monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)

    assert get_gemini_keys(require=False) == []
    with pytest.raises(MissingSecretError):
        get_gemini_keys(require=True)


def test_mask_secret_formatting():
    """Verifies credential masking formats."""
    long_key = "AQ.Ab8RN6LjmI2iajxSwxWCMspBb5gNbcocQvHNpKx3TFs8o0VY0g"
    assert mask_secret(long_key) == "AQ.A...0VY0g"
    assert mask_secret("short") == "***"
    assert mask_secret(None) == "<none>"
    assert mask_secret("") == "<none>"
    assert mask_secret("   ") == "<none>"


def test_qdrant_credentials_discovery(monkeypatch):
    """Discovers QDRANT_URL, QDRANT_API_KEY, and collection."""
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.tech:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret_key_qdrant")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_collection")

    creds = get_qdrant_credentials(require=True)
    assert creds.url == "https://example.qdrant.tech:6333"
    assert creds.api_key == "secret_key_qdrant"
    assert creds.collection == "custom_collection"

    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(MissingSecretError):
        get_qdrant_credentials(require=True)


def test_offline_mode_detection(monkeypatch):
    """Verifies offline mode flag checking."""
    monkeypatch.setenv("SECOND_BRAIN_OFFLINE", "1")
    assert is_offline_mode() is True

    monkeypatch.setenv("SECOND_BRAIN_OFFLINE", "0")
    monkeypatch.setenv("OFFLINE_MODE", "true")
    assert is_offline_mode() is True

    monkeypatch.delenv("SECOND_BRAIN_OFFLINE", raising=False)
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    assert is_offline_mode() is False


# ==============================================================================
# Category 5: BrainConfig.from_yaml and Input Normalization
# ==============================================================================

def test_brain_config_from_yaml_file(tmp_path):
    """BrainConfig.from_yaml loads and validates configuration from a file path."""
    yaml_file = tmp_path / "valid_config.yaml"
    yaml_file.write_text(
        "creator_name: 'From Yaml Creator'\nlanguages: ['english', 'spanish']\n",
        encoding="utf-8",
    )
    config = BrainConfig.from_yaml(yaml_file)
    assert config.creator_name == "From Yaml Creator"
    assert config.languages == ["english", "spanish"]


def test_brain_config_from_yaml_string():
    """BrainConfig.from_yaml loads and validates configuration from raw YAML string."""
    yaml_content = "creator_name: 'String Creator'\ntopics:\n  custom: ['alpha', 'beta']\n"
    config = BrainConfig.from_yaml(yaml_content)
    assert config.creator_name == "String Creator"
    assert "custom" in config.topics
    assert config.topics["custom"] == ["alpha", "beta"]


def test_brain_config_from_yaml_empty_defaults(tmp_path):
    """BrainConfig.from_yaml with empty file loads default configuration."""
    empty_file = tmp_path / "empty_cfg.yaml"
    empty_file.write_text("", encoding="utf-8")
    config = BrainConfig.from_yaml(empty_file)
    assert config.creator_name == "Second Brain Creator"
    assert "data_dir" in config.paths


def test_brain_config_from_yaml_with_data_dir_override(tmp_path):
    """BrainConfig.from_yaml applies data_dir_override properly."""
    yaml_file = tmp_path / "simple.yaml"
    yaml_file.write_text("creator_name: 'Override Creator'\n", encoding="utf-8")
    override_dir = tmp_path / "custom_data_dir"
    config = BrainConfig.from_yaml(yaml_file, data_dir_override=override_dir)
    assert config.paths["data_dir"] == str(override_dir)
    assert config.paths["vids_dir"] == str(override_dir / "vids")


def test_brain_config_from_yaml_invalid_scalar_raises_config_error(tmp_path):
    """BrainConfig.from_yaml rejects non-dict scalar payloads with ConfigError."""
    bad_file = tmp_path / "bad_scalar.yaml"
    bad_file.write_text("false\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML must contain a top-level mapping/dictionary"):
        BrainConfig.from_yaml(bad_file)


def test_brain_config_from_yaml_missing_file_raises_error(tmp_path):
    """BrainConfig.from_yaml raises FileNotFoundError for missing path."""
    missing = tmp_path / "missing_file.yaml"
    with pytest.raises(FileNotFoundError):
        BrainConfig.from_yaml(missing)


def test_brain_config_normalize_input_rejects_non_dict_primitives():
    """BrainConfig.model_validate rejects non-dict primitives with ConfigError."""
    for invalid in [False, 0, 0.0, "just a string", [1, 2, 3]]:
        with pytest.raises(ConfigError, match="YAML must contain a top-level mapping/dictionary"):
            BrainConfig.model_validate(invalid)


def test_brain_config_normalize_input_accepts_existing_brain_config():
    """BrainConfig.model_validate preserves existing BrainConfig instance."""
    original = BrainConfig(creator_name="Original")
    validated = BrainConfig.model_validate(original)
    assert validated.creator_name == "Original"

