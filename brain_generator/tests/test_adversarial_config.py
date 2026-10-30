"""Adversarial stress-testing suite for Configuration Engine (config.py) and Secrets Management (env.py).

Covers:
1. Fuzzing load_config: malformed YAML, deeply nested YAML, recursive structures,
   billion laughs expansion, non-ASCII Unicode keys, null bytes, non-dict payloads.
2. CLI override boundary conditions: non-existent directories, read-only paths,
   file conflicts, path traversal, unicode directories.
3. get_gemini_keys & secrets: malformed env keys, out-of-order numerical indices,
   whitespace-padded keys, dummy key filtering, deduplication, concurrent mutations.
"""

import os
from pathlib import Path
import stat
import threading
import time
import pytest
from pydantic import ValidationError

from second_brain_generator.config import (
    BrainConfig,
    ConfigError,
    load_config,
    scaffold_directories,
    DEFAULT_PATHS,
    DEFAULT_TOPICS,
)
from second_brain_generator.utils.dirs import (
    initialize_directories,
    validate_directories,
)
from second_brain_generator.utils.env import (
    get_gemini_keys,
    get_qdrant_credentials,
    load_environment,
    mask_secret,
    is_offline_mode,
    MissingSecretError,
)


# ==============================================================================
# 1. Adversarial Fuzzing for load_config & YAML Parser
# ==============================================================================

class TestLoadConfigFuzzing:
    """Fuzzing and adversarial stress tests for config.py / load_config."""

    @pytest.mark.parametrize(
        "malformed_syntax",
        [
            "creator_name: [unclosed list",
            "creator_name: {unclosed mapping",
            "foo: \"unclosed quote\nbar: 1",
            "a:\n\tb: tab_character_indent\n",
            "key: : dangling colon",
            "creator_handles:\n  - item1\n - bad_indent",
            "%YAML 1.2\n%TAG ! bad tag\n",
            "--- !!binary \"invalid_base64_%%%===\"",
            "::bad::yaml::format",
        ],
    )
    def test_fuzz_malformed_syntax_raises_config_error(self, tmp_path, malformed_syntax):
        """Malformed YAML syntax must consistently raise ConfigError without uncaught exceptions."""
        cfg_file = tmp_path / "syntax_err.yaml"
        cfg_file.write_text(malformed_syntax, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(config_path=cfg_file)

    def test_fuzz_binary_and_random_bytes(self, tmp_path):
        """Random binary bytes and headers must raise ConfigError rather than crashing."""
        for i, header in enumerate([
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",  # PNG header
            b"\x7fELF\x02\x01\x01\x00\x00\x00",      # ELF header
            os.urandom(512),                         # Random noise
            b"\xff\xfe\x00\x00invalid_utf8\x80\x81", # Invalid UTF-8
        ]):
            f = tmp_path / f"bin_{i}.yaml"
            f.write_bytes(header)
            with pytest.raises(ConfigError):
                load_config(config_path=f)

    @pytest.mark.parametrize(
        "scalar_payload",
        [
            "12345",
            "-99.99",
            "'just a plain string'",
            "[1, 2, 3, 'a list']",
            "true",
        ],
    )
    def test_fuzz_non_dict_yaml_scalars_raises_config_error(self, tmp_path, scalar_payload):
        """Root YAML content that is a scalar or list rather than a mapping must raise ConfigError."""
        f = tmp_path / "scalar.yaml"
        f.write_text(scalar_payload, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(config_path=f)

    @pytest.mark.parametrize("falsy_scalar", ["false", "0", "0.0", '""', "[]"])
    def test_fuzz_falsy_yaml_scalars_should_raise_config_error(self, tmp_path, falsy_scalar):
        """Falsy YAML scalars (false, 0, 0.0, '', []) must raise ConfigError just like truthy scalars."""
        f = tmp_path / "falsy_scalar.yaml"
        f.write_text(falsy_scalar, encoding="utf-8")
        with pytest.raises(ConfigError, match="YAML must contain a top-level mapping/dictionary"):
            load_config(config_path=f)

    def test_fuzz_deeply_nested_yaml_resilience(self, tmp_path):
        """Deeply nested structures are handled or gracefully rejected without uncaught recursion crashes."""
        # Test 100-level nesting (should parse and store extra fields)
        deep_100 = "root:\n" + "".join("  " * (i + 1) + f"k{i}:\n" for i in range(100)) + "  " * 101 + "leaf: 42\n"
        f1 = tmp_path / "deep_100.yaml"
        f1.write_text(deep_100, encoding="utf-8")
        cfg = load_config(config_path=f1)
        assert isinstance(cfg, BrainConfig)

        # Test extreme 1000-level nesting (should raise ConfigError cleanly if recursion limit hit)
        deep_1000 = "root:\n" + "".join("  " * (i + 1) + f"k{i}:\n" for i in range(1000)) + "  " * 1001 + "leaf: 42\n"
        f2 = tmp_path / "deep_1000.yaml"
        f2.write_text(deep_1000, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(config_path=f2)

    def test_fuzz_cyclic_anchors_and_billion_laughs(self, tmp_path):
        """Cyclic anchors and exponential entity expansion (billion laughs) are handled safely."""
        # 1. Cyclic anchor mapping
        cyclic_content = "node: &node\n  parent: *node\ncreator_name: 'Cyclic Node'\n"
        f_cyc = tmp_path / "cyclic.yaml"
        f_cyc.write_text(cyclic_content, encoding="utf-8")
        cfg = load_config(config_path=f_cyc)
        assert cfg.creator_name == "Cyclic Node"
        assert "node" in cfg.model_extra

        # 2. Billion laughs anchor expansion
        bl_content = """
a: &a ['lol','lol','lol','lol','lol','lol','lol','lol','lol','lol']
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]
creator_name: 'Expansion Test'
"""
        f_bl = tmp_path / "bl.yaml"
        f_bl.write_text(bl_content, encoding="utf-8")
        t0 = time.time()
        cfg_bl = load_config(config_path=f_bl)
        elapsed = time.time() - t0
        assert cfg_bl.creator_name == "Expansion Test"
        assert elapsed < 1.0  # Must not block or hang

    def test_fuzz_non_ascii_unicode_keys_and_values(self, tmp_path):
        """Config handles diverse Unicode keys, multilingual scripts, and emojis."""
        unicode_content = """
creator_name: 'कबीर'
creator_handles:
  - '@kabir_شاعر'
  - '@李白'
languages:
  - 'hindi'
  - 'urdu'
  - 'mandarin'
tone_guidelines:
  - 'रहस्यवादी एवं प्रत्यक्ष'
  - '富有哲理'
topics:
  मशीन_लर्निंग:
    - 'न्यूरल_नेटवर्क'
    - 'ट्रांसफॉर्मर'
  人工智能:
    - '大语言模型'
paths:
  डेटा_पथ: 'data/विशेष'
  🧠_brain: 'brain_data/emoji'
"""
        f = tmp_path / "unicode_config.yaml"
        f.write_text(unicode_content, encoding="utf-8")
        cfg = load_config(config_path=f)

        assert cfg.creator_name == "कबीर"
        assert "@李白" in cfg.creator_handles
        assert "मशीन_लर्निंग" in cfg.topics
        assert "न्यूरल_नेटवर्क" in cfg.topics["मशीन_लर्निंग"]
        assert "人工智能" in cfg.topics
        assert cfg.paths["डेटा_पथ"] == "data/विशेष"

        # Resolved path retains Unicode characters
        resolved = cfg.get_path("डेटा_पथ", base_dir=tmp_path)
        assert "विशेष" in str(resolved)

    def test_fuzz_null_bytes_in_content_and_keys(self, tmp_path):
        """Null bytes in YAML keys or values are caught and raise ConfigError."""
        # Null byte in value
        f_val = tmp_path / "null_val.yaml"
        f_val.write_bytes(b"creator_name: bad\x00value\n")
        with pytest.raises(ConfigError):
            load_config(config_path=f_val)

        # Null byte in key
        f_key = tmp_path / "null_key.yaml"
        f_key.write_bytes(b"bad\x00key: 12345\n")
        with pytest.raises(ConfigError):
            load_config(config_path=f_key)

    def test_fuzz_invalid_schema_types(self, tmp_path):
        """Incorrect data types in schema sections fail with ConfigError."""
        # Topics is a string instead of dict
        f1 = tmp_path / "bad_topics_str.yaml"
        f1.write_text("topics: 'not a dictionary'\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(config_path=f1)

        # Paths is an integer instead of dict
        f2 = tmp_path / "bad_paths_int.yaml"
        f2.write_text("paths: 99999\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(config_path=f2)

        # Topic values are non-list (e.g. integer)
        f3 = tmp_path / "bad_topic_val.yaml"
        f3.write_text("topics:\n  ai: 12345\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(config_path=f3)

        # Topic value is null (None)
        f4 = tmp_path / "null_topic_val.yaml"
        f4.write_text("topics:\n  ai:\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(config_path=f4)

    def test_unreadable_file_and_directory_path(self, tmp_path):
        """Unreadable file permissions or directory passed as config_path."""
        # Unreadable file
        f_unreadable = tmp_path / "unreadable.yaml"
        f_unreadable.write_text("creator_name: test", encoding="utf-8")
        os.chmod(f_unreadable, 0o000)
        try:
            with pytest.raises(ConfigError):
                load_config(config_path=f_unreadable)
        finally:
            os.chmod(f_unreadable, 0o644)

        # Directory passed as config_path with raise_if_missing=True
        with pytest.raises(FileNotFoundError):
            load_config(config_path=tmp_path, raise_if_missing=True)


# ==============================================================================
# 2. CLI Override Boundary Conditions
# ==============================================================================

class TestCliOverrideBoundaries:
    """Boundary condition tests for CLI overrides and directory scaffolding."""

    def test_cli_override_nonexistent_deep_directory(self, tmp_path):
        """Deeply nested non-existent directory override is accepted and scaffolded."""
        deep_target = tmp_path / "level1" / "level2" / "level3" / "custom_data"
        assert not deep_target.exists()

        cfg = load_config(data_dir_override=deep_target)
        assert Path(cfg.paths["data_dir"]) == deep_target

        dirs = scaffold_directories(base_dir=tmp_path, config=cfg)
        assert deep_target.is_dir()
        assert dirs["data_vids"].is_dir()
        assert dirs["data_vids"].exists()
        assert (deep_target / "vids").exists()

    def test_cli_override_file_collision_raises_error(self, tmp_path):
        """If a regular file exists at the override path, scaffolding raises NotADirectoryError."""
        file_path = tmp_path / "blocking_file.txt"
        file_path.write_text("I am a file, not a directory", encoding="utf-8")

        cfg = load_config(data_dir_override=file_path)
        with pytest.raises(NotADirectoryError):
            scaffold_directories(base_dir=tmp_path, config=cfg)

    def test_cli_override_readonly_destination_raises_permission_error(self, tmp_path):
        """Scaffolding into a read-only parent directory raises PermissionError."""
        ro_dir = tmp_path / "readonly_parent"
        ro_dir.mkdir()
        # Revoke write permissions
        os.chmod(ro_dir, 0o555)

        try:
            target_data = ro_dir / "new_data"
            cfg = load_config(data_dir_override=target_data)
            with pytest.raises(PermissionError):
                scaffold_directories(base_dir=ro_dir, config=cfg)
        finally:
            os.chmod(ro_dir, 0o755)

    def test_cli_override_empty_and_whitespace_strings(self, tmp_path):
        """Empty string override retains defaults; whitespace is isolated."""
        # Empty string
        cfg_empty = load_config(data_dir_override="")
        assert cfg_empty.paths["data_dir"] == "data"
        assert cfg_empty.paths["vids_dir"] == "data/vids"

        # Whitespace-only string
        cfg_ws = load_config(data_dir_override="   ")
        assert cfg_ws.paths["data_dir"] == "   "

    def test_cli_override_path_traversal_and_dots(self, tmp_path):
        """Path traversal sequences (..) in data_dir_override are resolved safely."""
        traversal_dir = tmp_path / "subdir" / ".." / "resolved_data"
        cfg = load_config(data_dir_override=traversal_dir)
        dirs = scaffold_directories(base_dir=tmp_path, config=cfg)
        assert (tmp_path / "resolved_data" / "vids").is_dir()
        assert dirs["data_vids"].is_dir()

    def test_validate_directories_on_read_only_text_dir(self, tmp_path):
        """validate_directories detects when output directories lack write permission."""
        dirs = initialize_directories(base_dir=tmp_path)
        assert validate_directories(base_dir=tmp_path).is_valid is True

        text_dir = dirs["brain_data_text"]
        os.chmod(text_dir, 0o555)
        try:
            report = validate_directories(base_dir=tmp_path)
            assert report.is_valid is False
            assert text_dir in report.unwritable_dirs
        finally:
            os.chmod(text_dir, 0o755)

    def test_validate_directories_on_read_only_brain_data_root(self, tmp_path):
        """validate_directories detects when brain_data root directory lacks write permission."""
        dirs = initialize_directories(base_dir=tmp_path)
        assert validate_directories(base_dir=tmp_path).is_valid is True

        brain_root = dirs["brain_data"]
        os.chmod(brain_root, 0o555)
        try:
            # If running as unprivileged user, 0o555 prevents write
            if not os.access(brain_root, os.W_OK):
                report = validate_directories(base_dir=tmp_path)
                assert report.is_valid is False
                assert brain_root in report.unwritable_dirs
        finally:
            os.chmod(brain_root, 0o755)

    def test_validate_directories_custom_dirname_unwritable(self, tmp_path):
        """validate_directories checks write permission dynamically when brain_data_dirname is custom."""
        dirs = initialize_directories(base_dir=tmp_path, brain_data_dirname="custom_vault")
        assert validate_directories(base_dir=tmp_path, brain_data_dirname="custom_vault").is_valid is True

        vault_root = dirs["brain_data"]
        os.chmod(vault_root, 0o555)
        try:
            if not os.access(vault_root, os.W_OK):
                report = validate_directories(base_dir=tmp_path, brain_data_dirname="custom_vault")
                assert report.is_valid is False
                assert vault_root in report.unwritable_dirs
        finally:
            os.chmod(vault_root, 0o755)

    def test_validate_directories_unwritable_deduplication(self, tmp_path):
        """validate_directories deduplicates unwritable list when directory has mode 0o000."""
        dirs = initialize_directories(base_dir=tmp_path)
        brain_root = dirs["brain_data"]
        os.chmod(brain_root, 0o000)
        try:
            if not os.access(brain_root, os.W_OK) and not os.access(brain_root, os.R_OK):
                report = validate_directories(base_dir=tmp_path)
                assert report.is_valid is False
                assert report.unwritable_dirs.count(brain_root) == 1
        finally:
            os.chmod(brain_root, 0o755)


# ==============================================================================
# 3. Secrets Management & get_gemini_keys Boundary Tests
# ==============================================================================

class TestSecretsAndGeminiKeys:
    """Stress testing secrets discovery, numeric ordering, padding, and mutations."""

    @pytest.fixture(autouse=True)
    def clean_gemini_env(self, monkeypatch):
        """Ensure clean environment before and after each test."""
        for k in list(os.environ.keys()):
            if "GEMINI" in k or "QDRANT" in k:
                monkeypatch.delenv(k, raising=False)

    def test_gemini_keys_malformed_variable_names_ignored(self, monkeypatch):
        """Malformed variable names are strictly ignored by regex discovery."""
        monkeypatch.setenv("GEMINI_API_KEY_", "bad_no_index")
        monkeypatch.setenv("GEMINI_API_KEY_abc", "bad_alphabetic_index")
        monkeypatch.setenv("GEMINI_API_KEY_-1", "bad_negative_index")
        monkeypatch.setenv("MY_GEMINI_API_KEY_1", "bad_prefix")
        monkeypatch.setenv("GEMINI_API_KEY_1_EXTRA", "bad_suffix")
        monkeypatch.setenv("gemini_api_key_1", "bad_lowercase")
        monkeypatch.setenv("GEMINI_API_KEY_1", "valid_key_one")

        keys = get_gemini_keys(require=False)
        assert keys == ["valid_key_one"]

    def test_gemini_keys_out_of_order_numerical_sorting(self, monkeypatch):
        """Numeric indices sort strictly by integer value (1, 2, ..., 10, 20, 100), not lexicographically."""
        # Lexicographical sort would produce: 1, 10, 100, 2, 20, 3
        # Numerical sort must produce: 0, 1, 2, 3, 10, 20, 100
        monkeypatch.setenv("GEMINI_API_KEY_100", "key_hundred")
        monkeypatch.setenv("GEMINI_API_KEY_10", "key_ten")
        monkeypatch.setenv("GEMINI_API_KEY_2", "key_two")
        monkeypatch.setenv("GEMINI_API_KEY_20", "key_twenty")
        monkeypatch.setenv("GEMINI_API_KEY_3", "key_three")
        monkeypatch.setenv("GEMINI_API_KEY_1", "key_one")
        monkeypatch.setenv("GEMINI_API_KEY", "key_base_zero")

        keys = get_gemini_keys(require=False)
        expected = [
            "key_base_zero",
            "key_one",
            "key_two",
            "key_three",
            "key_ten",
            "key_twenty",
            "key_hundred",
        ]
        assert keys == expected

    def test_gemini_keys_large_index_and_zero_index(self, monkeypatch):
        """Handles GEMINI_API_KEY_0 and huge numeric index values safely."""
        monkeypatch.setenv("GEMINI_API_KEY_0", "key_zero")
        monkeypatch.setenv("GEMINI_API_KEY", "key_base")
        monkeypatch.setenv("GEMINI_API_KEY_99999999999999999999", "key_huge")

        keys = get_gemini_keys(require=False)
        # Base key and key_0 both have index 0; order is preserved, huge index is last
        assert keys[0] == "key_base"
        assert keys[1] == "key_zero"
        assert keys[-1] == "key_huge"

    def test_gemini_keys_whitespace_padding_and_empty_filtering(self, monkeypatch):
        """Whitespace-padded keys are stripped; whitespace-only keys are discarded."""
        monkeypatch.setenv("GEMINI_API_KEY", "   spaced_base   ")
        monkeypatch.setenv("GEMINI_API_KEY_1", "\t\r\nkey_with_escapes\n")
        monkeypatch.setenv("GEMINI_API_KEY_2", "   \t   ")  # Whitespace only
        monkeypatch.setenv("GEMINI_API_KEY_3", "")          # Empty string

        keys = get_gemini_keys(require=False)
        assert keys == ["spaced_base", "key_with_escapes"]

    def test_gemini_keys_dummy_placeholder_case_insensitivity(self, monkeypatch):
        """All dummy/placeholder indicator strings are filtered regardless of casing."""
        dummy_samples = [
            "your_api_key",
            "YOUR_API_KEY",
            "Your_Gemini_Api_Key",
            "my_placeholder_key",
            "PLACEHOLDER",
            "xxx",
            "XXX",
            "my_key_xxx_dummy",
        ]
        for idx, dummy in enumerate(dummy_samples, start=1):
            monkeypatch.setenv(f"GEMINI_API_KEY_{idx}", dummy)

        monkeypatch.setenv("GEMINI_API_KEY", "valid_secret_key_12345")
        keys = get_gemini_keys(require=False)
        assert keys == ["valid_secret_key_12345"]

    def test_gemini_keys_deduplication_preserves_priority(self, monkeypatch):
        """Identical keys across different environment variables are deduplicated to their first occurrence."""
        monkeypatch.setenv("GEMINI_API_KEY", "shared_secret")
        monkeypatch.setenv("GEMINI_API_KEY_1", "   shared_secret   ")  # Matches when stripped
        monkeypatch.setenv("GEMINI_API_KEY_2", "unique_second")
        monkeypatch.setenv("GEMINI_API_KEY_3", "shared_secret")

        keys = get_gemini_keys(require=False)
        assert keys == ["shared_secret", "unique_second"]

    def test_gemini_keys_require_flag_enforcement(self, monkeypatch):
        """When require=True, raises MissingSecretError if only dummy or empty keys exist."""
        monkeypatch.setenv("GEMINI_API_KEY_1", "your_api_key")
        monkeypatch.setenv("GEMINI_API_KEY_2", "   ")
        with pytest.raises(MissingSecretError):
            get_gemini_keys(require=True)

    def test_gemini_keys_concurrent_env_mutation_safety(self, monkeypatch):
        """Concurrent mutations to os.environ during get_gemini_keys execute without raising RuntimeError."""
        monkeypatch.setenv("GEMINI_API_KEY_1", "stable_key_1")
        monkeypatch.setenv("GEMINI_API_KEY_2", "stable_key_2")

        stop_event = threading.Event()
        mutation_errors = []

        def background_mutator():
            i = 0
            while not stop_event.is_set():
                key = f"MUTATION_TEST_VAR_{i}"
                os.environ[key] = f"val_{i}"
                time.sleep(0.0001)
                os.environ.pop(key, None)
                i += 1

        t = threading.Thread(target=background_mutator)
        t.daemon = True
        t.start()

        try:
            for _ in range(100):
                keys = get_gemini_keys(require=False)
                assert "stable_key_1" in keys
                assert "stable_key_2" in keys
        except Exception as exc:
            mutation_errors.append(exc)
        finally:
            stop_event.set()
            t.join(timeout=2.0)

        assert len(mutation_errors) == 0, f"Concurrent mutation error: {mutation_errors}"

    def test_load_environment_malformed_dotenv(self, tmp_path):
        """load_environment gracefully survives malformed .env files with missing equals or bad lines."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
# Valid and malformed mixed content
BAD_LINE_NO_EQUALS
=empty_key
VALID_SECRET=unmasked_value_123
   SPACED_SECRET   =   spaced_value   
UNICODE_VAR=李白_कबीर
# Trailing comment
""", encoding="utf-8")

        result = load_environment(dotenv_path=env_file, override=True)
        assert result is True
        assert os.getenv("VALID_SECRET") == "unmasked_value_123"
        assert os.getenv("SPACED_SECRET") == "spaced_value"
        assert os.getenv("UNICODE_VAR") == "李白_कबीर"

    @pytest.mark.parametrize(
        "secret, expected",
        [
            (None, "<none>"),
            ("", "<none>"),
            ("   ", "<none>"),
            ("1", "***"),
            ("12345678", "***"),            # Exactly 8 characters
            ("123456789", "1234...56789"),   # Exactly 9 characters
            ("AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz", "AIza...vWxYz"),
            ("🔐secret_unicode_key_here🔑", "🔐sec...here🔑"),
        ],
    )
    def test_mask_secret_boundary_cases(self, secret, expected):
        """mask_secret accurately masks short, long, boundary, and Unicode secrets."""
        assert mask_secret(secret) == expected


# ==============================================================================
# 4. Additional Adversarial YAML Configurations (Iteration 2)
# ==============================================================================

class TestAdversarialYamlSequences:
    """Adversarial stress testing of deeply nested and malformed YAML sequences."""

    def test_deeply_nested_flow_sequences_at_root(self, tmp_path):
        """Deeply nested flow sequences (e.g. [[[[...]]]]) must raise ConfigError."""
        deep_seq = "[" * 50 + "1" + "]" * 50 + "\n"
        # 1. Via raw string in BrainConfig.from_yaml
        with pytest.raises(ConfigError, match="YAML must contain a top-level mapping/dictionary"):
            BrainConfig.from_yaml(deep_seq)

        # 2. Via file in load_config
        f = tmp_path / "deep_flow_seq.yaml"
        f.write_text(deep_seq, encoding="utf-8")
        with pytest.raises(ConfigError, match="YAML must contain a top-level mapping/dictionary"):
            load_config(config_path=f)

    def test_extreme_nested_sequence_recursion_limit(self, tmp_path):
        """Extreme nesting (1000 levels) must not cause unhandled crashes."""
        extreme_seq = "[" * 1000 + "1" + "]" * 1000 + "\n"
        with pytest.raises(ConfigError):
            BrainConfig.from_yaml(extreme_seq)

    def test_deeply_nested_sequence_in_schema_field(self):
        """Deeply nested sequence within a schema field (creator_handles) fails validation."""
        deep_block = "creator_name: DeepSeq\ncreator_handles:\n" + "".join("  " * (i + 1) + "- " for i in range(25)) + "item\n"
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            BrainConfig.from_yaml(deep_block)

    def test_deeply_nested_sequence_in_topics_field(self):
        """Deeply nested sequence within topics mapping fails validation with ConfigError."""
        deep_topic = "topics:\n  ai:\n" + "".join("    " * (i + 1) + "- " for i in range(20)) + "deep_item\n"
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            BrainConfig.from_yaml(deep_topic)

    def test_empty_nested_flow_sequences(self):
        """Empty nested flow sequences at root raise ConfigError."""
        for seq_str in ["[[]]\n", "[[[], []]]\n", "[[], [[]], [[[]]]]\n"]:
            with pytest.raises(ConfigError, match="YAML must contain a top-level mapping/dictionary"):
                BrainConfig.from_yaml(seq_str)

    def test_single_line_deeply_nested_sequence_without_newline(self):
        """Single-line nested sequence without newline (>255 chars) must raise ConfigError, not OSError."""
        single_line_seq = "[" * 150 + "1" + "]" * 150
        with pytest.raises(ConfigError):
            BrainConfig.from_yaml(single_line_seq)


class TestAdversarialYamlBinaryData:
    """Adversarial testing of YAML binary tags (!!binary) and byte payloads."""

    def test_binary_tag_at_root_mapping_rejection(self, tmp_path):
        """YAML root consisting of a !!binary scalar raises ConfigError."""
        bin_payload = "--- !!binary \"aGVsbG8=\"\n"
        with pytest.raises(ConfigError, match="YAML must contain a top-level mapping/dictionary, got bytes"):
            BrainConfig.from_yaml(bin_payload)

        f = tmp_path / "bin_root.yaml"
        f.write_text(bin_payload, encoding="utf-8")
        with pytest.raises(ConfigError, match="YAML must contain a top-level mapping/dictionary, got bytes"):
            load_config(config_path=f)

    def test_binary_tag_in_string_field_valid_utf8(self):
        """!!binary payload with valid UTF-8 base64 is coerced safely to string."""
        import base64
        b64 = base64.b64encode(b"Test Creator").decode("ascii")
        yml = f"creator_name: !!binary \"{b64}\"\n"
        cfg = BrainConfig.from_yaml(yml)
        assert cfg.creator_name == "Test Creator"

    def test_binary_tag_in_string_field_corrupted_base64(self):
        """!!binary payload with corrupted base64 characters raises ConfigError."""
        yml = "creator_name: !!binary \"invalid_base64_%%%===\"\n"
        with pytest.raises(ConfigError):
            BrainConfig.from_yaml(yml)

    def test_binary_tag_in_string_field_non_utf8_bytes(self):
        """!!binary payload containing non-UTF8 arbitrary bytes fails Pydantic string validation."""
        import base64
        raw_non_utf8 = b"\x80\x81\xff\xfe\xaa\xbb"
        b64 = base64.b64encode(raw_non_utf8).decode("ascii")
        yml = f"creator_name: !!binary \"{b64}\"\n"
        with pytest.raises(ConfigError, match="unable to parse raw data as a unicode string"):
            BrainConfig.from_yaml(yml)

    def test_binary_tag_with_embedded_null_byte_base64(self):
        """!!binary payload encoding a null byte inside valid base64 is parsed safely."""
        import base64
        b64 = base64.b64encode(b"hello\x00world").decode("ascii")
        yml = f"creator_name: !!binary \"{b64}\"\n"
        cfg = BrainConfig.from_yaml(yml)
        assert "\x00" in cfg.creator_name

    def test_binary_tag_as_topic_dictionary_key(self):
        """!!binary tag used as dictionary key is decoded cleanly."""
        import base64
        b64 = base64.b64encode(b"ai").decode("ascii")
        yml = f"topics:\n  !!binary \"{b64}\": [\"ml\", \"deep-learning\"]\n"
        cfg = BrainConfig.from_yaml(yml)
        assert "ai" in cfg.topics
        assert cfg.topics["ai"] == ["ml", "deep-learning"]

    def test_binary_tag_large_payload_no_oom(self):
        """Large !!binary payload (500KB) is handled without excessive memory or crashing."""
        import base64
        large_bytes = b"X" * (500 * 1024)
        b64 = base64.b64encode(large_bytes).decode("ascii")
        yml = f"bio: !!binary \"{b64}\"\n"
        cfg = BrainConfig.from_yaml(yml)
        assert len(cfg.bio) == 500 * 1024


class TestAdversarialYamlTabsVsSpaces:
    """Adversarial testing of tabs vs spaces indentation and placement in YAML."""

    def test_tab_indentation_rejected(self):
        """Tab character used for YAML block indentation raises ConfigError."""
        yml = "creator_name: TabTest\n\tcreator_handles:\n\t  - handle1\n"
        with pytest.raises(ConfigError, match="YAML syntax error"):
            BrainConfig.from_yaml(yml)

    def test_tab_after_mapping_colon_rejected(self):
        """Tab character immediately after mapping colon raises ConfigError."""
        yml = "creator_name:\t\"ColonTab\"\n"
        with pytest.raises(ConfigError, match="YAML syntax error"):
            BrainConfig.from_yaml(yml)

    def test_tab_in_flow_mapping_rejected(self):
        """Tab character inside flow mapping raises ConfigError."""
        yml = "{\tcreator_name:\t\"FlowTab\"\t}\n"
        with pytest.raises(ConfigError, match="YAML syntax error"):
            BrainConfig.from_yaml(yml)

    def test_tab_in_sequence_indentation_rejected(self):
        """Tab character in sequence item indentation raises ConfigError."""
        yml = "creator_handles:\n\t- item1\n\t- item2\n"
        with pytest.raises(ConfigError, match="YAML syntax error"):
            BrainConfig.from_yaml(yml)

    def test_tab_trailing_outside_quotes_rejected(self):
        """Trailing tab characters outside quotes raise ConfigError."""
        yml = "creator_name: \"ValidName\"\t\t\n"
        with pytest.raises(ConfigError, match="YAML syntax error"):
            BrainConfig.from_yaml(yml)

    def test_tab_inside_quoted_strings_preserved(self):
        """Tab characters inside double or single quoted strings are safely parsed and preserved."""
        yml = "creator_name: \"First\\tLast\"\nbio: 'Bio with\\ttab'\n"
        cfg = BrainConfig.from_yaml(yml)
        assert "\t" in cfg.creator_name
        assert "First\tLast" == cfg.creator_name

    def test_mixed_spaces_and_tabs_indentation_rejected(self):
        """Mixed spaces and tabs in line indentation raise ConfigError."""
        yml = "creator_name: Mixed\n  \tlanguages:\n    - english\n"
        with pytest.raises(ConfigError, match="YAML syntax error"):
            BrainConfig.from_yaml(yml)


class TestAdversarialYamlAnchorsAndAliases:
    """Adversarial testing of anchors (&), aliases (*), and merge keys (<<)."""

    def test_undefined_anchor_alias_raises_config_error(self):
        """Referencing an undefined alias raises ConfigError."""
        yml = "creator_name: *nonexistent_anchor\n"
        with pytest.raises(ConfigError, match="found undefined alias"):
            BrainConfig.from_yaml(yml)

    def test_forward_reference_alias_raises_config_error(self):
        """Referencing an alias before its anchor definition raises ConfigError."""
        yml = "creator_name: *future_anchor\nfuture_anchor: &future_anchor 'Defined Later'\n"
        with pytest.raises(ConfigError, match="found undefined alias"):
            BrainConfig.from_yaml(yml)

    def test_merge_key_single_anchor_dict(self):
        """Single merge key (<<: *anchor) correctly inherits mapping fields."""
        yml = """
base_persona: &base
  languages: ["english", "french"]
  tone_guidelines: ["precise", "concise"]
<<: *base
creator_name: "Merge User"
"""
        cfg = BrainConfig.from_yaml(yml)
        assert cfg.creator_name == "Merge User"
        assert cfg.languages == ["english", "french"]
        assert cfg.tone_guidelines == ["precise", "concise"]

    def test_merge_key_multiple_anchors_list(self):
        """Multiple merge keys (<<: [*a, *b]) merge in sequence."""
        yml = """
lang_base: &lang_base
  languages: ["urdu", "hindi"]
tone_base: &tone_base
  tone_guidelines: ["philosophical"]
creator_name: "Multi Merge"
<<: [*lang_base, *tone_base]
"""
        cfg = BrainConfig.from_yaml(yml)
        assert cfg.creator_name == "Multi Merge"
        assert cfg.languages == ["urdu", "hindi"]
        assert cfg.tone_guidelines == ["philosophical"]

    def test_merge_key_override_precedence(self):
        """Explicit keys in the mapping take precedence over merged anchor keys."""
        yml = """
defaults: &defaults
  creator_name: "Default Name"
  languages: ["english"]
<<: *defaults
creator_name: "Overridden Name"
"""
        cfg = BrainConfig.from_yaml(yml)
        assert cfg.creator_name == "Overridden Name"

    def test_alias_type_mismatch_raises_config_error(self):
        """Alias resolving to unexpected type for a field raises ConfigError."""
        yml = """
dict_anchor: &da
  key: value
languages: *da
"""
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            BrainConfig.from_yaml(yml)

    def test_recursive_sequence_anchor_raises_config_error(self):
        """Recursive sequence anchor (&s - *s) at root raises ConfigError."""
        yml = "&seq\n- *seq\n"
        with pytest.raises(ConfigError, match="YAML must contain a top-level mapping/dictionary"):
            BrainConfig.from_yaml(yml)


class TestAdversarialCyclicAndSelfReferences:
    """Adversarial testing of circular and recursive structures."""

    def test_cyclic_mapping_in_validated_topics_raises_config_error(self):
        """Cyclic dictionary in validated topics structure fails schema validation."""
        yml = "topics: &top\n  ai:\n    - *top\n"
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            BrainConfig.from_yaml(yml)

    def test_cyclic_mapping_in_paths_raises_config_error(self):
        """Cyclic dictionary in paths fails schema validation."""
        yml = "paths: &p\n  data_dir: *p\n"
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            BrainConfig.from_yaml(yml)

    def test_cyclic_sequence_in_handles_raises_config_error(self):
        """Self-referencing sequence inside list field fails schema validation."""
        yml = "creator_handles: &h\n  - \"@handle\"\n  - *h\n"
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            BrainConfig.from_yaml(yml)

    def test_cyclic_mapping_in_extra_fields_safe_dump(self):
        """Cyclic mapping in extra fields parses and serializes via model_dump without recursion crash."""
        yml = "node: &node\n  parent: *node\ncreator_name: 'Cyclic Extra'\n"
        cfg = BrainConfig.from_yaml(yml)
        assert cfg.creator_name == "Cyclic Extra"
        dumped = cfg.model_dump()
        assert "node" in dumped


class TestCredentialLeakagePrevention:
    """Stress tests verifying secrets and credentials are never leaked in error output."""

    def test_yaml_syntax_error_does_not_leak_env_secrets(self, monkeypatch):
        """YAML syntax errors must never leak environment variable secrets."""
        secret_token = "SUPER_SECRET_ENV_GEMINI_KEY_ABC123"
        monkeypatch.setenv("GEMINI_API_KEY", secret_token)
        monkeypatch.setenv("SECRET_TOKEN", secret_token)

        try:
            BrainConfig.from_yaml("creator_name: [unclosed mapping\n")
            pytest.fail("Expected ConfigError")
        except ConfigError as exc:
            assert secret_token not in str(exc)
            assert secret_token not in repr(exc)

    def test_yaml_validation_error_does_not_leak_env_secrets(self, monkeypatch):
        """Schema validation errors must never leak environment secrets."""
        secret_token = "QDRANT_SUPER_SECRET_KEY_XYZ789"
        monkeypatch.setenv("QDRANT_API_KEY", secret_token)

        try:
            BrainConfig.from_yaml("paths: 99999\n")
            pytest.fail("Expected ConfigError")
        except ConfigError as exc:
            assert secret_token not in str(exc)

    def test_from_yaml_does_not_inject_unexpected_secrets_into_config(self, monkeypatch):
        """BrainConfig.from_yaml must not inject environment secrets into model fields."""
        secret_token = "AIzaSyNeverInjectMeDirectly"
        monkeypatch.setenv("GEMINI_API_KEY", secret_token)

        cfg = BrainConfig.from_yaml("creator_name: 'Safe Creator'\n")
        assert secret_token not in str(cfg.model_dump())


class TestPathVsYamlStringBoundarySafety:
    """Stress testing boundary cases where strings may be confused as paths or cause OS-level crashes."""

    def test_single_line_yaml_containing_slashes_treated_as_yaml(self):
        """Valid single-line YAML containing slashes must be parsed as YAML content, not file path."""
        yml = '{"creator_name": "user/admin", "languages": ["en"]}'
        cfg = BrainConfig.from_yaml(yml)
        assert cfg.creator_name == "user/admin"

    def test_single_line_yaml_over_255_chars_without_newline(self):
        """Single-line YAML string longer than 255 chars must not crash with OSError [Errno 36]."""
        long_inline = '{"creator_name": "' + "a" * 300 + '"}'
        cfg = BrainConfig.from_yaml(long_inline)
        assert len(cfg.creator_name) == 300

    def test_load_config_path_over_255_chars_handled_safely(self):
        """load_config with a path string longer than 255 chars must not crash with OSError."""
        long_path = "a" * 300
        # When raise_if_missing is False, it should use default BrainConfig fallback
        cfg = load_config(config_path=long_path, raise_if_missing=False)
        assert isinstance(cfg, BrainConfig)

