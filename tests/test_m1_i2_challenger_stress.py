"""Adversarial challenger test suite for Milestone 1 Iteration 2.

Tests:
1. Multi-threaded os.environ mutation stress testing (20 threads) during get_gemini_keys and load_config.
2. Directory validation edge conditions: read-only directories, missing parents, symlinked output directories,
   and workspace paths containing 'brain_data' substring.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import shutil
import threading
import time
from typing import List

import pytest

from second_brain_generator.config import BrainConfig, ConfigError, load_config
from second_brain_generator.utils.dirs import (
    initialize_directories,
    validate_directories,
    DirectoryValidationReport,
)
from second_brain_generator.utils.env import (
    get_gemini_keys,
    get_qdrant_credentials,
    is_offline_mode,
    load_environment,
    MissingSecretError,
)


# ==============================================================================
# Suite 1: 20-Thread os.environ Aggressive Mutation Stress Harness
# ==============================================================================

class TestConcurrentEnvMutationStress:
    """Stress tests verifying thread-safety of get_gemini_keys and load_config under 20-thread mutations."""

    def test_20_threads_aggressive_environ_mutation_zero_key_errors(self, tmp_path):
        """20 mutator threads aggressively adding/deleting env vars while callers invoke
        get_gemini_keys and load_config. Verifies 0 KeyError exceptions occur."""
        stop_event = threading.Event()
        caller_key_errors = []
        caller_runtime_errors = []
        caller_unexpected_exceptions = []
        mutations_count = [0] * 20
        caller_ops_count = [0] * 8

        tmp_cfg = tmp_path / "stress_cfg.yaml"
        tmp_cfg.write_text("creator_name: StressUser\nlanguages: ['en']\n", encoding="utf-8")

        def safe_del_env(key: str) -> None:
            try:
                del os.environ[key]
            except KeyError:
                pass

        def mutator_worker(tid: int):
            rng = random.Random(tid * 333 + 7)
            count = 0
            while not stop_event.is_set():
                action = rng.randint(0, 5)
                if action == 0:
                    # Random variable add/delete
                    k = f"STRESS_MUT_{tid}_{rng.randint(0, 50)}"
                    os.environ[k] = f"val_{rng.random()}"
                    if rng.random() > 0.4:
                        safe_del_env(k)
                elif action == 1:
                    # Numbered Gemini keys
                    idx = rng.randint(0, 60)
                    k = f"GEMINI_API_KEY_{idx}" if idx > 0 else "GEMINI_API_KEY"
                    os.environ[k] = f"AIzaSyTestKey_{tid}_{idx}_{rng.randint(100, 999)}"
                    if rng.random() > 0.4:
                        safe_del_env(k)
                elif action == 2:
                    # Dummy or malformed keys
                    k = f"GEMINI_API_KEY_{rng.randint(0, 30)}"
                    os.environ[k] = "   your_api_key   "
                    if rng.random() > 0.5:
                        safe_del_env(k)
                elif action == 3:
                    # BRAIN_CONFIG_PATH
                    if rng.random() > 0.5:
                        os.environ["BRAIN_CONFIG_PATH"] = str(tmp_cfg)
                    else:
                        safe_del_env("BRAIN_CONFIG_PATH")
                elif action == 4:
                    # Other app environment variables
                    os.environ["QDRANT_URL"] = "http://localhost:6333"
                    os.environ["SECOND_BRAIN_OFFLINE"] = "true" if rng.random() > 0.5 else "false"
                    if rng.random() > 0.5:
                        safe_del_env("QDRANT_URL")
                        safe_del_env("SECOND_BRAIN_OFFLINE")
                elif action == 5:
                    # Mass prune of mutation vars
                    for var in list(os.environ):
                        if var.startswith("STRESS_MUT_"):
                            safe_del_env(var)
                count += 1
            mutations_count[tid] = count

        def caller_worker(cid: int):
            rng = random.Random(cid * 888 + 19)
            count = 0
            while not stop_event.is_set():
                call_type = rng.randint(0, 4)
                try:
                    if call_type == 0:
                        keys = get_gemini_keys(require=False)
                        assert isinstance(keys, list)
                    elif call_type == 1:
                        try:
                            keys = get_gemini_keys(require=True)
                            assert isinstance(keys, list)
                        except MissingSecretError:
                            pass
                    elif call_type == 2:
                        try:
                            cfg = load_config(config_path=None)
                            assert isinstance(cfg, BrainConfig)
                        except (ConfigError, FileNotFoundError):
                            pass
                    elif call_type == 3:
                        cfg = load_config(config_path=tmp_cfg)
                        assert isinstance(cfg, BrainConfig)
                    elif call_type == 4:
                        _ = get_qdrant_credentials()
                        _ = is_offline_mode()
                    count += 1
                except KeyError as e:
                    caller_key_errors.append((cid, e))
                except RuntimeError as e:
                    caller_runtime_errors.append((cid, e))
                except Exception as e:
                    caller_unexpected_exceptions.append((cid, type(e).__name__, str(e)))
            caller_ops_count[cid] = count

        mutator_threads = [threading.Thread(target=mutator_worker, args=(i,)) for i in range(20)]
        caller_threads = [threading.Thread(target=caller_worker, args=(i,)) for i in range(8)]

        for t in mutator_threads + caller_threads:
            t.daemon = True
            t.start()

        # Let the stress harness run for 3 seconds under heavy contention
        time.sleep(3.0)
        stop_event.set()

        for t in mutator_threads + caller_threads:
            t.join(timeout=3.0)

        # Cleanup created stress variables
        for k in list(os.environ):
            if k.startswith("STRESS_MUT_") or k.startswith("GEMINI_API_KEY_"):
                safe_del_env(k)
        safe_del_env("BRAIN_CONFIG_PATH")
        safe_del_env("QDRANT_URL")
        safe_del_env("SECOND_BRAIN_OFFLINE")

        total_mutations = sum(mutations_count)
        total_caller_ops = sum(caller_ops_count)

        assert total_mutations > 10_000, f"Expected >10k mutations, got {total_mutations}"
        assert total_caller_ops > 100, f"Expected >100 caller ops, got {total_caller_ops}"
        assert len(caller_key_errors) == 0, f"Encountered KeyErrors: {caller_key_errors}"
        assert len(caller_runtime_errors) == 0, f"Encountered RuntimeErrors: {caller_runtime_errors}"
        assert len(caller_unexpected_exceptions) == 0, f"Encountered unexpected exceptions: {caller_unexpected_exceptions}"


# ==============================================================================
# Suite 2: Directory Validation Edge Conditions
# ==============================================================================

class TestDirectoryValidationEdgeConditions:
    """Stress testing validate_directories against read-only dirs, missing parents, symlinks, and path edge cases."""

    # --------------------------------------------------------------------------
    # 2.1 Read-Only Directory Boundaries
    # --------------------------------------------------------------------------

    def test_read_only_brain_data_root_fails_validation(self, tmp_path):
        """Output directory brain_data without write permission is flagged as unwritable."""
        dirs = initialize_directories(base_dir=tmp_path)
        brain_root = dirs["brain_data"]
        os.chmod(brain_root, 0o555)
        try:
            if not os.access(brain_root, os.W_OK):
                report = validate_directories(base_dir=tmp_path)
                assert report.is_valid is False
                assert brain_root in report.unwritable_dirs
        finally:
            os.chmod(brain_root, 0o755)

    def test_read_only_brain_data_text_subdirectory_fails_validation(self, tmp_path):
        """Subdirectory brain_data/text without write permission is flagged as unwritable."""
        dirs = initialize_directories(base_dir=tmp_path)
        text_dir = dirs["brain_data_text"]
        os.chmod(text_dir, 0o555)
        try:
            if not os.access(text_dir, os.W_OK):
                report = validate_directories(base_dir=tmp_path)
                assert report.is_valid is False
                assert text_dir in report.unwritable_dirs
        finally:
            os.chmod(text_dir, 0o755)

    def test_read_only_input_directory_passes_validation(self, tmp_path):
        """Input data directories (e.g. data/vids) only require read access; read-only mode passes."""
        dirs = initialize_directories(base_dir=tmp_path)
        vids_dir = dirs["data_vids"]
        os.chmod(vids_dir, 0o555)
        try:
            if not os.access(vids_dir, os.W_OK) and os.access(vids_dir, os.R_OK):
                report = validate_directories(base_dir=tmp_path)
                assert report.is_valid is True
                assert vids_dir not in report.unwritable_dirs
        finally:
            os.chmod(vids_dir, 0o755)

    def test_unreadable_input_directory_fails_validation(self, tmp_path):
        """Input directory without read permission fails directory validation."""
        dirs = initialize_directories(base_dir=tmp_path)
        vids_dir = dirs["data_vids"]
        os.chmod(vids_dir, 0o000)
        try:
            if not os.access(vids_dir, os.R_OK):
                report = validate_directories(base_dir=tmp_path)
                assert report.is_valid is False
                assert vids_dir in report.unwritable_dirs
        finally:
            os.chmod(vids_dir, 0o755)

    def test_inaccessible_parent_directory_mode_000_fails_cleanly(self, tmp_path):
        """Parent directory with mode 000 catches PermissionError and reports unwritable without crashing."""
        parent = tmp_path / "inaccessible_parent"
        parent.mkdir()
        child = parent / "workspace"
        os.chmod(parent, 0o000)
        try:
            if not os.access(parent, os.R_OK) and not os.access(parent, os.X_OK):
                report = validate_directories(base_dir=child)
                assert report.is_valid is False
                assert len(report.unwritable_dirs) > 0
                assert len(report.conflicts) == 0
        finally:
            os.chmod(parent, 0o755)

    # --------------------------------------------------------------------------
    # 2.2 Missing Parents Boundaries
    # --------------------------------------------------------------------------

    def test_missing_base_directory_reports_all_missing(self, tmp_path):
        """Non-existent deeply nested base directory reports all expected paths as missing."""
        missing_base = tmp_path / "lvl1" / "lvl2" / "lvl3"
        assert not missing_base.exists()

        report = validate_directories(base_dir=missing_base)
        assert report.is_valid is False
        assert len(report.missing_dirs) == 7
        assert len(report.unwritable_dirs) == 0
        assert len(report.conflicts) == 0

    def test_parent_is_regular_file_reports_missing_cleanly(self, tmp_path):
        """If a parent path component exists as a regular file, reports missing without crashing."""
        blocker = tmp_path / "file_blocking_parent"
        blocker.write_text("blocker content", encoding="utf-8")
        nested_base = blocker / "nested_workspace"

        report = validate_directories(base_dir=nested_base)
        assert report.is_valid is False
        assert len(report.missing_dirs) == 7
        assert len(report.conflicts) == 0

    # --------------------------------------------------------------------------
    # 2.3 Symlinked Output Directory Boundaries
    # --------------------------------------------------------------------------

    def test_symlinked_brain_data_to_writable_directory_passes(self, tmp_path):
        """Symlinked brain_data directory pointing to an external writable target passes validation."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_storage = tmp_path / "external_storage" / "brain_data"
        external_storage.mkdir(parents=True)
        (external_storage / "text").mkdir()

        # Create input directories in workspace
        for sub in ["vids", "audios", "text", "md", "comments"]:
            (workspace / "data" / sub).mkdir(parents=True)

        # Symlink brain_data -> external_storage
        (workspace / "brain_data").symlink_to(external_storage)

        report = validate_directories(base_dir=workspace)
        assert report.is_valid is True
        assert len(report.missing_dirs) == 0
        assert len(report.unwritable_dirs) == 0

    def test_symlinked_brain_data_to_read_only_directory_fails(self, tmp_path):
        """Symlinked brain_data directory pointing to external read-only directory fails validation."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_storage = tmp_path / "external_storage" / "brain_data"
        external_storage.mkdir(parents=True)
        (external_storage / "text").mkdir()

        for sub in ["vids", "audios", "text", "md", "comments"]:
            (workspace / "data" / sub).mkdir(parents=True)

        (workspace / "brain_data").symlink_to(external_storage)

        os.chmod(external_storage, 0o555)
        try:
            if not os.access(external_storage, os.W_OK):
                report = validate_directories(base_dir=workspace)
                assert report.is_valid is False
                assert external_storage.resolve() in report.unwritable_dirs
        finally:
            os.chmod(external_storage, 0o755)

    def test_symlinked_brain_data_text_subfolder_read_only_fails(self, tmp_path):
        """Output subdirectory symlinked to a read-only target is flagged as unwritable."""
        dirs = initialize_directories(base_dir=tmp_path)
        shutil.rmtree(dirs["brain_data_text"])

        external_text = tmp_path / "external_text"
        external_text.mkdir()
        dirs["brain_data_text"].symlink_to(external_text)

        os.chmod(external_text, 0o555)
        try:
            if not os.access(external_text, os.W_OK):
                report = validate_directories(base_dir=tmp_path)
                assert report.is_valid is False
                assert (
                    dirs["brain_data_text"] in report.unwritable_dirs
                    or external_text.resolve() in report.unwritable_dirs
                )
        finally:
            os.chmod(external_text, 0o755)

    def test_broken_symlink_as_brain_data_fails_validation(self, tmp_path):
        """Broken symlink for brain_data is reported as missing."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for sub in ["vids", "audios", "text", "md", "comments"]:
            (workspace / "data" / sub).mkdir(parents=True)

        (workspace / "brain_data").symlink_to(tmp_path / "nonexistent_target")

        report = validate_directories(base_dir=workspace)
        assert report.is_valid is False
        assert any("nonexistent_target" in str(p) for p in report.missing_dirs)

    def test_symlink_to_file_as_brain_data_reports_conflict(self, tmp_path):
        """Symlink pointing to a regular file instead of a directory reports a conflict."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for sub in ["vids", "audios", "text", "md", "comments"]:
            (workspace / "data" / sub).mkdir(parents=True)

        target_file = tmp_path / "regular_file.txt"
        target_file.write_text("I am not a directory", encoding="utf-8")
        (workspace / "brain_data").symlink_to(target_file)

        report = validate_directories(base_dir=workspace)
        assert report.is_valid is False
        assert target_file.resolve() in report.conflicts

    # --------------------------------------------------------------------------
    # 2.4 Substring Collision Vulnerability Challenge
    # --------------------------------------------------------------------------

    def test_workspace_path_containing_brain_data_substring_with_readonly_input(self, tmp_path):
        """Adversarial challenge: when workspace path contains 'brain_data' (e.g. /home/user/my_brain_data_repo),
        read-only input directories (e.g. data/vids) must NOT be falsely marked as unwritable output directories."""
        workspace = tmp_path / "project_brain_data_pipeline"
        workspace.mkdir()
        initialize_directories(workspace)

        vids_dir = workspace / "data" / "vids"
        os.chmod(vids_dir, 0o555)
        try:
            if not os.access(vids_dir, os.W_OK) and os.access(vids_dir, os.R_OK):
                report = validate_directories(workspace)
                # An input directory should never be required to have write permissions!
                # If ('brain_data' in str(p)) is used loosely, this assertion will fail.
                assert vids_dir not in report.unwritable_dirs, (
                    f"False positive: Input directory {vids_dir} was classified as an unwritable "
                    f"output directory because the workspace path contains 'brain_data'!"
                )
                assert report.is_valid is True
        finally:
            os.chmod(vids_dir, 0o755)
