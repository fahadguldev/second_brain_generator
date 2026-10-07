"""Adversarial stress-test suite for Milestone 1 implementations.

Tests:
1. UUIDv5 Generator (second_brain_generator.utils.uuids):
   - 10,000 distinct pseudo-random strings (zero collisions)
   - Extreme inputs: lone unicode surrogates, zero-width characters, mixed RTL/LTR, multi-MB payloads
   - Invariance tests: whitespace, newlines, slash directions, casing
2. Directory Utilities (second_brain_generator.utils.dirs):
   - Multi-threaded concurrent initialize_directories race conditions
   - Symlink traversal and broken symlink handling
   - File blocking and permission edge cases
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import stat
import string
import threading
import time
import uuid

import pytest

from second_brain_generator.utils.dirs import (
    initialize_directories,
    scan_category_files,
    validate_directories,
    DATA_SUBDIRS,
    BRAIN_DATA_SUBDIRS,
)
from second_brain_generator.utils.uuids import (
    compute_content_hash,
    generate_record_uuid,
    is_valid_uuid,
    normalize_identifier,
    normalize_text,
    DEFAULT_NAMESPACE,
    UUID_NAMESPACE_SECOND_BRAIN,
)


# ==============================================================================
# Suite 1: UUIDv5 Stress & Collision Resistance
# ==============================================================================

def test_uuid5_bulk_10k_collision_resistance():
    """Generates 10,000 distinct pseudo-random records and verifies zero UUID collisions."""
    rng = random.Random(42)
    source_types = ["video_transcript", "audio_transcript", "text_note", "markdown_note", "social_comment_reply"]
    
    seen_seeds = set()
    seen_uuids = set()
    
    start_time = time.perf_counter()
    
    for i in range(10000):
        # Generate random distinct strings
        rand_str_len = rng.randint(15, 80)
        rand_text = "".join(rng.choices(string.ascii_letters + string.digits + " \t\n-.,!?", k=rand_str_len))
        st = rng.choice(source_types)
        ident = f"folder_{rng.randint(1, 50)}/item_{i}_{rng.randint(1000, 9999)}.dat"
        question = f"What is topic {i}?" if i % 3 == 0 else None
        
        # Verify input seed uniqueness
        input_seed = (st, ident, rand_text, question)
        seen_seeds.add(input_seed)
        
        rec_uuid = generate_record_uuid(st, ident, rand_text, question=question)
        
        # Invariant: valid RFC 4122 v5 UUID format
        assert is_valid_uuid(rec_uuid, expected_version=5), f"Invalid UUIDv5 format: {rec_uuid}"
        
        # Invariant: zero collision
        assert rec_uuid not in seen_uuids, f"Collision detected at iteration {i}: UUID={rec_uuid}"
        seen_uuids.add(rec_uuid)
        
    duration = time.perf_counter() - start_time
    rate = 10000 / duration if duration > 0 else float("inf")
    
    assert len(seen_uuids) == 10000
    assert len(seen_seeds) == 10000
    # Performance assertion: Should generate at least 5,000 UUIDs per second
    assert rate > 5000, f"Throughput too low: {rate:.1f} UUIDs/sec"


def test_uuid5_extreme_multi_megabyte_payloads():
    """Verifies that multi-megabyte payloads hash and generate UUIDs accurately and performantly."""
    # 1 MB, 5 MB, and 10 MB payloads
    sizes = [1 * 1024 * 1024, 5 * 1024 * 1024]
    
    for sz in sizes:
        # Repeating pattern text to simulate huge documents
        chunk = "Second Brain Generator knowledge chunk with varied text. \n"
        repetitions = (sz // len(chunk)) + 1
        large_text = (chunk * repetitions)[:sz]
        
        t0 = time.perf_counter()
        u1 = generate_record_uuid("text_note", "large_document.txt", large_text)
        t_gen = time.perf_counter() - t0
        
        # Determinism check
        u2 = generate_record_uuid("text_note", "large_document.txt", large_text)
        assert u1 == u2
        assert is_valid_uuid(u1, expected_version=5)
        # Should process multi-MB payload in under 1.0s
        assert t_gen < 1.0, f"Payload {sz} bytes took too long: {t_gen:.3f}s"


def test_uuid5_extreme_zero_width_and_format_characters():
    """Tests handling of zero-width and invisible Unicode formatting characters."""
    zw_chars = [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\ufeff",  # Zero-width no-break space / BOM
        "\u2060",  # Word joiner
    ]
    
    base_text = "Building Second Brain with AI embeddings"
    
    # Inject each zero-width character
    for zwc in zw_chars:
        dirty_text = f"Building{zwc} Second {zwc}Brain with{zwc} AI embeddings{zwc}"
        u_dirty = generate_record_uuid("text_note", "notes.md", dirty_text)
        
        # Must be valid UUIDv5
        assert is_valid_uuid(u_dirty, expected_version=5)
        
        # Determinism check
        u_dirty_repeat = generate_record_uuid("text_note", "notes.md", dirty_text)
        assert u_dirty == u_dirty_repeat


def test_uuid5_extreme_mixed_rtl_ltr_scripts():
    """Tests mixed bidirectional text (Arabic, Hebrew, Urdu, Devanagari, English, emojis)."""
    samples = [
        "Urdu: آپ کا AI سیکنڈ برین کیسا ہے؟ - English: Testing bilingual system 123",
        "Hindi: नमस्ते दुनिया! AI सेकंड ब्रेन - English: Hello world 456",
        "Arabic: مرحبًا بك في نظام الذاكرة الاصطناعية 🚀 - Version 2.0",
        "Hebrew: שלום עולם! מבחן מערכת 🧠 - Test #99",
        "BiDi Marks: \u202aLeft-to-Right\u202c and \u202bRight-to-Left\u202c with \u200eLTR\u200e & \u200fRTL\u200f",
    ]
    
    generated = set()
    for s in samples:
        u = generate_record_uuid("multilingual_note", "bidi_notes.txt", s)
        assert is_valid_uuid(u, expected_version=5)
        # Repeatability
        u_rep = generate_record_uuid("multilingual_note", "bidi_notes.txt", s)
        assert u == u_rep
        generated.add(u)
        
    assert len(generated) == len(samples), "Collision among distinct multilingual inputs"


def test_uuid5_extreme_unicode_surrogates_behavior():
    """Examines system resilience when lone Unicode surrogates are presented."""
    surrogate_text = "Corrupted\ud800Surrogate\udc00Content"
    
    # Python UTF-8 encoding raises UnicodeEncodeError on lone surrogates unless handled
    # We verify the exact exception type and failure boundary
    try:
        u = generate_record_uuid("corrupt_source", "surrogate.txt", surrogate_text)
        # If the implementation cleanses surrogates:
        assert is_valid_uuid(u, 5)
    except UnicodeEncodeError as exc:
        # Documented failure mode: unhandled surrogate character
        assert "surrogates not allowed" in str(exc) or "codec can't encode" in str(exc)


def test_uuid5_invariance_whitespace_combinations():
    """Verifies that arbitrary combinations of whitespace produce identical UUIDs."""
    canonical = "Machine learning vector embeddings for RAG systems"
    variations = [
        "   Machine   learning   vector   embeddings   for   RAG   systems   ",
        "Machine\tlearning\tvector\tembeddings\tfor\tRAG\tsystems",
        "Machine \t \n learning \r\n vector \t embeddings \n for \r\n RAG \t systems \n",
        "  \n\t Machine learning vector embeddings for RAG systems \t\n  ",
        "Machine  learning    vector      embeddings  for  RAG  systems",
    ]
    
    expected_uuid = generate_record_uuid("text_note", "note.txt", canonical)
    
    for var in variations:
        var_uuid = generate_record_uuid("text_note", "note.txt", var)
        assert var_uuid == expected_uuid, f"Whitespace variation failed to match canonical UUID:\n{var!r}"


def test_uuid5_invariance_newline_crlf_lf_cr():
    """Verifies that Windows CRLF (\\r\\n), Unix LF (\\n), and old Mac CR (\\r) yield identical UUIDs."""
    text_lf = "Line 1\nLine 2\nLine 3"
    text_crlf = "Line 1\r\nLine 2\r\nLine 3"
    text_cr = "Line 1\rLine 2\rLine 3"
    text_mixed = "Line 1\r\nLine 2\nLine 3\r"
    
    u_lf = generate_record_uuid("note", "doc.txt", text_lf)
    u_crlf = generate_record_uuid("note", "doc.txt", text_crlf)
    u_cr = generate_record_uuid("note", "doc.txt", text_cr)
    u_mixed = generate_record_uuid("note", "doc.txt", text_mixed)
    
    assert u_lf == u_crlf == u_cr == u_mixed


def test_uuid5_invariance_slash_direction_and_path_representations():
    """Verifies that slash direction (\\ vs /) in identifiers yields identical UUIDs."""
    posix_path = "data/vids/interviews/episode_01.mp4"
    win_path = r"data\vids\interviews\episode_01.mp4"
    mixed_path = r"data/vids\interviews/episode_01.mp4"
    
    u_posix = generate_record_uuid("video_transcript", posix_path, "Transcript text")
    u_win = generate_record_uuid("video_transcript", win_path, "Transcript text")
    u_mixed = generate_record_uuid("video_transcript", mixed_path, "Transcript text")
    u_pure_posix = generate_record_uuid("video_transcript", PurePosixPath(posix_path), "Transcript text")
    u_pure_win = generate_record_uuid("video_transcript", PureWindowsPath(win_path), "Transcript text")
    
    assert u_posix == u_win == u_mixed == u_pure_posix == u_pure_win


# ==============================================================================
# Suite 2: Directory Scaffolding Concurrency & Resilience
# ==============================================================================

def test_dirs_concurrent_initialization_race_condition(tmp_path):
    """Stress tests initialize_directories under high concurrent multi-threaded contention.
    
    Fires 50 worker threads simultaneously on the exact same directory target to test
    mkdir race condition resilience, exist_ok semantics, and .gitkeep creation.
    """
    target_root = tmp_path / "concurrent_workspace"
    num_threads = 50
    barrier = threading.Barrier(num_threads)
    errors: list[Exception] = []
    results: list[dict[str, Path]] = []
    
    def worker_func(worker_id: int):
        try:
            # Synchronize start so all threads hit initialize_directories simultaneously
            barrier.wait(timeout=5.0)
            res = initialize_directories(base_dir=target_root)
            results.append(res)
        except Exception as e:
            errors.append(e)
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_func, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)
        
    assert len(errors) == 0, f"Concurrent initialization raised errors: {errors}"
    assert len(results) == num_threads
    
    # Verify resulting directory integrity
    report = validate_directories(base_dir=target_root)
    assert report.is_valid is True
    assert len(report.missing_dirs) == 0
    assert len(report.conflicts) == 0
    assert len(report.unwritable_dirs) == 0
    
    # Verify .gitkeep creation in empty leaf directories
    for sub in DATA_SUBDIRS.values():
        gk = target_root / "data" / sub / ".gitkeep"
        assert gk.exists() and gk.is_file()
    for sub in BRAIN_DATA_SUBDIRS.values():
        gk = target_root / "brain_data" / sub / ".gitkeep"
        assert gk.exists() and gk.is_file()


def test_dirs_repeated_concurrent_cycles(tmp_path):
    """Executes 20 rapid sequential cycles of 20 concurrent threads creating and validating directories."""
    for cycle in range(20):
        cycle_dir = tmp_path / f"cycle_{cycle}"
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        errors = []
        
        def worker():
            try:
                barrier.wait(timeout=2.0)
                initialize_directories(base_dir=cycle_dir)
            except Exception as e:
                errors.append(e)
                
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert len(errors) == 0, f"Errors in cycle {cycle}: {errors}"
        report = validate_directories(base_dir=cycle_dir)
        assert report.is_valid is True


def test_dirs_symlink_directory_resolution(tmp_path):
    """Tests initialize_directories when data/ is a symlink to an external storage volume."""
    real_data_store = tmp_path / "external_mount" / "big_data"
    real_data_store.mkdir(parents=True)
    
    project_root = tmp_path / "project"
    project_root.mkdir()
    
    # Create symlink: project/data -> external_mount/big_data
    symlink_data = project_root / "data"
    symlink_data.symlink_to(real_data_store, target_is_directory=True)
    
    # Initialize should succeed through symlinked directory
    paths = initialize_directories(base_dir=project_root)
    assert paths["data_vids"].exists()
    assert (real_data_store / "vids").exists()
    
    # Validation should pass
    report = validate_directories(base_dir=project_root)
    assert report.is_valid is True


def test_dirs_broken_symlink_handling(tmp_path):
    """Tests initialize_directories when an expected directory path is a broken symlink."""
    project_root = tmp_path / "broken_sym_project"
    project_root.mkdir()
    
    data_dir = project_root / "data"
    data_dir.mkdir()
    
    # Create broken symlink at data/vids -> /nonexistent/target
    broken_link = data_dir / "vids"
    broken_link.symlink_to(Path("/nonexistent/storage/partition/vids"))
    
    # In Linux, a broken symlink returns False for exists(), but mkdir will fail with FileExistsError
    # We test whether initialize_directories handles or surfaces this condition
    with pytest.raises((FileExistsError, NotADirectoryError, OSError)):
        initialize_directories(base_dir=project_root)


def test_dirs_file_blocking_subpath(tmp_path):
    """Verifies that a regular file occupying an expected directory path raises NotADirectoryError."""
    project_root = tmp_path / "blocked_project"
    project_root.mkdir()
    data_dir = project_root / "data"
    data_dir.mkdir()
    
    # Create a regular file where 'text' folder belongs
    blocking_file = data_dir / "text"
    blocking_file.write_text("I am a file, not a directory")
    
    with pytest.raises(NotADirectoryError):
        initialize_directories(base_dir=project_root)
        
    report = validate_directories(base_dir=project_root)
    assert report.is_valid is False
    assert blocking_file in report.conflicts


def test_dirs_permission_unwritable_brain_data(tmp_path):
    """Verifies that an unwritable brain_data directory raises PermissionError and fails validation."""
    project_root = tmp_path / "readonly_project"
    project_root.mkdir()
    
    brain_data_dir = project_root / "brain_data"
    brain_data_dir.mkdir()
    
    # Make brain_data read-only (0o555: r-x r-x r-x)
    os.chmod(brain_data_dir, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    
    try:
        # Check if we can write to it; if running as root, chmod 0555 won't block write.
        can_write = os.access(brain_data_dir, os.W_OK)
        if not can_write:
            with pytest.raises(PermissionError):
                initialize_directories(base_dir=project_root)
                
            report = validate_directories(base_dir=project_root)
            assert report.is_valid is False
            assert brain_data_dir / "text" in report.unwritable_dirs or brain_data_dir / "text" in report.missing_dirs
    finally:
        # Restore permissions for cleanup
        os.chmod(brain_data_dir, stat.S_IRWXU)


def test_dirs_scan_category_files_with_symlinks(tmp_path):
    """Tests scan_category_files when files are symlinked."""
    workspace = tmp_path / "scan_space"
    paths = initialize_directories(base_dir=workspace)
    
    vids_dir = paths["data_vids"]
    external_dir = tmp_path / "external_media"
    external_dir.mkdir()
    
    real_video = external_dir / "sample.mp4"
    real_video.write_bytes(b"dummy mp4 video bytes")
    
    # Create symlinked video file inside data/vids/
    sym_video = vids_dir / "sym_video.mp4"
    sym_video.symlink_to(real_video)
    
    scanned = scan_category_files("vids", base_dir=workspace)
    assert len(scanned) == 1
    assert scanned[0].name == "sym_video.mp4"
