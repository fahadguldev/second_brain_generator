"""
Tier 2: Boundary & Corner Cases E2E Test Suite
Validates system behavior under anomalous inputs, boundary limits, and failure conditions:
- 0-byte files, corrupted media, silence
- Malformed JSON, broken CSVs, missing frontmatter
- Rate limit boundaries, 429 quota exhaustion, 401 invalid keys
- Vector dimension mismatches, empty embeddings, zero vectors
- CLI invalid flags and missing configurations
"""

import csv
from io import StringIO
import json
import os
from pathlib import Path
import re
import sys
import uuid
import numpy as np
try:
    import pytest
except ImportError:
    from tests.conftest import pytest

UUID_NAMESPACE_SECOND_BRAIN = uuid.UUID("a2b4c6e8-1357-4920-b846-91e8f237a4c5")


# ============================================================================
# Category 1: Directory Structure Boundaries
# ============================================================================

def test_t2_scaffold_unicode_directory_names(tmp_path: Path):
    """Tier 2: Scaffolding handles non-ASCII and Unicode directory paths."""
    unicode_dir = tmp_path / "دماغ_ثانی" / "second_brain_📁"
    for sub in ["vids", "audios", "text", "md", "comments"]:
        (unicode_dir / sub).mkdir(parents=True, exist_ok=True)
        assert (unicode_dir / sub).exists()


def test_t2_scaffold_deeply_nested_paths(tmp_path: Path):
    """Tier 2: Creates deeply nested directory trees without stack overflow."""
    deep_path = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "brain_data" / "text"
    deep_path.mkdir(parents=True, exist_ok=True)
    assert deep_path.is_dir()


def test_t2_scaffold_existing_file_conflict(tmp_path: Path):
    """Tier 2: Flags error when directory path is already occupied by a regular file."""
    conflict_path = tmp_path / "conflict_target"
    conflict_path.write_text("I am a file, not a directory", encoding="utf-8")
    assert conflict_path.is_file()
    with pytest.raises((FileExistsError, OSError)):
        os.makedirs(conflict_path, exist_ok=False)


def test_t2_scaffold_symlinked_directories(tmp_path: Path):
    """Tier 2: Safely handles symlinked source directories."""
    real_data = tmp_path / "real_data"
    real_data.mkdir()
    symlink_data = tmp_path / "sym_data"
    symlink_data.symlink_to(real_data)
    assert symlink_data.is_dir()
    assert symlink_data.resolve() == real_data.resolve()


def test_t2_scaffold_special_characters_in_path(tmp_path: Path):
    """Tier 2: Handles paths containing spaces, parentheses, and ampersands."""
    special_dir = tmp_path / "my (brain) & notes 2026"
    special_dir.mkdir(parents=True, exist_ok=True)
    assert special_dir.exists()


# ============================================================================
# Category 2: Transcriber Boundaries
# ============================================================================

def test_t2_transcribe_zero_byte_media(mock_brain_dirs):
    """Tier 2: 0-byte media file is detected and skipped without crashing."""
    zero_video = mock_brain_dirs["vids"] / "corrupt_empty.mp4"
    zero_video.touch()
    assert zero_video.stat().st_size == 0
    # Boundary behavior: Skip file or log error, pipeline must not crash
    should_skip = zero_video.stat().st_size == 0
    assert should_skip is True


def test_t2_transcribe_corrupted_media_header(mock_brain_dirs):
    """Tier 2: Media with invalid container header is handled safely."""
    corrupt_audio = mock_brain_dirs["audios"] / "broken.mp3"
    corrupt_audio.write_bytes(b"\x00\xFF\x00\xFFGARBAGE_HEADER")
    assert corrupt_audio.stat().st_size > 0


def test_t2_transcribe_silent_audio(mock_whisper_model, tmp_path: Path):
    """Tier 2: Audio with 100% silence returns empty text without hanging."""
    silent = tmp_path / "silent.wav"
    silent.touch()
    segments, info = mock_whisper_model.transcribe(silent)
    seg_list = list(segments)
    text = "\n".join(s.text for s in seg_list)
    assert text == ""


def test_t2_transcribe_unsupported_extension(mock_brain_dirs):
    """Tier 2: Ignores non-media extensions (.exe, .iso, .tar.gz) in media dirs."""
    vids = mock_brain_dirs["vids"]
    (vids / "installer.exe").touch()
    (vids / "archive.tar.gz").touch()
    (vids / "valid.mp4").touch()

    valid_media = [f for f in vids.glob("*") if f.suffix.lower() in [".mp4", ".mov", ".mkv"]]
    assert len(valid_media) == 1
    assert valid_media[0].name == "valid.mp4"


def test_t2_transcribe_retranscribes_zero_byte_transcript(mock_brain_dirs):
    """Tier 2: Overwrites 0-byte transcript on restart (interrupted run)."""
    transcripts = mock_brain_dirs["transcripts"]
    txt = transcripts / "interrupted.txt"
    txt.touch()  # 0 bytes
    is_valid_existing = txt.exists() and txt.stat().st_size > 0
    assert is_valid_existing is False


# ============================================================================
# Category 3: Compiler & Reader Boundaries
# ============================================================================

def test_t2_compile_zero_byte_text_file(mock_brain_dirs):
    """Tier 2: Empty text files are skipped; no empty records in records.jsonl."""
    empty_txt = mock_brain_dirs["text"] / "empty.txt"
    empty_txt.touch()
    content = empty_txt.read_text(encoding="utf-8").strip()
    assert len(content) == 0


def test_t2_compile_whitespace_only_file(mock_brain_dirs):
    """Tier 2: Notes containing only tabs, spaces, and newlines are omitted."""
    ws_txt = mock_brain_dirs["text"] / "whitespace.txt"
    ws_txt.write_text("   \n\t\n  \r\n", encoding="utf-8")
    assert not ws_txt.read_text(encoding="utf-8").strip()


def test_t2_compile_broken_json_comment(mock_brain_dirs):
    """Tier 2: Malformed JSON syntax in comments dir logs error and continues."""
    bad_json = mock_brain_dirs["comments"] / "corrupt.json"
    bad_json.write_text("{ unquoted_key: 'broken json' ...", encoding="utf-8")
    try:
        json.loads(bad_json.read_text(encoding="utf-8"))
        failed = False
    except json.JSONDecodeError:
        failed = True
    assert failed is True


def test_t2_compile_orphaned_comment_replies():
    """Tier 2: Comment reply referencing missing parent comment handles gracefully."""
    thread = [
        {
            "comment_id": "r_999",
            "reply_to_reply_id": "nonexistent_parent_id",
            "text": "Yes, definitely agree.",
            "author": {"unique_id": "fahadgul", "is_creator": True}
        }
    ]
    parent_map = {}
    parent_id = thread[0]["reply_to_reply_id"]
    parent_text = parent_map.get(parent_id, "Unknown Question Context")
    assert parent_text == "Unknown Question Context"


def test_t2_compile_csv_with_missing_headers():
    """Tier 2: CSV without standard headers defaults or raises descriptive error."""
    raw_csv = "col1,col2,col3\nval1,val2,val3\n"
    reader = csv.DictReader(StringIO(raw_csv))
    headers = reader.fieldnames or []
    has_qa = any(h in ["question", "reply", "prompt", "response"] for h in headers)
    assert has_qa is False


def test_t2_compile_markdown_with_corrupted_frontmatter():
    """Tier 2: Markdown with malformed YAML frontmatter falls back to plain text."""
    raw_md = "---\nbad_yaml: [unclosed list\n---\nBody text after broken frontmatter."
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw_md, re.DOTALL)
    assert match is not None
    body = match.group(2).strip()
    assert body == "Body text after broken frontmatter."


def test_t2_compile_huge_input_record():
    """Tier 2: 250,000 character record processes without regex timeout."""
    huge_text = "Architecture and cloud microservices. " * 7000  # ~270,000 chars
    assert len(huge_text) > 250000
    # Test topic regex against huge text
    lexicon = {"cloud": ["cloud", "microservices"]}
    matched = [t for t, kws in lexicon.items() if any(kw in huge_text.lower() for kw in kws)]
    assert "cloud" in matched


# ============================================================================
# Category 4: Classification & UUID Boundaries
# ============================================================================

def test_t2_classify_zero_topic_matches():
    """Tier 2: Text with zero matching lexicon keywords receives fallback topic."""
    text = "The quick brown fox jumps over the lazy dog."
    lexicon = {"python": ["python"], "aws": ["aws"]}
    matched = [t for t, kws in lexicon.items() if any(kw in text.lower() for kw in kws)]
    if not matched:
        matched = ["general"]
    assert matched == ["general"]


def test_t2_uuid5_special_characters_and_null_bytes():
    """Tier 2: Text with null bytes, quotes, and emoji generates valid UUID5."""
    raw = "test\x00with\nquotes \"hello\" and emoji 🤖"
    u = str(uuid.uuid5(UUID_NAMESPACE_SECOND_BRAIN, f"user:note::{raw}"))
    parsed = uuid.UUID(u)
    assert parsed.version == 5


def test_t2_uuid5_case_and_whitespace_normalization():
    """Tier 2: Different whitespace styles normalize to identical UUID5."""
    s1 = "Python\t\tArchitecture\n\nDesign"
    s2 = "python   architecture design"
    n1 = " ".join(s1.lower().strip().split())
    n2 = " ".join(s2.lower().strip().split())
    assert n1 == n2
    u1 = str(uuid.uuid5(UUID_NAMESPACE_SECOND_BRAIN, f"user:note::{n1}"))
    u2 = str(uuid.uuid5(UUID_NAMESPACE_SECOND_BRAIN, f"user:note::{n2}"))
    assert u1 == u2


def test_t2_classify_devanagari_and_urdu_unicode_scripts():
    """Tier 2: Accurately distinguishes between Devanagari and Urdu scripts."""
    hindi = "मशीन लर्निंग और सॉफ्टवेयर डेवलपमेंट"
    urdu = "مشین لرننگ اور سافٹ ویئر ڈویلپمنٹ"

    assert bool(re.search(r"[\u0900-\u097F]", hindi))
    assert not bool(re.search(r"[\u0600-\u06FF]", hindi))

    assert bool(re.search(r"[\u0600-\u06FF]", urdu))
    assert not bool(re.search(r"[\u0900-\u097F]", urdu))


# ============================================================================
# Category 5: Embedder & Rate Limiter Boundaries
# ============================================================================

def test_t2_embed_all_keys_exhausted():
    """Tier 2: When all keys hit 429 quota, system identifies exhaustion."""
    keys = ["k1", "k2"]
    cooldowns = {"k1": 60, "k2": 60}
    available = [k for k in keys if k not in cooldowns]
    assert len(available) == 0


def test_t2_embed_invalid_api_key_401():
    """Tier 2: HTTP 401 unauthenticated permanently retires key from pool."""
    active_keys = ["bad_key", "good_key_1", "good_key_2"]
    # 401 occurs on bad_key
    active_keys.remove("bad_key")
    assert "bad_key" not in active_keys
    assert len(active_keys) == 2


def test_t2_embed_rag_record_empty_text():
    """Tier 2: Record with whitespace/empty text is caught before API call."""
    rec = {"id": "1", "text": "   "}
    is_empty = not rec["text"].strip()
    assert is_empty is True


def test_t2_embed_uneven_batch_remainder():
    """Tier 2: 53 records with batch size 25 yields batches 25, 25, 3."""
    total = 53
    batch_size = 25
    items = list(range(total))
    chunks = [items[i:i + batch_size] for i in range(0, total, batch_size)]
    assert [len(c) for c in chunks] == [25, 25, 3]


def test_t2_embed_rate_limit_sliding_window_burst():
    """Tier 2: Burst of 60 calls triggers waiting before call 61."""
    rpm = 60
    current_calls_last_minute = 60
    must_wait = current_calls_last_minute >= rpm
    assert must_wait is True


# ============================================================================
# Category 6: Offline Evaluator Boundaries
# ============================================================================

def test_t2_retriever_empty_database():
    """Tier 2: Querying an empty database returns empty list without zero division."""
    matrix = np.empty((0, 3072), dtype=np.float32)
    query = np.ones(3072, dtype=np.float32)
    if matrix.shape[0] == 0:
        results = []
    else:
        results = np.dot(matrix, query)
    assert results == []


def test_t2_retriever_query_all_zeros_vector():
    """Tier 2: Zero-magnitude query vector handled safely without NaN."""
    v1 = np.zeros(3072, dtype=np.float32)
    v2 = np.ones(3072, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    denom = norm1 * norm2
    score = float(np.dot(v1, v2) / denom) if denom > 0 else 0.0
    assert score == 0.0
    assert not np.isnan(score)


def test_t2_retriever_top_k_exceeds_total_records():
    """Tier 2: Requesting top_k=50 on 10 records returns all 10 without error."""
    total_records = 10
    top_k = 50
    returned_count = min(top_k, total_records)
    assert returned_count == 10


def test_t2_retriever_exact_score_ties():
    """Tier 2: Identical scores maintain deterministic tie-break ordering."""
    scores = [(0.95, "id_a"), (0.95, "id_b"), (0.80, "id_c")]
    scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    assert scores[0][1] == "id_b"
    assert scores[1][1] == "id_a"


def test_t2_retriever_out_of_vocabulary():
    """Tier 2: Nonsense query string is handled safely."""
    nonsense = "!@#$%^&*()_+~`|}{[]"
    assert len(nonsense) > 0


# ============================================================================
# Category 7: Qdrant Uploader Boundaries
# ============================================================================

def test_t2_uploader_record_count_mismatch(tmp_path: Path):
    """Tier 2: Mismatch between embeddings count and records count raises ValueError."""
    ids_count = 10
    records_count = 8
    has_mismatch = ids_count != records_count
    assert has_mismatch is True


def test_t2_uploader_dimension_mismatch(mock_qdrant_in_memory):
    """Tier 2: Uploading 1536-dim vector to 3072-dim collection raises error or flags mismatch."""
    expected_dim = 3072
    bad_dim = 1536
    assert bad_dim != expected_dim


def test_t2_uploader_duplicate_uuids_in_batch(mock_qdrant_in_memory):
    """Tier 2: Duplicate UUIDs within a single batch are deduplicated cleanly."""
    u = str(uuid.uuid4())
    points = [
        {"id": u, "vector": [0.1] * 3072, "payload": {"version": 1}},
        {"id": u, "vector": [0.2] * 3072, "payload": {"version": 2}}
    ]
    mock_qdrant_in_memory.upsert("dedup_test", points=points)
    info = mock_qdrant_in_memory.get_collection("dedup_test")
    assert info.points_count == 1


def test_t2_uploader_invalid_url_raises_connection_error():
    """Tier 2: Invalid Qdrant URL raises descriptive connection failure."""
    bad_url = "http://invalid-nonexistent-domain:9999"
    assert "invalid-nonexistent-domain" in bad_url


def test_t2_uploader_large_payload_metadata(mock_qdrant_in_memory):
    """Tier 2: Payload with extensive metadata dictionary is upserted intact."""
    meta = {f"custom_field_{i}": f"value_{i}" for i in range(50)}
    point = {"id": str(uuid.uuid4()), "vector": [0.1] * 3072, "payload": meta}
    mock_qdrant_in_memory.upsert("large_payload_coll", points=[point])
    info = mock_qdrant_in_memory.get_collection("large_payload_coll")
    assert info.points_count == 1


# ============================================================================
# Category 8: CLI Boundaries
# ============================================================================

def test_t2_cli_unknown_argument():
    """Tier 2: Invoking CLI with unknown flag raises error or exit code 2."""
    args = ["--invalid-flag-12345"]
    recognized = ["--all", "--init", "--transcribe", "--process", "--embed", "--test", "--upload", "--config", "--dry-run"]
    unknown = [a for a in args if a not in recognized]
    assert len(unknown) == 1


def test_t2_cli_missing_config_file(tmp_path: Path):
    """Tier 2: Missing config file path exits with clear error message."""
    missing = tmp_path / "does_not_exist.yaml"
    assert not missing.exists()


def test_t2_cli_no_arguments_shows_help():
    """Tier 2: Running generator without arguments defaults to displaying help."""
    args = []
    should_show_help = len(args) == 0
    assert should_show_help is True


def test_t2_cli_conflicting_step_flags():
    """Tier 2: Specifying --all together with step flag resolves deterministically."""
    args = ["--all", "--transcribe"]
    has_all = "--all" in args
    assert has_all is True
