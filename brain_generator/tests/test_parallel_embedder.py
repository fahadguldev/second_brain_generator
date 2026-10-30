"""Comprehensive test suite for token-aware multi-key rotation and parallel embedding."""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pytest

from second_brain_generator.embedder.key_manager import KeyManager, estimate_tokens
from second_brain_generator.embedder.embedder import GeminiEmbedder, _deterministic_mock_vector


# ============================================================================
# Token Estimation Tests
# ============================================================================

def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0


def test_estimate_tokens_short_english():
    text = "hello world"
    tokens = estimate_tokens(text)
    assert tokens >= 2
    assert tokens <= 10


def test_estimate_tokens_multilingual_hindi_urdu():
    text = "Travelling to Taiwan with Research Samples | Ask Ganjiswag #153"
    tokens = estimate_tokens(text)
    assert tokens > 0

    hindi_text = "ठीक है। पिछले एपिसोड में लोग अपनी यात्राओं और उपलब्धियों को भी साझा कर सकते हैं।"
    h_tokens = estimate_tokens(hindi_text)
    assert h_tokens > len(hindi_text.split())  # Should provide safe headroom


# ============================================================================
# KeyManager TPM & Rate Limiting Tests
# ============================================================================

def test_key_manager_initialization():
    km = KeyManager(api_keys=["key1", "key2"], tpm_limit=30000, rpm_limit=60)
    assert km.api_keys == ["key1", "key2"]
    assert km.tpm_limit == 30000
    assert km.rpm_limit == 60


def test_key_manager_round_robin():
    km = KeyManager(api_keys=["k1", "k2"], tpm_limit=50000)
    k_first = km.get_key(estimated_tokens=100)
    k_second = km.get_key(estimated_tokens=100)
    assert k_first == "k1"
    assert k_second == "k2"


def test_key_manager_blocks_when_tpm_exceeded():
    # Only 1 key with 10,000 TPM limit
    km = KeyManager(api_keys=["k1"], tpm_limit=10000)

    # First call uses 8,000 tokens -> should succeed
    k = km.get_key(estimated_tokens=8000)
    assert k == "k1"

    # Second call needs 3,000 tokens (8000 + 3000 = 11000 > 10000) -> should be blocked
    k_blocked = km.get_key(estimated_tokens=3000)
    assert k_blocked is None


def test_key_manager_rotates_to_second_key_when_first_tpm_exceeded():
    # Two keys with 10,000 TPM limit each
    km = KeyManager(api_keys=["k1", "k2"], tpm_limit=10000)

    # Key 1 consumes 9,000 tokens
    k = km.get_key(estimated_tokens=9000)
    assert k == "k1"

    # Next request needs 4,000 tokens. k2 should be chosen because k1 would exceed 10,000
    k2 = km.get_key(estimated_tokens=4000)
    assert k2 == "k2"


def test_key_manager_sliding_window_expiration(monkeypatch):
    km = KeyManager(api_keys=["k1"], tpm_limit=10000)

    current_time = 1000.0
    monkeypatch.setattr(time, "time", lambda: current_time)

    # Consume 9,000 tokens at t=1000
    assert km.get_key(estimated_tokens=9000) == "k1"
    assert km.get_key(estimated_tokens=3000) is None

    # Advance time by 61 seconds (t=1061.0)
    current_time = 1061.0

    # Old window expired; key should now have full capacity
    assert km.get_key(estimated_tokens=3000) == "k1"


def test_key_manager_cooldown_and_eviction():
    km = KeyManager(api_keys=["good", "bad"], cooldown_seconds=60.0)

    # Evict bad key
    km.evict_key("bad")
    assert km.api_keys == ["good"]

    # Mark good key in cooldown
    km.mark_cooldown("good", duration=10.0)
    assert km.get_key() is None


def test_key_manager_status():
    km = KeyManager(api_keys=["AIzaSyTestKey123456"], tpm_limit=28000)
    km.get_key(estimated_tokens=500)
    status = km.get_key_status()
    assert len(status) == 1
    assert status[0]["rpm"] == 1
    assert status[0]["tpm"] == 500


# ============================================================================
# GeminiEmbedder Parallel Execution Tests
# ============================================================================

def test_embedder_deterministic_offline_vector():
    vec = _deterministic_mock_vector("test text", dim=3072)
    assert vec.shape == (3072,)
    assert vec.dtype == np.float32
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5


def test_embedder_sequential_offline_execution(tmp_path: Path):
    embedder = GeminiEmbedder(
        api_keys=[],
        batch_size=2,
        dimension=128,
        max_workers=1,
    )
    texts = ["chunk 1", "chunk 2", "chunk 3", "chunk 4"]
    ids = ["id_1", "id_2", "id_3", "id_4"]
    ckpt_path = tmp_path / "checkpoint.npz"

    embeddings = embedder.embed_texts(texts=texts, ids=ids, checkpoint_path=ckpt_path)
    assert embeddings.shape == (4, 128)
    assert not np.isnan(embeddings).any()


def test_embedder_parallel_execution_preserves_order(tmp_path: Path):
    # Test with multiple mock workers and small batch size
    embedder = GeminiEmbedder(
        api_keys=[],
        batch_size=2,
        dimension=64,
        max_workers=4,
    )
    texts = [f"Distinct chunk text number {i}" for i in range(20)]
    ids = [f"id_{i}" for i in range(20)]
    ckpt_path = tmp_path / "parallel_ckpt.npz"

    parallel_embeddings = embedder.embed_texts(texts=texts, ids=ids, checkpoint_path=ckpt_path)
    assert parallel_embeddings.shape == (20, 64)

    # Verify deterministic ordering: row i must match single mock vector for texts[i]
    for i in range(20):
        expected_vec = _deterministic_mock_vector(texts[i], dim=64)
        np.testing.assert_allclose(parallel_embeddings[i], expected_vec, atol=1e-6)


def test_embedder_checkpoint_resumption(tmp_path: Path):
    ckpt_path = tmp_path / "resume_ckpt.npz"
    dim = 64

    # Pre-save 2 cached embeddings
    cached_ids = np.array(["id_0", "id_1"], dtype=object)
    cached_vecs = np.zeros((2, dim), dtype=np.float32)
    cached_vecs[0, 0] = 1.0
    cached_vecs[1, 1] = 1.0
    np.savez_compressed(ckpt_path, ids=cached_ids, embeddings=cached_vecs)

    embedder = GeminiEmbedder(api_keys=[], batch_size=2, dimension=dim, max_workers=2)
    texts = ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]
    ids = ["id_0", "id_1", "id_2", "id_3"]

    results = embedder.embed_texts(texts=texts, ids=ids, checkpoint_path=ckpt_path)
    assert results.shape == (4, dim)

    # First two vectors must match the pre-cached checkpoint vectors
    assert results[0, 0] == 1.0
    assert results[1, 1] == 1.0

    # Remaining two vectors should be computed
    expected_vec_2 = _deterministic_mock_vector("chunk 2", dim=dim)
    np.testing.assert_allclose(results[2], expected_vec_2, atol=1e-6)


def test_embedder_save_npz_cleans_up_checkpoint(tmp_path: Path):
    embedder = GeminiEmbedder(api_keys=[], dimension=64)
    out_file = tmp_path / "final_embeddings.npz"
    ckpt_file = tmp_path / "temp_ckpt.npz"
    ckpt_file.touch()

    ids = ["a", "b"]
    vecs = np.ones((2, 64), dtype=np.float32)
    embedder.save_npz(out_file, ids=ids, embeddings=vecs, checkpoint_path=ckpt_file)

    assert out_file.exists()
    assert not ckpt_file.exists()

    with np.load(out_file, allow_pickle=True) as data:
        assert list(data["ids"]) == ["a", "b"]
        assert data["embeddings"].shape == (2, 64)
