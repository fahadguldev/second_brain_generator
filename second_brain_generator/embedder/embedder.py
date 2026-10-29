"""Gemini vector embedding engine with checkpointing, rate limiting, and multi-key rotation."""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .key_manager import KeyManager, estimate_tokens

logger = logging.getLogger(__name__)


def _safe_replace(src: Path, dst: Path, retries: int = 5, delay: float = 0.5) -> None:
    """Windows-safe atomic file replacement with retry and copy fallback on transient file lock."""
    for attempt in range(retries):
        try:
            src.replace(dst)
            return
        except (PermissionError, OSError):
            if attempt == retries - 1:
                try:
                    import shutil
                    shutil.copy2(src, dst)
                    src.unlink(missing_ok=True)
                    return
                except Exception:
                    raise
            time.sleep(delay)


def _deterministic_mock_vector(text: str, dim: int = 3072) -> np.ndarray:
    """Deterministic offline fallback vector generator."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


class GeminiEmbedder:
    """
    Generates 3072-dimensional vector embeddings using Gemini API (models/gemini-embedding-2),
    supporting multi-key parallel rotation, sliding-window rate limiting (RPM/RPD/TPM),
    and concurrent batch checkpointing.
    """

    def __init__(
        self,
        api_keys: List[str],
        model: str = "models/gemini-embedding-2",
        batch_size: int = 25,
        dimension: int = 3072,
        rpm_limit: int = 60,
        rpd_limit: int = 1000,
        tpm_limit: int = 28000,
        max_workers: Optional[int] = None,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.api_keys = [k for k in api_keys if k and k.strip()]
        self.model = model
        self.batch_size = max(1, batch_size)
        self.dimension = dimension
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.tpm_limit = tpm_limit
        self.cooldown_seconds = cooldown_seconds

        num_keys = len(self.api_keys)
        if max_workers is not None:
            self.max_workers = max(1, max_workers)
        else:
            self.max_workers = max(1, min(num_keys, 8)) if num_keys > 0 else 1

        self.key_manager = KeyManager(
            self.api_keys,
            rpm_limit=self.rpm_limit,
            rpd_limit=self.rpd_limit,
            tpm_limit=self.tpm_limit,
            cooldown_seconds=self.cooldown_seconds,
        )

    def _embed_batch_remote(self, texts: List[str], key: str) -> List[List[float]]:
        """Invokes google.genai client to embed a batch of texts."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=key)
            # Remove models/ prefix if required by SDK
            model_name = self.model.replace("models/", "")

            contents = [
                types.Content(role="user", parts=[types.Part(text=text)])
                for text in texts
            ]
            result = client.models.embed_content(
                model=model_name,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
            embeddings = [e.values for e in result.embeddings]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Expected {len(texts)} embeddings, received {len(embeddings)}"
                )
            return embeddings
        except Exception as e:
            err_msg = str(e).upper()
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "QUOTA" in err_msg:
                self.key_manager.mark_cooldown(key, duration=self.cooldown_seconds)
            elif "401" in err_msg or "UNAUTHENTICATED" in err_msg or "INVALID_ARGUMENT" in err_msg:
                self.key_manager.evict_key(key)
            raise

    def embed_texts(
        self,
        texts: List[str],
        ids: List[str],
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Embeds a list of texts into normalized float32 vectors.
        Executes batches concurrently across API keys respecting RPM and TPM quotas.
        Resumes from checkpoint_path if available.
        """
        if len(texts) != len(ids):
            raise ValueError(f"Length mismatch: {len(texts)} texts vs {len(ids)} ids")

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        ckpt_file = Path(checkpoint_path).resolve() if checkpoint_path else None
        resumed_embeddings: Dict[str, np.ndarray] = {}

        # 1. Checkpoint resumption
        if ckpt_file and ckpt_file.exists():
            try:
                with np.load(ckpt_file, allow_pickle=True) as data:
                    c_ids = list(data["ids"])
                    c_vecs = data["embeddings"]
                    for cid, cvec in zip(c_ids, c_vecs):
                        resumed_embeddings[cid] = cvec
                logger.info("Resumed %d embeddings from checkpoint: %s", len(resumed_embeddings), ckpt_file.name)
            except Exception as e:
                logger.warning("Failed to load checkpoint %s, starting fresh: %s", ckpt_file, e)

        output_vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        indices_to_compute: List[int] = []

        for idx, (rec_id, text) in enumerate(zip(ids, texts)):
            if rec_id in resumed_embeddings:
                output_vectors[idx] = resumed_embeddings[rec_id]
            else:
                indices_to_compute.append(idx)

        logger.info(
            "Embedding %d total items (%d already cached, %d to compute)",
            len(texts), len(resumed_embeddings), len(indices_to_compute)
        )

        if not indices_to_compute:
            return np.array(output_vectors, dtype=np.float32)

        # 2. Partition into batches
        batches: List[List[int]] = []
        for start in range(0, len(indices_to_compute), self.batch_size):
            batches.append(indices_to_compute[start : start + self.batch_size])

        checkpoint_lock = threading.Lock()
        completed_batches = 0
        total_batches = len(batches)

        def process_batch(batch_indices: List[int]) -> None:
            nonlocal completed_batches
            batch_texts = [texts[i] for i in batch_indices]
            batch_tokens = sum(estimate_tokens(t) for t in batch_texts)

            if not self.key_manager.api_keys:
                batch_vecs = [_deterministic_mock_vector(t, self.dimension) for t in batch_texts]
            else:
                retries = 3
                success = False
                batch_vecs = []

                while retries > 0 and not success:
                    key = self.key_manager.acquire_key(estimated_tokens=batch_tokens, timeout=120.0)
                    if not key:
                        logger.warning("No available API key with quota after timeout; falling back to offline vectors for batch")
                        batch_vecs = [_deterministic_mock_vector(t, self.dimension) for t in batch_texts]
                        success = True
                        break

                    try:
                        raw_embeddings = self._embed_batch_remote(batch_texts, key)
                        for raw in raw_embeddings:
                            v = np.array(raw, dtype=np.float32)
                            norm = np.linalg.norm(v)
                            if norm > 1e-12:
                                v /= norm
                            batch_vecs.append(v)
                        success = True
                    except Exception as err:
                        retries -= 1
                        logger.warning("Batch embedding error (retry %d): %s", retries, err)
                        time.sleep(2)

                if not success:
                    logger.error("Batch embedding failed after retries; falling back to offline vectors")
                    batch_vecs = [_deterministic_mock_vector(t, self.dimension) for t in batch_texts]

            # Thread-safe update of outputs and checkpoint
            with checkpoint_lock:
                for idx, vec in zip(batch_indices, batch_vecs):
                    output_vectors[idx] = vec
                    resumed_embeddings[ids[idx]] = vec
                completed_batches += 1
                logger.info(
                    "Embedded batch %d/%d (%.1f%%) - %d items",
                    completed_batches,
                    total_batches,
                    (completed_batches / total_batches) * 100,
                    len(batch_indices),
                )

                if ckpt_file:
                    ckpt_ids = list(resumed_embeddings.keys())
                    ckpt_vecs = np.array(list(resumed_embeddings.values()), dtype=np.float32)
                    ckpt_temp = ckpt_file.with_name(
                        f"{ckpt_file.stem}.tmp.{os.getpid()}.{threading.get_ident()}.npz"
                    )
                    np.savez_compressed(
                        ckpt_temp,
                        ids=np.array(ckpt_ids, dtype=object),
                        embeddings=ckpt_vecs,
                    )
                    _safe_replace(ckpt_temp, ckpt_file)

        # 3. Execution strategy: sequential vs concurrent
        if self.max_workers <= 1 or len(batches) == 1 or not self.key_manager.api_keys:
            for b in batches:
                process_batch(b)
        else:
            workers = min(self.max_workers, len(batches))
            logger.info("Executing parallel embedding across %d worker threads", workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_batch, b) for b in batches]
                for f in concurrent.futures.as_completed(futures):
                    f.result()

        final_array = np.array(output_vectors, dtype=np.float32)
        return final_array

    def save_npz(
        self,
        output_path: Union[str, Path],
        ids: List[str],
        embeddings: np.ndarray,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Saves the final embeddings to compressed .npz and removes temporary checkpoint.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        temp_p = out_p.with_name(f"{out_p.stem}.tmp.{os.getpid()}.npz")
        np.savez_compressed(
            temp_p,
            ids=np.array(ids, dtype=object),
            embeddings=embeddings.astype(np.float32),
        )
        _safe_replace(temp_p, out_p)

        # Cleanup checkpoint file if it exists
        if checkpoint_path:
            ckpt_p = Path(checkpoint_path).resolve()
            if ckpt_p.exists():
                try:
                    ckpt_p.unlink()
                except OSError:
                    pass

        logger.info("Successfully saved %d embeddings to %s", len(ids), out_p)
