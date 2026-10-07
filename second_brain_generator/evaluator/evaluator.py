"""Offline cosine similarity retrieval and benchmark query evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class BrainEvaluator:
    """
    Evaluates retrieval performance using local NumPy matrix-vector cosine similarity
    against records loaded from disk.
    """

    def __init__(
        self,
        embeddings_path: Union[str, Path],
        records_path: Union[str, Path],
    ) -> None:
        self.embeddings_path = Path(embeddings_path).resolve()
        self.records_path = Path(records_path).resolve()

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        if not self.records_path.exists():
            raise FileNotFoundError(f"Records file not found: {self.records_path}")

        # Load embeddings
        with np.load(self.embeddings_path, allow_pickle=True) as data:
            self.ids = list(data["ids"])
            self.embeddings = data["embeddings"].astype(np.float32)

        # Load records mapping id -> record dict
        self.records: Dict[str, Dict[str, Any]] = {}
        with open(self.records_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    rec = json.loads(line_str)
                    self.records[rec["id"]] = rec

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Computes cosine similarity of query_embedding across all records and returns top_k matches.
        """
        if len(self.embeddings) == 0:
            return []

        q_vec = query_embedding.astype(np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 1e-12:
            q_vec = q_vec / norm

        # Cosine similarity dot product
        scores = np.dot(self.embeddings, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            rec_id = self.ids[idx]
            rec = self.records.get(rec_id, {})
            score = float(scores[idx])

            text = rec.get("text") or rec.get("content", {}).get("text", "")
            results.append({
                "id": rec_id,
                "score": score,
                "text": text,
                "record": rec,
                "metadata": rec.get("metadata") or rec.get("classification", {}),
            })

        return results

    def evaluate_benchmarks(
        self,
        benchmark_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Runs evaluation queries and returns summary metrics (mean similarity score).
        """
        queries = [
            "How do I start with cloud computing?",
            "What is the best way to learn Python?",
            "Should I prioritize degree or skills?",
            "How does serverless work?",
        ]

        if benchmark_file and Path(benchmark_file).exists():
            with open(benchmark_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    queries = [q if isinstance(q, str) else q.get("query", "") for q in loaded]

        total_score = 0.0
        count = 0

        for q in queries:
            if not q:
                continue
            # Offline pseudo-query vector representation from random hash if not embedded
            from ..embedder.embedder import _deterministic_mock_vector
            q_vec = _deterministic_mock_vector(q, self.embeddings.shape[1] if len(self.embeddings) > 0 else 3072)
            results = self.search(q_vec, top_k=1)
            if results:
                total_score += results[0]["score"]
                count += 1

        mean_score = (total_score / count) if count > 0 else 0.0
        return {
            "total_queries": float(count),
            "mean_similarity_score": round(mean_score, 4),
        }
