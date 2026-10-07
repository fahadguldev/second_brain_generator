"""Qdrant batch point upserter and collection manager."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class QdrantUploader:
    """
    Connects to Qdrant, verifies collection configuration (Cosine, 3072 dim),
    and batch-upserts points with deterministic UUIDs and complete payloads.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "second_brain",
        client: Optional[Any] = None,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self._client = client

    @property
    def client(self):
        """Lazy client loader supporting in-memory, local URL, or remote cloud."""
        if self._client is None:
            from qdrant_client import QdrantClient

            if not self.url or self.url == ":memory:":
                logger.info("Initializing in-memory Qdrant client")
                self._client = QdrantClient(":memory:")
            else:
                logger.info("Connecting to Qdrant at %s", self.url)
                self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=120)
        return self._client

    def ensure_collection(self, vector_size: int = 3072, distance: str = "Cosine") -> None:
        """
        Creates the Qdrant collection if it does not already exist.
        """
        from qdrant_client import models

        # Map distance string
        dist_map = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }
        dist_enum = dist_map.get(distance.lower(), models.Distance.COSINE)

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info(
                "Creating Qdrant collection '%s' (vector_size=%d, distance=%s)",
                self.collection_name, vector_size, distance
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=dist_enum),
            )
        else:
            logger.info("Collection '%s' already exists in Qdrant", self.collection_name)

    def upload_from_disk(
        self,
        embeddings_path: Union[str, Path],
        records_path: Union[str, Path],
        batch_size: int = 100,
    ) -> int:
        """
        Uploads local embeddings and records into Qdrant collection in batches.
        """
        from qdrant_client import models

        emb_p = Path(embeddings_path).resolve()
        rec_p = Path(records_path).resolve()

        if not emb_p.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_p}")
        if not rec_p.exists():
            raise FileNotFoundError(f"Records file not found: {rec_p}")

        # Load embeddings
        with np.load(emb_p, allow_pickle=True) as data:
            ids = list(data["ids"])
            embeddings = data["embeddings"]

        # Load records
        records = []
        with open(rec_p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(json.loads(line_str))

        if len(ids) != len(embeddings):
            raise ValueError(f"IDs count ({len(ids)}) does not match embeddings count ({len(embeddings)})")

        if len(records) != len(embeddings):
            raise ValueError(f"Records count ({len(records)}) does not match embeddings count ({len(embeddings)})")

        self.ensure_collection(vector_size=embeddings.shape[1], distance="Cosine")

        total = len(embeddings)
        uploaded = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = []

            for i in range(start, end):
                rec = records[i]
                rec_id = str(ids[i])
                payload = rec.get("metadata") or rec

                # Ensure search text is preserved in payload
                if "text" in rec and "text" not in payload:
                    payload["text"] = rec["text"]

                point = models.PointStruct(
                    id=rec_id,
                    vector=embeddings[i].tolist(),
                    payload=payload,
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            uploaded += len(points)
            logger.info("Uploaded %d/%d points to Qdrant", uploaded, total)

        logger.info("Upload complete: %d total points in '%s'", uploaded, self.collection_name)
        return uploaded
