"""Clean checkpoint by stripping local fallback mock vectors and keeping only genuine Gemini embeddings."""

import json
from pathlib import Path
import numpy as np
from second_brain_generator.embedder.embedder import _deterministic_mock_vector

def purge_mock_vectors():
    base = Path(__file__).parent / "brain_data"
    ckpt_path = base / "gemini_embeddings_checkpoint.npz"
    rag_path = base / "rag_records.jsonl"

    if not ckpt_path.exists():
        print("Checkpoint file not found.")
        return

    text_by_id = {}
    with open(rag_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text_by_id[item["id"]] = item["text"]

    with np.load(ckpt_path, allow_pickle=True) as data:
        ids = list(data["ids"])
        vecs = data["embeddings"]

    clean_ids = []
    clean_vecs = []
    mock_count = 0

    for rec_id, vec in zip(ids, vecs):
        text = text_by_id.get(rec_id, "")
        mock_vec = _deterministic_mock_vector(text, dim=3072)
        if np.allclose(vec, mock_vec, atol=1e-5):
            mock_count += 1
        else:
            clean_ids.append(rec_id)
            clean_vecs.append(vec)

    print(f"Original total in checkpoint: {len(ids)}")
    print(f"Genuine Gemini vectors found: {len(clean_ids)}")
    print(f"Mock/fallback vectors purged: {mock_count}")

    # Backup original before modifying
    backup_path = ckpt_path.with_name("gemini_embeddings_checkpoint.pre_purge.npz")
    ckpt_path.replace(backup_path)

    # Save cleaned checkpoint
    np.savez_compressed(
        ckpt_path,
        ids=np.array(clean_ids, dtype=object),
        embeddings=np.array(clean_vecs, dtype=np.float32)
    )
    print(f"Saved clean checkpoint with {len(clean_ids)} genuine vectors to {ckpt_path.name}")

if __name__ == "__main__":
    purge_mock_vectors()
