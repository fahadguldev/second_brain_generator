import json
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "second_brain")

EMBEDDINGS_FILE = "knowledge_base/gemini_embeddings.npz"
RECORDS_FILE = "knowledge_base/rag_records.jsonl"

# ---------------------------------------------------------
# Connect to Qdrant
# ---------------------------------------------------------

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120  # Optional: set a timeout for requests
)

# ---------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------

data = np.load(EMBEDDINGS_FILE)

ids = data["ids"]
embeddings = data["embeddings"]

print(f"Loaded {len(embeddings)} embeddings")
print(f"Embedding dimension: {embeddings.shape[1]}")

# ---------------------------------------------------------
# Load records
# ---------------------------------------------------------

records = []

with open(RECORDS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# ---------------------------------------------------------
# Safety check
# ---------------------------------------------------------

if len(ids) != len(embeddings):
    raise ValueError("IDs and embeddings count do not match")

if len(records) != len(embeddings):
    raise ValueError(
        f"Records ({len(records)}) and embeddings ({len(embeddings)}) "
        "do not have the same count"
    )

# ---------------------------------------------------------
# Upload in batches
# ---------------------------------------------------------

BATCH_SIZE = 100

for start in range(0, len(embeddings), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(embeddings))

    points = []

    for i in range(start, end):

        payload = records[i]

        point = models.PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload=payload,
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Uploaded {end}/{len(embeddings)}")

print("\nMigration completed successfully!")

# ---------------------------------------------------------
# Verify
# ---------------------------------------------------------

info = client.get_collection(COLLECTION_NAME)

print(f"Points in Qdrant: {info.points_count}")
