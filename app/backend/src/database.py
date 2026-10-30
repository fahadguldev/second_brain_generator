from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
)
from src.config import settings

_client = None


def get_client():
    global _client

    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=120,
        )

    return _client


def get_collection_name() -> str:
    return settings.QDRANT_COLLECTION


def ensure_collection(vector_size: int = 3072) -> bool:
    """Check that the Qdrant collection exists and has the expected dimensions."""
    try:
        client = get_client()
        info = client.get_collection(
            collection_name=get_collection_name()
        )
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            first_param = next(iter(vectors.values()), None)
            actual_size = getattr(first_param, "size", None) if first_param else None
        else:
            actual_size = getattr(vectors, "size", None)
        if actual_size != vector_size:
            raise RuntimeError(
                f"Qdrant vector size is {actual_size}; expected {vector_size}"
            )

        # Source-filtered retrieval needs a keyword index on metadata.source.type.
        try:
            client.create_payload_index(
                collection_name=get_collection_name(),
                field_name="metadata.source.type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_exc:
            print(f"Notice: could not ensure payload index on metadata.source.type: {idx_exc}")

        # Tag-routed retrieval filters on metadata.topics (array of keywords).
        try:
            client.create_payload_index(
                collection_name=get_collection_name(),
                field_name="metadata.topics",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_exc:
            print(f"Notice: could not ensure payload index on metadata.topics: {idx_exc}")

        return True

    except Exception as e:
        print(f"Could not connect to Qdrant collection: {e}")
        return False


def get_collection_stats() -> dict:
    """Get collection statistics including total vector count and metadata."""
    client = _get_client()
    if client is None:
        return {"status": "error", "error": "Client not initialized", "points_count": 0}
    try:
        coll_name = get_collection_name()
        info = client.get_collection(collection_name=coll_name)
        count_res = client.count(collection_name=coll_name)
        points_count = count_res.count if hasattr(count_res, "count") else (getattr(info, "points_count", 0) or 0)
        indexed_vectors_count = getattr(info, "indexed_vectors_count", points_count)
        vectors_param = getattr(info.config.params, "vectors", None)
        if isinstance(vectors_param, dict):
            first_param = next(iter(vectors_param.values()), None)
            vector_size = getattr(first_param, "size", 3072) if first_param else 3072
        else:
            vector_size = getattr(vectors_param, "size", 3072)
        return {
            "status": "ok",
            "collection": coll_name,
            "points_count": points_count,
            "indexed_vectors_count": indexed_vectors_count,
            "vector_size": vector_size,
        }
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return {"status": "error", "error": str(e), "points_count": 0}


def _get_client():
    """Get the Qdrant client."""
    try:
        return get_client()
    except Exception:
        return None


def insert_points(points: list) -> bool:
    """Insert points into the collection."""
    if not points:
        return True

    client = _get_client()

    if client is None:
        return False

    try:
        client.upsert(
            collection_name=get_collection_name(),
            points=points,
            wait=True,
        )
        return True
    except Exception as e:
        print(f"Error inserting points: {e}")
        return False


def delete_points(point_ids: list[str]) -> bool:
    """Delete known derived points after a replacement has been indexed."""
    if not point_ids:
        return True
    client = _get_client()
    if client is None:
        return False
    try:
        from qdrant_client.models import PointIdsList
        client.delete(
            collection_name=get_collection_name(),
            points_selector=PointIdsList(points=point_ids),
        )
        return True
    except Exception as e:
        print(f"Error deleting points: {e}")
        return False

def search_points(
    query_vector: list[float],
    top_k: int = 5,
    filter_dict: dict | None = None,
    source_types: list[str] | None = None,
    topics: list[str] | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """Search for similar vectors."""

    client = _get_client()

    if client is None:
        return []

    must_filters = []

    if source_types:
        must_filters.append(
            FieldCondition(
                key="metadata.source.type",
                match=MatchAny(any=source_types),
            )
        )

    # Tag routing: keep only chunks tagged with at least one matched topic.
    if topics:
        must_filters.append(
            FieldCondition(
                key="metadata.topics",
                match=MatchAny(any=topics),
            )
        )

    if filter_dict:
        for key, value in filter_dict.items():
            must_filters.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

    query_filter = (
        Filter(must=must_filters)
        if must_filters
        else None
    )

    if score_threshold is None:
        score_threshold = settings.RAG_SCORE_THRESHOLD

    try:
        results = client.query_points(
            collection_name=get_collection_name(),
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            for point in results.points
        ]

    except Exception as e:
        print(f"Error searching points: {e}")
        return []
