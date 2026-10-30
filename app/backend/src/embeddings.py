import time
from typing import List
from src.config import settings
from src.gemini_keys import gemini_key_manager

_embedding_model = settings.EMBEDDING_MODEL


def embed(text: str) -> List[float]:
    """Generate embedding for a single text string."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    
    try:
        result = gemini_key_manager.run(
            lambda client: client.models.embed_content(
                model=_embedding_model,
                contents=text,
            )
        )
        vector = result.embeddings[0].values
        if len(vector) != settings.EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding dimension is {len(vector)}; expected {settings.EMBEDDING_DIMENSION}"
            )
        return vector
    except Exception as e:
        print(f"Embedding error: {e}")
        raise RuntimeError("Embedding generation failed") from e


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""
    embeddings = []
    for text in texts:
        emb = embed(text)
        embeddings.append(emb)
        time.sleep(0.1)
    return embeddings


class EmbeddingGenerator:
    """Wrapper for embedding operations."""
    
    def embed(self, text: str) -> List[float]:
        return embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return embed_batch(texts)


# Global instance
embedding_generator = EmbeddingGenerator()
