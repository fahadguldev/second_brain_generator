import os
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://second-brain-phi-opal.vercel.app",
    "https://second-brain-backend-phi.vercel.app",
)


def _parse_cors_origins(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [
        origin
        for origin in (value.strip().rstrip("/").lower() for value in raw.split(","))
        if origin and "://" in origin and "/" not in origin.partition("://")[2]
    ]


class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./second_brain.db")
    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "change-me-in-production")
    COOKIE_SECURE: bool = (os.getenv("COOKIE_SECURE") or os.getenv("COOKIES_SECURE", "False")).lower() == "true"
    COOKIE_SAMESITE: str = os.getenv("COOKIE_SAMESITE", "lax")
    COOKIE_DOMAIN: str | None = os.getenv("COOKIE_DOMAIN") or None
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "").rstrip("/")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    ADMIN_EMAILS: set[str] = {
        email.strip().lower() for email in os.getenv("ADMIN_EMAILS", "").split(",") if email.strip()
    }

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "second_brain")

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEYS: list[str] = list(dict.fromkeys(
        val.strip()
        for k, val in os.environ.items()
        if k.startswith("GEMINI_API_KEY") and val.strip()
    ))

    # ElevenLabs Voice API (TTS & Conversational AI)
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_API_KEYS: list[str] = list(dict.fromkeys(
        [val.strip() for k, val in os.environ.items() if k.startswith("ELEVENLABS_API_KEY") and val.strip()] +
        [k.strip() for k in os.getenv("ELEVENLABS_API_KEYS", "").split(",") if k.strip()]
    ))
    # Default ElevenLabs voice (e.g. JBFqnCBsd6RMkjVDRZzb (George), or custom cloned voice)
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    ELEVENLABS_MODEL_ID: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    ELEVENLABS_CONVAI_AGENT_ID: str = os.getenv("ELEVENLABS_CONVAI_AGENT_ID", "")
    ELEVENLABS_STABILITY: float = float(os.getenv("ELEVENLABS_STABILITY", "0.5"))
    ELEVENLABS_SIMILARITY_BOOST: float = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))
    ELEVENLABS_SPEED: float = float(os.getenv("ELEVENLABS_SPEED", "1.0"))

    # Model names from Google AI Studio
    # - Embedding model: models/gemini-embedding-2 produces 3072-dim vectors (matches knowledge_base)
    # - Generation model: gemini-3.1-lite
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gemini-3.1-flash-lite")
    # Final (post-rerank) minimum score for a retrieved chunk to count as evidence.
    RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.55"))
    # Which Qdrant payload source types are searchable at query time, comma-separated.
    # "all" disables the filter. With tag routing enabled the whole corpus is
    # searchable; source trust is enforced by boosts + the noise slot cap instead.
    RAG_SOURCES: str = os.getenv("RAG_SOURCES", "all")
    # Route retrieval by intent: classify the query against the brain_config topic
    # lexicon and filter candidates to matching topics first.
    RAG_USE_TAGS: bool = os.getenv("RAG_USE_TAGS", "True").lower() == "true"
    # If the tag-filtered search returns fewer hits than this, retry without tags.
    RAG_TAG_MIN_HITS: int = int(os.getenv("RAG_TAG_MIN_HITS", "2"))
    # Max legacy/social points allowed in the final top-k; remaining slots go to
    # curated knowledge. Keeps curated docs from being crowded out while still
    # letting legacy topics (e.g. university advice in DMs) reach the context.
    RAG_NOISE_MAX: int = int(os.getenv("RAG_NOISE_MAX", "4"))

    # Rate limiting
    RATE_LIMIT_PER_MIN: int = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
    RATE_LIMIT_PER_DAY: int = int(os.getenv("RATE_LIMIT_PER_DAY", "1000"))

    # CORS
    CORS_ORIGINS: list[str] = list(
        dict.fromkeys(_parse_cors_origins(os.getenv("CORS_ORIGINS")) + list(_DEFAULT_CORS_ORIGINS))
    )

    # App & Brain Config
    BRAIN_CONFIG_PATH: str = os.getenv("BRAIN_CONFIG_PATH", "brain_config.yaml")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
