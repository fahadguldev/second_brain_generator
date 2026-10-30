import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional
from src.config import settings
from src.elevenlabs_service import (
    elevenlabs_key_manager,
    is_elevenlabs_configured,
    synthesize_speech_elevenlabs,
    synthesize_speech_stream_elevenlabs,
    get_convai_signed_url,
    ElevenLabsConfigurationError,
    ElevenLabsServiceError,
)

logger = logging.getLogger("second_brain.voice")

# Type aliases for backward compatibility
VoiceConfigurationError = ElevenLabsConfigurationError
VoiceServiceError = ElevenLabsServiceError


def _get_audio_cache_dir() -> Path:
    """
    Determine a writable audio cache directory.
    In serverless environments (e.g. Vercel, AWS Lambda), deployment root is read-only,
    so we safely fallback to the system temp directory.
    """
    if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        cache_dir = Path(tempfile.gettempdir()) / "second_brain_audio_cache"
    else:
        cache_dir = Path("audio_cache")

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as exc:
        logger.warning(f"Could not create cache directory at {cache_dir} ({exc}); falling back to temp dir.")
        cache_dir = Path(tempfile.gettempdir()) / "second_brain_audio_cache"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create temp audio cache directory: {e}")
    return cache_dir


AUDIO_CACHE_DIR = _get_audio_cache_dir()


def is_voice_configured() -> bool:
    """Check if ElevenLabs credentials are configured."""
    return is_elevenlabs_configured()


def get_voice_status() -> Dict[str, Any]:
    """Return public ElevenLabs configuration and key pool health status."""
    configured = is_elevenlabs_configured()
    return {
        "provider": "elevenlabs",
        "configured": configured,
        "voice_id": settings.ELEVENLABS_VOICE_ID if configured else None,
        "model_id": settings.ELEVENLABS_MODEL_ID if configured else None,
        "speed": settings.ELEVENLABS_SPEED if configured else 1.0,
        "has_convai_agent": bool(settings.ELEVENLABS_CONVAI_AGENT_ID.strip()),
        "convai_agent_id": settings.ELEVENLABS_CONVAI_AGENT_ID if settings.ELEVENLABS_CONVAI_AGENT_ID else None,
        "key_pool": elevenlabs_key_manager.get_status_summary(),
    }


def _compute_cache_key(
    text: str,
    voice_id: Optional[str] = None,
    speed: float = 1.0,
    model_id: Optional[str] = None,
) -> str:
    """Generate deterministic hash for text and synthesis parameters."""
    target_model = model_id or getattr(settings, "ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    raw = f"elevenlabs:{voice_id or 'default'}:{target_model}:{speed}:{text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def synthesize_speech(
    text: str,
    voice_id: Optional[str] = None,
    speed: Optional[float] = None,
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
) -> bytes:
    """
    Synthesize speech using ElevenLabs Text-to-Speech API
    with persistent disk caching and key rotation.
    Returns audio bytes (audio/mpeg).
    """
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text cannot be empty.")

    if not is_voice_configured():
        raise ElevenLabsConfigurationError(
            "ElevenLabs API key is not configured. Set ELEVENLABS_API_KEY in backend/.env"
        )

    target_voice = voice_id or settings.ELEVENLABS_VOICE_ID or "JBFqnCBsd6RMkjVDRZzb"
    target_speed = speed if speed is not None else settings.ELEVENLABS_SPEED
    target_model = model_id or settings.ELEVENLABS_MODEL_ID or "eleven_multilingual_v2"

    cache_key = _compute_cache_key(clean_text, target_voice, target_speed, model_id=target_model)
    cache_path = AUDIO_CACHE_DIR / f"{cache_key}.mp3"

    try:
        if cache_path.exists():
            logger.info(f"ElevenLabs TTS cache hit for key {cache_key[:12]}")
            return cache_path.read_bytes()
    except Exception as exc:
        logger.warning(f"Failed reading audio cache: {exc}")

    audio_bytes, _ = await synthesize_speech_elevenlabs(
        text=clean_text,
        voice_id=target_voice,
        model_id=target_model,
        speed=target_speed,
    )

    try:
        cache_path.write_bytes(audio_bytes)
    except Exception as cache_err:
        logger.warning(f"Failed to write audio cache: {cache_err}")

    return audio_bytes


async def synthesize_speech_stream(
    text: str,
    voice_id: Optional[str] = None,
    speed: Optional[float] = None,
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Stream audio chunks from ElevenLabs for real-time playback.
    """
    async for chunk in synthesize_speech_stream_elevenlabs(
        text=text, voice_id=voice_id, model_id=model_id, speed=speed
    ):
        yield chunk
