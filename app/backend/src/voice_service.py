import base64
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
import httpx
from src.config import settings
from src.jsf_keys import jsf_key_manager
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

JSF_API_BASE = "https://www.jsflabs.io/api/v1"
JSF_FAST_TTS_ENDPOINT = "https://www.jsflabs.io/api/jsf-tts-3"


def _get_audio_cache_dir() -> Path:
    """
    Determine a writable audio cache directory.
    In serverless environments (e.g. Vercel, AWS Lambda), the deployment root is read-only,
    so we safely fallback to the system temp directory (/tmp/second_brain_audio_cache).
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


class VoiceConfigurationError(Exception):
    """Raised when voice credentials or configuration are missing."""
    pass


class VoiceServiceError(Exception):
    """Raised when Voice API call fails across all available keys or endpoints."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def get_active_provider() -> str:
    """
    Determine the active voice provider:
    1. If explicitly set to 'elevenlabs', use ElevenLabs.
    2. If explicitly set to 'jsflabs', use JSF Labs.
    3. If set to 'auto' (default), prioritize ElevenLabs if configured, else JSF Labs.
    """
    req_provider = (getattr(settings, "VOICE_PROVIDER", "auto") or "auto").lower().strip()
    if req_provider == "elevenlabs":
        return "elevenlabs"
    if req_provider == "jsflabs":
        return "jsflabs"

    # auto
    if is_elevenlabs_configured():
        return "elevenlabs"
    return "jsflabs"


def is_jsf_configured() -> bool:
    """Check if any JSF Labs API keys or User IDs are configured."""
    return bool(getattr(settings, "JSF_USER_ID", "").strip()) or jsf_key_manager.has_keys


def is_voice_configured() -> bool:
    """Check if the currently active voice provider is configured."""
    provider = get_active_provider()
    if provider == "elevenlabs":
        return is_elevenlabs_configured()
    return is_jsf_configured()


def get_voice_status() -> Dict[str, Any]:
    """Return public voice configuration, active provider, and key pool health status."""
    provider = get_active_provider()
    if provider == "elevenlabs":
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

    # JSF Labs
    configured = is_jsf_configured()
    return {
        "provider": "jsflabs",
        "configured": configured,
        "has_user_id": bool(settings.JSF_USER_ID.strip()),
        "voice_id": settings.JSF_VOICE_ID if configured else None,
        "speed": settings.JSF_TTS_SPEED if configured else 1.0,
        "key_pool": jsf_key_manager.get_status_summary(),
    }


def _compute_cache_key(
    text: str,
    voice_id: Optional[str] = None,
    speed: float = 1.0,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
) -> str:
    """Generate deterministic hash for text and synthesis parameters."""
    p = provider or get_active_provider()
    if p == "jsflabs":
        raw = f"jsflabs:{voice_id or 'default'}:{speed}:{text.strip()}"
    else:
        raw = f"elevenlabs:{voice_id or 'default'}:{model_id or 'default'}:{speed}:{text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _synthesize_via_fast_inference(
    text: str,
    voice_id: str,
    speed: float,
    user_id: str,
) -> bytes:
    """Call the JSF Labs Fast Inference endpoint."""
    payload = {
        "userId": user_id,
        "text": text,
        "voiceId": voice_id,
        "speed": speed,
    }
    headers = {
        "Content-Type": "application/json",
        "Origin": "https://www.jsflabs.io",
        "Referer": "https://www.jsflabs.io/dashboard/fast-inference",
    }
    if jsf_key_manager.has_keys:
        try:
            key_state = jsf_key_manager.reserve_key()
            headers["x-api-key"] = key_state.api_key
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(JSF_FAST_TTS_ENDPOINT, json=payload, headers=headers)
        if response.status_code not in (200, 201):
            err_msg = response.text
            try:
                err_json = response.json()
                err_msg = err_json.get("error") or err_json.get("message") or err_msg
            except Exception:
                pass
            raise VoiceServiceError(
                f"Fast inference error ({response.status_code}): {err_msg}",
                status_code=response.status_code,
            )

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            res_json = response.json()
            audio_url = res_json.get("audioUrl") or res_json.get("audio_url") or res_json.get("url")
            if audio_url:
                audio_res = await client.get(audio_url, timeout=30.0)
                if audio_res.status_code == 200:
                    return audio_res.content
            b64_audio = res_json.get("audio") or res_json.get("audioBase64") or res_json.get("base64")
            if b64_audio:
                return base64.b64decode(b64_audio)
            raise VoiceServiceError(f"Could not extract audio from response: {response.text}")
        return response.content


async def _synthesize_jsf(
    text: str,
    voice_id: Optional[str] = None,
    speed: Optional[float] = None,
) -> bytes:
    """Execute JSF Labs TTS synthesis."""
    if not is_jsf_configured():
        raise VoiceConfigurationError(
            "JSF Labs credentials are not configured. Set JSF_API_KEY or JSF_USER_ID in backend/.env"
        )

    target_voice_id = voice_id or settings.JSF_VOICE_ID or "jsf_default_male"
    target_speed = speed if speed is not None else settings.JSF_TTS_SPEED

    # Try fast inference first if user_id present
    if settings.JSF_USER_ID.strip():
        try:
            return await _synthesize_via_fast_inference(
                text=text,
                voice_id=target_voice_id,
                speed=target_speed,
                user_id=settings.JSF_USER_ID.strip(),
            )
        except Exception as e:
            logger.warning(f"Fast inference endpoint failed ({e}); falling back to standard API gateway.")

    if not jsf_key_manager.has_keys:
        raise VoiceServiceError("No active JSF Labs API keys available.", status_code=503)

    url = f"{JSF_API_BASE}/text-to-speech"
    payload = {
        "text": text,
        "voice_id": target_voice_id,
        "speed": target_speed,
    }
    attempts = max(1, jsf_key_manager.total_keys())
    errors = []

    for _ in range(attempts):
        key_state = jsf_key_manager.reserve_key()
        active_key = key_state.api_key
        headers = {
            "x-api-key": active_key,
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code in (429, 401, 402, 403):
                    jsf_key_manager.mark_key_failure(active_key, status_code=response.status_code, reason=response.text)
                    errors.append(f"Key ...{active_key[-6:]} [{response.status_code}]")
                    continue
                if response.status_code not in (200, 201):
                    jsf_key_manager.mark_key_failure(active_key, status_code=response.status_code, reason=response.text)
                    errors.append(f"Key ...{active_key[-6:]} [{response.status_code}]")
                    continue

                jsf_key_manager.mark_key_success(active_key)
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    res_data = response.json()
                    audio_url = res_data.get("audio_url") or res_data.get("url")
                    if not audio_url:
                        raise VoiceServiceError("JSF Labs returned no audio_url")
                    audio_res = await client.get(audio_url, timeout=30.0)
                    return audio_res.content
                return response.content
        except httpx.RequestError as exc:
            jsf_key_manager.mark_key_failure(active_key, status_code=502, reason=str(exc))
            errors.append(f"Key ...{active_key[-6:]} Network Error: {exc}")
            continue

    raise VoiceServiceError(f"All JSF keys exhausted. Errors: {errors}", status_code=502)


async def synthesize_speech(
    text: str,
    voice_id: Optional[str] = None,
    speed: Optional[float] = None,
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
) -> bytes:
    """
    Synthesize speech using active provider (ElevenLabs or JSF Labs)
    with persistent disk caching and key rotation.
    Returns audio bytes.
    """
    if not is_voice_configured():
        raise VoiceConfigurationError(
            "Voice credentials are not configured. Set ELEVENLABS_API_KEY or JSF_API_KEY in backend/.env"
        )

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text cannot be empty.")

    active_p = (provider or get_active_provider()).lower()

    if active_p == "elevenlabs":
        target_voice = voice_id or settings.ELEVENLABS_VOICE_ID or "JBFqnCBsd6RMkjVDRZzb"
        target_speed = speed if speed is not None else settings.ELEVENLABS_SPEED
        target_model = model_id or settings.ELEVENLABS_MODEL_ID or "eleven_multilingual_v2"
        ext = "mp3"
    else:
        target_voice = voice_id or settings.JSF_VOICE_ID or "jsf_default_male"
        target_speed = speed if speed is not None else settings.JSF_TTS_SPEED
        target_model = "default"
        ext = "wav"

    cache_key = _compute_cache_key(clean_text, target_voice, target_speed, provider=active_p, model_id=target_model)
    cache_path = AUDIO_CACHE_DIR / f"{cache_key}.{ext}"

    # Also check .wav if ext is .mp3 or vice-versa for resilience
    alt_ext = "wav" if ext == "mp3" else "mp3"
    alt_cache_path = AUDIO_CACHE_DIR / f"{cache_key}.{alt_ext}"

    try:
        if cache_path.exists():
            logger.info(f"TTS cache hit for key {cache_key[:12]} ({active_p})")
            return cache_path.read_bytes()
        if alt_cache_path.exists():
            logger.info(f"TTS alt cache hit for key {cache_key[:12]}")
            return alt_cache_path.read_bytes()
    except Exception as exc:
        logger.warning(f"Failed reading audio cache: {exc}")

    # Synthesize via active provider
    if active_p == "elevenlabs":
        try:
            audio_bytes, _ = await synthesize_speech_elevenlabs(
                text=clean_text,
                voice_id=target_voice,
                model_id=target_model,
                speed=target_speed,
            )
        except (ElevenLabsConfigurationError, ElevenLabsServiceError) as e:
            if (getattr(settings, "VOICE_PROVIDER", "auto") or "auto").lower() == "auto" and is_jsf_configured():
                logger.warning(f"ElevenLabs synthesis failed ({e}); falling back to JSF Labs.")
                audio_bytes = await _synthesize_jsf(clean_text, voice_id=None, speed=speed)
                cache_path = AUDIO_CACHE_DIR / f"{cache_key}.wav"
            else:
                raise
    else:
        audio_bytes = await _synthesize_jsf(clean_text, voice_id=target_voice, speed=target_speed)

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
    Stream audio chunks for real-time playback.
    """
    active_p = (provider or get_active_provider()).lower()
    if active_p == "elevenlabs" and is_elevenlabs_configured():
        async for chunk in synthesize_speech_stream_elevenlabs(
            text=text, voice_id=voice_id, model_id=model_id, speed=speed
        ):
            yield chunk
    else:
        audio_bytes = await synthesize_speech(text, voice_id=voice_id, speed=speed, provider=active_p)
        chunk_size = 16384
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]


async def clone_voice(
    audio_bytes: bytes,
    name: str,
    gender: str = "Male",
    language: str = "en",
    filename: str = "sample.wav",
) -> Dict[str, Any]:
    """Clone a voice using JSF Labs voice cloning endpoint."""
    if not is_jsf_configured():
        raise VoiceConfigurationError("JSF Labs credentials are not configured.")

    url = f"{JSF_API_BASE}/voice-clone"
    files = {"audio": (filename, audio_bytes, "audio/wav")}
    data = {"name": name, "gender": gender, "language": language}
    key_state = jsf_key_manager.reserve_key()
    headers = {"x-api-key": key_state.api_key}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, files=files, data=data)
        if response.status_code not in (200, 201):
            raise VoiceServiceError(
                f"JSF Labs voice cloning failed [{response.status_code}]: {response.text}",
                status_code=response.status_code,
            )
        res_json = response.json()
        return {
            "success": True,
            "voice_id": res_json.get("voice_id"),
            "name": name,
            "gender": gender,
            "language": language,
        }
