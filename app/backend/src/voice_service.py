import base64
import hashlib
import os
import tempfile
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional
import httpx
from src.config import settings
from src.jsf_keys import jsf_key_manager

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
    """Raised when JSF Labs credentials or configuration are missing."""
    pass


class VoiceServiceError(Exception):
    """Raised when JSF Labs API call fails across all available keys or endpoints."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def is_voice_configured() -> bool:
    """Check if any JSF Labs API keys or User IDs are configured."""
    return bool(settings.JSF_USER_ID.strip()) or jsf_key_manager.has_keys


def get_voice_status() -> Dict[str, Any]:
    """Return public voice configuration and key pool health status."""
    configured = is_voice_configured()
    return {
        "provider": "jsflabs",
        "configured": configured,
        "has_user_id": bool(settings.JSF_USER_ID.strip()),
        "voice_id": settings.JSF_VOICE_ID if configured else None,
        "speed": settings.JSF_TTS_SPEED if configured else 1.0,
        "key_pool": jsf_key_manager.get_status_summary(),
    }


def _compute_cache_key(text: str, voice_id: Optional[str], speed: float) -> str:
    """Generate deterministic hash for text and synthesis parameters."""
    raw = f"jsflabs:{voice_id or 'default'}:{speed}:{text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _synthesize_via_fast_inference(
    text: str,
    voice_id: str,
    speed: float,
    user_id: str,
) -> bytes:
    """
    Call the JSF Labs Fast Inference endpoint:
    POST https://www.jsflabs.io/api/jsf-tts-3
    Uses dashboard user session credits directly.
    """
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
    # Also attach API key if available
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
            raise VoiceServiceError(f"Fast inference error ({response.status_code}): {err_msg}", status_code=response.status_code)

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            res_json = response.json()
            # Check for URL
            audio_url = res_json.get("audioUrl") or res_json.get("audio_url") or res_json.get("url")
            if audio_url:
                audio_res = await client.get(audio_url, timeout=30.0)
                if audio_res.status_code == 200:
                    return audio_res.content
            # Check for base64 encoded audio
            b64_audio = res_json.get("audio") or res_json.get("audioBase64") or res_json.get("base64")
            if b64_audio:
                return base64.b64decode(b64_audio)
            raise VoiceServiceError(f"Could not extract audio from response: {response.text}")
        else:
            return response.content


async def synthesize_speech(
    text: str,
    voice_id: Optional[str] = None,
    speed: Optional[float] = None,
) -> bytes:
    """
    Synthesize speech using JSF Labs TTS API with persistent disk caching
    and automatic recurring API key rotation on expiry, quota, or rate-limits.
    """
    if not is_voice_configured():
        raise VoiceConfigurationError(
            "JSF Labs credentials are not configured. Set JSF_API_KEY or JSF_USER_ID in backend/.env"
        )

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text cannot be empty.")

    target_voice_id = voice_id or settings.JSF_VOICE_ID or "jsf_default_male"
    target_speed = speed if speed is not None else settings.JSF_TTS_SPEED

    # Check local disk cache first
    cache_key = _compute_cache_key(clean_text, target_voice_id, target_speed)
    cache_path = AUDIO_CACHE_DIR / f"{cache_key}.wav"

    try:
        if cache_path.exists():
            logger.info(f"JSF Labs TTS cache hit for key {cache_key[:12]}")
            return cache_path.read_bytes()
    except Exception as exc:
        logger.warning(f"Failed reading audio cache: {exc}")

    # 1. If JSF_USER_ID is provided, try the fast inference endpoint first
    if settings.JSF_USER_ID.strip():
        try:
            logger.info("Synthesizing via JSF Labs Fast Inference endpoint (/api/jsf-tts-3)...")
            audio_bytes = await _synthesize_via_fast_inference(
                text=clean_text,
                voice_id=target_voice_id,
                speed=target_speed,
                user_id=settings.JSF_USER_ID.strip(),
            )
            try:
                cache_path.write_bytes(audio_bytes)
            except Exception as cache_err:
                logger.warning(f"Failed to write audio cache: {cache_err}")
            return audio_bytes
        except Exception as e:
            logger.warning(f"Fast inference endpoint failed ({e}); falling back to standard API gateway.")

    # 2. Standard Developer API with recurring key pool rotation
    if not jsf_key_manager.has_keys:
        raise VoiceServiceError("No active JSF Labs API keys available.", status_code=503)

    url = f"{JSF_API_BASE}/text-to-speech"
    payload: Dict[str, Any] = {
        "text": clean_text,
        "voice_id": target_voice_id,
        "speed": target_speed,
    }

    total_keys = jsf_key_manager.total_keys()
    attempts = max(1, total_keys)
    errors: list[str] = []

    for attempt_idx in range(attempts):
        try:
            key_state = jsf_key_manager.reserve_key()
            active_key = key_state.api_key
        except RuntimeError as reserve_err:
            raise VoiceServiceError(str(reserve_err), status_code=429) from reserve_err

        headers = {
            "x-api-key": active_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)

                err_detail = response.text
                retry_after = None
                try:
                    err_json = response.json()
                    err_obj = err_json.get("error", {})
                    err_detail = err_obj.get("message") or err_json.get("message") or response.text
                    retry_after = err_obj.get("retryAfterSeconds")
                except Exception:
                    pass

                try:
                    if not retry_after:
                        retry_after = int(response.headers.get("Retry-After", 0)) or None
                except Exception:
                    pass

                # Handle Rate Limiting (429) or Auth Expiry / Insufficient Credits (401, 402, 403)
                if response.status_code in (429, 401, 402, 403):
                    err_msg = f"Key ...{active_key[-6:]} returned [{response.status_code}]: {err_detail}"
                    logger.warning(f"Rotating JSF Labs key: {err_msg}")
                    jsf_key_manager.mark_key_failure(
                        active_key,
                        status_code=response.status_code,
                        reason=err_detail,
                        retry_after_seconds=retry_after,
                    )
                    errors.append(f"Key ...{active_key[-6:]} [{response.status_code}]: {err_detail}")
                    continue

                if response.status_code not in (200, 201):
                    jsf_key_manager.mark_key_failure(active_key, status_code=response.status_code, reason=err_detail)
                    errors.append(f"Key ...{active_key[-6:]} [{response.status_code}]: {err_detail}")
                    continue

                # Success
                jsf_key_manager.mark_key_success(active_key)
                content_type = response.headers.get("Content-Type", "")
                audio_bytes: bytes

                if "application/json" in content_type:
                    res_data = response.json()
                    audio_url = res_data.get("audio_url") or res_data.get("url")
                    if not audio_url:
                        raise VoiceServiceError(f"JSF Labs returned no audio_url: {response.text}")

                    audio_res = await client.get(audio_url, timeout=30.0)
                    if audio_res.status_code != 200:
                        raise VoiceServiceError(
                            f"Failed to download audio from {audio_url} [{audio_res.status_code}]"
                        )
                    audio_bytes = audio_res.content
                else:
                    audio_bytes = response.content

                try:
                    cache_path.write_bytes(audio_bytes)
                except Exception as cache_err:
                    logger.warning(f"Failed to write audio cache: {cache_err}")
                logger.info(f"Synthesized and cached JSF Labs TTS ({len(audio_bytes)} bytes) using key ...{active_key[-6:]}")
                return audio_bytes

        except httpx.RequestError as exc:
            logger.exception("HTTP network error communicating with JSF Labs")
            jsf_key_manager.mark_key_failure(active_key, status_code=502, reason=str(exc))
            errors.append(f"Key ...{active_key[-6:]} Network Error: {str(exc)}")
            continue

    raise VoiceServiceError(
        f"All JSF Labs API keys failed or were exhausted. Errors: {' | '.join(errors)}",
        status_code=502,
    )


async def synthesize_speech_stream(
    text: str,
    voice_id: Optional[str] = None,
    speed: Optional[float] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Stream audio chunks for real-time playback.
    """
    audio_bytes = await synthesize_speech(text, voice_id=voice_id, speed=speed)
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
    """
    Clone a voice using JSF Labs voice cloning endpoint:
    POST https://www.jsflabs.io/api/v1/voice-clone
    Returns portable voice_id.
    """
    if not is_voice_configured():
        raise VoiceConfigurationError("JSF Labs credentials are not configured.")

    url = f"{JSF_API_BASE}/voice-clone"
    files = {
        "audio": (filename, audio_bytes, "audio/wav"),
    }
    data = {
        "name": name,
        "gender": gender,
        "language": language,
    }

    key_state = jsf_key_manager.reserve_key()
    headers = {
        "x-api-key": key_state.api_key,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, files=files, data=data)
        if response.status_code not in (200, 201):
            raise VoiceServiceError(
                f"JSF Labs voice cloning failed [{response.status_code}]: {response.text}",
                status_code=response.status_code,
            )
        res_json = response.json()
        voice_id = res_json.get("voice_id")
        return {
            "success": True,
            "voice_id": voice_id,
            "name": name,
            "gender": gender,
            "language": language,
        }
