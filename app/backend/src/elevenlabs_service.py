import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx
from src.config import settings

logger = logging.getLogger("second_brain.elevenlabs")

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"


class ElevenLabsConfigurationError(Exception):
    """Raised when ElevenLabs credentials or voice configuration are missing."""
    pass


class ElevenLabsServiceError(Exception):
    """Raised when ElevenLabs API call fails across all available keys or endpoints."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class ElevenLabsKeyStatus:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.failures = 0
        self.successes = 0
        self.last_failure_at: Optional[float] = None
        self.last_used_at: Optional[float] = None
        self.retry_after_until: Optional[float] = None
        self.permanent_exhausted = False
        self.last_error: Optional[str] = None

    @property
    def is_available(self) -> bool:
        if self.permanent_exhausted:
            return False
        if self.retry_after_until and time.time() < self.retry_after_until:
            return False
        return True


class ElevenLabsKeyManager:
    """
    Manages ElevenLabs API keys with automatic recurring rotation,
    rate-limit backoff, and exhaustion tracking.
    """
    def __init__(self, keys: Optional[List[str]] = None):
        self._keys: List[ElevenLabsKeyStatus] = []
        self._current_index = 0
        if keys:
            self.load_keys(keys)

    def load_keys(self, keys: List[str]) -> None:
        seen = set()
        loaded = []
        for raw in keys:
            k = raw.strip()
            if k and k not in seen:
                seen.add(k)
                loaded.append(ElevenLabsKeyStatus(k))
        self._keys = loaded

    @property
    def has_keys(self) -> bool:
        return any(k.is_available for k in self._keys)

    def total_keys(self) -> int:
        return len(self._keys)

    def reserve_key(self) -> ElevenLabsKeyStatus:
        if not self._keys:
            raise ElevenLabsConfigurationError("No ElevenLabs API keys configured.")

        total = len(self._keys)
        for offset in range(total):
            idx = (self._current_index + offset) % total
            candidate = self._keys[idx]
            if candidate.is_available:
                self._current_index = (idx + 1) % total
                candidate.last_used_at = time.time()
                return candidate

        earliest_retry = min(
            (k.retry_after_until for k in self._keys if k.retry_after_until),
            default=None,
        )
        wait_s = max(1.0, earliest_retry - time.time()) if earliest_retry else 30.0
        raise ElevenLabsServiceError(
            f"All ElevenLabs API keys temporarily exhausted or rate limited. Retry in {int(wait_s)}s.",
            status_code=429,
        )

    def mark_success(self, api_key: str) -> None:
        for k in self._keys:
            if k.api_key == api_key:
                k.successes += 1
                k.failures = 0
                k.last_error = None
                k.retry_after_until = None
                break

    def mark_failure(
        self,
        api_key: str,
        status_code: int,
        reason: str,
        retry_after_seconds: Optional[int] = None,
    ) -> None:
        now = time.time()
        for k in self._keys:
            if k.api_key == api_key:
                k.failures += 1
                k.last_failure_at = now
                k.last_error = reason

                if status_code in (401, 403):
                    logger.error(f"ElevenLabs key ...{api_key[-6:]} permanently disabled (auth/permission error {status_code}): {reason}")
                    k.permanent_exhausted = True
                elif status_code in (429, 402):
                    backoff = retry_after_seconds if retry_after_seconds else 60
                    logger.warning(f"ElevenLabs key ...{api_key[-6:]} quota or rate limited ({status_code}); pausing for {backoff}s: {reason}")
                    k.retry_after_until = now + backoff
                else:
                    k.retry_after_until = now + 10
                break

    def get_status_summary(self) -> Dict[str, Any]:
        return {
            "total_keys": len(self._keys),
            "available_keys": sum(1 for k in self._keys if k.is_available),
            "keys": [
                {
                    "masked": f"...{k.api_key[-6:]}" if len(k.api_key) >= 6 else "***",
                    "available": k.is_available,
                    "successes": k.successes,
                    "failures": k.failures,
                    "permanent_exhausted": k.permanent_exhausted,
                    "last_error": k.last_error,
                }
                for k in self._keys
            ],
        }


# Global singleton key manager for ElevenLabs
elevenlabs_key_manager = ElevenLabsKeyManager()


def init_elevenlabs_keys() -> None:
    """Initialize ElevenLabs keys from config settings and environment."""
    keys: List[str] = []
    if hasattr(settings, "ELEVENLABS_API_KEYS") and settings.ELEVENLABS_API_KEYS:
        keys.extend(settings.ELEVENLABS_API_KEYS)
    if hasattr(settings, "ELEVENLABS_API_KEY") and settings.ELEVENLABS_API_KEY:
        keys.append(settings.ELEVENLABS_API_KEY)
    # Also inspect environment variables matching ELEVENLABS_API_KEY*
    for k, v in os.environ.items():
        if k.startswith("ELEVENLABS_API_KEY") and v.strip():
            keys.append(v.strip())
    elevenlabs_key_manager.load_keys(keys)


init_elevenlabs_keys()


def is_elevenlabs_configured() -> bool:
    """Check if valid ElevenLabs credentials are provided."""
    return elevenlabs_key_manager.has_keys


async def synthesize_speech_elevenlabs(
    text: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    speed: Optional[float] = None,
) -> tuple[bytes, str]:
    """
    Synthesize speech using ElevenLabs Text-to-Speech API with
    automatic recurring API key rotation on quota exhaustion.
    Returns (audio_bytes, media_type).
    """
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text cannot be empty.")

    if not is_elevenlabs_configured():
        raise ElevenLabsConfigurationError(
            "ElevenLabs API key is not configured. Set ELEVENLABS_API_KEY in backend/.env"
        )

    target_voice_id = (
        voice_id or getattr(settings, "ELEVENLABS_VOICE_ID", None) or "JBFqnCBsd6RMkjVDRZzb"
    )
    target_model_id = (
        model_id or getattr(settings, "ELEVENLABS_MODEL_ID", None) or "eleven_multilingual_v2"
    )
    target_speed = speed if speed is not None else getattr(settings, "ELEVENLABS_SPEED", 1.0)
    stability = getattr(settings, "ELEVENLABS_STABILITY", 0.5)
    similarity_boost = getattr(settings, "ELEVENLABS_SIMILARITY_BOOST", 0.75)

    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{target_voice_id}"
    payload: Dict[str, Any] = {
        "text": clean_text,
        "model_id": target_model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "speed": target_speed,
        },
    }

    total_keys = elevenlabs_key_manager.total_keys()
    attempts = max(1, total_keys)
    errors: list[str] = []

    for _ in range(attempts):
        try:
            key_state = elevenlabs_key_manager.reserve_key()
            active_key = key_state.api_key
        except RuntimeError as reserve_err:
            raise ElevenLabsServiceError(str(reserve_err), status_code=429) from reserve_err

        headers = {
            "xi-api-key": active_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code in (429, 401, 402, 403):
                    err_msg = response.text
                    retry_after = None
                    try:
                        err_json = response.json()
                        err_msg = err_json.get("detail", {}).get("message") or str(err_json)
                    except Exception:
                        pass
                    try:
                        retry_after = int(response.headers.get("Retry-After", 0)) or None
                    except Exception:
                        pass

                    logger.warning(f"Rotating ElevenLabs key: status {response.status_code} - {err_msg}")
                    elevenlabs_key_manager.mark_failure(
                        active_key,
                        status_code=response.status_code,
                        reason=err_msg,
                        retry_after_seconds=retry_after,
                    )
                    errors.append(f"Key ...{active_key[-6:]} [{response.status_code}]: {err_msg}")
                    continue

                if response.status_code not in (200, 201):
                    err_msg = response.text
                    elevenlabs_key_manager.mark_failure(
                        active_key, status_code=response.status_code, reason=err_msg
                    )
                    errors.append(f"Key ...{active_key[-6:]} [{response.status_code}]: {err_msg}")
                    continue

                # Success
                elevenlabs_key_manager.mark_success(active_key)
                audio_bytes = response.content
                logger.info(
                    f"Synthesized ElevenLabs speech ({len(audio_bytes)} bytes) using voice {target_voice_id}"
                )
                return audio_bytes, "audio/mpeg"

        except httpx.RequestError as exc:
            logger.exception("HTTP network error communicating with ElevenLabs")
            elevenlabs_key_manager.mark_failure(active_key, status_code=502, reason=str(exc))
            errors.append(f"Key ...{active_key[-6:]} Network Error: {str(exc)}")
            continue

    raise ElevenLabsServiceError(
        f"All ElevenLabs API keys failed or were exhausted. Errors: {' | '.join(errors)}",
        status_code=502,
    )


async def synthesize_speech_stream_elevenlabs(
    text: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    speed: Optional[float] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Stream audio chunks from ElevenLabs Text-to-Speech stream endpoint.
    """
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text cannot be empty.")

    if not is_elevenlabs_configured():
        raise ElevenLabsConfigurationError("ElevenLabs credentials not configured.")

    target_voice_id = (
        voice_id or getattr(settings, "ELEVENLABS_VOICE_ID", None) or "JBFqnCBsd6RMkjVDRZzb"
    )
    target_model_id = (
        model_id or getattr(settings, "ELEVENLABS_MODEL_ID", None) or "eleven_multilingual_v2"
    )
    target_speed = speed if speed is not None else getattr(settings, "ELEVENLABS_SPEED", 1.0)
    stability = getattr(settings, "ELEVENLABS_STABILITY", 0.5)
    similarity_boost = getattr(settings, "ELEVENLABS_SIMILARITY_BOOST", 0.75)

    key_state = elevenlabs_key_manager.reserve_key()
    headers = {
        "xi-api-key": key_state.api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": clean_text,
        "model_id": target_model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "speed": target_speed,
        },
    }
    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{target_voice_id}/stream"

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                err_text = await response.aread()
                elevenlabs_key_manager.mark_failure(
                    key_state.api_key, status_code=response.status_code, reason=err_text.decode("utf-8", "ignore")
                )
                raise ElevenLabsServiceError(
                    f"ElevenLabs stream error [{response.status_code}]: {err_text.decode('utf-8', 'ignore')}",
                    status_code=response.status_code,
                )
            elevenlabs_key_manager.mark_success(key_state.api_key)
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk


async def get_convai_signed_url(agent_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a secure signed WebSocket URL for ElevenLabs Conversational AI Agent:
    GET https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?agent_id={agent_id}
    """
    if not is_elevenlabs_configured():
        raise ElevenLabsConfigurationError("ElevenLabs API key is not configured.")

    target_agent_id = agent_id or getattr(settings, "ELEVENLABS_CONVAI_AGENT_ID", None)
    if not target_agent_id or not target_agent_id.strip():
        raise ElevenLabsConfigurationError(
            "ElevenLabs Conversational AI Agent ID is not configured. Set ELEVENLABS_CONVAI_AGENT_ID in backend/.env"
        )

    key_state = elevenlabs_key_manager.reserve_key()
    headers = {
        "xi-api-key": key_state.api_key,
    }
    url = f"{ELEVENLABS_API_BASE}/convai/conversation/get-signed-url"
    params = {"agent_id": target_agent_id.strip()}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers, params=params)
        if response.status_code != 200:
            err_msg = response.text
            try:
                err_msg = response.json().get("detail", {}).get("message") or response.text
            except Exception:
                pass
            elevenlabs_key_manager.mark_failure(
                key_state.api_key, status_code=response.status_code, reason=err_msg
            )
            raise ElevenLabsServiceError(
                f"Failed to obtain signed URL for Conversational AI [{response.status_code}]: {err_msg}",
                status_code=response.status_code,
            )

        elevenlabs_key_manager.mark_success(key_state.api_key)
        data = response.json()
        return {
            "signed_url": data.get("signed_url"),
            "agent_id": target_agent_id.strip(),
        }
