import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, List, Optional, Tuple, TypeVar
from src.config import settings

logger = logging.getLogger("second_brain.jsf_keys")

T = TypeVar("T")


@dataclass
class KeyState:
    api_key: str
    minute_calls: deque[float] = field(default_factory=deque)
    cooldown_until: float = 0.0
    failed_attempts: int = 0
    last_error: Optional[str] = None


class JSFKeyManager:
    """
    Manages a pool of recurring/rotating JSF Labs API keys.
    Automatically rotates to the next available key if a key is rate-limited (429),
    expired/unauthorized (401/403), or out of credits (402).
    """

    def __init__(self, keys: Optional[List[str]] = None) -> None:
        raw_keys = keys if keys is not None else settings.JSF_API_KEYS
        unique_keys = list(dict.fromkeys(k.strip() for k in raw_keys if k and k.strip()))
        self._states = [KeyState(api_key=key) for key in unique_keys]
        self._lock = Lock()
        self._next_index = 0

    @property
    def has_keys(self) -> bool:
        return len(self._states) > 0

    def total_keys(self) -> int:
        return len(self._states)

    def active_keys_count(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for s in self._states if s.cooldown_until <= now)

    def _prune(self, state: KeyState, now: float) -> None:
        # JSF Labs rate limit is 50 requests per minute
        minute_cutoff = now - 60.0
        while state.minute_calls and state.minute_calls[0] < minute_cutoff:
            state.minute_calls.popleft()

    def reserve_key(self) -> KeyState:
        with self._lock:
            now = time.time()
            count = len(self._states)
            if count == 0:
                raise RuntimeError("No JSF Labs API keys configured in environment.")

            # Search round-robin for a healthy key
            for offset in range(count):
                idx = (self._next_index + offset) % count
                state = self._states[idx]
                self._prune(state, now)

                # Skip if in cooldown
                if state.cooldown_until > now:
                    continue

                # Skip if sliding window rate limit (50 req/min) hit
                if len(state.minute_calls) >= 50:
                    continue

                state.minute_calls.append(now)
                self._next_index = (idx + 1) % count
                return state

        # If all keys are currently cooling down, return the one that will recover earliest
        earliest = min(self._states, key=lambda s: s.cooldown_until)
        wait_seconds = max(0.5, earliest.cooldown_until - time.time())
        raise RuntimeError(
            f"All JSF Labs API keys are rate-limited or cooling down. Next key recovers in {wait_seconds:.1f}s."
        )

    def mark_key_failure(
        self,
        api_key: str,
        status_code: int,
        reason: str = "",
        retry_after_seconds: Optional[int] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            for state in self._states:
                if state.api_key == api_key:
                    state.failed_attempts += 1
                    state.last_error = f"[{status_code}] {reason}"

                    if status_code == 429:
                        # Rate limit: respect Retry-After or default 60s cooldown
                        cooldown = float(retry_after_seconds) if retry_after_seconds else 60.0
                        state.cooldown_until = now + cooldown
                        logger.warning(
                            f"JSF Labs key ...{api_key[-6:]} rate-limited (429). Cooling down for {cooldown}s."
                        )
                    elif status_code in (401, 403):
                        # Key expired or invalid: cooldown for 1 hour so other keys take over
                        state.cooldown_until = now + 3600.0
                        logger.error(
                            f"JSF Labs key ...{api_key[-6:]} authentication failed ({status_code}). Cooled down for 1hr."
                        )
                    elif status_code == 402:
                        # Out of credits: cooldown for 2 hours
                        state.cooldown_until = now + 7200.0
                        logger.error(
                            f"JSF Labs key ...{api_key[-6:]} out of credits (402). Cooled down for 2hr."
                        )
                    else:
                        # General failure cooldown
                        state.cooldown_until = now + 30.0
                    break

    def mark_key_success(self, api_key: str) -> None:
        with self._lock:
            for state in self._states:
                if state.api_key == api_key:
                    state.failed_attempts = 0
                    state.last_error = None
                    break

    def get_status_summary(self) -> dict:
        now = time.time()
        with self._lock:
            return {
                "total_keys": len(self._states),
                "healthy_keys": sum(1 for s in self._states if s.cooldown_until <= now),
                "cooling_down_keys": sum(1 for s in self._states if s.cooldown_until > now),
            }


# Global singleton key manager
jsf_key_manager = JSFKeyManager()
