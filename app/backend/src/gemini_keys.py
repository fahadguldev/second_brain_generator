import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Iterable, Iterator, TypeVar

from google import genai as genai_lib

from src.config import settings

T = TypeVar("T")


@dataclass
class KeyState:
    api_key: str
    minute_calls: deque[float] = field(default_factory=deque)
    day_calls: deque[float] = field(default_factory=deque)
    cooldown_until: float = 0.0


class GeminiKeyManager:
    def __init__(self) -> None:
        unique_keys = list(dict.fromkeys(settings.GEMINI_API_KEYS))
        self._states = [KeyState(api_key=key) for key in unique_keys]
        self._clients: dict[str, genai_lib.Client] = {}
        self._lock = Lock()
        self._next_index = 0

    def _client_for(self, api_key: str) -> genai_lib.Client:
        client = self._clients.get(api_key)
        if client is None:
            client = genai_lib.Client(api_key=api_key)
            self._clients[api_key] = client
        return client

    def _prune(self, state: KeyState, now: float) -> None:
        minute_cutoff = now - 60
        day_cutoff = now - 86400

        while state.minute_calls and state.minute_calls[0] < minute_cutoff:
            state.minute_calls.popleft()
        while state.day_calls and state.day_calls[0] < day_cutoff:
            state.day_calls.popleft()

    def _reserve_key(self) -> tuple[KeyState, genai_lib.Client]:
        with self._lock:
            now = time.time()
            key_count = len(self._states)

            for offset in range(key_count):
                index = (self._next_index + offset) % key_count
                state = self._states[index]
                self._prune(state, now)

                if state.cooldown_until > now:
                    continue
                if len(state.minute_calls) >= settings.RATE_LIMIT_PER_MIN:
                    continue
                if len(state.day_calls) >= settings.RATE_LIMIT_PER_DAY:
                    continue

                state.minute_calls.append(now)
                state.day_calls.append(now)
                self._next_index = (index + 1) % key_count
                return state, self._client_for(state.api_key)

        raise RuntimeError("All Gemini API keys are rate limited or cooling down.")

    def run(self, operation: Callable[[genai_lib.Client], T]) -> T:
        if not self._states:
            raise RuntimeError(
                "Gemini is not configured. Set GEMINI_API_KEY or GEMINI_API_KEY_1..5."
            )
        errors: list[str] = []

        for _ in range(len(self._states)):
            state, client = self._reserve_key()
            try:
                return operation(client)
            except Exception as exc:
                errors.append(str(exc))
                with self._lock:
                    state.cooldown_until = time.time() + 60
                print(f"Gemini API key failed ({exc}); trying next key.")

        raise RuntimeError(f"All Gemini API keys failed. Last errors: {' | '.join(errors)}")

    def stream(self, operation: Callable[[genai_lib.Client], Iterable[T]]) -> Iterator[T]:
        """Retry a stream only when no output has reached the caller."""
        if not self._states:
            raise RuntimeError(
                "Gemini is not configured. Set GEMINI_API_KEY or GEMINI_API_KEY_1..5."
            )
        errors: list[str] = []
        for _ in range(len(self._states)):
            state, client = self._reserve_key()
            emitted = False
            try:
                for chunk in operation(client):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                errors.append(str(exc))
                with self._lock:
                    state.cooldown_until = time.time() + 60
                if emitted:
                    raise
                print(f"Gemini API key failed ({exc}); trying next key.")
        raise RuntimeError(f"All Gemini API keys failed. Last errors: {' | '.join(errors)}")


gemini_key_manager = GeminiKeyManager()
