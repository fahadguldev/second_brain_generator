"""Multi-key manager with cooldown, eviction, and sliding-window rate limiting (RPM/RPD/TPM)."""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Conservative token estimator for multilingual and mixed English/Urdu/Hindi text.
    Provides safety headroom for character-heavy scripts and Roman Urdu subwords.
    """
    if not text or not text.strip():
        return 0
    clean = text.strip()
    words = len(clean.split())
    chars = len(clean)
    # 2.5 characters per token provides safe headroom for multilingual subwords
    char_estimate = int(chars / 2.5) + 1
    return max(1, words, char_estimate)


class KeyManager:
    """
    Manages a pool of Gemini API keys with round-robin load distribution,
    sliding-window rate limiting (RPM/RPD/TPM), 60s cooldown on 429 quota exhaustion,
    and permanent key eviction on 401 authentication errors.
    """

    def __init__(
        self,
        api_keys: List[str],
        rpm_limit: int = 60,
        rpd_limit: int = 1000,
        tpm_limit: int = 28000,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.api_keys: List[str] = [k.strip() for k in api_keys if k and k.strip()]
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.tpm_limit = tpm_limit
        self.cooldown_seconds = cooldown_seconds

        self._lock = threading.Lock()
        self._index = 0
        self._cooldowns: Dict[str, float] = {}  # key -> cooldown_expiry
        self._rpm_windows: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)
        self._rpd_windows: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)
        self._tpm_windows: Dict[str, Deque[Tuple[float, int]]] = collections.defaultdict(collections.deque)

    def mark_cooldown(self, key: str, duration: Optional[float] = None) -> None:
        """Puts a key on cooldown due to rate limit (429)."""
        dur = duration if duration is not None else self.cooldown_seconds
        with self._lock:
            self._cooldowns[key] = time.time() + dur
            logger.warning("Key marked for %.1fs cooldown due to 429 quota limit", dur)

    def evict_key(self, key: str) -> None:
        """Permanently removes an invalid key (401)."""
        with self._lock:
            if key in self.api_keys:
                self.api_keys.remove(key)
                self._cooldowns.pop(key, None)
                self._rpm_windows.pop(key, None)
                self._rpd_windows.pop(key, None)
                self._tpm_windows.pop(key, None)
                logger.error("Key permanently evicted due to 401 authentication failure")

    def get_key(self, estimated_tokens: int = 0) -> Optional[str]:
        """
        Retrieves the next available key that is not in cooldown and within rate limits (RPM, RPD, TPM).
        Returns None if all keys are unavailable.
        """
        with self._lock:
            if not self.api_keys:
                return None

            now = time.time()
            n = len(self.api_keys)

            for _ in range(n):
                key = self.api_keys[self._index % n]
                self._index = (self._index + 1) % n

                # 1. Check cooldown
                if key in self._cooldowns:
                    if now < self._cooldowns[key]:
                        continue
                    else:
                        del self._cooldowns[key]

                # 2. Check sliding window RPM (past 60s)
                rpm_q = self._rpm_windows[key]
                while rpm_q and now - rpm_q[0] > 60.0:
                    rpm_q.popleft()
                if len(rpm_q) >= self.rpm_limit:
                    continue

                # 3. Check sliding window RPD (past 86400s)
                rpd_q = self._rpd_windows[key]
                while rpd_q and now - rpd_q[0] > 86400.0:
                    rpd_q.popleft()
                if len(rpd_q) >= self.rpd_limit:
                    continue

                # 4. Check sliding window TPM (past 60s)
                tpm_q = self._tpm_windows[key]
                while tpm_q and now - tpm_q[0][0] > 60.0:
                    tpm_q.popleft()
                current_tpm = sum(tok for _, tok in tpm_q)
                if current_tpm + estimated_tokens > self.tpm_limit:
                    continue

                # Key is valid and within limits; record usage
                rpm_q.append(now)
                rpd_q.append(now)
                tpm_q.append((now, estimated_tokens))
                return key

            return None

    def acquire_key(
        self,
        estimated_tokens: int = 0,
        timeout: float = 60.0,
        retry_interval: float = 1.0,
    ) -> Optional[str]:
        """
        Attempts to acquire a valid key within limits, waiting up to `timeout` seconds
        if keys are temporarily in cooldown or at capacity.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            key = self.get_key(estimated_tokens=estimated_tokens)
            if key is not None:
                return key
            with self._lock:
                if not self.api_keys:
                    return None
            time.sleep(retry_interval)
        return None

    def get_key_status(self) -> List[Dict[str, Any]]:
        """Returns snapshot of key status, cooldowns, RPM and TPM for monitoring."""
        with self._lock:
            now = time.time()
            status: List[Dict[str, Any]] = []
            for k in self.api_keys:
                cooldown_remaining = max(0.0, self._cooldowns.get(k, 0.0) - now)
                rpm_count = len([t for t in self._rpm_windows[k] if now - t <= 60.0])
                tpm_count = sum(tok for t, tok in self._tpm_windows[k] if now - t <= 60.0)
                status.append({
                    "key_masked": f"{k[:4]}...{k[-4:]}" if len(k) > 8 else "***",
                    "cooldown_remaining": cooldown_remaining,
                    "rpm": rpm_count,
                    "tpm": tpm_count,
                })
            return status
