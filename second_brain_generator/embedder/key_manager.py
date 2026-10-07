"""Multi-key manager with cooldown, eviction, and sliding-window rate limiting."""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class KeyManager:
    """
    Manages a pool of Gemini API keys with round-robin load distribution,
    sliding-window rate limiting (RPM/RPD), 60s cooldown on 429 quota exhaustion,
    and permanent key eviction on 401 authentication errors.
    """

    def __init__(
        self,
        api_keys: List[str],
        rpm_limit: int = 60,
        rpd_limit: int = 1000,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.api_keys: List[str] = [k.strip() for k in api_keys if k and k.strip()]
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.cooldown_seconds = cooldown_seconds

        self._lock = threading.Lock()
        self._index = 0
        self._cooldowns: Dict[str, float] = {}  # key -> cooldown_expiry
        self._rpm_windows: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)
        self._rpd_windows: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)

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
                logger.error("Key permanently evicted due to 401 authentication failure")

    def get_key(self) -> Optional[str]:
        """
        Retrieves the next available key that is not in cooldown and within rate limits.
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

                # Check cooldown
                if key in self._cooldowns:
                    if now < self._cooldowns[key]:
                        continue
                    else:
                        del self._cooldowns[key]

                # Check sliding window RPM (past 60s)
                rpm_q = self._rpm_windows[key]
                while rpm_q and now - rpm_q[0] > 60.0:
                    rpm_q.popleft()
                if len(rpm_q) >= self.rpm_limit:
                    continue

                # Check sliding window RPD (past 86400s)
                rpd_q = self._rpd_windows[key]
                while rpd_q and now - rpd_q[0] > 86400.0:
                    rpd_q.popleft()
                if len(rpd_q) >= self.rpd_limit:
                    continue

                # Key is available; record call timestamp
                rpm_q.append(now)
                rpd_q.append(now)
                return key

            return None
