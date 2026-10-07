"""Environment and secrets manager for second_brain_generator."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import List, Optional

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    # Graceful fallback if dotenv is not yet installed
    def find_dotenv(filename: str = ".env", raise_error_if_not_found: bool = False, usecwd: bool = False) -> str:
        cur = Path.cwd()
        for parent in [cur, *cur.parents]:
            candidate = parent / filename
            if candidate.is_file():
                return str(candidate)
        return ""

    def load_dotenv(dotenv_path: Optional[str | Path] = None, override: bool = False, **kwargs) -> bool:
        if not dotenv_path:
            dotenv_path = find_dotenv()
        if not dotenv_path:
            return False
        p = Path(dotenv_path)
        if not p.is_file():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if override or k not in os.environ:
                        os.environ[k] = v
            return True
        except Exception:
            return False

logger = logging.getLogger(__name__)

GEMINI_KEY_PATTERN = re.compile(r"^GEMINI_API_KEY_(\d+)$")
DUMMY_KEY_INDICATORS = {"your_api_key", "your_gemini_api_key", "placeholder", "xxx"}


class MissingSecretError(Exception):
    """Raised when a required secret is absent from the environment."""
    pass


@dataclass(frozen=True)
class QdrantCredentials:
    url: Optional[str]
    api_key: Optional[str]
    collection: str


def mask_secret(secret: Optional[str]) -> str:
    """
    Safely masks an API key or token for logs and status outputs.

    Examples:
        >>> mask_secret("AQ.Ab8RN6LjmI2iajxSwxWCMspBb5gNbcocQvHNpKx3TFs8o0VY0g")
        'AQ.A...0VY0g'
        >>> mask_secret("short")
        '***'
        >>> mask_secret(None)
        '<none>'
        >>> mask_secret("")
        '<none>'
    """
    if not secret:
        return "<none>"
    s = secret.strip()
    if not s:
        return "<none>"
    if len(s) <= 8:
        return "***"
    return f"{s[:4]}...{s[-5:]}"


def load_environment(
    dotenv_path: Optional[Union[str, Path]] = None,
    override: bool = False,
) -> bool:
    """
    Load environment variables from a .env file.

    Args:
        dotenv_path: Explicit path to .env. If None, searches upwards from cwd.
        override: Whether to override existing environment variables.

    Returns:
        bool: True if a .env file was located and loaded, False otherwise.
    """
    if dotenv_path is not None:
        try:
            target = Path(dotenv_path).resolve()
            if target.is_file():
                loaded = load_dotenv(dotenv_path=target, override=override)
                logger.debug("Loaded .env from %s", target)
                return True
        except (OSError, ValueError):
            pass
        logger.debug("Specified .env path does not exist: %s", dotenv_path)
        return False

    try:
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(dotenv_path=found, override=override)
            logger.debug("Discovered and loaded .env from %s", found)
            return True
    except Exception as err:
        logger.debug("Failed while discovering .env: %s", err)

    return False


def get_gemini_keys(
    require: bool = False,
    reload_env: bool = False,
) -> List[str]:
    """
    Discover all configured Gemini API keys (GEMINI_API_KEY, GEMINI_API_KEY_1..N).

    Discovers arbitrary N numbered keys, sorts them numerically by index (0, 1, 2, ..., 10, ...),
    and deduplicates identical keys while preserving priority order.

    Args:
        require: If True, raises MissingSecretError when no valid keys are found.
        reload_env: If True, reloads .env from disk before checking.

    Returns:
        List[str]: Discovered unique Gemini API keys.
    """
    if reload_env:
        load_environment()

    candidate_keys: List[tuple[int, str]] = []

    # 1. Base key: GEMINI_API_KEY
    base_key = os.getenv("GEMINI_API_KEY", "").strip()
    if base_key and not any(ind in base_key.lower() for ind in DUMMY_KEY_INDICATORS):
        candidate_keys.append((0, base_key))

    # 2. Numbered keys: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ..., GEMINI_API_KEY_N
    # Iterate over an atomic snapshot list of environment variable names to prevent
    # KeyError crashes when variables are deleted concurrently by background threads.
    for env_var in list(os.environ):
        match = GEMINI_KEY_PATTERN.match(env_var)
        if match:
            idx = int(match.group(1))
            val = os.environ.get(env_var)
            if val:
                val_clean = val.strip()
                if val_clean and not any(ind in val_clean.lower() for ind in DUMMY_KEY_INDICATORS):
                    candidate_keys.append((idx, val_clean))

    # Sort numerically by index (0 first, then 1, 2, ..., 10, ...)
    candidate_keys.sort(key=lambda item: item[0])

    # Deduplicate while preserving order
    unique_keys: List[str] = []
    seen = set()
    for _, key in candidate_keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)

    if not unique_keys:
        msg = (
            "No Gemini API keys detected. Set GEMINI_API_KEY or GEMINI_API_KEY_1..N "
            "in your .env file or environment variables."
        )
        if require:
            raise MissingSecretError(msg)
        if not is_offline_mode():
            logger.warning("%s Running in offline/unauthenticated mode.", msg)

    return unique_keys


def get_qdrant_credentials(
    require: bool = False,
    reload_env: bool = False,
    default_collection: str = "second_brain",
) -> QdrantCredentials:
    """
    Discover Qdrant connection credentials from environment variables.

    Args:
        require: If True, raises MissingSecretError when QDRANT_URL is absent.
        reload_env: If True, reloads .env from disk before checking.
        default_collection: Default collection name if QDRANT_COLLECTION is unset.

    Returns:
        QdrantCredentials: Dataclass with url, api_key, and collection.
    """
    if reload_env:
        load_environment()

    url = os.getenv("QDRANT_URL", "").strip() or None
    api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    collection = os.getenv("QDRANT_COLLECTION", "").strip() or default_collection

    if not url and require:
        raise MissingSecretError(
            "QDRANT_URL is not set. Please provide your Qdrant cluster endpoint "
            "in .env or via the QDRANT_URL environment variable."
        )

    return QdrantCredentials(url=url, api_key=api_key, collection=collection)


def is_offline_mode() -> bool:
    """
    Check if offline testing/mock mode is active.
    """
    val = os.getenv("SECOND_BRAIN_OFFLINE", "").strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    val_alt = os.getenv("OFFLINE_MODE", "").strip().lower()
    return val_alt in {"1", "true", "yes", "on"}


# Auto-load environment on module import safely
load_environment()
