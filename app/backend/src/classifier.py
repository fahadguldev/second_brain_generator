"""Topic classification, language detection, and brain_config.yaml loader for Second Brain App."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

from src.config import settings

logger = logging.getLogger(__name__)

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
EMOJI_OR_PUNCT = re.compile(r"^[\W\d\s\u00a9-\u329f]*$")

HINGLISH_MARKERS: Set[str] = {
    "kya", "hai", "nahi", "nhi", "mein", "main", "bhai", "kar", "ka", "ki", "ke",
    "ho", "hain", "aap", "tum", "diya", "tha", "the", "hoga", "hoon", "ko",
    "se", "ab", "to", "bhi", "krske", "chahiye", "chahye", "ja", "ga", "raha",
    "rahy", "rahay", "rahe", "dono", "sath", "saath", "milke", "apna", "apni",
    "mere", "mera", "meri", "log", "krna", "karna", "karo", "krta", "karta",
    "krte", "sakta", "sakti", "sakte", "bna", "bana", "zarur", "zyada", "kam",
    "samajh", "behtar", "shuru", "khud", "dijiye", "kijiye", "krlo", "denge",
    "milta", "milte", "yar", "dear", "agr", "qk", "hy", "nai", "skty", "waghera",
}

HINGLISH_IGNORE: Set[str] = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "it", "for", "with"
}

_DEFAULT_PERSONAL_TOPICS: Set[str] = {
    "family", "marriage", "studies", "future", "personal", "life"
}

_cached_config: Optional[Dict[str, Any]] = None
_cached_mtime: float = 0.0


def _find_config_path() -> Optional[Path]:
    """Finds brain_config.yaml across standard locations."""
    candidates = []
    if getattr(settings, "BRAIN_CONFIG_PATH", None):
        candidates.append(Path(settings.BRAIN_CONFIG_PATH))

    # Current working directory / backend dir
    cwd = Path.cwd()
    candidates.append(cwd / "brain_config.yaml")
    candidates.append(cwd / "backend" / "brain_config.yaml")
    
    # Directory relative to this file
    this_dir = Path(__file__).resolve().parent.parent  # backend/
    candidates.append(this_dir / "brain_config.yaml")
    candidates.append(this_dir.parent / "brain_config.yaml")
    candidates.append(this_dir.parent.parent / "second_brain_generator" / "brain_config.yaml")

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p.resolve()
        except Exception:
            continue
    return None


def get_brain_config() -> Dict[str, Any]:
    """
    Loads and caches brain_config.yaml, reloading automatically when modified.
    """
    global _cached_config, _cached_mtime

    config_path = _find_config_path()
    if not config_path:
        return {}

    try:
        mtime = config_path.stat().st_mtime
        if _cached_config is not None and mtime == _cached_mtime:
            return _cached_config

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        _cached_config = data
        _cached_mtime = mtime
        logger.info("Loaded brain_config from %s (topics: %d)", config_path, len(data.get("topics", {})))
        return _cached_config
    except Exception as e:
        logger.warning("Failed to load brain_config from %s: %s", config_path, e)
        return _cached_config or {}


def get_topics_lexicon() -> Dict[str, List[str]]:
    """Returns the topic taxonomy dictionary from brain_config.yaml."""
    cfg = get_brain_config()
    return cfg.get("topics", {})


def get_personal_topics() -> Set[str]:
    """Returns set of topics designated as personal domain."""
    cfg = get_brain_config()
    configured = cfg.get("personal_topics")
    if configured and isinstance(configured, list):
        return {str(t).lower() for t in configured}
    return _DEFAULT_PERSONAL_TOPICS


def classify_topics(
    text: str,
    question: Optional[str] = None,
    filename: str = "",
    lexicon: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Matches text, optional question, and filename against configured topic patterns.
    """
    if lexicon is None:
        lexicon = get_topics_lexicon()

    if not lexicon:
        return []

    combined = f"{text or ''} {question or ''} {filename or ''}".replace("_", " ").replace("-", " ").lower()
    matched: List[str] = []

    for topic, patterns in lexicon.items():
        for pat in patterns:
            safe_pat = rf"\b{re.escape(pat.lower())}\b" if pat.isalnum() else re.escape(pat.lower())
            if re.search(safe_pat, combined):
                matched.append(topic)
                break

    return sorted(list(set(matched)))


def determine_domain(topics: List[str]) -> str:
    """
    Determines a record domain by majority vote across matched categories.

    Topic classification returns one entry per matched category. Each category
    belongs to a domain through ``personal_topics``; unmatched category names
    are treated as professional. A tie uses the configured professional
    default.
    """
    personal = get_personal_topics()
    normalized_topics = {topic.lower() for topic in topics}
    personal_count = len(normalized_topics & personal)
    professional_count = len(normalized_topics) - personal_count
    return "personal" if personal_count > professional_count else "professional"


def detect_language(text: Optional[str]) -> Optional[str]:
    """Detects language: english, hinglish, hindi, or urdu."""
    if not text or not str(text).strip():
        return None

    cleaned = str(text).strip()
    if EMOJI_OR_PUNCT.match(cleaned):
        return None

    if DEVANAGARI.search(cleaned):
        return "hindi"

    if ARABIC.search(cleaned):
        return "urdu"

    tokens = [t.lower() for t in re.findall(r"[\w'\u2019]+", cleaned)]
    if not tokens:
        return None

    sig = [t for t in tokens if t not in HINGLISH_IGNORE]
    hits = sum(1 for t in sig if t in HINGLISH_MARKERS)

    return "hinglish" if hits >= 2 else "english"


def list_categories() -> List[Dict[str, Any]]:
    """Returns all categories with their keywords, domain, and count."""
    cfg = get_brain_config()
    topics = cfg.get("topics", {})
    personal_set = get_personal_topics()

    result = []
    for name, patterns in topics.items():
        domain = "personal" if name.lower() in personal_set else "professional"
        result.append({
            "name": name,
            "keywords": patterns or [],
            "domain": domain,
        })
    return sorted(result, key=lambda x: x["name"])


def save_brain_config(data: Dict[str, Any]) -> bool:
    """Writes updated brain_config dictionary back to file and updates cache."""
    global _cached_config, _cached_mtime
    config_path = _find_config_path()
    if not config_path:
        config_path = Path(__file__).resolve().parent.parent / "brain_config.yaml"

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        _cached_config = data
        _cached_mtime = config_path.stat().st_mtime
        return True
    except Exception as e:
        logger.error("Failed to save brain_config to %s: %s", config_path, e)
        return False


def save_category(name: str, keywords: List[str], domain: str = "professional") -> Dict[str, Any]:
    """Adds or updates a category in brain_config.yaml."""
    cfg = get_brain_config()
    if "topics" not in cfg or not isinstance(cfg["topics"], dict):
        cfg["topics"] = {}

    clean_name = name.strip().lower()
    clean_kws = [k.strip().lower() for k in keywords if k and k.strip()]

    cfg["topics"][clean_name] = clean_kws

    # Update personal_topics list
    personal_list = list(cfg.get("personal_topics") or [])
    if domain == "personal":
        if clean_name not in [p.lower() for p in personal_list]:
            personal_list.append(clean_name)
    else:
        personal_list = [p for p in personal_list if p.lower() != clean_name]
    cfg["personal_topics"] = personal_list

    save_brain_config(cfg)
    return {
        "name": clean_name,
        "keywords": clean_kws,
        "domain": domain,
    }


def remove_category(name: str) -> bool:
    """Deletes a category from brain_config.yaml."""
    cfg = get_brain_config()
    clean_name = name.strip().lower()
    topics = cfg.get("topics", {})
    if clean_name in topics:
        del topics[clean_name]
        cfg["topics"] = topics

        # Remove from personal_topics
        personal_list = list(cfg.get("personal_topics") or [])
        cfg["personal_topics"] = [p for p in personal_list if p.lower() != clean_name]

        save_brain_config(cfg)
        return True
    return False
