"""Language detection, deflection filtering, and topic classification."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

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

DEFLECTION_PATTERNS = [
    r"\bcheck\s+dm\b",
    r"\bchk\s+dm\b",
    r"\bcheck\s+inbox\b",
    r"\binbox\s+me\b",
    r"\bcheck\s+your\s+dm\b",
    r"\bdm\s+karo\b",
    r"\bdm\s+krlo\b",
    r"\bsent\s+dm\b",
]

OPINION_PATTERNS = [
    r"\bi think\b", r"\bin my opinion\b", r"\bi feel\b", r"\bmy view\b",
    r"\bmere khayal\b", r"\bmujhe lagta\b", r"\bmjhe lgta\b"
]

EXPERIENCE_PATTERNS = [
    r"\bi worked\b", r"\bi have been\b", r"\bmy experience\b", r"\bwhen i was\b",
    r"\bmain ne\b", r"\bmaine\b", r"\bmera experience\b"
]


def detect_language(text: Optional[str]) -> Optional[str]:
    """
    Detects language among: english, hinglish, hindi, urdu.
    Returns None if text is empty or purely emojis/symbols.
    """
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


def is_deflection(text: Optional[str]) -> bool:
    """Detects boilerplate DM deflection responses like 'check dm', 'chk dm'."""
    if not text:
        return False
    low = text.lower().strip()
    return any(re.search(pat, low) for pat in DEFLECTION_PATTERNS)


def is_style_only(text: Optional[str], question: Optional[str] = None) -> bool:
    """Identifies short reaction responses (e.g. 'good', 'sure', emojis) that are style-only."""
    if not text:
        return True
    cleaned = text.strip()
    if EMOJI_OR_PUNCT.match(cleaned):
        return True
    if len(cleaned) <= 25 and not question:
        return True
    return False


def make_topics(text: str, lexicon: Dict[str, List[str]], filename: str = "") -> List[str]:
    """Matches text and filename against configurable topic lexicon."""
    topics: List[str] = []
    if not text:
        return topics

    combined = f"{text} {filename}".lower()
    for topic, patterns in lexicon.items():
        for pat in patterns:
            # Word boundary search if simple word
            safe_pat = rf"\b{re.escape(pat)}\b" if pat.isalnum() else re.escape(pat)
            if re.search(safe_pat, combined):
                topics.append(topic)
                break

    return sorted(list(set(topics)))


def infer_booleans(text: str) -> Tuple[bool, bool]:
    """Infers (is_experience, is_opinion) flags from text."""
    low = text.lower()
    opinion = bool(re.search(r"(?:%s)" % "|".join(OPINION_PATTERNS), low))
    experience = bool(re.search(r"(?:%s)" % "|".join(EXPERIENCE_PATTERNS), low))
    return experience, opinion
