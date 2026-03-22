from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _get_ngrams(text: str, n: int = 5) -> set[str]:
    """Extract word n-grams from text."""
    words = _normalize(text).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _text_hash(text: str) -> str:
    """Get a hash of normalized text."""
    return hashlib.md5(_normalize(text).encode()).hexdigest()
