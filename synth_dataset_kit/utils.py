"""Small shared helpers."""

from __future__ import annotations

import re


def safe_slug(value: str, max_length: int = 64) -> str:
    """Convert free text to a filesystem-friendly, stable slug."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "dataset"
    if len(cleaned) <= max_length:
        return cleaned
    trimmed = cleaned[:max_length].rstrip("_")
    return trimmed or "dataset"
