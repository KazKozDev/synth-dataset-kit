from __future__ import annotations

import logging
from typing import Any

# ─── BUILT-IN BENCHMARK SIGNATURES ──────────────────────────────────────────
# We store characteristic n-gram patterns from popular benchmarks.
# These are NOT the full benchmarks — just enough to detect overlap.
# In production, you'd load full benchmark sets from HuggingFace.

BENCHMARK_SIGNATURES: dict[str, list[str]] = {
    "mmlu": [
        "the longest wavelength of light",
        "which of the following is not a characteristic",
        "a longest wavelength",
        "according to piaget",
        "the supreme court case",
        "which of the following statements about",
        "in the context of operant conditioning",
        "the primary function of",
        "which of the following best describes",
        "a search warrant is not required",
    ],
    "gsm8k": [
        "how many more apples",
        "janet's ducks lay 16 eggs per day",
        "a robe takes 2 bolts of blue fiber",
        "josh decides to try flipping a house",
        "every day, wendi feeds each of her chickens",
        "kylar went to the store to buy glasses",
        "toulouse has twice as many sheep as charleston",
        "carla is downloading a 200 GB file",
        "john drives for 3 hours at a speed of 60 mph",
        "elaine initially had 20 pokemon cards",
    ],
    "humaneval": [
        "def has_close_elements",
        "def separate_paren_groups",
        "def truncate_number",
        "def below_zero",
        "def mean_absolute_deviation",
        "def intersperse",
        "def parse_nested_parens",
        "def filter_by_substring",
        "def sum_product",
        "def rolling_max",
    ],
    "arc": [
        "which of these would help to prevent infections",
        "when a guitar string is plucked",
        "which of the following is a fossil fuel",
        "a student is studying the revolution of earth",
        "what tool is used to measure the volume of a liquid",
        "which property of air does a barometer measure",
        "which of the following is the best example of erosion",
        "a group of students is testing different materials",
    ],
    "hellaswag": [
        "a woman is outside with a bucket",
        "the man is sitting on a roof",
        "a lady walks to a drum set",
        "two women are sitting on a bed",
        "a man is standing in a kitchen",
    ],
}


logger = logging.getLogger(__name__)


def _extract_mmlu(record: dict[str, Any]) -> str:
    parts = [str(record.get("question", ""))]
    choices = record.get("choices") or []
    if isinstance(choices, list):
        parts.extend(str(choice) for choice in choices)
    return " ".join(part for part in parts if part).strip()


def _extract_gsm8k(record: dict[str, Any]) -> str:
    return str(record.get("question", "")).strip()


def _extract_humaneval(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(field, "")).strip()
        for field in ["prompt", "canonical_solution", "test"]
        if record.get(field)
    )


def _extract_arc(record: dict[str, Any]) -> str:
    parts = [str(record.get("question", ""))]
    choices = record.get("choices", {})
    if isinstance(choices, dict):
        parts.extend(str(text) for text in choices.get("text", []) if text)
    return " ".join(part for part in parts if part).strip()


def _extract_hellaswag(record: dict[str, Any]) -> str:
    parts = [str(record.get("ctx", ""))]
    parts.extend(str(ending) for ending in record.get("endings", []) if ending)
    return " ".join(part for part in parts if part).strip()


BENCHMARK_DATASETS: dict[str, dict[str, Any]] = {
    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "validation",
        "extractor": _extract_mmlu,
    },
    "gsm8k": {
        "path": "gsm8k",
        "name": "main",
        "split": "test",
        "extractor": _extract_gsm8k,
    },
    "humaneval": {
        "path": "openai/openai_humaneval",
        "split": "test",
        "extractor": _extract_humaneval,
    },
    "arc": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "validation",
        "extractor": _extract_arc,
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "split": "validation",
        "extractor": _extract_hellaswag,
    },
}

CACHE_MANIFEST_VERSION = 1


def _extract_mmlu(record: dict[str, Any]) -> str:
    parts = [str(record.get("question", ""))]
    choices = record.get("choices") or []
    if isinstance(choices, list):
        parts.extend(str(choice) for choice in choices)
    return " ".join(part for part in parts if part).strip()


def _extract_gsm8k(record: dict[str, Any]) -> str:
    return str(record.get("question", "")).strip()


def _extract_humaneval(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(field, "")).strip()
        for field in ["prompt", "canonical_solution", "test"]
        if record.get(field)
    )


def _extract_arc(record: dict[str, Any]) -> str:
    parts = [str(record.get("question", ""))]
    choices = record.get("choices", {})
    if isinstance(choices, dict):
        parts.extend(str(text) for text in choices.get("text", []) if text)
    return " ".join(part for part in parts if part).strip()


def _extract_hellaswag(record: dict[str, Any]) -> str:
    parts = [str(record.get("ctx", ""))]
    parts.extend(str(ending) for ending in record.get("endings", []) if ending)
    return " ".join(part for part in parts if part).strip()
