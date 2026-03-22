"""Benchmark decontamination: detect overlap with known evaluation datasets.
Checks generated examples against MMLU, GSM8K, HumanEval, ARC, HellaSwag
using n-gram overlap and (optionally) embedding similarity.
"""

from __future__ import annotations

from .core import Decontaminator
from .signatures import BENCHMARK_SIGNATURES

__all__ = ["Decontaminator", "BENCHMARK_SIGNATURES"]
