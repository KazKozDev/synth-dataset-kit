from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .signatures import BENCHMARK_DATASETS, BENCHMARK_SIGNATURES

logger = logging.getLogger(__name__)


class DecontaminationLoaderMixin:
    """DecontaminationLoaderMixin functionality."""

    def _load_benchmark_texts(self, bench_name: str) -> tuple[list[str], str, str | None]:
        """Load benchmark texts from datasets when available, fallback to signatures."""
        if self.use_benchmark_datasets:
            texts, source, error = self._load_from_cache_or_datasets(bench_name)
            if texts:
                logger.info(f"Loaded {len(texts)} benchmark samples for {bench_name} from {source}")
                return texts, source, None
            return list(BENCHMARK_SIGNATURES.get(bench_name, [])), "fallback_signatures", error

        return (
            list(BENCHMARK_SIGNATURES.get(bench_name, [])),
            "fallback_signatures",
            "benchmark dataset loading disabled",
        )

    def _load_from_cache_or_datasets(self, bench_name: str) -> tuple[list[str], str, str | None]:
        """Load benchmark texts from cache first, then datasets."""
        cached_texts = self._load_cached_texts(bench_name)
        if cached_texts:
            logger.info(f"Loaded {len(cached_texts)} benchmark samples for {bench_name} from cache")
            return cached_texts, "datasets_cache", None

        texts, error = self._load_from_datasets(bench_name)
        if texts:
            self._store_cached_texts(bench_name, texts)
            return texts, "datasets", None
        return [], "fallback_signatures", error

    def _load_from_datasets(self, bench_name: str) -> tuple[list[str], str | None]:
        """Try to load benchmark samples via HuggingFace datasets."""
        config = BENCHMARK_DATASETS.get(bench_name)
        if not config:
            return [], "no dataset config registered"

        try:
            from datasets import load_dataset
        except ImportError:
            logger.debug("datasets not installed; skipping benchmark dataset loading")
            return [], "datasets not installed"

        try:
            dataset = load_dataset(
                path=config["path"],
                name=config.get("name"),
                split=config.get("split", "train"),
            )
        except Exception as e:
            logger.warning(f"Failed to load benchmark dataset for {bench_name}: {e}")
            return [], f"datasets load failed: {e}"

        extractor: Callable[[dict[str, Any]], str] = config["extractor"]
        texts: list[str] = []
        for index, record in enumerate(dataset):
            if (
                not self.load_full_benchmark_corpus
                and self.benchmark_sample_limit > 0
                and index >= self.benchmark_sample_limit
            ):
                break
            text = extractor(record)
            if text:
                texts.append(text)

        if not texts:
            return [], "datasets loaded but no benchmark samples extracted"
        return texts, None
