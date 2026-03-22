from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from synth_dataset_kit.models import Dataset, Example

from .cache import DecontaminationCacheMixin
from .embeddings import DecontaminationEmbeddingMixin
from .loaders import DecontaminationLoaderMixin
from .signatures import BENCHMARK_SIGNATURES
from .utils import _get_ngrams, _normalize, _text_hash

logger = logging.getLogger(__name__)


class Decontaminator(
    DecontaminationLoaderMixin, DecontaminationEmbeddingMixin, DecontaminationCacheMixin
):
    """Check datasets for benchmark contamination."""

    def __init__(
        self,
        benchmarks: list[str] | None = None,
        similarity_threshold: float = 0.85,
        ngram_size: int = 5,
        use_benchmark_datasets: bool = True,
        load_full_benchmark_corpus: bool = False,
        benchmark_sample_limit: int = 200,
        method: str = "hybrid",
        cache_dir: str = ".sdk_cache/benchmarks",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_index_backend: str = "json",
        embedding_top_k: int = 3,
        review_threshold: float = 0.92,
        hard_fail_methods: list[str] | None = None,
        review_methods: list[str] | None = None,
    ):
        self.benchmarks = benchmarks or list(BENCHMARK_SIGNATURES.keys())
        self.threshold = similarity_threshold
        self.ngram_size = ngram_size
        self.use_benchmark_datasets = use_benchmark_datasets
        self.load_full_benchmark_corpus = load_full_benchmark_corpus
        self.benchmark_sample_limit = benchmark_sample_limit
        self.method = method
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.embedding_index_backend = embedding_index_backend.lower().strip()
        self.embedding_top_k = max(1, embedding_top_k)
        self.review_threshold = review_threshold
        self.hard_fail_methods = set(hard_fail_methods or ["exact", "ngram"])
        self.review_methods = set(review_methods or ["substring", "embedding"])
        self.cache_manifest = self._load_cache_manifest()

        # Materialize benchmark texts from datasets when available, otherwise use signatures.
        self._benchmark_texts: dict[str, list[str]] = {}
        self.benchmark_sources: dict[str, str] = {}
        self.benchmark_sample_counts: dict[str, int] = {}
        self.benchmark_load_errors: dict[str, str] = {}
        self._benchmark_hashes: dict[str, set[str]] = {}
        self._benchmark_ngrams: dict[str, list[set[str]]] = {}
        for bench_name in self.benchmarks:
            texts, source, error = self._load_benchmark_texts(bench_name)
            self._benchmark_texts[bench_name] = texts
            self.benchmark_sources[bench_name] = source
            self.benchmark_sample_counts[bench_name] = len(texts)
            if error:
                self.benchmark_load_errors[bench_name] = error
            self._benchmark_hashes[bench_name] = {_text_hash(sig) for sig in texts}
            self._benchmark_ngrams[bench_name] = [
                _get_ngrams(sig, self.ngram_size) for sig in texts
            ]

        logger.info(f"Decontaminator initialized with {len(self.benchmarks)} benchmarks")

    def check_example(self, example: Example) -> list[str]:
        """Check a single example against all benchmarks.

        Returns list of benchmark names where contamination was detected.
        """
        flags, _ = self.check_example_with_evidence(example)
        return flags

    def check_example_with_evidence(
        self,
        example: Example,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Check a single example and return flags with structured evidence."""
        flags: list[str] = []
        evidence: list[dict[str, Any]] = []
        user_text = example.user_message
        asst_text = example.assistant_message
        combined = f"{user_text} {asst_text}"
        combined_ngrams = _get_ngrams(combined, self.ngram_size)
        combined_norm = _normalize(combined)
        combined_hash = _text_hash(combined)

        for bench_name in self.benchmarks:
            bench_texts = self._benchmark_texts.get(bench_name, [])
            bench_ngrams = self._benchmark_ngrams.get(bench_name, [])
            bench_hashes = self._benchmark_hashes.get(bench_name, set())

            exact_match = False
            ngram_match = False

            if self.method in {"exact", "hybrid"} and combined_hash in bench_hashes:
                flags.append(bench_name)
                exact_match = True
                evidence.append(
                    {
                        "benchmark": bench_name,
                        "method": "exact",
                        "confidence": 1.0,
                        "benchmark_item_index": 0,
                        "reason": "normalized_exact_match",
                    }
                )

            if self.method in {"ngram", "hybrid", "embedding"} and not exact_match:
                for index, sig_ngrams in enumerate(bench_ngrams):
                    if not sig_ngrams:
                        continue
                    overlap = sig_ngrams & combined_ngrams
                    score = len(overlap) / len(sig_ngrams)
                    if score >= self.threshold:
                        if bench_name not in flags:
                            flags.append(bench_name)
                        ngram_match = True
                        evidence.append(
                            {
                                "benchmark": bench_name,
                                "method": "ngram",
                                "confidence": round(score, 4),
                                "benchmark_item_index": index,
                                "matched_text": bench_texts[index][:200]
                                if index < len(bench_texts)
                                else "",
                                "reason": f"{self.ngram_size}gram_overlap",
                            }
                        )
                        break

            if self.method in {"ngram", "hybrid"} and not exact_match and not ngram_match:
                for index, sig in enumerate(bench_texts):
                    if _normalize(sig) in combined_norm:
                        if bench_name not in flags:
                            flags.append(bench_name)
                        evidence.append(
                            {
                                "benchmark": bench_name,
                                "method": "substring",
                                "confidence": 0.99,
                                "benchmark_item_index": index,
                                "matched_text": sig[:200],
                                "reason": "benchmark_text_substring_match",
                            }
                        )
                        break

        return flags, evidence

    def check_dataset(self, dataset: Dataset) -> Dataset:
        """Check all examples in a dataset and flag contaminated ones."""
        if self.method == "embedding":
            dataset = self.check_with_embeddings(dataset)
            return self._apply_decision_policy(dataset)
        if self.method == "hybrid":
            dataset = self._apply_rule_based_checks(dataset)
            dataset = self.check_with_embeddings(dataset, append=True)
            return self._apply_decision_policy(dataset)
        dataset = self._apply_rule_based_checks(dataset)
        return self._apply_decision_policy(dataset)

    def _apply_rule_based_checks(self, dataset: Dataset) -> Dataset:
        """Apply exact/ngram-based checks and attach evidence."""
        total_flags = 0
        flag_counts: dict[str, int] = defaultdict(int)

        for example in dataset.examples:
            flags, evidence = self.check_example_with_evidence(example)
            example.decontamination_flags = flags
            example.decontamination_evidence = evidence
            if flags:
                total_flags += 1
                for f in flags:
                    flag_counts[f] += 1

        if total_flags > 0:
            logger.warning(f"Contamination detected: {total_flags}/{dataset.size} examples flagged")
            for bench, count in flag_counts.items():
                logger.warning(f"  {bench}: {count} matches")
        else:
            logger.info(f"No contamination detected in {dataset.size} examples")

        return dataset

    def _apply_decision_policy(self, dataset: Dataset) -> Dataset:
        """Convert raw evidence into a consistent block/review/clean verdict."""
        for example in dataset.examples:
            evidence = list(example.decontamination_evidence or [])
            if not evidence:
                example.metadata["contamination_verdict"] = "clean"
                example.metadata["contamination_confidence"] = 0.0
                example.metadata["contamination_reason_codes"] = []
                continue

            max_confidence = max(float(item.get("confidence", 0.0) or 0.0) for item in evidence)
            reason_codes = [
                f"{item.get('benchmark', 'unknown')}:{item.get('method', 'unknown')}"
                for item in evidence
            ]
            methods = {str(item.get("method", "unknown")) for item in evidence}
            benchmarks_by_method: dict[str, set[str]] = defaultdict(set)
            for item in evidence:
                benchmarks_by_method[str(item.get("method", "unknown"))].add(
                    str(item.get("benchmark", "unknown"))
                )

            exact_or_hard = any(method in self.hard_fail_methods for method in methods)
            multi_signal = any(
                len(methods_for_bench) > 1
                for methods_for_bench in self._methods_by_benchmark(evidence).values()
            )
            review_match = any(
                (
                    method in self.review_methods
                    and any(
                        float(item.get("confidence", 0.0) or 0.0) >= self.review_threshold
                        and str(item.get("method", "unknown")) == method
                        for item in evidence
                    )
                )
                for method in methods
            )

            verdict = "clean"
            if exact_or_hard or multi_signal:
                verdict = "block"
            elif review_match:
                verdict = "review"

            example.metadata["contamination_verdict"] = verdict
            example.metadata["contamination_confidence"] = round(max_confidence, 4)
            example.metadata["contamination_reason_codes"] = sorted(set(reason_codes))
            example.metadata["contamination_methods"] = sorted(methods)
            example.metadata["contamination_benchmarks"] = sorted(
                {
                    benchmark
                    for method_benchmarks in benchmarks_by_method.values()
                    for benchmark in method_benchmarks
                }
            )
        return dataset

    def _methods_by_benchmark(
        self,
        evidence: list[dict[str, Any]],
    ) -> dict[str, set[str]]:
        """Group evidence methods by benchmark."""
        grouped: dict[str, set[str]] = defaultdict(set)
        for item in evidence:
            grouped[str(item.get("benchmark", "unknown"))].add(str(item.get("method", "unknown")))
        return grouped
