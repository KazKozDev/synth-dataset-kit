"""Benchmark decontamination: detect overlap with known evaluation datasets.

This is a KEY differentiator — no other synthetic data tool does this automatically.
Checks generated examples against MMLU, GSM8K, HumanEval, ARC, HellaSwag
using n-gram overlap and (optionally) embedding similarity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from synth_dataset_kit.models import Dataset, Example

logger = logging.getLogger(__name__)

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


class Decontaminator:
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
                _get_ngrams(sig, self.ngram_size)
                for sig in texts
            ]

        logger.info(
            f"Decontaminator initialized with {len(self.benchmarks)} benchmarks"
        )

    def _cache_manifest_file(self) -> Path:
        return self.cache_dir / "cache_manifest.json"

    def _cache_scope_signature(self) -> dict[str, Any]:
        return {
            "benchmarks": sorted(self.benchmarks),
            "load_full_benchmark_corpus": self.load_full_benchmark_corpus,
            "benchmark_sample_limit": self.benchmark_sample_limit,
            "embedding_model": self.embedding_model,
            "embedding_index_backend": self.embedding_index_backend,
        }

    def _load_cache_manifest(self) -> dict[str, Any]:
        path = self._cache_manifest_file()
        if not path.exists():
            return {
                "version": CACHE_MANIFEST_VERSION,
                "scope": self._cache_scope_signature(),
                "artifacts": {},
            }
        try:
            with open(path) as f:
                payload = json.load(f)
            if int(payload.get("version", 0)) != CACHE_MANIFEST_VERSION:
                return {
                    "version": CACHE_MANIFEST_VERSION,
                    "scope": self._cache_scope_signature(),
                    "artifacts": {},
                }
            payload["scope"] = self._cache_scope_signature()
            payload.setdefault("artifacts", {})
            return payload
        except Exception:
            return {
                "version": CACHE_MANIFEST_VERSION,
                "scope": self._cache_scope_signature(),
                "artifacts": {},
            }

    def _save_cache_manifest(self) -> None:
        path = self._cache_manifest_file()
        self.cache_manifest["version"] = CACHE_MANIFEST_VERSION
        self.cache_manifest["scope"] = self._cache_scope_signature()
        with open(path, "w") as f:
            json.dump(self.cache_manifest, f, indent=2, ensure_ascii=False)

    def _update_manifest_artifact(self, key: str, metadata: dict[str, Any]) -> None:
        self.cache_manifest.setdefault("artifacts", {})
        self.cache_manifest["artifacts"][key] = metadata
        self._save_cache_manifest()

    def _artifact_manifest(self, key: str) -> dict[str, Any]:
        return dict(self.cache_manifest.get("artifacts", {}).get(key, {}))

    def _texts_hash(self, texts: list[str]) -> str:
        return hashlib.md5("\n".join(_normalize(text) for text in texts).encode()).hexdigest()

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
                                "matched_text": bench_texts[index][:200] if index < len(bench_texts) else "",
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
            logger.warning(
                f"Contamination detected: {total_flags}/{dataset.size} examples flagged"
            )
            for bench, count in flag_counts.items():
                logger.warning(f"  {bench}: {count} matches")
        else:
            logger.info(f"No contamination detected in {dataset.size} examples")

        return dataset

    def check_with_embeddings(self, dataset: Dataset, append: bool = False) -> Dataset:
        """Enhanced decontamination using sentence embeddings.

        Requires: pip install synth-dataset-kit[decontamination]
        """
        model = self._load_sentence_transformer()
        if model is None:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install synth-dataset-kit[decontamination]"
            )
            if append:
                return dataset
            return self._apply_rule_based_checks(dataset)

        logger.info("Running embedding-based decontamination...")

        bench_texts, bench_labels, bench_item_indexes, embedding_index, backend = self._load_or_build_embedding_index(model)

        if not bench_texts:
            return dataset

        # Encode dataset examples
        example_texts = [
            f"{e.user_message} {e.assistant_message}" for e in dataset.examples
        ]
        example_embeddings = model.encode(example_texts, normalize_embeddings=True)

        for i, example in enumerate(dataset.examples):
            top_k = min(self.embedding_top_k, len(bench_texts))
            top_scores, top_indices = self._search_embedding_index(
                embedding_index,
                example_embeddings[i],
                top_k,
                backend,
            )
            matched = False
            if not append:
                example.decontamination_evidence = []
            for match_rank, (similarity, match_idx) in enumerate(zip(top_scores, top_indices), start=1):
                similarity = float(similarity)
                match_idx = int(match_idx)
                if match_idx < 0:
                    continue
                if similarity < self.threshold:
                    continue
                matched = True
                bench_name = bench_labels[match_idx]
                if bench_name not in example.decontamination_flags:
                    example.decontamination_flags.append(bench_name)
                example.decontamination_evidence.append(
                    {
                        "benchmark": bench_name,
                        "method": "embedding",
                        "confidence": round(similarity, 4),
                        "benchmark_item_index": bench_item_indexes[match_idx],
                        "matched_text": bench_texts[match_idx][:200],
                        "match_rank": match_rank,
                        "reason": "semantic_embedding_similarity",
                    }
                )
                logger.debug(
                    f"Embedding match: example {example.id} ↔ {bench_name} "
                    f"(sim={similarity:.3f}, rank={match_rank})"
                )
            if matched:
                example.metadata["embedding_top_k_matches"] = [
                    {
                        "benchmark": bench_labels[idx],
                        "benchmark_item_index": bench_item_indexes[idx],
                        "similarity": round(float(score), 4),
                        "match_rank": rank,
                    }
                    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1)
                    if int(idx) >= 0 and float(score) >= self.threshold
                ]

        flagged = sum(1 for e in dataset.examples if e.decontamination_flags)
        logger.info(f"Embedding decontamination: {flagged}/{dataset.size} flagged")
        return dataset

    def _load_benchmark_texts(self, bench_name: str) -> tuple[list[str], str, str | None]:
        """Load benchmark texts from datasets when available, fallback to signatures."""
        if self.use_benchmark_datasets:
            texts, source, error = self._load_from_cache_or_datasets(bench_name)
            if texts:
                logger.info(
                    f"Loaded {len(texts)} benchmark samples for {bench_name} from {source}"
                )
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
            if not self.load_full_benchmark_corpus and self.benchmark_sample_limit > 0 and index >= self.benchmark_sample_limit:
                break
            text = extractor(record)
            if text:
                texts.append(text)

        if not texts:
            return [], "datasets loaded but no benchmark samples extracted"
        return texts, None

    def _cache_file(self, bench_name: str) -> Path:
        """Return cache file path for a benchmark."""
        limit = "full" if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0 else str(self.benchmark_sample_limit)
        return self.cache_dir / f"{bench_name}_{limit}.json"

    def _embedding_cache_file(self, bench_name: str) -> Path:
        """Return cache file path for benchmark embeddings."""
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.embedding_model)
        limit = "full" if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0 else str(self.benchmark_sample_limit)
        return self.cache_dir / f"{bench_name}_{limit}_{safe_model}_embeddings.json"

    def _embedding_index_file(self) -> Path:
        """Return cache file path for the combined benchmark embedding index."""
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.embedding_model)
        limit = "full" if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0 else str(self.benchmark_sample_limit)
        bench_key = "_".join(sorted(self.benchmarks))
        return self.cache_dir / f"{bench_key}_{limit}_{safe_model}_index.json"

    def _faiss_index_file(self) -> Path:
        """Return cache file path for the FAISS ANN index."""
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.embedding_model)
        limit = "full" if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0 else str(self.benchmark_sample_limit)
        bench_key = "_".join(sorted(self.benchmarks))
        return self.cache_dir / f"{bench_key}_{limit}_{safe_model}_index.faiss"

    def _faiss_metadata_file(self) -> Path:
        """Return sidecar metadata file for the FAISS index."""
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.embedding_model)
        limit = "full" if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0 else str(self.benchmark_sample_limit)
        bench_key = "_".join(sorted(self.benchmarks))
        return self.cache_dir / f"{bench_key}_{limit}_{safe_model}_index_meta.json"

    def _load_cached_texts(self, bench_name: str) -> list[str]:
        """Load cached benchmark texts if present."""
        cache_file = self._cache_file(bench_name)
        if not cache_file.exists():
            return []
        try:
            with open(cache_file) as f:
                payload = json.load(f)
            texts = payload.get("texts", [])
            manifest = self._artifact_manifest(f"texts:{bench_name}")
            if (
                int(payload.get("version", CACHE_MANIFEST_VERSION)) == CACHE_MANIFEST_VERSION
                and manifest.get("file") == cache_file.name
                and isinstance(texts, list)
            ):
                return [str(text) for text in texts if text]
        except Exception as e:
            logger.warning(f"Failed to load benchmark cache for {bench_name}: {e}")
        return []

    def _store_cached_texts(self, bench_name: str, texts: list[str]) -> None:
        """Persist benchmark texts for reuse."""
        cache_file = self._cache_file(bench_name)
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "version": CACHE_MANIFEST_VERSION,
                        "benchmark": bench_name,
                        "texts": texts,
                        "texts_hash": self._texts_hash(texts),
                    },
                    f,
                    ensure_ascii=False,
                )
            self._update_manifest_artifact(
                f"texts:{bench_name}",
                {
                    "type": "texts",
                    "file": cache_file.name,
                    "benchmark": bench_name,
                    "texts_count": len(texts),
                    "texts_hash": self._texts_hash(texts),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store benchmark cache for {bench_name}: {e}")

    def _load_sentence_transformer(self) -> Any | None:
        """Load the configured sentence-transformers model if available."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        return SentenceTransformer(self.embedding_model)

    def _load_faiss(self) -> Any | None:
        """Load FAISS if installed."""
        try:
            import faiss
        except ImportError:
            return None
        return faiss

    def _load_cached_embeddings(self, bench_name: str) -> list[list[float]]:
        """Load cached benchmark embeddings if present."""
        cache_file = self._embedding_cache_file(bench_name)
        if not cache_file.exists():
            return []
        try:
            with open(cache_file) as f:
                payload = json.load(f)
            vectors = payload.get("embeddings", [])
            manifest = self._artifact_manifest(f"embeddings:{bench_name}")
            if (
                int(payload.get("version", CACHE_MANIFEST_VERSION)) == CACHE_MANIFEST_VERSION
                and manifest.get("file") == cache_file.name
                and payload.get("texts_hash") == self._texts_hash(self._benchmark_texts.get(bench_name, []))
                and payload.get("model") == self.embedding_model
                and isinstance(vectors, list)
            ):
                return [
                    [float(value) for value in vector]
                    for vector in vectors
                    if isinstance(vector, list) and vector
                ]
        except Exception as e:
            logger.warning(f"Failed to load embedding cache for {bench_name}: {e}")
        return []

    def _store_cached_embeddings(
        self,
        bench_name: str,
        texts: list[str],
        embeddings: Any,
    ) -> None:
        """Persist benchmark embeddings for reuse."""
        cache_file = self._embedding_cache_file(bench_name)
        try:
            serializable = []
            for vector in embeddings:
                if hasattr(vector, "tolist"):
                    serializable.append(vector.tolist())
                else:
                    serializable.append([float(value) for value in vector])
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "version": CACHE_MANIFEST_VERSION,
                        "benchmark": bench_name,
                        "model": self.embedding_model,
                        "texts_count": len(texts),
                        "texts_hash": self._texts_hash(texts),
                        "embeddings": serializable,
                    },
                    f,
                    ensure_ascii=False,
                )
            self._update_manifest_artifact(
                f"embeddings:{bench_name}",
                {
                    "type": "embeddings",
                    "file": cache_file.name,
                    "benchmark": bench_name,
                    "model": self.embedding_model,
                    "texts_count": len(texts),
                    "texts_hash": self._texts_hash(texts),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store embedding cache for {bench_name}: {e}")

    def _load_cached_embedding_index(
        self,
    ) -> tuple[list[str], list[str], list[int], list[list[float]]] | None:
        """Load the combined embedding index if present."""
        cache_file = self._embedding_index_file()
        if not cache_file.exists():
            return None
        try:
            with open(cache_file) as f:
                payload = json.load(f)
            manifest = self._artifact_manifest("index:json")
            texts = [str(text) for text in payload.get("texts", []) if text]
            labels = [str(label) for label in payload.get("labels", [])]
            item_indexes = [int(index) for index in payload.get("item_indexes", [])]
            embeddings = [
                [float(value) for value in vector]
                for vector in payload.get("embeddings", [])
                if isinstance(vector, list) and vector
            ]
            if (
                int(payload.get("version", CACHE_MANIFEST_VERSION)) == CACHE_MANIFEST_VERSION
                and payload.get("scope") == self._cache_scope_signature()
                and manifest.get("file") == cache_file.name
                and len(texts) == len(labels) == len(item_indexes) == len(embeddings)
            ):
                return texts, labels, item_indexes, embeddings
        except Exception as e:
            logger.warning(f"Failed to load embedding index cache: {e}")
        return None

    def _store_cached_embedding_index(
        self,
        texts: list[str],
        labels: list[str],
        item_indexes: list[int],
        embeddings: Any,
    ) -> None:
        """Persist the combined benchmark embedding index for reuse."""
        cache_file = self._embedding_index_file()
        try:
            serializable = []
            for vector in embeddings:
                if hasattr(vector, "tolist"):
                    serializable.append(vector.tolist())
                else:
                    serializable.append([float(value) for value in vector])
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "version": CACHE_MANIFEST_VERSION,
                        "scope": self._cache_scope_signature(),
                        "texts": texts,
                        "labels": labels,
                        "item_indexes": item_indexes,
                        "embeddings": serializable,
                    },
                    f,
                    ensure_ascii=False,
                )
            self._update_manifest_artifact(
                "index:json",
                {
                    "type": "index",
                    "backend": "json",
                    "file": cache_file.name,
                    "entries": len(texts),
                    "scope": self._cache_scope_signature(),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store embedding index cache: {e}")

    def _load_cached_faiss_index(
        self,
    ) -> tuple[list[str], list[str], list[int], Any] | None:
        """Load a FAISS index plus sidecar metadata if present."""
        faiss = self._load_faiss()
        if faiss is None:
            return None
        index_file = self._faiss_index_file()
        metadata_file = self._faiss_metadata_file()
        if not index_file.exists() or not metadata_file.exists():
            return None
        try:
            with open(metadata_file) as f:
                payload = json.load(f)
            manifest = self._artifact_manifest("index:faiss")
            texts = [str(text) for text in payload.get("texts", []) if text]
            labels = [str(label) for label in payload.get("labels", [])]
            item_indexes = [int(index) for index in payload.get("item_indexes", [])]
            if (
                int(payload.get("version", CACHE_MANIFEST_VERSION)) != CACHE_MANIFEST_VERSION
                or payload.get("scope") != self._cache_scope_signature()
                or manifest.get("file") != index_file.name
            ):
                return None
            if len(texts) != len(labels) or len(texts) != len(item_indexes):
                return None
            index = faiss.read_index(str(index_file))
            return texts, labels, item_indexes, index
        except Exception as e:
            logger.warning(f"Failed to load FAISS index cache: {e}")
            return None

    def _store_cached_faiss_index(
        self,
        texts: list[str],
        labels: list[str],
        item_indexes: list[int],
        embeddings: Any,
    ) -> bool:
        """Persist a FAISS ANN index plus metadata sidecar."""
        faiss = self._load_faiss()
        if faiss is None:
            return False
        try:
            import numpy as np

            array = np.array(embeddings, dtype="float32")
            if array.ndim != 2 or array.shape[0] == 0:
                return False
            index = faiss.IndexFlatIP(array.shape[1])
            index.add(array)
            faiss.write_index(index, str(self._faiss_index_file()))
            with open(self._faiss_metadata_file(), "w") as f:
                json.dump(
                    {
                        "version": CACHE_MANIFEST_VERSION,
                        "texts": texts,
                        "labels": labels,
                        "item_indexes": item_indexes,
                        "backend": "faiss",
                        "scope": self._cache_scope_signature(),
                    },
                    f,
                    ensure_ascii=False,
                )
            self._update_manifest_artifact(
                "index:faiss",
                {
                    "type": "index",
                    "backend": "faiss",
                    "file": self._faiss_index_file().name,
                    "metadata_file": self._faiss_metadata_file().name,
                    "entries": len(texts),
                    "scope": self._cache_scope_signature(),
                },
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to store FAISS index cache: {e}")
            return False

    def _load_or_build_embedding_index(
        self,
        model: Any,
    ) -> tuple[list[str], list[str], list[int], Any, str]:
        """Load the combined benchmark embedding index from disk or build it once."""
        import numpy as np

        if self.embedding_index_backend == "faiss":
            cached_faiss = self._load_cached_faiss_index()
            if cached_faiss is not None:
                texts, labels, item_indexes, index = cached_faiss
                return texts, labels, item_indexes, index, "faiss"
            if self._load_faiss() is None:
                logger.warning(
                    "FAISS backend requested but faiss is not installed; falling back to JSON index"
                )

        cached = self._load_cached_embedding_index()
        if cached is not None:
            texts, labels, item_indexes, embeddings = cached
            return texts, labels, item_indexes, np.array(embeddings), "json"

        bench_texts: list[str] = []
        bench_labels: list[str] = []
        bench_item_indexes: list[int] = []
        bench_embeddings: list[Any] = []

        for bench_name in self.benchmarks:
            texts = self._benchmark_texts.get(bench_name, [])
            if not texts:
                continue

            cached_embeddings = self._load_cached_embeddings(bench_name)
            if len(cached_embeddings) == len(texts):
                embeddings = np.array(cached_embeddings)
            else:
                embeddings = model.encode(texts, normalize_embeddings=True)
                self._store_cached_embeddings(bench_name, texts, embeddings)

            bench_texts.extend(texts)
            bench_labels.extend([bench_name] * len(texts))
            bench_item_indexes.extend(list(range(len(texts))))
            bench_embeddings.extend(list(embeddings))

        if bench_texts and bench_embeddings:
            if self.embedding_index_backend == "faiss":
                stored = self._store_cached_faiss_index(
                    bench_texts,
                    bench_labels,
                    bench_item_indexes,
                    bench_embeddings,
                )
                if stored:
                    cached_faiss = self._load_cached_faiss_index()
                    if cached_faiss is not None:
                        texts, labels, item_indexes, index = cached_faiss
                        return texts, labels, item_indexes, index, "faiss"
            self._store_cached_embedding_index(
                bench_texts,
                bench_labels,
                bench_item_indexes,
                bench_embeddings,
            )
        return bench_texts, bench_labels, bench_item_indexes, np.array(bench_embeddings), "json"

    def _search_embedding_index(
        self,
        embedding_index: Any,
        query_embedding: Any,
        top_k: int,
        backend: str,
    ) -> tuple[list[float], list[int]]:
        """Search the configured embedding index and return top-k scores and indices."""
        if backend == "faiss":
            import numpy as np

            query = np.array([query_embedding], dtype="float32")
            scores, indices = embedding_index.search(query, top_k)
            return [float(value) for value in scores[0]], [int(value) for value in indices[0]]

        import numpy as np

        similarities = np.dot(query_embedding, embedding_index.T)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = [float(similarities[idx]) for idx in top_indices]
        return top_scores, [int(idx) for idx in top_indices]

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
            multi_signal = any(len(methods_for_bench) > 1 for methods_for_bench in self._methods_by_benchmark(evidence).values())
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
            grouped[str(item.get("benchmark", "unknown"))].add(
                str(item.get("method", "unknown"))
            )
        return grouped
