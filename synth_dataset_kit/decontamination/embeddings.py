from __future__ import annotations

import logging
from typing import Any

from synth_dataset_kit.models import Dataset

logger = logging.getLogger(__name__)


class DecontaminationEmbeddingMixin:
    """DecontaminationEmbeddingMixin functionality."""

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

        bench_texts, bench_labels, bench_item_indexes, embedding_index, backend = (
            self._load_or_build_embedding_index(model)
        )

        if not bench_texts:
            return dataset

        # Encode dataset examples
        example_texts = [f"{e.user_message} {e.assistant_message}" for e in dataset.examples]
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
            for match_rank, (similarity, match_idx) in enumerate(
                zip(top_scores, top_indices, strict=False), start=1
            ):
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
                    for rank, (score, idx) in enumerate(
                        zip(top_scores, top_indices, strict=False), start=1
                    )
                    if int(idx) >= 0 and float(score) >= self.threshold
                ]

        flagged = sum(1 for e in dataset.examples if e.decontamination_flags)
        logger.info(f"Embedding decontamination: {flagged}/{dataset.size} flagged")
        return dataset

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
