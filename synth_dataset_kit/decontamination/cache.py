from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from .signatures import CACHE_MANIFEST_VERSION
from .utils import _normalize

logger = logging.getLogger(__name__)


class DecontaminationCacheMixin:
    """DecontaminationCacheMixin functionality."""

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

    def _cache_file(self, bench_name: str) -> Path:
        """Return cache file path for a benchmark."""
        limit = (
            "full"
            if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0
            else str(self.benchmark_sample_limit)
        )
        return self.cache_dir / f"{bench_name}_{limit}.json"

    def _embedding_cache_file(self, bench_name: str) -> Path:
        """Return cache file path for benchmark embeddings."""
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.embedding_model)
        limit = (
            "full"
            if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0
            else str(self.benchmark_sample_limit)
        )
        return self.cache_dir / f"{bench_name}_{limit}_{safe_model}_embeddings.json"

    def _embedding_index_file(self) -> Path:
        """Return cache file path for the combined benchmark embedding index."""
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.embedding_model)
        limit = (
            "full"
            if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0
            else str(self.benchmark_sample_limit)
        )
        bench_key = "_".join(sorted(self.benchmarks))
        return self.cache_dir / f"{bench_key}_{limit}_{safe_model}_index.json"

    def _faiss_index_file(self) -> Path:
        """Return cache file path for the FAISS ANN index."""
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.embedding_model)
        limit = (
            "full"
            if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0
            else str(self.benchmark_sample_limit)
        )
        bench_key = "_".join(sorted(self.benchmarks))
        return self.cache_dir / f"{bench_key}_{limit}_{safe_model}_index.faiss"

    def _faiss_metadata_file(self) -> Path:
        """Return sidecar metadata file for the FAISS index."""
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.embedding_model)
        limit = (
            "full"
            if self.load_full_benchmark_corpus or self.benchmark_sample_limit <= 0
            else str(self.benchmark_sample_limit)
        )
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
                and payload.get("texts_hash")
                == self._texts_hash(self._benchmark_texts.get(bench_name, []))
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
