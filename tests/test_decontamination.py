import json
import sys
import tempfile
import types
from pathlib import Path

from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.decontamination import BENCHMARK_SIGNATURES, Decontaminator
from synth_dataset_kit.generators.seed_expander import SeedExpander
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, Role


class _RoundRobinExpander(SeedExpander):
    def __init__(self, rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rounds = list(rounds)

    def generate_candidates(
        self,
        seeds,
        analysis,
        target_count=None,
        accepted_examples=None,
        focus_cluster_ids=None,
    ):
        if not self._rounds:
            return []
        return self._rounds.pop(0)


class _TrackingRoundRobinExpander(_RoundRobinExpander):
    def __init__(self, rounds, *args, **kwargs):
        super().__init__(rounds, *args, **kwargs)
        self.focus_history = []

    def generate_candidates(
        self,
        seeds,
        analysis,
        target_count=None,
        accepted_examples=None,
        focus_cluster_ids=None,
    ):
        self.focus_history.append(list(focus_cluster_ids or []))
        return super().generate_candidates(
            seeds,
            analysis,
            target_count=target_count,
            accepted_examples=accepted_examples,
            focus_cluster_ids=focus_cluster_ids,
        )


def _make_test_dataset() -> Dataset:
    ds = Dataset(name="test_export")
    for i in range(3):
        ds.add(
            Example(
                messages=[
                    Message(role=Role.USER, content=f"Question {i}"),
                    Message(role=Role.ASSISTANT, content=f"Answer {i}"),
                ]
            )
        )
    return ds


class _DummyClient:
    def complete_json(self, messages, temperature=None):
        return {
            "relevance": 8,
            "accuracy": 8,
            "completeness": 8,
            "naturalness": 8,
            "helpfulness": 8,
            "overall": 8,
            "issues": [],
            "has_pii": False,
            "has_toxic_content": False,
        }


class _FakeLLMClient(LLMClient):
    def __init__(self):
        self.config = SDKConfig().llm

    def list_models(self):
        return [
            {"name": "llama3.1:8b"},
            {"name": "qwen2.5-coder:7b"},
            {"name": "mistral:7b"},
        ]


def test_decontaminator_clean():
    decon = Decontaminator(benchmarks=["mmlu", "gsm8k"])
    ex = Example(
        messages=[
            Message(role=Role.USER, content="How do I make pasta?"),
            Message(role=Role.ASSISTANT, content="Boil water, add pasta, cook 8-10 mins."),
        ]
    )
    flags = decon.check_example(ex)
    assert len(flags) == 0


def test_decontaminator_catches_gsm8k():
    decon = Decontaminator(benchmarks=["gsm8k"], use_benchmark_datasets=False)
    ex = Example(
        messages=[
            Message(
                role=Role.USER,
                content="Janet's ducks lay 16 eggs per day. She eats three for breakfast.",
            ),
            Message(role=Role.ASSISTANT, content="Let me calculate that."),
        ]
    )
    flags = decon.check_example(ex)
    assert "gsm8k" in flags


def test_decontaminator_catches_humaneval():
    decon = Decontaminator(benchmarks=["humaneval"], use_benchmark_datasets=False)
    ex = Example(
        messages=[
            Message(
                role=Role.USER,
                content="Write a function def has_close_elements that checks if any two numbers are close.",
            ),
            Message(role=Role.ASSISTANT, content="Here's the implementation..."),
        ]
    )
    flags = decon.check_example(ex)
    assert "humaneval" in flags


def test_decontaminator_dataset():
    decon = Decontaminator(use_benchmark_datasets=False)
    ds = Dataset(name="test")
    ds.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Normal question about cooking"),
                Message(role=Role.ASSISTANT, content="Here's a recipe..."),
            ]
        )
    )
    ds.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Janet's ducks lay 16 eggs per day"),
                Message(role=Role.ASSISTANT, content="The answer is..."),
            ]
        )
    )
    ds = decon.check_dataset(ds)
    contaminated = [e for e in ds.examples if e.decontamination_flags]
    assert len(contaminated) == 1


def test_decontaminator_falls_back_to_builtin_signatures():
    decon = Decontaminator(
        benchmarks=["gsm8k"],
        use_benchmark_datasets=False,
    )
    assert decon._benchmark_texts["gsm8k"] == BENCHMARK_SIGNATURES["gsm8k"]
    assert decon.benchmark_sources["gsm8k"] == "fallback_signatures"
    assert decon.benchmark_sample_counts["gsm8k"] == len(BENCHMARK_SIGNATURES["gsm8k"])
    assert decon.benchmark_load_errors["gsm8k"] == "benchmark dataset loading disabled"


def test_decontaminator_loads_benchmark_samples_from_datasets():
    fake_module = types.SimpleNamespace()

    def fake_load_dataset(path, name=None, split=None):
        assert path == "gsm8k"
        return [
            {"question": "Janet's ducks lay 16 eggs per day."},
            {"question": "A baker sells 12 loaves every morning."},
        ]

    fake_module.load_dataset = fake_load_dataset
    original = sys.modules.get("datasets")
    sys.modules["datasets"] = fake_module
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            decon = Decontaminator(
                benchmarks=["gsm8k"],
                use_benchmark_datasets=True,
                benchmark_sample_limit=1,
                cache_dir=tmpdir,
            )
            assert decon._benchmark_texts["gsm8k"] == ["Janet's ducks lay 16 eggs per day."]
            assert decon.benchmark_sources["gsm8k"] == "datasets"
            assert decon.benchmark_sample_counts["gsm8k"] == 1
            assert "gsm8k" not in decon.benchmark_load_errors
        finally:
            if original is None:
                del sys.modules["datasets"]
            else:
                sys.modules["datasets"] = original


def test_decontaminator_loads_benchmark_samples_from_cache():
    fake_module = types.SimpleNamespace()

    def fake_load_dataset(path, name=None, split=None):
        assert path == "gsm8k"
        return [
            {"question": "Janet's ducks lay 16 eggs per day."},
            {"question": "A baker sells 12 loaves every morning."},
        ]

    fake_module.load_dataset = fake_load_dataset
    original = sys.modules.get("datasets")

    with tempfile.TemporaryDirectory() as tmpdir:
        sys.modules["datasets"] = fake_module
        try:
            first = Decontaminator(
                benchmarks=["gsm8k"],
                use_benchmark_datasets=True,
                benchmark_sample_limit=1,
                cache_dir=tmpdir,
            )
            cache_file = Path(tmpdir) / "gsm8k_1.json"
            assert cache_file.exists()
            assert first.benchmark_sources["gsm8k"] == "datasets"
        finally:
            if original is None:
                del sys.modules["datasets"]
            else:
                sys.modules["datasets"] = original

        second = Decontaminator(
            benchmarks=["gsm8k"],
            use_benchmark_datasets=True,
            benchmark_sample_limit=1,
            cache_dir=tmpdir,
        )
        assert second._benchmark_texts["gsm8k"] == ["Janet's ducks lay 16 eggs per day."]
        assert second.benchmark_sources["gsm8k"] == "datasets_cache"
        assert second.benchmark_sample_counts["gsm8k"] == 1
        assert "gsm8k" not in second.benchmark_load_errors


def test_decontaminator_hybrid_attaches_evidence():
    decon = Decontaminator(
        benchmarks=["gsm8k"],
        method="hybrid",
        use_benchmark_datasets=False,
    )
    dataset = Dataset(name="hybrid")
    dataset.add(
        Example(
            messages=[
                Message(
                    role=Role.USER,
                    content="Janet's ducks lay 16 eggs per day. She eats three for breakfast.",
                ),
                Message(role=Role.ASSISTANT, content="Let me calculate that."),
            ]
        )
    )

    checked = decon.check_dataset(dataset)

    assert checked.examples[0].decontamination_flags == ["gsm8k"]
    assert checked.examples[0].decontamination_evidence
    assert checked.examples[0].decontamination_evidence[0]["benchmark"] == "gsm8k"
    assert checked.examples[0].decontamination_evidence[0]["method"] in {
        "ngram",
        "substring",
        "embedding",
    }
    assert checked.examples[0].metadata["contamination_verdict"] == "block"
    assert "gsm8k:ngram" in checked.examples[0].metadata["contamination_reason_codes"]


def test_decontaminator_decision_policy_marks_review_for_embedding_signal():
    decon = Decontaminator(
        benchmarks=["gsm8k"],
        review_threshold=0.92,
        hard_fail_methods=["exact", "ngram"],
        review_methods=["embedding"],
    )
    dataset = Dataset(name="review_signal")
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Question"),
                Message(role=Role.ASSISTANT, content="Answer"),
            ],
            decontamination_flags=["gsm8k"],
            decontamination_evidence=[
                {
                    "benchmark": "gsm8k",
                    "method": "embedding",
                    "confidence": 0.97,
                    "matched_text": "Janet's ducks lay 16 eggs per day.",
                }
            ],
        )
    )

    checked = decon._apply_decision_policy(dataset)

    assert checked.examples[0].metadata["contamination_verdict"] == "review"
    assert checked.examples[0].metadata["contamination_confidence"] == 0.97
    assert checked.examples[0].metadata["contamination_methods"] == ["embedding"]


def test_decontaminator_embedding_cache_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        decon = Decontaminator(
            benchmarks=["gsm8k"],
            use_benchmark_datasets=False,
            cache_dir=tmpdir,
            benchmark_sample_limit=3,
        )
        texts = list(decon._benchmark_texts["gsm8k"])
        decon._store_cached_embeddings(
            "gsm8k",
            texts,
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        )
        loaded = decon._load_cached_embeddings("gsm8k")
        assert loaded == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        assert decon._embedding_cache_file("gsm8k").exists()
        manifest = json.loads((Path(tmpdir) / "cache_manifest.json").read_text())
        assert manifest["version"] == 1
        assert "embeddings:gsm8k" in manifest["artifacts"]


def test_decontaminator_embedding_index_cache_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        decon = Decontaminator(
            benchmarks=["gsm8k", "mmlu"],
            use_benchmark_datasets=False,
            cache_dir=tmpdir,
            benchmark_sample_limit=3,
        )
        decon._store_cached_embedding_index(
            texts=["q1", "q2"],
            labels=["gsm8k", "mmlu"],
            item_indexes=[0, 1],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
        loaded = decon._load_cached_embedding_index()
        assert loaded is not None
        texts, labels, item_indexes, embeddings = loaded
        assert texts == ["q1", "q2"]
        assert labels == ["gsm8k", "mmlu"]
        assert item_indexes == [0, 1]
        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert decon._embedding_index_file().exists()
        manifest = json.loads((Path(tmpdir) / "cache_manifest.json").read_text())
        assert manifest["artifacts"]["index:json"]["backend"] == "json"


def test_decontaminator_cache_manifest_invalidation_on_scope_change():
    with tempfile.TemporaryDirectory() as tmpdir:
        first = Decontaminator(
            benchmarks=["gsm8k"],
            use_benchmark_datasets=False,
            cache_dir=tmpdir,
            benchmark_sample_limit=3,
        )
        first._store_cached_embedding_index(
            texts=["q1"],
            labels=["gsm8k"],
            item_indexes=[0],
            embeddings=[[0.1, 0.2]],
        )
        second = Decontaminator(
            benchmarks=["gsm8k"],
            use_benchmark_datasets=False,
            cache_dir=tmpdir,
            benchmark_sample_limit=4,
        )
        assert second._load_cached_embedding_index() is None


def test_decontaminator_faiss_index_cache_roundtrip():
    import numpy as np

    class _FakeFaissIndex:
        def __init__(self, dim):
            self.dim = dim
            self.vectors = np.empty((0, dim), dtype="float32")

        def add(self, array):
            self.vectors = np.array(array, dtype="float32")

        def search(self, query, top_k):
            scores = np.dot(query, self.vectors.T)
            top_indices = np.argsort(scores[0])[-top_k:][::-1]
            top_scores = scores[:, top_indices]
            return top_scores, np.array([top_indices], dtype="int64")

    def _write_index(index, path):
        Path(path).write_text(json.dumps({"vectors": index.vectors.tolist(), "dim": index.dim}))

    def _read_index(path):
        payload = json.loads(Path(path).read_text())
        index = _FakeFaissIndex(payload["dim"])
        index.add(np.array(payload["vectors"], dtype="float32"))
        return index

    fake_faiss = types.SimpleNamespace(
        IndexFlatIP=_FakeFaissIndex,
        write_index=_write_index,
        read_index=_read_index,
    )
    original = sys.modules.get("faiss")
    sys.modules["faiss"] = fake_faiss
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            decon = Decontaminator(
                benchmarks=["gsm8k", "mmlu"],
                use_benchmark_datasets=False,
                cache_dir=tmpdir,
                benchmark_sample_limit=3,
                embedding_index_backend="faiss",
            )
            stored = decon._store_cached_faiss_index(
                texts=["q1", "q2"],
                labels=["gsm8k", "mmlu"],
                item_indexes=[0, 1],
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
            )
            assert stored is True
            loaded = decon._load_cached_faiss_index()
            assert loaded is not None
            texts, labels, item_indexes, index = loaded
            assert texts == ["q1", "q2"]
            assert labels == ["gsm8k", "mmlu"]
            assert item_indexes == [0, 1]
            scores, indices = decon._search_embedding_index(index, np.array([0.3, 0.4]), 1, "faiss")
            assert len(scores) == 1
            assert indices == [1]
            assert decon._faiss_index_file().exists()
            assert decon._faiss_metadata_file().exists()
            manifest = json.loads((Path(tmpdir) / "cache_manifest.json").read_text())
            assert manifest["artifacts"]["index:faiss"]["backend"] == "faiss"
        finally:
            if original is None:
                del sys.modules["faiss"]
            else:
                sys.modules["faiss"] = original


def test_decontaminator_embedding_top_k_matches():
    import numpy as np

    class _FakeEmbeddingModel:
        def encode(self, texts, normalize_embeddings=True):
            vectors = []
            for text in texts:
                normalized = str(text).lower()
                if "janet" in normalized:
                    vectors.append([1.0, 0.0])
                elif "apples" in normalized:
                    vectors.append([0.96, 0.04])
                else:
                    vectors.append([0.0, 1.0])
            return np.array(vectors)

    decon = Decontaminator(
        benchmarks=["gsm8k"],
        use_benchmark_datasets=False,
        method="embedding",
        embedding_top_k=2,
        similarity_threshold=0.8,
    )
    decon._load_sentence_transformer = lambda: _FakeEmbeddingModel()
    decon._load_or_build_embedding_index = lambda model: (
        ["Janet's ducks lay 16 eggs per day.", "How many more apples are needed?"],
        ["gsm8k", "gsm8k"],
        [0, 1],
        np.array([[1.0, 0.0], [0.96, 0.04]]),
        "json",
    )

    dataset = Dataset(name="embedding_top_k")
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Janet's ducks lay 16 eggs per day."),
                Message(role=Role.ASSISTANT, content="Let's calculate the answer."),
            ]
        )
    )

    checked = decon.check_dataset(dataset)

    evidence = [
        item
        for item in checked.examples[0].decontamination_evidence
        if item["method"] == "embedding"
    ]
    assert len(evidence) == 2
    assert evidence[0]["match_rank"] == 1
    assert evidence[1]["match_rank"] == 2
    assert checked.examples[0].metadata["embedding_top_k_matches"][0]["match_rank"] == 1
    assert checked.examples[0].metadata["contamination_verdict"] == "review"
