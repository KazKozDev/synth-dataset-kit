import tempfile
from pathlib import Path

from synth_dataset_kit.cli import (
    _default_demo_seed_path,
)
from synth_dataset_kit.config import LLMProvider, SDKConfig
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


def test_default_config():
    config = SDKConfig()
    assert config.llm.provider == LLMProvider.OLLAMA
    assert config.generation.num_examples == 100
    assert config.generation.divergence_threshold == 0.15
    assert config.generation.focus_top_k_clusters == 2
    assert config.generation.rebalancing_strategy == "strict"
    assert config.generation.graph_neighbor_k == 3
    assert config.generation.distance_allocator_weight == 0.6
    assert config.quality.min_score == 7.5
    assert "mmlu" in config.decontamination.benchmarks


def test_default_demo_seed_path_exists():
    path = _default_demo_seed_path()
    assert path is not None
    assert path.name == "customer_support_seeds.jsonl"


def test_config_for_provider():
    config = SDKConfig.default_for_provider("openai")
    assert config.llm.provider == LLMProvider.OPENAI
    assert "openai.com" in config.llm.api_base

    config = SDKConfig.default_for_provider("ollama")
    assert config.llm.provider == LLMProvider.OLLAMA
    assert "11434" in config.llm.api_base


def test_config_yaml_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    config = SDKConfig.default_for_provider("openai")
    config.to_yaml(path)
    loaded = SDKConfig.from_yaml(path)
    assert loaded.llm.model == config.llm.model
    assert loaded.llm.provider.value == config.llm.provider.value
    Path(path).unlink()
