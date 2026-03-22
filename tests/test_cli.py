from synth_dataset_kit.cli import (
    _recommend_benchmark_result,
    _select_benchmark_models,
)
from synth_dataset_kit.config import SDKConfig
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


def test_select_benchmark_models_prefers_recommended_model():
    models = [
        {"name": "llama3.1:8b"},
        {"name": "mistral:7b"},
        {"name": "qwen2.5-coder:7b"},
    ]
    selected = _select_benchmark_models(
        models,
        domain="customer support",
        top_n=2,
        recommended_model="llama3.1:8b",
    )
    assert selected == ["llama3.1:8b", "mistral:7b"]


def test_recommend_benchmark_result_balances_quality_pass_rate_and_speed():
    recommendation = _recommend_benchmark_result(
        [
            {
                "model": "llama3.1:8b",
                "status": "ok",
                "avg_quality_score": 8.4,
                "pass_rate": 0.9,
                "examples_per_second": 1.1,
                "contamination_hits": 0,
            },
            {
                "model": "mistral:7b",
                "status": "ok",
                "avg_quality_score": 7.8,
                "pass_rate": 0.8,
                "examples_per_second": 2.4,
                "contamination_hits": 0,
            },
        ]
    )
    assert recommendation is not None
    assert recommendation["model"] == "llama3.1:8b"
    assert "benchmark_score" in recommendation
