import tempfile
from pathlib import Path

from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.evaluation import (
    compare_models_on_holdout,
    evaluate_prediction_dataset,
    export_uplift_results,
)
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


def test_evaluate_prediction_dataset():
    holdout = Dataset(
        name="holdout",
        examples=[
            Example(
                messages=[
                    Message(role=Role.USER, content="Refund"),
                    Message(
                        role=Role.ASSISTANT,
                        content="Share your order number so support can review the refund.",
                    ),
                ],
                metadata={"topic": "refunds"},
            )
        ],
    )
    predictions = Dataset(
        name="predictions",
        examples=[
            Example(
                messages=[
                    Message(role=Role.USER, content="Refund"),
                    Message(
                        role=Role.ASSISTANT,
                        content="Please share your order number so support can review the refund.",
                    ),
                ]
            )
        ],
    )
    metrics = evaluate_prediction_dataset(predictions, holdout)
    assert metrics["examples"] == 1
    assert metrics["task_success_rate"] > 0
    assert metrics["pass_rate"] > 0


def test_compare_models_on_holdout_and_export():
    class _BaseClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def complete(self, messages, temperature=None, max_tokens=None, response_format=None):
            prompt = messages[-1]["content"].lower()
            if "charged twice" in prompt:
                return "Contact support."
            return "Please contact support."

    class _FineTunedClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def complete(self, messages, temperature=None, max_tokens=None, response_format=None):
            prompt = messages[-1]["content"].lower()
            if "charged twice" in prompt:
                return "Please share your order number and payment details so support can review the duplicate charge."
            return "Please share your order number so support can investigate and help."

    import synth_dataset_kit.evaluation as evaluation_module

    original_client = evaluation_module.LLMClient
    with tempfile.TemporaryDirectory() as tmpdir:
        holdout_path = Path(tmpdir) / "holdout.jsonl"
        holdout_path.write_text(
            '{"messages":[{"role":"user","content":"I was charged twice for the same order. Can you help me fix that?"},{"role":"assistant","content":"Please share your order number and payment details so support can review the duplicate charge."}]}\n',
            encoding="utf-8",
        )
        try:
            base_cfg = SDKConfig()
            base_cfg.llm.model = "base-model"
            finetuned_cfg = SDKConfig()
            finetuned_cfg.llm.model = "fine-model"

            def _client_factory(llm_config):
                if llm_config.model == "base-model":
                    return _BaseClient()
                return _FineTunedClient()

            evaluation_module.LLMClient = _client_factory
            results = compare_models_on_holdout(base_cfg, finetuned_cfg, str(holdout_path))
            assert results["uplift"]["task_success_rate_delta"] >= 0
            files = export_uplift_results(results, tmpdir, name="uplift_test")
            assert Path(files[0]).exists()
            assert Path(files[1]).exists()
        finally:
            evaluation_module.LLMClient = original_client
