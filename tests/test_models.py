from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.generators.seed_expander import SeedExpander
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, Role
from synth_dataset_kit.utils import safe_slug


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


def test_example_creation():
    ex = Example(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
    )
    assert ex.user_message == "Hello"
    assert ex.assistant_message == "Hi there!"
    assert len(ex.id) == 12


def test_safe_slug_shortens_and_normalizes_names():
    slug = safe_slug(
        "E-commerce and SaaS customer support (orders, shipping, billing, subscriptions, app troubleshooting)"
    )
    assert len(slug) <= 64
    assert "(" not in slug
    assert " " not in slug


def test_dataset_operations():
    ds = Dataset(name="test")
    ds.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Q1"),
                Message(role=Role.ASSISTANT, content="A1"),
            ],
            quality_score=8.0,
        )
    )
    ds.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Q2"),
                Message(role=Role.ASSISTANT, content="A2"),
            ],
            quality_score=4.0,
        )
    )
    assert ds.size == 2

    filtered = ds.filter_by_quality(7.0)
    assert filtered.size == 1
    assert filtered.examples[0].user_message == "Q1"


def test_dataset_remove_contaminated():
    ds = Dataset(name="test")
    ds.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Clean"),
                Message(role=Role.ASSISTANT, content="Clean answer"),
            ],
            decontamination_flags=[],
        )
    )
    ds.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Contaminated"),
                Message(role=Role.ASSISTANT, content="Bad answer"),
            ],
            decontamination_flags=["mmlu"],
        )
    )
    clean = ds.remove_contaminated()
    assert clean.size == 1
    assert clean.examples[0].user_message == "Clean"


def test_dataset_remove_contaminated_keeps_review_verdict():
    ds = Dataset(name="test")
    ds.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Needs review"),
                Message(role=Role.ASSISTANT, content="Potentially similar answer"),
            ],
            decontamination_flags=["gsm8k"],
            metadata={"contamination_verdict": "review"},
        )
    )
    ds.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Blocked"),
                Message(role=Role.ASSISTANT, content="Definitely contaminated"),
            ],
            decontamination_flags=["mmlu"],
            metadata={"contamination_verdict": "block"},
        )
    )
    clean = ds.remove_contaminated()
    assert clean.size == 1
    assert clean.examples[0].metadata["contamination_verdict"] == "review"
