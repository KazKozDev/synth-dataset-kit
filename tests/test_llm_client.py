from unittest.mock import patch

import synth_dataset_kit.llm_client as llm_client_module
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


def test_anthropic_complete_uses_native_messages_api():
    cfg = SDKConfig.default_for_provider("anthropic")
    cfg.llm.api_key = "test-key"
    cfg.llm.timeout = 1
    client = LLMClient(cfg.llm)

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"content":[{"type":"text","text":"native anthropic response"}]}'

    with patch.object(llm_client_module, "urlopen", return_value=_FakeResponse()) as mocked_urlopen:
        result = client.complete(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ],
            temperature=0.2,
            max_tokens=32,
        )

    assert result == "native anthropic response"
    request = mocked_urlopen.call_args.args[0]
    assert request.full_url.endswith("/messages")
    assert request.headers["X-api-key"] == "test-key"


def test_complete_json_repairs_control_characters_and_trailing_commas():
    client = _FakeLLMClient()
    raw = '{"examples":[{"user":"Line 1\nLine 2","assistant":"Answer",},],}'
    parsed = client._parse_json_response(raw)
    assert isinstance(parsed, dict)
    assert parsed["examples"][0]["user"] == "Line 1\nLine 2"
