import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from synth_dataset_kit.cli import (
    _resolve_artifact_group,
)
from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.generators.seed_expander import SeedExpander
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, Role
from synth_dataset_kit.publishing import (
    publish_huggingface_bundle,
    resolve_hf_token,
)
from synth_dataset_kit.training import TrainingJob, save_training_job


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


def test_save_training_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        job = TrainingJob(
            dataset_path="dataset.jsonl",
            base_model="llama3.1:8b",
            output_dir=tmpdir,
        )
        files = save_training_job(job)
        assert Path(files[0]).exists()
        assert Path(files[1]).exists()
        payload = json.loads(Path(files[0]).read_text(encoding="utf-8"))
        assert payload["base_model"] == "llama3.1:8b"


def test_resolve_artifact_group_prefers_latest_group_in_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        older = tmp / "older_run_candidates.jsonl"
        older.write_text('{"user":"q","assistant":"a"}\n')
        (tmp / "older_run_accepted.jsonl").write_text('{"user":"q","assistant":"a"}\n')
        (tmp / "older_run_rejected.jsonl").write_text('{"user":"q","assistant":"a"}\n')
        time.sleep(0.01)
        newer = tmp / "newer_run_candidates.jsonl"
        newer.write_text('{"user":"q","assistant":"a"}\n')
        (tmp / "newer_run_accepted.jsonl").write_text('{"user":"q","assistant":"a"}\n')
        (tmp / "newer_run_rejected.jsonl").write_text('{"user":"q","assistant":"a"}\n')

        base_name, artifact_paths = _resolve_artifact_group(tmp)

        assert base_name == "newer_run"
        assert artifact_paths["candidates"] == newer
        assert artifact_paths["accepted"] == tmp / "newer_run_accepted.jsonl"


def test_resolve_hf_token_prefers_explicit_value():
    assert resolve_hf_token("abc123") == "abc123"


def test_publish_huggingface_bundle_requires_token():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict("os.environ", {}, clear=True):
            try:
                publish_huggingface_bundle(
                    bundle_dir=tmpdir,
                    repo_id="user/test-dataset",
                    token=None,
                )
            except RuntimeError as exc:
                assert "HF_TOKEN" in str(exc)
            else:
                raise AssertionError("expected missing-token RuntimeError")
