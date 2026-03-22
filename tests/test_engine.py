import importlib
import json
import tempfile
from pathlib import Path

from synth_dataset_kit.cli import (
    _artifact_base_name,
    _artifact_example_preview,
    _artifact_summary,
    _sort_artifact_examples,
)
from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.engine import DatasetEngine
from synth_dataset_kit.evaluation import (
    build_metric_validation_report,
)
from synth_dataset_kit.generators.seed_expander import SeedExpander, _parse_example, load_seed_file
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, Role
from synth_dataset_kit.publishing import (
    build_publish_manifest,
    write_publish_manifest,
)
from synth_dataset_kit.training import TrainingJob


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


def test_final_export_dataset_includes_seed_examples(tmp_path: Path):
    seed_path = tmp_path / "seeds.jsonl"
    seed_path.write_text(
        "\n".join(
            [
                json.dumps({"user": "Seed question 1", "assistant": "Seed answer 1"}),
                json.dumps({"user": "Seed question 2", "assistant": "Seed answer 2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    generated_dataset = Dataset(
        name="test_dataset",
        examples=[
            Example(
                messages=[
                    Message(role=Role.USER, content="Generated question"),
                    Message(role=Role.ASSISTANT, content="Generated answer"),
                ],
                metadata={"cluster_id": "c1"},
                quality_score=8.2,
            )
        ],
    )

    engine = DatasetEngine(SDKConfig())
    final_dataset, seed_count = engine._final_export_dataset(
        generated_dataset,
        seed_file=str(seed_path),
    )

    assert seed_count == 2
    assert final_dataset.size == 3
    assert final_dataset.examples[0].metadata["source"] == "seed"
    assert final_dataset.examples[1].metadata["source"] == "seed"
    assert final_dataset.examples[2].metadata["source"] == "generated"
    assert final_dataset.examples[2].metadata["generation_source"] == "generated"
    assert final_dataset.config_snapshot["seed_examples_included"] == 2
    assert final_dataset.config_snapshot["generated_examples_retained"] == 1
    assert final_dataset.config_snapshot["final_export_examples"] == 3


def test_engine_export_forces_metadata_when_seed_examples_included(tmp_path: Path):
    engine = DatasetEngine(SDKConfig())
    dataset = Dataset(
        name="seed_merge_test",
        config_snapshot={"seed_examples_included": 2},
        examples=[
            Example(
                messages=[
                    Message(role=Role.USER, content="Question"),
                    Message(role=Role.ASSISTANT, content="Answer"),
                ],
                metadata={"source": "seed"},
            )
        ],
    )

    output_files = engine.export(
        dataset,
        format="jsonl",
        output_dir=str(tmp_path),
    )

    exported = Path(output_files[0]).read_text(encoding="utf-8").strip()
    record = json.loads(exported)
    assert record["metadata"]["source"] == "seed"


def test_parse_openai_format():
    data = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
    }
    ex = _parse_example(data)
    assert ex is not None
    assert ex.user_message == "Hello"


def test_parse_alpaca_format():
    data = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
    ex = _parse_example(data)
    assert ex is not None
    assert "Hello" in ex.user_message
    assert ex.assistant_message == "Hola"


def test_parse_sharegpt_format():
    data = {
        "conversations": [
            {"from": "human", "value": "What's 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
    }
    ex = _parse_example(data)
    assert ex is not None
    assert ex.user_message == "What's 2+2?"


def test_parse_simple_format():
    data = {"user": "Question", "assistant": "Answer"}
    ex = _parse_example(data)
    assert ex is not None
    assert ex.user_message == "Question"


def test_load_seed_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"user": "Q1", "assistant": "A1"}) + "\n")
        f.write(json.dumps({"user": "Q2", "assistant": "A2"}) + "\n")
        path = f.name
    examples = load_seed_file(path)
    assert len(examples) == 2
    Path(path).unlink()


def test_build_metric_validation_report():
    runs = [
        {
            "name": "run_a",
            "eval_summary": {
                "distribution_validation": {
                    "internal_distribution_match_score": 70.0,
                    "reference_alignment_score": 72.0,
                    "graph_coverage_score": 68.0,
                    "calibration_error": 2.0,
                    "validated_distribution_match_score": 71.0,
                }
            },
            "uplift": {
                "uplift": {
                    "task_success_rate_delta": 0.02,
                    "pass_rate_delta": 0.01,
                    "avg_token_f1_delta": 0.03,
                }
            },
        },
        {
            "name": "run_b",
            "eval_summary": {
                "distribution_validation": {
                    "internal_distribution_match_score": 82.0,
                    "reference_alignment_score": 84.0,
                    "graph_coverage_score": 79.0,
                    "calibration_error": 2.0,
                    "validated_distribution_match_score": 82.9,
                }
            },
            "uplift": {
                "uplift": {
                    "task_success_rate_delta": 0.08,
                    "pass_rate_delta": 0.05,
                    "avg_token_f1_delta": 0.07,
                }
            },
        },
    ]

    report = build_metric_validation_report(runs)

    assert report["runs_analyzed"] == 2
    assert report["avg_calibration_error"] == 2.0
    assert report["correlations"]["validated_match_vs_task_success_delta"] is not None
    assert runs[0]["validation_summary"]["validated_distribution_match_score"] == 71.0


def test_run_training_job_requires_dependencies():
    training_module = importlib.import_module("synth_dataset_kit.training")
    job = TrainingJob(
        dataset_path="dataset.jsonl",
        base_model="llama3.1:8b",
        output_dir="./output/finetune",
    )
    try:
        training_module.run_training_job(job)
    except RuntimeError as exc:
        assert "training dependencies" in str(exc).lower()
    except (ValueError, FileNotFoundError):
        pass


def test_artifact_summary_counts_rejection_reasons():
    dataset = Dataset(name="artifact_summary")
    dataset.artifacts = {
        "candidates": [
            Example(
                messages=[
                    Message(role=Role.USER, content="c1"),
                    Message(role=Role.ASSISTANT, content="candidate one with enough words."),
                ]
            ),
            Example(
                messages=[
                    Message(role=Role.USER, content="c2"),
                    Message(role=Role.ASSISTANT, content="candidate two with enough words."),
                ]
            ),
            Example(
                messages=[
                    Message(role=Role.USER, content="c3"),
                    Message(role=Role.ASSISTANT, content="candidate three with enough words."),
                ]
            ),
        ],
        "accepted": [
            Example(
                messages=[
                    Message(role=Role.USER, content="a1"),
                    Message(role=Role.ASSISTANT, content="accepted example with enough words."),
                ]
            )
        ],
        "rejected": [
            Example(
                messages=[
                    Message(role=Role.USER, content="r1"),
                    Message(role=Role.ASSISTANT, content="rejected example one with enough words."),
                ],
                metadata={"rejection_reasons": ["below_min_quality:7.0", "assistant_too_short"]},
            ),
            Example(
                messages=[
                    Message(role=Role.USER, content="r2"),
                    Message(role=Role.ASSISTANT, content="rejected example two with enough words."),
                ],
                metadata={"rejection_reasons": ["below_min_quality:7.0"]},
            ),
        ],
    }

    summary = _artifact_summary(dataset)

    assert summary["candidates"] == 3
    assert summary["accepted"] == 1
    assert summary["rejected"] == 2
    assert summary["top_rejection_reasons"][0] == ("below_min_quality:7.0", 2)


def test_artifact_base_name_parses_artifact_file():
    assert _artifact_base_name(Path("expanded_support_candidates.jsonl")) == "expanded_support"
    assert _artifact_base_name(Path("expanded_support_quality_report.json")) == "expanded_support"
    assert _artifact_base_name(Path("expanded_support.jsonl")) == "expanded_support"


def test_artifact_example_preview_exposes_key_fields():
    example = Example(
        messages=[
            Message(role=Role.USER, content="How do I cancel my subscription?"),
            Message(
                role=Role.ASSISTANT,
                content="Open billing settings, choose cancel, and confirm the cancellation before the renewal date.",
            ),
        ],
        quality_score=8.5,
        metadata={
            "selection_decision": "rejected",
            "rejection_reasons": ["below_min_quality:9.0"],
            "topic": "billing",
            "persona": "beginner",
            "difficulty": "easy",
        },
        decontamination_flags=["gsm8k"],
        decontamination_evidence=[
            {
                "benchmark": "gsm8k",
                "method": "ngram",
                "confidence": 0.91,
            }
        ],
    )

    preview = _artifact_example_preview(example)

    assert preview["quality_score"] == 8.5
    assert preview["selection_decision"] == "rejected"
    assert preview["rejection_reasons"] == ["below_min_quality:9.0"]
    assert preview["topic"] == "billing"
    assert preview["decontamination_flags"] == ["gsm8k"]
    assert preview["decontamination_methods"] == ["ngram"]
    assert preview["decontamination_evidence_summary"] == ["gsm8k:ngram:0.91"]


def test_sort_artifact_examples_by_score_topic_reason():
    examples = [
        Example(
            messages=[
                Message(role=Role.USER, content="Q1"),
                Message(role=Role.ASSISTANT, content="A1"),
            ],
            quality_score=7.2,
            metadata={"topic": "zeta", "rejection_reasons": ["weak_answer"]},
        ),
        Example(
            messages=[
                Message(role=Role.USER, content="Q2"),
                Message(role=Role.ASSISTANT, content="A2"),
            ],
            quality_score=9.1,
            metadata={"topic": "alpha", "rejection_reasons": ["assistant_too_short"]},
        ),
        Example(
            messages=[
                Message(role=Role.USER, content="Q3"),
                Message(role=Role.ASSISTANT, content="A3"),
            ],
            quality_score=8.0,
            metadata={"topic": "beta", "rejection_reasons": ["below_min_quality:7.0"]},
        ),
    ]

    by_score = _sort_artifact_examples(examples, "score")
    assert [e.user_message for e in by_score] == ["Q2", "Q3", "Q1"]

    by_topic = _sort_artifact_examples(examples, "topic")
    assert [e.user_message for e in by_topic] == ["Q2", "Q3", "Q1"]

    by_reason = _sort_artifact_examples(examples, "reason")
    assert [e.user_message for e in by_reason] == ["Q2", "Q3", "Q1"]


def test_ollama_recommend_model_prefers_coder_for_code_domains():
    client = _FakeLLMClient()
    assert client.recommend_model("python code generation") == "qwen2.5-coder:7b"
    assert client.recommend_model("customer support") == "llama3.1:8b"


def test_ollama_recommend_model_prefers_saved_benchmark_result():
    client = _FakeLLMClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        client._recommendation_cache_path = Path(tmpdir) / "ollama_models.json"
        client.save_benchmark_recommendation(
            "customer support",
            {
                "model": "mistral:7b",
                "benchmark_score": 0.91,
                "avg_quality_score": 8.1,
                "pass_rate": 0.88,
                "examples_per_second": 2.3,
            },
        )
        assert client.recommend_model("customer support") == "mistral:7b"


def test_build_and_write_publish_manifest():
    manifest = build_publish_manifest(
        repo_id="user/test-dataset",
        bundle_dir="/tmp/bundle",
        private=False,
        pushed=False,
        uploaded_files=4,
    )
    assert manifest["repo_id"] == "user/test-dataset"
    assert manifest["dataset_url"] == "https://huggingface.co/datasets/user/test-dataset"

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = write_publish_manifest(tmpdir, manifest)
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert payload["uploaded_files"] == 4
