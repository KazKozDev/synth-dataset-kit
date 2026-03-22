import json
import tempfile
from pathlib import Path

from synth_dataset_kit.cli import (
    _export_artifact_csv,
)
from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.evaluation import (
    export_metric_validation_report,
)
from synth_dataset_kit.exporters import (
    export_alpaca,
    export_case_study_bundle,
    export_chatml,
    export_eval_summary,
    export_huggingface_bundle,
    export_jsonl,
    export_pipeline_artifacts,
    export_proof_bundle,
    export_quality_report_json,
    export_sharegpt,
)
from synth_dataset_kit.generators.seed_expander import SeedExpander
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, QualityReport, Role


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


def test_export_jsonl():
    ds = _make_test_dataset()
    ds.examples[0].decontamination_flags = ["gsm8k"]
    ds.examples[0].decontamination_evidence = [
        {
            "benchmark": "gsm8k",
            "method": "ngram",
            "confidence": 0.91,
            "matched_text": "Janet's ducks lay 16 eggs per day.",
        }
    ]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    export_jsonl(ds, path, include_metadata=True)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert "messages" in record
    assert record["decontamination_flags"] == ["gsm8k"]
    assert record["decontamination_evidence"][0]["method"] == "ngram"
    Path(path).unlink()


def test_export_alpaca():
    ds = _make_test_dataset()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    export_alpaca(ds, path)
    with open(path) as f:
        records = json.load(f)
    assert len(records) == 3
    assert "instruction" in records[0]
    Path(path).unlink()


def test_export_sharegpt():
    ds = _make_test_dataset()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    export_sharegpt(ds, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert "conversations" in record
    Path(path).unlink()


def test_export_chatml():
    ds = _make_test_dataset()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    export_chatml(ds, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert "<|im_start|>" in record["text"]
    Path(path).unlink()


def test_export_huggingface_bundle():
    ds = _make_test_dataset()
    report = QualityReport(
        dataset_name="test_export",
        total_examples=3,
        passed_examples=3,
        failed_examples=0,
        avg_quality_score=8.2,
        contamination_hits=0,
        distribution_divergence=0.05,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_huggingface_bundle(ds, tmpdir, include_metadata=True, quality_report=report)
        bundle_dir = Path(paths[0])
        assert bundle_dir.exists()
        assert (bundle_dir / "train.jsonl").exists()
        assert (bundle_dir / "README.md").exists()
        assert (bundle_dir / "dataset_metadata.json").exists()
        assert (bundle_dir / "eval_summary.json").exists()
        card = (bundle_dir / "README.md").read_text(encoding="utf-8")
        assert "Data Generation Process" in card
        assert "Distribution Alignment" in card
        eval_summary = json.loads((bundle_dir / "eval_summary.json").read_text(encoding="utf-8"))
        assert eval_summary["distribution_divergence"] == 0.05
        assert eval_summary["examples"] == 3


def test_export_huggingface_bundle_with_baseline_comparison():
    ds = _make_test_dataset()
    report = QualityReport(
        dataset_name="generated_export",
        total_examples=3,
        passed_examples=3,
        failed_examples=0,
        avg_quality_score=8.0,
        lexical_diversity=0.44,
        contamination_hits=0,
        distribution_divergence=0.04,
    )
    baseline = _make_test_dataset()
    baseline.name = "baseline_seed_set"
    baseline_report = QualityReport(
        dataset_name="baseline_seed_set",
        total_examples=3,
        passed_examples=2,
        failed_examples=1,
        avg_quality_score=6.5,
        lexical_diversity=0.28,
        contamination_hits=1,
        distribution_divergence=0.11,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_huggingface_bundle(
            ds,
            tmpdir,
            include_metadata=True,
            quality_report=report,
            baseline_dataset=baseline,
            baseline_report=baseline_report,
        )
        bundle_dir = Path(paths[0])
        card = (bundle_dir / "README.md").read_text(encoding="utf-8")
        assert "Baseline Comparison" in card
        eval_summary = json.loads((bundle_dir / "eval_summary.json").read_text(encoding="utf-8"))
        assert eval_summary["baseline_comparison"]["baseline_dataset_name"] == "baseline_seed_set"
        assert eval_summary["baseline_comparison"]["avg_quality_delta"] == 1.5


def test_export_case_study_bundle():
    ds = _make_test_dataset()
    report = QualityReport(
        dataset_name="test_export",
        total_examples=3,
        passed_examples=3,
        failed_examples=0,
        avg_quality_score=8.2,
        contamination_hits=0,
        distribution_divergence=0.1,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_case_study_bundle(ds, report, tmpdir)
        assert Path(path).exists()
        contents = Path(path).read_text(encoding="utf-8")
        assert "Case Study" in contents
        assert "Distribution divergence" in contents


def test_export_eval_summary():
    ds = _make_test_dataset()
    report = QualityReport(
        dataset_name="test_export",
        total_examples=3,
        passed_examples=2,
        failed_examples=1,
        avg_quality_score=7.4,
        contamination_hits=1,
        contamination_verdicts={"clean": 2, "review": 1},
        distribution_divergence=0.12,
        issue_counts={"assistant_too_short": 1},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_eval_summary(ds, report, tmpdir)
        assert len(paths) == 2
        payload = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        assert payload["avg_quality_score"] == 7.4
        assert payload["distribution_divergence"] == 0.12
        markdown = Path(paths[1]).read_text(encoding="utf-8")
        assert "Eval Summary" in markdown
        assert "Top Issues" in markdown


def test_export_eval_summary_with_baseline_comparison():
    ds = _make_test_dataset()
    report = QualityReport(
        dataset_name="generated_export",
        total_examples=3,
        passed_examples=3,
        failed_examples=0,
        avg_quality_score=8.1,
        lexical_diversity=0.42,
        contamination_hits=0,
        contamination_verdicts={"clean": 3},
        distribution_divergence=0.08,
        issue_counts={"assistant_too_short": 1},
    )
    baseline = _make_test_dataset()
    baseline.name = "baseline_seed_set"
    baseline_report = QualityReport(
        dataset_name="baseline_seed_set",
        total_examples=3,
        passed_examples=2,
        failed_examples=1,
        avg_quality_score=6.9,
        lexical_diversity=0.31,
        contamination_hits=1,
        contamination_verdicts={"clean": 2, "review": 1},
        distribution_divergence=0.16,
        issue_counts={"assistant_too_short": 2},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_eval_summary(
            ds,
            report,
            tmpdir,
            baseline_dataset=baseline,
            baseline_report=baseline_report,
        )
        payload = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        comparison = payload["baseline_comparison"]
        assert comparison["baseline_dataset_name"] == "baseline_seed_set"
        assert comparison["avg_quality_delta"] == 1.2
        assert comparison["pass_rate_delta"] == round(1.0 - (2 / 3), 4)
        assert comparison["distribution_divergence_delta"] == -0.08
        markdown = Path(paths[1]).read_text(encoding="utf-8")
        assert "Baseline Comparison" in markdown
        assert "Avg quality delta" in markdown


def test_export_eval_summary_with_reference_comparison():
    ds = _make_test_dataset()
    ds.examples[0].metadata["cluster_id"] = "billing__c0__medium__procedural"
    ds.examples[1].metadata["cluster_id"] = "billing__c0__medium__procedural"
    ds.examples[2].metadata["cluster_id"] = "refunds__c1__easy__concise"
    report = QualityReport(
        dataset_name="generated_export",
        total_examples=3,
        passed_examples=3,
        failed_examples=0,
        avg_quality_score=8.1,
        avg_user_length=11.0,
        avg_assistant_length=48.0,
        diversity_score=0.88,
        lexical_diversity=0.42,
        topic_coverage={"billing": 2, "refunds": 1},
        difficulty_distribution={"easy": 1, "medium": 2},
    )
    reference = _make_test_dataset()
    reference.name = "reference_support_set"
    reference.examples[0].metadata["cluster_id"] = "billing__c0__medium__procedural"
    reference.examples[1].metadata["cluster_id"] = "technical__c2__medium__detailed"
    reference.examples[2].metadata["cluster_id"] = "technical__c2__medium__detailed"
    reference_report = QualityReport(
        dataset_name="reference_support_set",
        total_examples=3,
        passed_examples=3,
        failed_examples=0,
        avg_quality_score=7.7,
        avg_user_length=10.0,
        avg_assistant_length=44.0,
        diversity_score=0.81,
        lexical_diversity=0.35,
        topic_coverage={"billing": 1, "technical": 2},
        difficulty_distribution={"easy": 2, "medium": 1},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_eval_summary(
            ds,
            report,
            tmpdir,
            reference_dataset=reference,
            reference_report=reference_report,
        )
        payload = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        comparison = payload["reference_comparison"]
        assert comparison["reference_dataset_name"] == "reference_support_set"
        assert comparison["avg_assistant_length_delta"] == 4.0
        assert comparison["topic_overlap_ratio"] == 0.5
        assert "style_distribution_distance" in comparison
        assert "cluster_coverage_distance" in comparison
        assert "exact_overlap_ratio" in comparison
        assert "near_overlap_ratio" in comparison
        assert "semantic_overlap_ratio" in comparison
        assert "reference_alignment_score" in comparison
        validation = payload["distribution_validation"]
        assert validation["status"] == "reference_calibrated"
        assert "validated_distribution_match_score" in validation
        assert "calibration_error" in validation
        markdown = Path(paths[1]).read_text(encoding="utf-8")
        assert "Reference Dataset Comparison" in markdown
        assert "Distribution Validation" in markdown
        assert "Topic overlap ratio" in markdown
        assert "Style distribution distance" in markdown
        assert "Reference alignment score" in markdown


def test_export_proof_bundle():
    ds = _make_test_dataset()
    report = QualityReport(
        dataset_name="proof_dataset",
        total_examples=3,
        passed_examples=2,
        failed_examples=1,
        avg_quality_score=7.8,
        lexical_diversity=0.41,
        distribution_divergence=0.09,
        contamination_hits=0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        holdout_path = Path(tmpdir) / "holdout.jsonl"
        holdout_path.write_text('{"user":"u","assistant":"a"}\n', encoding="utf-8")
        paths = export_proof_bundle(
            ds,
            report,
            tmpdir,
            base_model="llama3.1:8b",
            trainer="unsloth",
            holdout_path=str(holdout_path),
        )
        bundle_dir = Path(paths[0])
        assert bundle_dir.exists()
        assert (bundle_dir / "proof_summary.json").exists()
        assert (bundle_dir / "run_finetune.sh").exists()
        assert (bundle_dir / "run_eval.sh").exists()
        assert (bundle_dir / "support_eval_rubric.json").exists()
        assert (bundle_dir / "holdout.jsonl").exists()
        markdown = (bundle_dir / "proof_summary.md").read_text(encoding="utf-8")
        assert "Proof Bundle" in markdown
        assert "Recommended Flow" in markdown
        assert "Included holdout file" in markdown


def test_export_metric_validation_report():
    report = {
        "runs_analyzed": 2,
        "avg_calibration_error": 1.5,
        "avg_validated_distribution_match_score": 77.0,
        "avg_reference_alignment_score": 78.5,
        "correlations": {
            "validated_match_vs_task_success_delta": 0.91,
            "validated_match_vs_pass_rate_delta": 0.88,
            "validated_match_vs_token_f1_delta": 0.86,
        },
        "runs": [
            {
                "name": "run_a",
                "validation_summary": {
                    "validated_distribution_match_score": 74.0,
                    "task_success_rate_delta": 0.04,
                    "pass_rate_delta": 0.03,
                    "avg_token_f1_delta": 0.02,
                },
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        files = export_metric_validation_report(report, tmpdir, name="metric_validation")
        payload = json.loads(Path(files[0]).read_text(encoding="utf-8"))
        assert payload["runs_analyzed"] == 2
        markdown = Path(files[1]).read_text(encoding="utf-8")
        assert "Metric Validation Report" in markdown
        assert "Validated match vs task success delta" in markdown
        assert "run_a" in markdown


def test_export_pipeline_artifacts():
    dataset = Dataset(name="artifact_test")
    dataset.artifacts = {
        "candidates": [
            Example(
                messages=[
                    Message(role=Role.USER, content="Candidate question"),
                    Message(
                        role=Role.ASSISTANT, content="Candidate answer with enough words to export."
                    ),
                ],
                metadata={"selection_decision": "candidate"},
            )
        ],
        "accepted": [
            Example(
                messages=[
                    Message(role=Role.USER, content="Accepted question"),
                    Message(
                        role=Role.ASSISTANT, content="Accepted answer with enough words to export."
                    ),
                ],
                metadata={"selection_decision": "accepted"},
            )
        ],
        "rejected": [
            Example(
                messages=[
                    Message(role=Role.USER, content="Rejected question"),
                    Message(
                        role=Role.ASSISTANT, content="Rejected answer with enough words to export."
                    ),
                ],
                metadata={
                    "selection_decision": "rejected",
                    "rejection_reasons": ["below_min_quality:7.0"],
                },
            )
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_pipeline_artifacts(dataset, tmpdir)
        assert len(paths) == 3
        rejected_path = [p for p in paths if p.endswith("_rejected.jsonl")][0]
        with open(rejected_path) as f:
            record = json.loads(f.readline())
        assert record["metadata"]["selection_decision"] == "rejected"
        assert "below_min_quality:7.0" in record["metadata"]["rejection_reasons"]


def test_export_artifact_csv_writes_rows():
    examples = [
        Example(
            messages=[
                Message(role=Role.USER, content="How do I cancel?"),
                Message(
                    role=Role.ASSISTANT, content="Open settings and cancel before the renewal date."
                ),
            ],
            quality_score=6.5,
            metadata={
                "selection_decision": "rejected",
                "rejection_reasons": ["below_min_quality:7.0", "assistant_too_short"],
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
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "rejected.csv"
        exported = _export_artifact_csv(examples, str(csv_path))
        assert exported == str(csv_path)
        with open(csv_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        assert "selection_decision" in lines[0]
        assert "decontamination_methods" in lines[0]
        assert "below_min_quality:7.0 | assistant_too_short" in lines[1]
        assert "gsm8k:ngram:0.91" in lines[1]


def test_export_quality_report_json():
    report = QualityReport(
        dataset_name="test",
        total_examples=2,
        passed_examples=1,
        failed_examples=1,
        avg_quality_score=6.0,
        issue_counts={"assistant_too_short": 1},
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    export_quality_report_json(report, path)
    with open(path) as f:
        data = json.load(f)
    assert data["dataset_name"] == "test"
    assert data["issue_counts"]["assistant_too_short"] == 1
    Path(path).unlink()


def test_export_huggingface_bundle_includes_yaml_metadata_and_quality_files():
    dataset = Dataset(
        name="customer_support_demo",
        examples=[
            Example(
                messages=[
                    Message(role=Role.USER, content="Where is my refund?"),
                    Message(
                        role=Role.ASSISTANT,
                        content="I checked your order and your refund is in progress.",
                    ),
                ],
                quality_score=8.5,
            )
        ],
    )
    report = QualityReport(
        dataset_name="customer_support_demo",
        total_examples=1,
        passed_examples=1,
        failed_examples=0,
        avg_quality_score=8.5,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        files = export_huggingface_bundle(dataset, tmpdir, quality_report=report)
        readme_path = Path(files[1])
        readme = readme_path.read_text(encoding="utf-8")
        assert readme.startswith("---\n")
        assert "license: mit" in readme
        assert "task_categories:" in readme
        assert "size_categories:" in readme
        assert (readme_path.parent / "customer_support_demo_quality_report.html").exists()
        assert (readme_path.parent / "customer_support_demo_quality_report.json").exists()
