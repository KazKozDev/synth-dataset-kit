from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.generators.seed_expander import SeedExpander
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, QualityReport, Role
from synth_dataset_kit.quality import QualityJudge


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


def test_quality_report():
    report = QualityReport(
        dataset_name="test",
        total_examples=100,
        passed_examples=85,
        failed_examples=15,
        avg_quality_score=7.5,
        diversity_score=0.82,
    )
    assert report.passed_examples + report.failed_examples == report.total_examples


def test_quality_report_includes_difficulty_and_diversity_metrics():
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    dataset = Dataset(name="quality_metrics")
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Reset my password"),
                Message(
                    role=Role.ASSISTANT,
                    content="Open security settings, request a reset email, confirm the code, and choose a new password.",
                ),
            ],
            quality_score=7.5,
            metadata={"topic": "auth", "difficulty": "easy", "quality_issues": []},
        )
    )
    dataset.add(
        Example(
            messages=[
                Message(
                    role=Role.USER,
                    content="Explain how to migrate a multi-tenant billing setup without losing invoices",
                ),
                Message(
                    role=Role.ASSISTANT,
                    content="Audit the schema, export invoice history, stage the tenant mapping, validate payment provider identifiers, and run a reconciliation pass before cutover.",
                ),
            ],
            quality_score=8.8,
            metadata={
                "topic": "billing",
                "difficulty": "hard",
                "quality_issues": ["complex_domain"],
            },
        )
    )

    report = judge.generate_report(dataset)

    assert report.lexical_diversity > 0
    assert report.self_bleu_proxy >= 0
    assert report.difficulty_distribution["easy"] == 1
    assert report.difficulty_distribution["hard"] == 1
    assert report.topic_heatmap["auth"]["easy"] == 1
    assert report.topic_heatmap["billing"]["hard"] == 1


def test_quality_report_includes_benchmark_sources():
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    dataset = Dataset(name="decon_sources")
    dataset.config_snapshot["benchmark_sources"] = {
        "gsm8k": "datasets",
        "mmlu": "fallback_signatures",
    }
    dataset.config_snapshot["benchmark_sample_counts"] = {
        "gsm8k": 128,
        "mmlu": 10,
    }
    dataset.config_snapshot["benchmark_load_errors"] = {
        "mmlu": "datasets not installed",
    }
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Question about math"),
                Message(
                    role=Role.ASSISTANT,
                    content="Answer with enough words for testing quality report generation.",
                ),
            ],
            quality_score=7.0,
            metadata={"quality_issues": []},
        )
    )

    report = judge.generate_report(dataset)

    assert report.benchmark_sources["gsm8k"] == "datasets"
    assert report.benchmark_sources["mmlu"] == "fallback_signatures"
    assert report.benchmark_sample_counts["gsm8k"] == 128
    assert report.benchmark_sample_counts["mmlu"] == 10
    assert report.benchmark_load_errors["mmlu"] == "datasets not installed"


def test_quality_report_includes_distribution_divergence():
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    dataset = Dataset(name="distribution_report")
    dataset.config_snapshot["seed_distribution_profile"] = {
        "clusters": [
            {"cluster_id": "billing__c0__medium__procedural", "target_examples": 4},
            {"cluster_id": "account__c1__easy__concise", "target_examples": 2},
        ]
    }
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Refund request"),
                Message(
                    role=Role.ASSISTANT,
                    content="Open billing settings and submit the refund request.",
                ),
            ],
            quality_score=8.0,
            metadata={"quality_issues": [], "cluster_id": "billing__c0__medium__procedural"},
        )
    )
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Reset password"),
                Message(
                    role=Role.ASSISTANT, content="Use the reset link and create a new password."
                ),
            ],
            quality_score=7.8,
            metadata={"quality_issues": [], "cluster_id": "account__c1__easy__concise"},
        )
    )

    report = judge.generate_report(dataset)

    assert report.seed_cluster_distribution["billing__c0__medium__procedural"] == 4
    assert report.generated_cluster_distribution["billing__c0__medium__procedural"] == 1
    assert report.underrepresented_clusters["billing__c0__medium__procedural"] == 3
    assert report.distribution_divergence > 0


def test_quality_report_includes_distribution_match_and_semantic_coverage():
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    dataset = Dataset(name="distribution_kpi_report")
    dataset.config_snapshot["seed_distribution_profile"] = {
        "clusters": [
            {
                "cluster_id": "billing__c0__medium__procedural",
                "target_examples": 4,
                "semantic_cluster": 0,
            },
            {
                "cluster_id": "account__c1__easy__concise",
                "target_examples": 2,
                "semantic_cluster": 1,
            },
        ]
    }
    dataset.config_snapshot["final_distribution_status"] = {
        "distribution_divergence": 0.1667,
        "distribution_match_score": 69.67,
        "semantic_cluster_target_distribution": {"0": 4, "1": 2},
        "semantic_cluster_generated_distribution": {"0": 1, "1": 1},
        "semantic_coverage_score": 0.3333,
        "semantic_coverage_gaps": {"0": 3, "1": 1},
        "graph_coverage_score": 0.455,
        "graph_frontier_clusters": ["billing__c0__medium__procedural"],
        "gaps": {
            "billing__c0__medium__procedural": 3,
            "account__c1__easy__concise": 1,
        },
    }
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Refund request"),
                Message(
                    role=Role.ASSISTANT,
                    content="Open billing settings and submit the refund request.",
                ),
            ],
            quality_score=8.0,
            metadata={
                "quality_issues": [],
                "cluster_id": "billing__c0__medium__procedural",
                "semantic_cluster": 0,
            },
        )
    )
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Reset password"),
                Message(
                    role=Role.ASSISTANT, content="Use the reset link and create a new password."
                ),
            ],
            quality_score=7.8,
            metadata={
                "quality_issues": [],
                "cluster_id": "account__c1__easy__concise",
                "semantic_cluster": 1,
            },
        )
    )

    report = judge.generate_report(dataset)

    assert report.distribution_match_score == 69.67
    assert report.semantic_cluster_target_distribution == {"0": 4, "1": 2}
    assert report.semantic_cluster_generated_distribution == {"0": 1, "1": 1}
    assert report.semantic_coverage_score == 0.3333
    assert report.semantic_coverage_gaps == {"0": 3, "1": 1}
    assert report.graph_coverage_score == 0.455
    assert report.graph_frontier_clusters == ["billing__c0__medium__procedural"]


def test_quality_report_includes_rebalancing_history():
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    dataset = Dataset(name="rebalancing_report")
    dataset.config_snapshot["rebalancing_history"] = [
        {
            "round": 1,
            "requested": 4,
            "accepted_total": 2,
            "rejected_batch": 1,
            "distribution_divergence": 0.33,
        },
        {
            "round": 2,
            "requested": 2,
            "accepted_total": 3,
            "rejected_batch": 0,
            "distribution_divergence": 0.0,
        },
    ]
    dataset.config_snapshot["final_distribution_status"] = {
        "distribution_divergence": 0.0,
        "gaps": {},
    }
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Refund request"),
                Message(
                    role=Role.ASSISTANT,
                    content="Open billing settings and submit the refund request.",
                ),
            ],
            quality_score=8.0,
            metadata={"quality_issues": []},
        )
    )

    report = judge.generate_report(dataset)

    assert len(report.rebalancing_history) == 2
    assert report.rebalancing_history[-1]["distribution_divergence"] == 0.0
    assert report.final_distribution_status["gaps"] == {}


def test_quality_report_html_includes_distribution_match_and_semantic_coverage(tmp_path):
    report = QualityReport(
        dataset_name="html_distribution_report",
        total_examples=10,
        passed_examples=8,
        failed_examples=2,
        avg_quality_score=7.9,
        distribution_divergence=0.12,
        distribution_match_score=84.5,
        seed_cluster_distribution={"billing__c0__medium__procedural": 6},
        generated_cluster_distribution={"billing__c0__medium__procedural": 5},
        underrepresented_clusters={"billing__c0__medium__procedural": 1},
        semantic_cluster_target_distribution={"0": 6},
        semantic_cluster_generated_distribution={"0": 5},
        semantic_coverage_score=0.8333,
        semantic_coverage_gaps={"0": 1},
        graph_coverage_score=0.8125,
        graph_frontier_clusters=["billing__c0__medium__procedural"],
    )

    from synth_dataset_kit.exporters import export_quality_report_html

    output_path = tmp_path / "quality.html"
    export_quality_report_html(report, str(output_path))
    html = output_path.read_text()

    assert "Match Score: 84.50/100" in html
    assert "Semantic Coverage: 83.33%" in html
    assert "Graph Coverage: 81.25%" in html
    assert "Semantic Coverage</h2>" in html
    assert "Graph Frontier</h2>" in html
    assert "cluster_0" in html


def test_quality_report_includes_contamination_methods_and_evidence():
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    dataset = Dataset(name="hybrid_evidence")
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Janet's ducks lay 16 eggs per day."),
                Message(role=Role.ASSISTANT, content="Let me calculate that."),
            ],
            quality_score=7.5,
            decontamination_flags=["gsm8k"],
            decontamination_evidence=[
                {
                    "benchmark": "gsm8k",
                    "method": "ngram",
                    "confidence": 0.91,
                    "matched_text": "Janet's ducks lay 16 eggs per day.",
                },
                {
                    "benchmark": "gsm8k",
                    "method": "embedding",
                    "confidence": 0.96,
                    "matched_text": "Janet's ducks lay 16 eggs per day.",
                },
            ],
            metadata={"quality_issues": [], "contamination_verdict": "review"},
        )
    )

    report = judge.generate_report(dataset)

    assert report.contamination_hits == 1
    assert report.contamination_verdicts["review"] == 1
    assert report.contamination_methods["ngram"] == 1
    assert report.contamination_methods["embedding"] == 1
    assert report.contamination_method_benchmarks["gsm8k"]["ngram"] == 1
    assert report.contamination_method_benchmarks["gsm8k"]["embedding"] == 1
    assert report.contamination_evidence_samples[0]["benchmark"] == "gsm8k"
    assert report.contamination_evidence_samples[0]["method"] in {"ngram", "embedding"}


def test_quality_judge_adds_rule_issues():
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    example = Example(
        messages=[
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Short."),
        ]
    )
    score = judge.score_example(example)
    assert score < 8
    assert "user_too_short" in example.metadata["quality_issues"]
    assert "assistant_too_short" in example.metadata["quality_issues"]
