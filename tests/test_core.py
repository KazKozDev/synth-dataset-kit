"""Tests for synth-dataset-kit core functionality."""

import json
import importlib
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import patch

from synth_dataset_kit.config import SDKConfig, LLMProvider
from synth_dataset_kit.engine import DatasetEngine
from synth_dataset_kit.cli import (
    _artifact_base_name,
    _artifact_example_preview,
    _export_artifact_csv,
    _artifact_summary,
    _default_demo_seed_path,
    _recommend_benchmark_result,
    _resolve_artifact_group,
    _select_benchmark_models,
    _sort_artifact_examples,
)
from synth_dataset_kit.models import Dataset, Example, Message, Role, QualityReport
from synth_dataset_kit.llm_client import LLMClient
import synth_dataset_kit.llm_client as llm_client_module
from synth_dataset_kit.decontamination import Decontaminator, BENCHMARK_SIGNATURES
from synth_dataset_kit.evaluation import (
    build_metric_validation_report,
    compare_models_on_holdout,
    evaluate_prediction_dataset,
    export_metric_validation_report,
    export_uplift_results,
)
from synth_dataset_kit.training import TrainingJob, save_training_job
from synth_dataset_kit.publishing import (
    build_publish_manifest,
    publish_huggingface_bundle,
    resolve_hf_token,
    write_publish_manifest,
)
from synth_dataset_kit.exporters import (
    export_alpaca,
    export_chatml,
    export_case_study_bundle,
    export_eval_summary,
    export_huggingface_bundle,
    export_jsonl,
    export_pipeline_artifacts,
    export_proof_bundle,
    export_quality_report_json,
    export_sharegpt,
)
from synth_dataset_kit.generators.seed_expander import SeedExpander, load_seed_file, _parse_example
from synth_dataset_kit.quality import QualityJudge
from synth_dataset_kit.support_cleanup import soften_support_answer, targeted_support_answer_review
from synth_dataset_kit.utils import safe_slug


# ─── CONFIG TESTS ────────────────────────────────────────────────────────────


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


# ─── MODEL TESTS ─────────────────────────────────────────────────────────────


def test_example_creation():
    ex = Example(messages=[
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.ASSISTANT, content="Hi there!"),
    ])
    assert ex.user_message == "Hello"
    assert ex.assistant_message == "Hi there!"
    assert len(ex.id) == 12


def test_safe_slug_shortens_and_normalizes_names():
    slug = safe_slug("E-commerce and SaaS customer support (orders, shipping, billing, subscriptions, app troubleshooting)")
    assert len(slug) <= 64
    assert "(" not in slug
    assert " " not in slug


def test_dataset_operations():
    ds = Dataset(name="test")
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Q1"),
            Message(role=Role.ASSISTANT, content="A1"),
        ],
        quality_score=8.0,
    ))
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Q2"),
            Message(role=Role.ASSISTANT, content="A2"),
        ],
        quality_score=4.0,
    ))
    assert ds.size == 2

    filtered = ds.filter_by_quality(7.0)
    assert filtered.size == 1
    assert filtered.examples[0].user_message == "Q1"


def test_dataset_remove_contaminated():
    ds = Dataset(name="test")
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Clean"),
            Message(role=Role.ASSISTANT, content="Clean answer"),
        ],
        decontamination_flags=[],
    ))
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Contaminated"),
            Message(role=Role.ASSISTANT, content="Bad answer"),
        ],
        decontamination_flags=["mmlu"],
    ))
    clean = ds.remove_contaminated()
    assert clean.size == 1
    assert clean.examples[0].user_message == "Clean"


def test_dataset_remove_contaminated_keeps_review_verdict():
    ds = Dataset(name="test")
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Needs review"),
            Message(role=Role.ASSISTANT, content="Potentially similar answer"),
        ],
        decontamination_flags=["gsm8k"],
        metadata={"contamination_verdict": "review"},
    ))
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Blocked"),
            Message(role=Role.ASSISTANT, content="Definitely contaminated"),
        ],
        decontamination_flags=["mmlu"],
        metadata={"contamination_verdict": "block"},
    ))
    clean = ds.remove_contaminated()
    assert clean.size == 1
    assert clean.examples[0].metadata["contamination_verdict"] == "review"


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


def test_soften_support_answer_reduces_overconfident_language():
    original = "I'll issue a refund right away and you'll receive it within 24 hours. I can check the order now."
    softened = soften_support_answer(original)
    assert "I'll issue" not in softened
    assert "right away" not in softened
    assert "Support can issue" in softened
    assert "typically within 24 hours" in softened


def test_targeted_support_answer_review_is_more_conservative():
    original = "Let me check this on our side. I'll escalate it and I'll issue a refund within 24 hours."
    reviewed = targeted_support_answer_review(original)
    assert "Let me" not in reviewed
    assert "I'll escalate" not in reviewed
    assert "I'll issue" not in reviewed
    assert "on our side" not in reviewed
    assert "Support should check" in reviewed or "The next step is to" in reviewed


# ─── PARSER TESTS ────────────────────────────────────────────────────────────


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


def test_seed_expander_dedup_candidates():
    seed = Example(messages=[
        Message(role=Role.USER, content="Question"),
        Message(role=Role.ASSISTANT, content="Answer"),
    ])
    candidate_1 = Example(messages=[
        Message(role=Role.USER, content="Question"),
        Message(role=Role.ASSISTANT, content="Answer"),
    ])
    candidate_2 = Example(messages=[
        Message(role=Role.USER, content="New question"),
        Message(role=Role.ASSISTANT, content="New answer"),
    ])
    candidate_3 = Example(messages=[
        Message(role=Role.USER, content="New question"),
        Message(role=Role.ASSISTANT, content="New answer"),
    ])

    expander = SeedExpander(_DummyClient(), SDKConfig().generation)
    deduped = expander.dedup_candidates([seed], [candidate_1, candidate_2, candidate_3])

    assert len(deduped) == 1
    assert deduped[0].user_message == "New question"
    assert deduped[0].metadata["pipeline_stage"] == "deduped"


def test_seed_expander_finalize_dataset():
    expander = SeedExpander(_DummyClient(), SDKConfig().generation.model_copy(update={"num_examples": 1}))
    candidate = Example(messages=[
        Message(role=Role.USER, content="User"),
        Message(role=Role.ASSISTANT, content="Assistant response with enough words."),
    ])

    dataset = expander.finalize_dataset({"domain": "customer support"}, [candidate])

    assert dataset.name == "expanded_customer_support"
    assert dataset.size == 1
    assert dataset.examples[0].metadata["pipeline_stage"] == "finalized"


def test_seed_expander_builds_distribution_profile():
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation.model_copy(update={"num_examples": 6}),
    )
    billing = Example(
        messages=[
            Message(role=Role.USER, content="How do I get a refund for a duplicate charge?"),
            Message(
                role=Role.ASSISTANT,
                content="Open billing settings, review the duplicate charge, submit the refund request, and wait for confirmation.",
            ),
        ]
    )
    account = Example(
        messages=[
            Message(role=Role.USER, content="Reset my password"),
            Message(role=Role.ASSISTANT, content="Use the reset link and choose a new password."),
        ]
    )
    analysis = {"domain": "support", "topics_covered": ["billing", "account"]}

    profile = expander.build_seed_distribution_profile([billing, billing.model_copy(), account], analysis)

    assert profile["cluster_count"] >= 2
    billing_seed_count = sum(
        int(cluster["seed_count"])
        for cluster in profile["clusters"]
        if cluster["topic"] == "billing"
    )
    account_target = sum(
        int(cluster["target_examples"])
        for cluster in profile["clusters"]
        if cluster["topic"] == "account"
    )
    billing_target = sum(
        int(cluster["target_examples"])
        for cluster in profile["clusters"]
        if cluster["topic"] == "billing"
    )
    assert billing_seed_count == 2
    assert billing_target >= account_target
    assert profile["semantic_graph"]["cluster_count"] >= 2
    assert profile["semantic_graph"]["allocator"] in {"lexical_fallback", "embedding_centroid"}


def test_seed_expander_semantic_graph_builds_neighbors_without_embeddings():
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation.model_copy(update={"graph_neighbor_k": 2}),
    )
    clusters = [
        {
            "cluster_id": "billing__c0__medium__procedural",
            "topic": "billing",
            "style": "procedural",
            "difficulty": "medium",
            "target_examples": 2,
        },
        {
            "cluster_id": "refunds__c0__hard__detailed",
            "topic": "refunds",
            "style": "detailed",
            "difficulty": "hard",
            "target_examples": 1,
        },
        {
            "cluster_id": "account__c1__easy__concise",
            "topic": "account",
            "style": "concise",
            "difficulty": "easy",
            "target_examples": 1,
        },
    ]

    graph = expander._build_semantic_coverage_graph(
        clusters,
        {
            "billing__c0__medium__procedural": ["refund duplicate charge billing support"],
            "refunds__c0__hard__detailed": ["refund request escalation charge dispute"],
            "account__c1__easy__concise": ["reset password account login"],
        },
    )

    assert graph["allocator"] == "lexical_fallback"
    assert len(graph["neighbors"]["billing__c0__medium__procedural"]) == 2


def test_seed_expander_generation_plan_prioritizes_underrepresented_clusters():
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation.model_copy(update={"num_examples": 6}),
    )
    analysis = {
        "domain": "support",
        "seed_distribution_profile": {
            "clusters": [
                {
                    "cluster_id": "billing__medium__procedural",
                    "topic": "billing",
                    "style": "procedural",
                    "difficulty": "medium",
                    "seed_count": 2,
                    "target_examples": 4,
                },
                {
                    "cluster_id": "account__easy__concise",
                    "topic": "account",
                    "style": "concise",
                    "difficulty": "easy",
                    "seed_count": 1,
                    "target_examples": 2,
                },
            ]
        },
    }
    accepted = [
        Example(
            messages=[
                Message(role=Role.USER, content="accepted"),
                Message(role=Role.ASSISTANT, content="accepted response"),
            ],
            metadata={"cluster_id": "billing__medium__procedural"},
        )
    ]

    plan = expander.build_generation_plan(analysis, accepted_examples=accepted)
    billing_slots = [item for item in plan if item["cluster_id"] == "billing__medium__procedural"]
    account_slots = [item for item in plan if item["cluster_id"] == "account__easy__concise"]

    assert len(billing_slots) == 3
    assert len(account_slots) == 2
    assert all(item["style"] for item in plan)


def test_seed_expander_generation_plan_uses_semantic_adaptive_bonus():
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation.model_copy(
            update={
                "num_examples": 10,
                "semantic_focus_top_k": 3,
                "long_tail_boost": 0.5,
            }
        ),
    )
    analysis = {
        "domain": "support",
        "seed_distribution_profile": {
            "clusters": [
                {
                    "cluster_id": "billing__c0__medium__procedural",
                    "topic": "billing",
                    "style": "procedural",
                    "difficulty": "medium",
                    "seed_count": 3,
                    "target_examples": 4,
                    "semantic_cluster": 0,
                },
                {
                    "cluster_id": "refunds__c0__hard__detailed",
                    "topic": "refunds",
                    "style": "detailed",
                    "difficulty": "hard",
                    "seed_count": 1,
                    "target_examples": 2,
                    "semantic_cluster": 0,
                },
            ]
        },
    }
    accepted = [
        Example(
            messages=[
                Message(role=Role.USER, content="accepted"),
                Message(role=Role.ASSISTANT, content="accepted response"),
            ],
            metadata={"cluster_id": "billing__c0__medium__procedural"},
        )
    ]

    plan = expander.build_generation_plan(analysis, accepted_examples=accepted)
    billing_slots = [item for item in plan if item["cluster_id"] == "billing__c0__medium__procedural"]
    refund_slots = [item for item in plan if item["cluster_id"] == "refunds__c0__hard__detailed"]

    assert len(billing_slots) >= 3
    assert len(refund_slots) >= 3


def test_seed_expander_optimizer_prefers_graph_frontier_clusters():
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation.model_copy(update={"distance_allocator_weight": 1.0}),
    )
    clusters = [
        {
            "cluster_id": "billing__c0__medium__procedural",
            "target_examples": 2,
            "_priority_score": 1.0,
            "_adaptive_bonus": 0,
            "_graph_frontier_score": 0.8,
        },
        {
            "cluster_id": "refunds__c0__hard__detailed",
            "target_examples": 2,
            "_priority_score": 1.0,
            "_adaptive_bonus": 0,
            "_graph_frontier_score": 0.2,
        },
    ]

    allocations = expander._optimize_cluster_allocations(clusters, accepted_counts={})

    assert allocations["billing__c0__medium__procedural"] >= allocations["refunds__c0__hard__detailed"]


def test_seed_expander_generation_plan_limits_active_clusters_for_large_profiles():
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation.model_copy(
            update={"num_examples": 20, "max_active_clusters_per_round": 3}
        ),
    )
    clusters = []
    for index in range(8):
        clusters.append(
            {
                "cluster_id": f"topic_{index}__c{index}__medium__concise",
                "topic": f"topic_{index}",
                "style": "concise",
                "difficulty": "medium",
                "seed_count": 1,
                "target_examples": index + 1,
                "semantic_cluster": index,
            }
        )
    analysis = {
        "domain": "support",
        "seed_distribution_profile": {"clusters": clusters},
    }

    plan = expander.build_generation_plan(analysis, accepted_examples=[])
    active_clusters = {item["cluster_id"] for item in plan}

    assert len(active_clusters) == 3


def test_seed_expander_embedding_cluster_assignments_fallback_without_transformers():
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation.model_copy(update={"num_examples": 4}),
    )
    seeds = [
        Example(messages=[Message(role=Role.USER, content="Q1"), Message(role=Role.ASSISTANT, content="A1")]),
        Example(messages=[Message(role=Role.USER, content="Q2"), Message(role=Role.ASSISTANT, content="A2")]),
    ]

    assignments = expander._embedding_cluster_assignments(seeds)

    assert assignments == [0, 1]


def test_seed_expander_keep_best_filters_low_scores():
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation,
        quality_judge=judge,
        min_quality=7.0,
    )
    strong = Example(messages=[
        Message(role=Role.USER, content="Tell me how to reset my account password safely"),
        Message(role=Role.ASSISTANT, content="Open settings, choose security, request a reset link, and confirm the email before changing the password."),
    ])
    weak = Example(messages=[
        Message(role=Role.USER, content="Hi"),
        Message(role=Role.ASSISTANT, content="Short."),
    ])

    audited = expander.audit_candidates([strong, weak])
    kept = expander.keep_best(audited, limit=5)

    assert len(kept) == 1
    assert kept[0].user_message.startswith("Tell me how")
    assert kept[0].metadata["pipeline_stage"] == "keep_best"


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


def test_seed_expander_expand_refills_after_filtering():
    config = SDKConfig().generation.model_copy(update={"num_examples": 2, "batch_size": 2})
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    round_one = [
        Example(messages=[
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Short."),
        ]),
        Example(messages=[
            Message(role=Role.USER, content="Explain how to update my billing address in the app"),
            Message(role=Role.ASSISTANT, content="Open billing settings, edit the address form, save the new details, and verify the confirmation message."),
        ]),
    ]
    round_two = [
        Example(messages=[
            Message(role=Role.USER, content="How can I cancel my subscription before renewal?"),
            Message(role=Role.ASSISTANT, content="Go to subscription settings, choose cancel, review the renewal date, and confirm the cancellation to stop the next charge."),
        ]),
    ]
    expander = _RoundRobinExpander(
        [round_one, round_two],
        _DummyClient(),
        config,
        quality_judge=judge,
        min_quality=7.0,
        max_rounds=3,
    )
    seed = Example(messages=[
        Message(role=Role.USER, content="Seed question"),
        Message(role=Role.ASSISTANT, content="Seed answer with enough words to be valid."),
    ])

    dataset = expander.expand([seed], {"domain": "support"})

    assert dataset.size == 2
    assert all(example.metadata["pipeline_stage"] == "finalized" for example in dataset.examples)
    assert len(dataset.artifacts["candidates"]) == 3
    assert len(dataset.artifacts["accepted"]) == 2
    assert len(dataset.artifacts["rejected"]) == 1
    assert dataset.artifacts["rejected"][0].metadata["selection_decision"] == "rejected"


def test_seed_expander_expand_records_rebalancing_history():
    config = SDKConfig().generation.model_copy(update={"num_examples": 3, "batch_size": 2})
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    round_one = [
        Example(
            messages=[
                Message(role=Role.USER, content="Refund for duplicate charge"),
                Message(
                    role=Role.ASSISTANT,
                    content="Open billing settings, review the duplicate charge, and submit the refund request for processing.",
                ),
            ],
            metadata={"cluster_id": "billing__c0__medium__procedural"},
        ),
        Example(
            messages=[
                Message(role=Role.USER, content="How do I update my card?"),
                Message(
                    role=Role.ASSISTANT,
                    content="Go to billing settings, replace the saved card, verify the new details, and confirm the update.",
                ),
            ],
            metadata={"cluster_id": "billing__c0__medium__procedural"},
        ),
    ]
    round_two = [
        Example(
            messages=[
                Message(role=Role.USER, content="Reset my password"),
                Message(
                    role=Role.ASSISTANT,
                    content="Use the password reset link, verify your identity, and set a new password for the account.",
                ),
            ],
            metadata={"cluster_id": "account__c1__easy__concise"},
        )
    ]
    expander = _TrackingRoundRobinExpander(
        [round_one, round_two],
        _DummyClient(),
        config,
        quality_judge=judge,
        min_quality=7.0,
        max_rounds=3,
    )
    seed_billing = Example(
        messages=[
            Message(role=Role.USER, content="Billing issue with duplicate charge"),
            Message(role=Role.ASSISTANT, content="Review billing settings and request a refund."),
        ],
        metadata={"seed_cluster_id": "billing__c0__medium__procedural"},
    )
    seed_account = Example(
        messages=[
            Message(role=Role.USER, content="I forgot my password"),
            Message(role=Role.ASSISTANT, content="Use the reset link to recover your account."),
        ],
        metadata={"seed_cluster_id": "account__c1__easy__concise"},
    )
    analysis = {
        "domain": "support",
        "seed_distribution_profile": {
            "clusters": [
                {
                    "cluster_id": "billing__c0__medium__procedural",
                    "topic": "billing",
                    "style": "procedural",
                    "difficulty": "medium",
                    "seed_count": 2,
                    "target_examples": 2,
                },
                {
                    "cluster_id": "account__c1__easy__concise",
                    "topic": "account",
                    "style": "concise",
                    "difficulty": "easy",
                    "seed_count": 1,
                    "target_examples": 1,
                },
            ]
        },
    }

    dataset = expander.expand([seed_billing, seed_account], analysis)

    history = dataset.config_snapshot["rebalancing_history"]
    assert len(history) >= 2
    assert history[0]["accepted_total"] == 2
    assert history[1]["focus_cluster_ids"] == ["account__c1__easy__concise"]
    assert "semantic_coverage_score" in history[0]
    assert dataset.config_snapshot["final_distribution_status"]["gaps"] == {}
    assert expander.focus_history[1] == ["account__c1__easy__concise"]


def test_seed_expander_distribution_status_tracks_oversaturation():
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation.model_copy(update={"saturation_threshold": 1.0}),
    )
    analysis = {
        "seed_distribution_profile": {
            "clusters": [
                {
                    "cluster_id": "billing__c0__medium__procedural",
                    "target_examples": 1,
                    "semantic_cluster": 0,
                }
            ]
        }
    }
    accepted = [
        Example(
            messages=[
                Message(role=Role.USER, content="Refund"),
                Message(role=Role.ASSISTANT, content="Check billing and request a refund."),
            ],
            metadata={"cluster_id": "billing__c0__medium__procedural"},
        ),
        Example(
            messages=[
                Message(role=Role.USER, content="Refund again"),
                Message(role=Role.ASSISTANT, content="Open billing, review the charge, and submit another refund request."),
            ],
            metadata={"cluster_id": "billing__c0__medium__procedural"},
        ),
    ]

    status = expander._distribution_status(analysis, accepted)

    assert status["oversaturated_clusters"] == {"billing__c0__medium__procedural": 1}


def test_seed_expander_distribution_status_includes_graph_coverage():
    expander = SeedExpander(
        _DummyClient(),
        SDKConfig().generation,
    )
    analysis = {
        "seed_distribution_profile": {
            "clusters": [
                {
                    "cluster_id": "billing__c0__medium__procedural",
                    "target_examples": 2,
                    "semantic_cluster": 0,
                },
                {
                    "cluster_id": "refunds__c0__hard__detailed",
                    "target_examples": 1,
                    "semantic_cluster": 0,
                },
            ],
            "semantic_graph": {
                "neighbors": {
                    "billing__c0__medium__procedural": [
                        {
                            "cluster_id": "refunds__c0__hard__detailed",
                            "similarity": 0.9,
                            "distance": 0.1,
                        }
                    ],
                    "refunds__c0__hard__detailed": [
                        {
                            "cluster_id": "billing__c0__medium__procedural",
                            "similarity": 0.9,
                            "distance": 0.1,
                        }
                    ],
                }
            },
        }
    }
    accepted = [
        Example(
            messages=[
                Message(role=Role.USER, content="Refund"),
                Message(role=Role.ASSISTANT, content="Check billing and request a refund."),
            ],
            metadata={"cluster_id": "billing__c0__medium__procedural"},
        ),
    ]

    status = expander._distribution_status(analysis, accepted)

    assert status["graph_coverage_score"] > 0
    assert status["graph_frontier_clusters"]


# ─── DECONTAMINATION TESTS ──────────────────────────────────────────────────


def test_decontaminator_clean():
    decon = Decontaminator(benchmarks=["mmlu", "gsm8k"])
    ex = Example(messages=[
        Message(role=Role.USER, content="How do I make pasta?"),
        Message(role=Role.ASSISTANT, content="Boil water, add pasta, cook 8-10 mins."),
    ])
    flags = decon.check_example(ex)
    assert len(flags) == 0


def test_decontaminator_catches_gsm8k():
    decon = Decontaminator(benchmarks=["gsm8k"])
    ex = Example(messages=[
        Message(role=Role.USER, content="Janet's ducks lay 16 eggs per day. She eats three for breakfast."),
        Message(role=Role.ASSISTANT, content="Let me calculate that."),
    ])
    flags = decon.check_example(ex)
    assert "gsm8k" in flags


def test_decontaminator_catches_humaneval():
    decon = Decontaminator(benchmarks=["humaneval"])
    ex = Example(messages=[
        Message(role=Role.USER, content="Write a function def has_close_elements that checks if any two numbers are close."),
        Message(role=Role.ASSISTANT, content="Here's the implementation..."),
    ])
    flags = decon.check_example(ex)
    assert "humaneval" in flags


def test_decontaminator_dataset():
    decon = Decontaminator()
    ds = Dataset(name="test")
    ds.add(Example(messages=[
        Message(role=Role.USER, content="Normal question about cooking"),
        Message(role=Role.ASSISTANT, content="Here's a recipe..."),
    ]))
    ds.add(Example(messages=[
        Message(role=Role.USER, content="Janet's ducks lay 16 eggs per day"),
        Message(role=Role.ASSISTANT, content="The answer is..."),
    ]))
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
    assert checked.examples[0].decontamination_evidence[0]["method"] in {"ngram", "substring", "embedding"}
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

    evidence = [item for item in checked.examples[0].decontamination_evidence if item["method"] == "embedding"]
    assert len(evidence) == 2
    assert evidence[0]["match_rank"] == 1
    assert evidence[1]["match_rank"] == 2
    assert checked.examples[0].metadata["embedding_top_k_matches"][0]["match_rank"] == 1
    assert checked.examples[0].metadata["contamination_verdict"] == "review"


# ─── EXPORTER TESTS ─────────────────────────────────────────────────────────


def _make_test_dataset() -> Dataset:
    ds = Dataset(name="test_export")
    for i in range(3):
        ds.add(Example(messages=[
            Message(role=Role.USER, content=f"Question {i}"),
            Message(role=Role.ASSISTANT, content=f"Answer {i}"),
        ]))
    return ds


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


def test_evaluate_prediction_dataset():
    holdout = Dataset(
        name="holdout",
        examples=[
            Example(
                messages=[
                    Message(role=Role.USER, content="Refund"),
                    Message(role=Role.ASSISTANT, content="Share your order number so support can review the refund."),
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
                    Message(role=Role.ASSISTANT, content="Please share your order number so support can review the refund."),
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


def test_export_pipeline_artifacts():
    dataset = Dataset(name="artifact_test")
    dataset.artifacts = {
        "candidates": [
            Example(
                messages=[
                    Message(role=Role.USER, content="Candidate question"),
                    Message(role=Role.ASSISTANT, content="Candidate answer with enough words to export."),
                ],
                metadata={"selection_decision": "candidate"},
            )
        ],
        "accepted": [
            Example(
                messages=[
                    Message(role=Role.USER, content="Accepted question"),
                    Message(role=Role.ASSISTANT, content="Accepted answer with enough words to export."),
                ],
                metadata={"selection_decision": "accepted"},
            )
        ],
        "rejected": [
            Example(
                messages=[
                    Message(role=Role.USER, content="Rejected question"),
                    Message(role=Role.ASSISTANT, content="Rejected answer with enough words to export."),
                ],
                metadata={"selection_decision": "rejected", "rejection_reasons": ["below_min_quality:7.0"]},
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


def test_artifact_summary_counts_rejection_reasons():
    dataset = Dataset(name="artifact_summary")
    dataset.artifacts = {
        "candidates": [
            Example(messages=[
                Message(role=Role.USER, content="c1"),
                Message(role=Role.ASSISTANT, content="candidate one with enough words."),
            ]),
            Example(messages=[
                Message(role=Role.USER, content="c2"),
                Message(role=Role.ASSISTANT, content="candidate two with enough words."),
            ]),
            Example(messages=[
                Message(role=Role.USER, content="c3"),
                Message(role=Role.ASSISTANT, content="candidate three with enough words."),
            ]),
        ],
        "accepted": [
            Example(messages=[
                Message(role=Role.USER, content="a1"),
                Message(role=Role.ASSISTANT, content="accepted example with enough words."),
            ])
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


def test_artifact_example_preview_exposes_key_fields():
    example = Example(
        messages=[
            Message(role=Role.USER, content="How do I cancel my subscription?"),
            Message(role=Role.ASSISTANT, content="Open billing settings, choose cancel, and confirm the cancellation before the renewal date."),
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


def test_export_artifact_csv_writes_rows():
    examples = [
        Example(
            messages=[
                Message(role=Role.USER, content="How do I cancel?"),
                Message(role=Role.ASSISTANT, content="Open settings and cancel before the renewal date."),
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


# ─── QUALITY REPORT TESTS ───────────────────────────────────────────────────


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
                Message(role=Role.ASSISTANT, content="Open security settings, request a reset email, confirm the code, and choose a new password."),
            ],
            quality_score=7.5,
            metadata={"topic": "auth", "difficulty": "easy", "quality_issues": []},
        )
    )
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Explain how to migrate a multi-tenant billing setup without losing invoices"),
                Message(role=Role.ASSISTANT, content="Audit the schema, export invoice history, stage the tenant mapping, validate payment provider identifiers, and run a reconciliation pass before cutover."),
            ],
            quality_score=8.8,
            metadata={"topic": "billing", "difficulty": "hard", "quality_issues": ["complex_domain"]},
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
                Message(role=Role.ASSISTANT, content="Answer with enough words for testing quality report generation."),
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
                Message(role=Role.ASSISTANT, content="Open billing settings and submit the refund request."),
            ],
            quality_score=8.0,
            metadata={"quality_issues": [], "cluster_id": "billing__c0__medium__procedural"},
        )
    )
    dataset.add(
        Example(
            messages=[
                Message(role=Role.USER, content="Reset password"),
                Message(role=Role.ASSISTANT, content="Use the reset link and create a new password."),
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
                Message(role=Role.ASSISTANT, content="Open billing settings and submit the refund request."),
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
                Message(role=Role.ASSISTANT, content="Use the reset link and create a new password."),
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
                Message(role=Role.ASSISTANT, content="Open billing settings and submit the refund request."),
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
                    Message(role=Role.ASSISTANT, content="I checked your order and your refund is in progress."),
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


def test_resolve_hf_token_prefers_explicit_value():
    assert resolve_hf_token("abc123") == "abc123"


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


if __name__ == "__main__":
    # Run all tests
    import sys

    test_functions = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
