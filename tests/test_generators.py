import sys
from unittest.mock import patch

from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.generators.seed_expander import SeedExpander
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, Role
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


def test_seed_expander_dedup_candidates():
    seed = Example(
        messages=[
            Message(role=Role.USER, content="Question"),
            Message(role=Role.ASSISTANT, content="Answer"),
        ]
    )
    candidate_1 = Example(
        messages=[
            Message(role=Role.USER, content="Question"),
            Message(role=Role.ASSISTANT, content="Answer"),
        ]
    )
    candidate_2 = Example(
        messages=[
            Message(role=Role.USER, content="New question"),
            Message(role=Role.ASSISTANT, content="New answer"),
        ]
    )
    candidate_3 = Example(
        messages=[
            Message(role=Role.USER, content="New question"),
            Message(role=Role.ASSISTANT, content="New answer"),
        ]
    )

    expander = SeedExpander(_DummyClient(), SDKConfig().generation)
    deduped = expander.dedup_candidates([seed], [candidate_1, candidate_2, candidate_3])

    assert len(deduped) == 1
    assert deduped[0].user_message == "New question"
    assert deduped[0].metadata["pipeline_stage"] == "deduped"


def test_seed_expander_finalize_dataset():
    expander = SeedExpander(
        _DummyClient(), SDKConfig().generation.model_copy(update={"num_examples": 1})
    )
    candidate = Example(
        messages=[
            Message(role=Role.USER, content="User"),
            Message(role=Role.ASSISTANT, content="Assistant response with enough words."),
        ]
    )

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

    profile = expander.build_seed_distribution_profile(
        [billing, billing.model_copy(), account], analysis
    )

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

    # Force lexical fallback by hiding sentence_transformers import.
    with patch.dict(sys.modules, {"sentence_transformers": None}):
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
    billing_slots = [
        item for item in plan if item["cluster_id"] == "billing__c0__medium__procedural"
    ]
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

    assert (
        allocations["billing__c0__medium__procedural"] >= allocations["refunds__c0__hard__detailed"]
    )


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
        Example(
            messages=[
                Message(role=Role.USER, content="Q1"),
                Message(role=Role.ASSISTANT, content="A1"),
            ]
        ),
        Example(
            messages=[
                Message(role=Role.USER, content="Q2"),
                Message(role=Role.ASSISTANT, content="A2"),
            ]
        ),
    ]

    # Force fallback path by hiding sentence_transformers import.
    with patch.dict(sys.modules, {"sentence_transformers": None}):
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
    strong = Example(
        messages=[
            Message(role=Role.USER, content="Tell me how to reset my account password safely"),
            Message(
                role=Role.ASSISTANT,
                content="Open settings, choose security, request a reset link, and confirm the email before changing the password.",
            ),
        ]
    )
    weak = Example(
        messages=[
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Short."),
        ]
    )

    audited = expander.audit_candidates([strong, weak])
    kept = expander.keep_best(audited, limit=5)

    assert len(kept) == 1
    assert kept[0].user_message.startswith("Tell me how")
    assert kept[0].metadata["pipeline_stage"] == "keep_best"


def test_seed_expander_expand_refills_after_filtering():
    config = SDKConfig().generation.model_copy(update={"num_examples": 2, "batch_size": 2})
    judge = QualityJudge(_DummyClient(), SDKConfig().quality)
    round_one = [
        Example(
            messages=[
                Message(role=Role.USER, content="Hi"),
                Message(role=Role.ASSISTANT, content="Short."),
            ]
        ),
        Example(
            messages=[
                Message(
                    role=Role.USER, content="Explain how to update my billing address in the app"
                ),
                Message(
                    role=Role.ASSISTANT,
                    content="Open billing settings, edit the address form, save the new details, and verify the confirmation message.",
                ),
            ]
        ),
    ]
    round_two = [
        Example(
            messages=[
                Message(role=Role.USER, content="How can I cancel my subscription before renewal?"),
                Message(
                    role=Role.ASSISTANT,
                    content="Go to subscription settings, choose cancel, review the renewal date, and confirm the cancellation to stop the next charge.",
                ),
            ]
        ),
    ]
    expander = _RoundRobinExpander(
        [round_one, round_two],
        _DummyClient(),
        config,
        quality_judge=judge,
        min_quality=7.0,
        max_rounds=3,
    )
    seed = Example(
        messages=[
            Message(role=Role.USER, content="Seed question"),
            Message(role=Role.ASSISTANT, content="Seed answer with enough words to be valid."),
        ]
    )

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
                Message(
                    role=Role.ASSISTANT,
                    content="Open billing, review the charge, and submit another refund request.",
                ),
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
