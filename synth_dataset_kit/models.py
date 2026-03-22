"""Data models for the SDK."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Example(BaseModel):
    """A single training example (conversation turn pair)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[Message]
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_score: float | None = None
    decontamination_flags: list[str] = Field(default_factory=list)
    decontamination_evidence: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def user_message(self) -> str:
        for m in self.messages:
            if m.role == Role.USER:
                return m.content
        return ""

    @property
    def assistant_message(self) -> str:
        for m in self.messages:
            if m.role == Role.ASSISTANT:
                return m.content
        return ""


class Dataset(BaseModel):
    """A collection of training examples with metadata."""

    name: str = "synthetic_dataset"
    version: str = "1.0"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    generator: str = "synth-dataset-kit"
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    examples: list[Example] = Field(default_factory=list)
    artifacts: dict[str, list[Example]] = Field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.examples)

    def add(self, example: Example) -> None:
        self.examples.append(example)

    def filter_by_quality(self, min_score: float) -> Dataset:
        """Return a new dataset with only examples above the quality threshold."""
        filtered = [e for e in self.examples if (e.quality_score or 0) >= min_score]
        return Dataset(
            name=self.name,
            version=self.version,
            created_at=self.created_at,
            generator=self.generator,
            config_snapshot=self.config_snapshot,
            examples=filtered,
            artifacts=self.artifacts,
        )

    def remove_contaminated(self) -> Dataset:
        """Return a new dataset without contaminated examples."""
        clean: list[Example] = []
        for example in self.examples:
            verdict = str(example.metadata.get("contamination_verdict", "")).lower()
            if verdict == "block":
                continue
            if not verdict and example.decontamination_flags:
                continue
            clean.append(example)
        return Dataset(
            name=self.name,
            version=self.version,
            created_at=self.created_at,
            generator=self.generator,
            config_snapshot=self.config_snapshot,
            examples=clean,
            artifacts=self.artifacts,
        )


class QualityReport(BaseModel):
    """Quality analysis report for a dataset."""

    dataset_name: str
    total_examples: int
    passed_examples: int
    failed_examples: int
    avg_quality_score: float
    score_distribution: dict[str, int] = Field(default_factory=dict)
    avg_user_length: float = 0.0
    avg_assistant_length: float = 0.0
    diversity_score: float = 0.0
    self_bleu_proxy: float = 0.0
    lexical_diversity: float = 0.0
    embedding_diversity_score: float | None = None
    diversity_method: str = "ngram_jaccard"
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    topic_coverage: dict[str, int] = Field(default_factory=dict)
    topic_heatmap: dict[str, dict[str, int]] = Field(default_factory=dict)
    seed_cluster_distribution: dict[str, int] = Field(default_factory=dict)
    generated_cluster_distribution: dict[str, int] = Field(default_factory=dict)
    distribution_divergence: float = 0.0
    distribution_match_score: float = 0.0
    semantic_cluster_target_distribution: dict[str, int] = Field(default_factory=dict)
    semantic_cluster_generated_distribution: dict[str, int] = Field(default_factory=dict)
    semantic_coverage_score: float = 0.0
    semantic_coverage_gaps: dict[str, int] = Field(default_factory=dict)
    graph_coverage_score: float = 0.0
    graph_frontier_clusters: list[str] = Field(default_factory=list)
    underrepresented_clusters: dict[str, int] = Field(default_factory=dict)
    rebalancing_history: list[dict[str, Any]] = Field(default_factory=list)
    final_distribution_status: dict[str, Any] = Field(default_factory=dict)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    duplicate_groups: int = 0
    near_duplicate_examples: int = 0
    contamination_hits: int = 0
    contaminated_benchmarks: list[str] = Field(default_factory=list)
    contamination_verdicts: dict[str, int] = Field(default_factory=dict)
    contamination_methods: dict[str, int] = Field(default_factory=dict)
    contamination_method_benchmarks: dict[str, dict[str, int]] = Field(default_factory=dict)
    contamination_evidence_samples: list[dict[str, Any]] = Field(default_factory=list)
    benchmark_sources: dict[str, str] = Field(default_factory=dict)
    benchmark_sample_counts: dict[str, int] = Field(default_factory=dict)
    benchmark_load_errors: dict[str, str] = Field(default_factory=dict)
    audit_method: str = "llm_judge+rules"
    generation_time_seconds: float = 0.0
