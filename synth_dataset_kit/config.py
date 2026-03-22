"""Configuration for the SDK."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
    CUSTOM = "custom"  # Any OpenAI-compatible endpoint


# Sensible defaults for each provider
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "ollama": {
        "api_base": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "llama3.1:8b",
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "anthropic": {
        "api_base": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-20250514",
    },
    "vllm": {
        "api_base": "http://localhost:8000/v1",
        "api_key": "vllm",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "custom": {
        "api_base": "http://localhost:8000/v1",
        "api_key": "no-key",
        "model": "default",
    },
}


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.OLLAMA
    api_base: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.8
    max_tokens: int = 2048
    top_p: float = 0.95
    timeout: int = 120
    max_retries: int = 3
    concurrent_requests: int = 4


class GenerationConfig(BaseModel):
    num_examples: int = 100
    strategy: str = "auto"  # auto | seed_expand | topic_tree | domain_description
    domain: str = ""  # e.g., "customer_support", "coding", "medical"
    language: str = "en"
    system_prompt: str = ""
    seed_file: str = ""  # Path to seed examples (JSONL)
    personas: list[str] = Field(
        default_factory=lambda: ["beginner", "expert", "skeptic"]
    )
    difficulty_levels: list[str] = Field(
        default_factory=lambda: ["easy", "medium", "hard"]
    )
    batch_size: int = 5
    divergence_threshold: float = 0.15
    focus_top_k_clusters: int = 2
    rebalancing_strategy: str = "strict"  # strict | soft
    semantic_focus_top_k: int = 3
    max_active_clusters_per_round: int = 12
    saturation_threshold: float = 1.05
    long_tail_boost: float = 0.35
    graph_neighbor_k: int = 3
    distance_allocator_weight: float = 0.6


class QualityConfig(BaseModel):
    enabled: bool = True
    min_score: float = 7.5
    judge_model: str = ""  # Empty = use same model as generation
    check_toxicity: bool = True
    check_pii: bool = True
    check_duplicates: bool = True
    duplicate_threshold: float = 0.85


class DecontaminationConfig(BaseModel):
    enabled: bool = True
    benchmarks: list[str] = Field(
        default_factory=lambda: ["mmlu", "gsm8k", "humaneval", "arc", "hellaswag"]
    )
    similarity_threshold: float = 0.85
    method: str = "hybrid"  # exact | ngram | embedding | hybrid
    use_benchmark_datasets: bool = True
    load_full_benchmark_corpus: bool = False
    benchmark_sample_limit: int = 200
    cache_dir: str = ".sdk_cache/benchmarks"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_index_backend: str = "json"  # json | faiss
    embedding_top_k: int = 3
    review_threshold: float = 0.92
    hard_fail_methods: list[str] = Field(default_factory=lambda: ["exact", "ngram"])
    review_methods: list[str] = Field(default_factory=lambda: ["substring", "embedding"])


class ExportConfig(BaseModel):
    format: str = "jsonl"  # jsonl | alpaca | chatml | openai | sharegpt | huggingface
    output_dir: str = "./output"
    include_metadata: bool = False
    include_quality_report: bool = True
    include_seed_examples: bool = True


class SDKConfig(BaseModel):
    """Main configuration object."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    decontamination: DecontaminationConfig = Field(default_factory=DecontaminationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SDKConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        data = self.model_dump(mode="json")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def default_for_provider(cls, provider: str) -> SDKConfig:
        """Create a config with sensible defaults for a given provider."""
        defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["ollama"])
        llm_config = LLMConfig(
            provider=LLMProvider(provider),
            api_base=defaults["api_base"],
            api_key=defaults.get("api_key", ""),
            model=defaults["model"],
        )
        return cls(llm=llm_config)
