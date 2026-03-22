"""
Synth Dataset Kit — Generate high-quality synthetic datasets for LLM fine-tuning.

Features:
  - Seed-to-dataset: amplify 20-50 real examples into thousands
  - Auto-decontamination: detect benchmark overlap (MMLU, GSM8K, HumanEval, etc.)
  - Quality reports: diversity metrics, topic coverage, difficulty distribution
  - Provider-agnostic: Ollama, OpenAI, Anthropic, vLLM, any OpenAI-compatible API
"""

__version__ = "0.1.0"

from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.engine import DatasetEngine

__all__ = ["SDKConfig", "DatasetEngine", "__version__"]
