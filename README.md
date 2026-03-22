# Synth Dataset Kit

CLI tool for generating high-quality synthetic datasets for LLM fine-tuning.

![CI](https://github.com/KazKozDev/synth-dataset-kit/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-orange)

## Highlights

- Amplifies 20–50 seed examples into thousands of training examples
- Generates datasets from a domain description using topic trees
- LLM-as-judge quality scoring with toxicity and PII checks
- Benchmark decontamination against MMLU, GSM8K, HumanEval, ARC, HellaSwag
- Exports to JSONL, Alpaca, ShareGPT, ChatML, and HuggingFace formats

<!-- TODO: Add demo GIF or screenshot -->

## Overview

`synth-dataset-kit` turns a small set of hand-crafted examples — or just a domain name — into a large, quality-filtered fine-tuning dataset. It connects to any OpenAI-compatible LLM endpoint (Ollama, OpenAI, Anthropic, vLLM) to generate examples, scores them with an LLM judge, filters out benchmark contamination, and exports to the format your trainer expects.

Aimed at ML engineers who need training data fast, without the cost of manual annotation at scale.

## Motivation

Manual annotation costs $1–5 per example on platforms like Scale AI or Labelbox — a 10k dataset runs $10k–50k before quality review. Self-Instruct and Alpaca-style pipelines generate data cheaply but ship it without any quality gate: no scoring, no dedup, no benchmark contamination check. The result is noisy datasets that hurt rather than help fine-tuning.

More recent tools like Magpie or WizardLM improve quality, but are tied to specific model families or require significant infrastructure. `synth-dataset-kit` takes a different approach: bring your own 20–50 seed examples, plug in any OpenAI-compatible endpoint (including local Ollama or vLLM), and get a scored, decontaminated, export-ready dataset in one command. The decontamination step alone typically removes 3–8% of generated examples that overlap with MMLU, GSM8K, or HumanEval — overlap that would otherwise inflate eval scores without reflecting real capability gains.

## Features

- `sdk generate --seeds` — expand seed JSONL into a larger dataset
- `sdk generate --domain` — generate from scratch given a domain description
- `sdk audit` — score dataset quality, flag toxicity, PII, and duplicates
- `sdk export` — convert to Alpaca, ShareGPT, ChatML, or HuggingFace format
- `sdk run` — full pipeline in one command: generate → audit → filter → export
- `sdk health` — verify LLM endpoint connectivity before starting a run
- YAML-based config with per-provider defaults (Ollama, OpenAI, Anthropic, vLLM)
- HTML quality report with diversity metrics, topic coverage, and score distribution

## Architecture

```
CLI (sdk)
  └── DatasetEngine
        ├── Generators
        │     ├── SeedExpander     — analyzes seeds, generates variations
        │     └── TopicTreeGenerator — builds topic tree from domain, generates per-node
        ├── QualityJudge           — LLM-as-judge scoring, toxicity/PII/dedup checks
        ├── Decontaminator         — embedding/n-gram similarity vs benchmark corpora
        └── Exporters              — JSONL, Alpaca, ShareGPT, ChatML, HuggingFace
```

Flow: seeds or domain → generation → quality scoring → decontamination → filtered export

## Tech Stack

- Python 3.10+
- Typer — CLI framework
- Pydantic v2 — config and data models
- Rich — terminal output and progress
- OpenAI Python SDK — unified LLM client
- sentence-transformers *(optional)* — embedding-based decontamination
- Jinja2 — prompt templates

## Configuration

`sdk init --provider <name>` creates `sdk_config.yaml` with provider defaults. All values have defaults and CLI flags override them at runtime.

<details>
<summary>Full sdk_config.yaml reference</summary>

```yaml
llm:
  provider: ollama          # ollama | openai | anthropic | vllm | custom
  api_base: http://localhost:11434/v1
  api_key: ollama
  model: llama3.1:8b
  temperature: 0.8
  max_tokens: 2048
  concurrent_requests: 4    # parallel generation requests

generation:
  num_examples: 100
  language: en
  system_prompt: ""         # injected as system message in every example
  personas: [beginner, expert, skeptic]
  difficulty_levels: [easy, medium, hard]
  batch_size: 5

quality:
  enabled: true
  min_score: 7.5            # 0–10 scale; examples below this are filtered out
  check_toxicity: true
  check_pii: true
  check_duplicates: true
  duplicate_threshold: 0.85

decontamination:
  enabled: true
  benchmarks: [mmlu, gsm8k, humaneval, arc, hellaswag]
  similarity_threshold: 0.85
  method: hybrid            # exact | ngram | embedding | hybrid
  embedding_model: all-MiniLM-L6-v2

export:
  format: jsonl             # jsonl | alpaca | chatml | sharegpt | huggingface
  output_dir: ./output
  include_metadata: false
  include_quality_report: true
```

</details>

## Quick Start

```bash
# Install
pip install synth-dataset-kit

# Or with decontamination support
pip install "synth-dataset-kit[decontamination]"
```

**With Ollama (local, no API key needed):**

```bash
# 1. Start Ollama and pull a model
ollama serve
ollama pull llama3.1:8b

# 2. Create config
sdk init --provider ollama

# 3. Generate from seeds
sdk generate --seeds examples/customer_support_seeds.jsonl --num 500

# 4. Or run the full pipeline in one shot
sdk run --seeds examples/customer_support_seeds.jsonl --num 500 --format alpaca
```

**With OpenAI:**

```bash
sdk init --provider openai
export OPENAI_API_KEY=sk-...
sdk run --domain "customer support chatbot" --num 1000 --format sharegpt
```

## Usage

```bash
# Check connectivity
sdk health

# Generate 200 examples from seeds, output as ShareGPT
sdk generate --seeds my_seeds.jsonl --num 200 --format sharegpt --output ./data

# Generate from domain description
sdk generate --domain "medical Q&A for primary care" --num 500

# Audit an existing dataset
sdk audit ./data/dataset.jsonl

# Export with quality filter (keep only score >= 8)
sdk export ./data/dataset.jsonl --format alpaca --min-quality 8.0

# Full pipeline: generate → audit → filter → export
sdk run --seeds my_seeds.jsonl --num 1000 --format jsonl --min-quality 7.0
```

Seed files are JSONL with OpenAI message format:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

See `examples/customer_support_seeds.jsonl` for a working example. The expanded dataset generated from these seeds is published on HuggingFace: [KazKozDev/synth-customer-support-expanded-R](https://huggingface.co/datasets/KazKozDev/synth-customer-support-expanded-R).

## Project Structure

```
synth_dataset_kit/
  cli.py              # CLI commands (init, generate, audit, export, run, health)
  engine.py           # DatasetEngine — main orchestrator
  config.py           # SDKConfig + per-provider defaults
  models.py           # Dataset, Example, QualityReport data models
  llm_client.py       # OpenAI-compatible LLM client
  prompts.py          # Jinja2 prompt templates
  generators/         # SeedExpander and TopicTreeGenerator
  quality/            # LLM-as-judge, toxicity, PII, dedup
  decontamination/    # Benchmark overlap detection
  exporters/          # Output format converters
examples/
  customer_support_seeds.jsonl
tests/
  test_core.py
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Contributing

1. Fork the repo and create a branch: `git checkout -b feat/your-feature`
2. Make your changes. Keep PRs focused — one feature or fix per PR.
3. Run the linter and tests before pushing:

```bash
ruff check synth_dataset_kit/
pytest tests/ -v
```

4. Open a PR with a short description of what changed and why.

Code style: `ruff` with `line-length = 100`, Python 3.10+ syntax. Type hints required for public functions. No new dependencies without discussion.

---

MIT - see LICENSE

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[LinkedIn](https://www.linkedin.com/in/kazkozdev/)
[Email](mailto:kazkozdev@gmail.com)
