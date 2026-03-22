# Synth Dataset Kit

CLI tool for generating high-quality synthetic datasets for LLM fine-tuning.

![CI](https://github.com/KazKozDev/synth-dataset-kit/actions/workflows/ci.yml/badge.svg)

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

Supervised fine-tuning requires labeled data, but collecting thousands of high-quality examples manually is expensive and slow. Existing synthetic data tools either require cloud APIs or produce low-quality outputs with no quality controls. `synth-dataset-kit` works with local models (Ollama, vLLM), integrates quality scoring and decontamination into the same pipeline, and gives you a clean dataset with a single command.

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
- [Typer](https://typer.tiangolo.com/) — CLI framework
- [Pydantic v2](https://docs.pydantic.dev/) — config and data models
- [Rich](https://github.com/Textualize/rich) — terminal output and progress
- [OpenAI Python SDK](https://github.com/openai/openai-python) — unified LLM client
- [sentence-transformers](https://www.sbert.net/) *(optional)* — embedding-based decontamination
- [Jinja2](https://jinja.palletsprojects.com/) — prompt templates

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

See `examples/customer_support_seeds.jsonl` for a working example.

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

Fork → branch → PR. Run `ruff check synth_dataset_kit/` before submitting.

---

MIT - see LICENSE

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[LinkedIn](https://www.linkedin.com/in/kazkozdev/)
[Email](mailto:kazkozdev@gmail.com)
