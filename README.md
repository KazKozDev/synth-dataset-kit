# Synth Dataset Kit

Generate high-quality synthetic datasets for LLM fine-tuning from seed examples or domain descriptions.

![CI](https://github.com/KazKozDev/synth-dataset-kit/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-orange)

## What it does

Takes 20-50 hand-crafted seed examples (or just a domain description) and expands them into hundreds or thousands of quality-filtered training examples. Connects to any OpenAI-compatible LLM (Ollama, OpenAI, Anthropic, vLLM) to generate, score with an LLM judge, check for benchmark contamination, and export in the format your trainer expects.

**Why?** Manual annotation costs $0.02-$5+ per example at scale. Naive synthetic pipelines (Self-Instruct, Alpaca-style) produce noise — duplicates, hallucinations, low-quality samples. This toolkit makes dataset quality measurable and controllable before you start training.

## Highlights

- **Seed expansion** — amplify 20-50 examples into thousands with controlled variation across topics, personas, and difficulty levels
- **Domain generation** — create datasets from scratch using hierarchical topic trees
- **Quality scoring** — LLM-as-judge (1-10 scale) + rule-based checks for toxicity, PII, duplicates, placeholders
- **Decontamination** — detect overlap with MMLU, GSM8K, HumanEval, ARC, HellaSwag via exact match, n-gram, and embedding similarity
- **Multiple formats** — export to JSONL, Alpaca, ShareGPT, ChatML, HuggingFace
- **HTML reports** — diversity metrics, topic coverage heatmaps, score distributions, contamination evidence
- **Provider-agnostic** — works with Ollama (local, free), OpenAI, Anthropic, vLLM, any OpenAI-compatible API

## Quick Start

```bash
pip install synth-dataset-kit

# With embedding-based decontamination
pip install "synth-dataset-kit[decontamination]"
```

### With Ollama (local, no API key)

```bash
ollama serve
ollama pull llama3.1:8b

sdk init --provider ollama
sdk run --seeds examples/customer_support_seeds.jsonl --num 500 --format alpaca
```

### With OpenAI

```bash
sdk init --provider openai
export OPENAI_API_KEY=sk-...
sdk run --domain "customer support chatbot" --num 1000 --format sharegpt
```

## Usage

```bash
# Check LLM connectivity
sdk health

# Generate from seed examples
sdk generate --seeds my_seeds.jsonl --num 200

# Generate from domain description
sdk generate --domain "medical Q&A for primary care" --num 500

# Audit an existing dataset
sdk audit ./data/dataset.jsonl

# Export with quality filter
sdk export ./data/dataset.jsonl --format alpaca --min-quality 8.0

# Full pipeline: generate -> audit -> filter -> export
sdk run --seeds my_seeds.jsonl --num 1000 --format jsonl --min-quality 7.0
```

### Seed file format

JSONL with any of these formats (auto-detected):

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"instruction": "...", "output": "..."}
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

See `examples/` for working seed files. The dataset generated from `customer_support_seeds.jsonl` is published on HuggingFace: [KazKozDev/synth-customer-support-expanded-R](https://huggingface.co/datasets/KazKozDev/synth-customer-support-expanded-R).

## Architecture

```
CLI (sdk)
  └── DatasetEngine
        ├── Generators
        │     ├── SeedExpander        — analyzes seeds, generates variations
        │     └── TopicTreeGenerator  — builds topic tree from domain, generates per-node
        ├── QualityJudge              — LLM-as-judge scoring + toxicity/PII/dedup checks
        ├── Decontaminator            — n-gram/embedding similarity vs benchmark corpora
        └── Exporters                 — JSONL, Alpaca, ShareGPT, ChatML, HuggingFace
```

Flow: seeds or domain &rarr; generation &rarr; quality scoring &rarr; decontamination &rarr; filtered export

## Configuration

`sdk init --provider <name>` creates `sdk_config.yaml` with provider defaults. All values have sensible defaults; CLI flags override them at runtime.

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
  min_score: 7.5            # 0-10 scale; examples below this are filtered out
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

## Project Structure

```
synth_dataset_kit/
  cli/                 # CLI commands (Typer + Rich)
  engine.py            # DatasetEngine — main orchestrator
  config.py            # SDKConfig + per-provider defaults
  models.py            # Dataset, Example, QualityReport (Pydantic)
  llm_client.py        # OpenAI-compatible LLM client with retry/concurrency
  prompts.py           # Jinja2 prompt templates
  generators/          # SeedExpander, TopicTreeGenerator
  quality/             # LLM-as-judge, toxicity, PII, dedup checks
  decontamination/     # Benchmark overlap detection (exact, n-gram, embedding)
  exporters/           # JSONL, Alpaca, ShareGPT, ChatML, HuggingFace, HTML reports
examples/              # Seed files for demos (customer support, coding tutor, FAQ, IT helpdesk)
tests/                 # Test suite (13 test modules)
```

## Tech Stack

- Python 3.10+
- Typer + Rich — CLI and terminal UI
- Pydantic v2 — config and data models
- OpenAI Python SDK — unified LLM client for all providers
- Jinja2 — prompt templates
- NumPy — numerical operations
- sentence-transformers *(optional)* — embedding-based decontamination

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Contributing

1. Fork and branch: `git checkout -b feat/your-feature`
2. Make changes — one feature or fix per PR
3. Lint and test:

```bash
ruff check synth_dataset_kit/
pytest tests/ -v
```

4. Open a PR with a short description of what changed and why

Code style: `ruff` with `line-length = 100`, Python 3.10+ syntax, type hints on public functions.

---

MIT - see LICENSE

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[LinkedIn](https://www.linkedin.com/in/kazkozdev/)
[Email](mailto:kazkozdev@gmail.com)
