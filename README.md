# Synth Dataset Kit

Turn `20-50` real examples into a reviewed fine-tuning dataset, proof bundle, and publish bundle.

## Highlights

- Seed set to dataset fast
- Reviewed outputs by default
- Real proof artifacts checked in
- Publish-ready Hugging Face bundle
- Local-first or API-backed providers

## Demo

Canonical customer-support showcase:

- Metrics: [showcase/customer_support_demo/SHOWCASE_METRICS.md](./showcase/customer_support_demo/SHOWCASE_METRICS.md)
- Proof: [showcase/customer_support_demo/proof/expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_proof/proof_summary.md](./showcase/customer_support_demo/proof/expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_proof/proof_summary.md)
- Publish bundle: [showcase/customer_support_demo/publish/expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_huggingface/README.md](./showcase/customer_support_demo/publish/expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_huggingface/README.md)

Checked-in run snapshot from March 21, 2026:

- `openai / gpt-5.2`
- `6` customer-support seed examples -> `10` retained examples
- `1.98 min` total runtime
- `8.1` average quality score
- `0` contamination hits

## Overview

Provides a CLI-first synthetic dataset tool for one narrow wedge: expanding your own small customer-support seed set into a reviewed dataset. Instead of making you wire a pipeline framework, it focuses on a direct flow: generate from seeds, audit quality, keep proof artifacts, and produce a publishable bundle. The repository already includes a real canonical showcase run, proof bundle, and Hugging Face-style export.

## What You Can Use It For

If you arrive at this repository with a real problem, the practical use cases are simple:

- You have a small set of your own examples and need more data in the same style
- You want a dataset, not just raw generated text
- You want quality checks before using the result for fine-tuning
- You want proof artifacts and publish-ready outputs next to the run

In plain terms: this tool helps you turn a small set of your own data into a larger reviewed dataset you can actually use.

## First Time Here?

- Run `sdk init`
- Try `sdk create --demo`
- Bring your own seed file and run `sdk create`
- Inspect the result with `sdk inspect`

## Motivation

Most competing tools feel like frameworks, pipeline kits, or research artifacts. That makes them flexible, but slow to start and hard to trust for a single practical job. This project is built for the opposite experience: bring a small set of your own examples, generate a larger reviewed dataset from them, and keep the quality report and proof artifacts next to the run. The goal is a usable tool, not a constructor.

## Features

- Accepts seed examples in JSONL and expands them into a larger dataset
- Supports `ollama`, `openai`, `anthropic`, `vllm`, and OpenAI-compatible APIs
- Scores outputs with an LLM judge and rule-based checks
- Checks contamination against configured benchmark signatures
- Writes `run_summary.json`, quality reports, and stage timings for every run
- Builds proof bundles with `proof_summary`, fine-tune script, and eval script templates
- Exports Hugging Face-style publish bundles with dataset card and metadata

## Architecture

Main components:

- `synth_dataset_kit/cli.py` - user-facing commands and runtime/reporting flow
- `synth_dataset_kit/engine.py` - orchestration for generate, audit, filter, and export
- `synth_dataset_kit/generators/seed_expander.py` - seed-to-dataset expansion logic
- `synth_dataset_kit/quality/` - scoring and report generation
- `synth_dataset_kit/decontamination/` - benchmark overlap checks
- `synth_dataset_kit/exporters/` - dataset, proof, and publish bundle writers

Flow:

`seed examples -> generation -> audit -> filter -> run summary -> proof/publish artifacts`

## Quick Start

1. Install the package:

```bash
pip install synth-dataset-kit[decontamination]
```

2. Initialize the provider config:

```bash
sdk init
```

3. Run the canonical demo:

```bash
sdk create --demo --num 10 --output ./showcase/customer_support_demo --showcase-summary
```

4. Inspect the result:

```bash
sdk inspect ./showcase/customer_support_demo
```

The LLM provider is selected in `sdk_config.yaml`. For local-first usage, use `ollama`. Otherwise point the tool at `openai`, `anthropic`, `vllm`, or another OpenAI-compatible API.

## Usage

Main guided flow:

```bash
sdk create
```

Fast demo path:

```bash
sdk create --demo --num 10 --output ./showcase/customer_support_demo --showcase-summary
```

Non-interactive scripting path:

```bash
sdk run --seeds examples/customer_support_seeds.jsonl --num 200 --output ./output/customer_support_demo
```

Seed file formats supported in JSONL:

- OpenAI / ChatML: `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`
- Simple: `{"user":"...", "assistant":"..."}`
- Alpaca: `{"instruction":"...", "input":"...", "output":"..."}`
- ShareGPT: `{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}`

Advanced commands:

- `sdk benchmark`
- `sdk eval <file>`
- `sdk proof <file>`
- `sdk export <file>`
- `sdk publish-hf`

## Project Structure

```text
synth_dataset_kit/
  cli.py
  engine.py
  config.py
  llm_client.py
  generators/
  quality/
  decontamination/
  exporters/
showcase/
  customer_support_demo/
docs/
examples/
tests/
```

## Status

- Stage: Beta
- Current wedge: customer-support seed expansion
- Checked-in proof: canonical showcase run, proof bundle, and publish bundle
- Still missing: direct built-in fine-tune uplift results for the canonical showcase

## Testing

```bash
pytest -q tests/test_core.py
```

## Useful Links

- Customer support showcase: [showcase/customer_support_demo/README.md](./showcase/customer_support_demo/README.md)
- Customer support case study: [docs/case-study-customer-support.md](./docs/case-study-customer-support.md)
- Support evaluation story: [docs/support-eval-story.md](./docs/support-eval-story.md)
- Publish runbook: [showcase/customer_support_demo/PUBLISHING.md](./showcase/customer_support_demo/PUBLISHING.md)
- Product positioning: [docs/product-positioning.md](./docs/product-positioning.md)

---

MIT - see LICENSE

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[LinkedIn](https://www.linkedin.com/in/kazkozdev/)
[Email](mailto:kazkozdev@gmail.com)
