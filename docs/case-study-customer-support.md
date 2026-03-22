# Customer Support Case Study

## Goal

Take a small set of real customer-support conversations and expand it into a larger instruction-tuning dataset that is:

- faster to inspect than a custom pipeline
- auditable before training
- comparable against a baseline or reference set
- ready for proof and publication

## Input

The intended starting point is a seed file like [examples/customer_support_seeds.jsonl](../examples/customer_support_seeds.jsonl) with 20-50 real examples covering:

- billing issues
- refunds
- account access
- delivery/support logistics
- escalation and frustrated-user scenarios

## Pipeline

The customer-support flow in this repository is:

1. `sdk go` or `sdk create`
2. `sdk inspect` to review accepted and rejected examples
3. `sdk benchmark` to compare local Ollama models
4. `sdk eval` to compare generated output against baseline/reference data
5. `sdk proof` to prepare a reproducible fine-tune/eval bundle
6. `sdk export --format huggingface` to prepare a publishable package

## Why This Vertical

Customer support is a strong first vertical because it has:

- clear user intents
- repeatable response styles
- a realistic need for seed amplification
- obvious quality risks like repetition, weak empathy, or over-generic answers
- a simple success criterion: useful support interactions that still look like support interactions

## What “Good” Looks Like

For this use case, a good generated dataset should:

- preserve support-specific style buckets such as concise replies, procedural instructions, and empathetic escalation handling
- keep topic coverage close to the seed/reference distribution
- avoid benchmark contamination
- make rejection reasons transparent
- produce a proof bundle that can be used in a real fine-tune/eval loop

## Current Repo Story

This repository already supports:

- customer-support-focused zero-to-dataset flow
- distribution-aware seed expansion
- local-model benchmarking
- baseline/reference dataset comparison
- proof and publish bundles

What is still missing is not the customer-support framing itself, but the final proof layer:

- a real checked-in showcase bundle from a local run
- before/after holdout evaluation on a fine-tuned support model

## Recommended Demo

For a live demo, the most convincing path is:

```bash
sdk init --provider ollama
sdk go --num 200
sdk inspect ./output/zero_to_dataset --show rejected --sort-by reason --limit 10
sdk benchmark --seeds examples/customer_support_seeds.jsonl --domain "customer support"
sdk eval ./output/zero_to_dataset/expanded_customer_support.jsonl \
  --baseline examples/customer_support_seeds.jsonl \
  --reference examples/customer_support_seeds.jsonl
sdk proof ./output/zero_to_dataset/expanded_customer_support.jsonl \
  --base-model llama3.1:8b \
  --trainer unsloth \
  --baseline examples/customer_support_seeds.jsonl \
  --reference examples/customer_support_seeds.jsonl
```

That path tells a coherent story:

- generate
- inspect
- compare
- prove

