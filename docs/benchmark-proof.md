# Benchmark Proof

This document turns the product claim into a reproducible benchmark.

## Claim

`install -> generate -> report -> publish`

The claim is not "synthetic data is possible".  
The claim is: a solo ML engineer can reach a usable, auditable dataset quickly without wiring a framework.

## Official Public Metric

A run counts as a **usable dataset** only when all of the following are true:

- the final export is non-empty
- all exported examples pass the configured quality gate
- average quality score is at least `7.5`
- contamination hits are `0`

This is the same gate exported in `eval_summary.json` and Hugging Face dataset bundles as `usable_dataset_gate`.

## Benchmark Variants

Run both variants separately.

### Variant A: Hosted Baseline

- provider: OpenAI-compatible
- model: stable small/medium production model
- goal: fastest proof of time-to-value

### Variant B: Local-First

- provider: Ollama
- model: installed local model selected by `sdk benchmark`
- goal: prove privacy-friendly local workflow

## Cold-Start Protocol

Measure from a clean environment:

1. create a fresh venv
2. install package and extras
3. initialize config
4. run generation from a checked-in seed set
5. verify quality gate
6. prepare publish bundle

## Commands

```bash
python -m venv .venv-proof
source .venv-proof/bin/activate
pip install -e .[decontamination,publish]

sdk init --provider ollama
sdk go --num 100 --output ./output/benchmark_proof

sdk publish-hf \
  ./output/benchmark_proof/expanded_customer_support.jsonl \
  --repo-id your-username/customer-support-proof \
  --output ./output/benchmark_publish \
  --plan-only
```

## What To Record

- install start time
- first successful `sdk init`
- first successful `sdk go`
- total wall-clock time to final dataset
- total wall-clock time to publish bundle
- model used
- provider used
- usable dataset gate result

## Proof Artifact Checklist

- terminal transcript or GIF
- final dataset path
- HTML quality report
- `eval_summary.json` or equivalent summary
- publish bundle path
- Hugging Face dataset URL when published

## Pass Criteria

The benchmark is considered successful when:

- the workflow completes without manual code edits
- the run passes the usable dataset gate
- the full proof artifact bundle exists
- the timings are recorded and attached to the showcase
