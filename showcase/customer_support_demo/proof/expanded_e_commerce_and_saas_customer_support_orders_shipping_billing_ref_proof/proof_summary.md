# Proof Bundle: expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref

This bundle is a reproducible starting point for `dataset -> fine-tune -> eval uplift`.

## Included Files

- `proof_summary.json`
- `run_finetune.sh`
- `run_eval.sh`
- `support_eval_rubric.json`

## Recommended Flow

1. Fine-tune the base model on the generated dataset.
2. Evaluate the base model and fine-tuned model on the same holdout set.
3. Record uplift metrics alongside the proof summary.

## Current Dataset Signals

- Avg quality score: 8.30
- Pass rate: 100.00%
- Lexical diversity: 0.4109
- Distribution divergence: 0.5000
- Contamination hits: 0

## Holdout

- Included holdout file: `customer_support_holdout.jsonl`
- Use the same holdout for both the base model and the fine-tuned model.

## Baseline Comparison

- Baseline dataset: customer_support_seeds
- Avg quality delta: +1.1333
- Pass rate delta: +50.00%

## Reference Comparison

- Reference dataset: customer_support_seeds
- Topic overlap ratio: 100.00%
- Style distribution distance: 0.7333
- Cluster coverage distance: 0.0000
- Difficulty profile distance: 1.0000
- Exact overlap ratio: 0.00%
- Near overlap ratio: 0.00%
- Semantic overlap ratio: n/a
- Reference alignment score: 65.13/100
