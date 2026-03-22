# Customer Support Showcase Bundle

This directory is the checked-in canonical showcase run for the customer-support vertical.

It is intentionally designed to mirror the output of the product story:

`create -> inspect -> proof -> publish`

## Current Canonical Bundle

- `expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref.jsonl`
- `run_summary.json`
- `expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_accepted.jsonl`
- `expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_candidates.jsonl`
- `expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_quality_report.html`
- `expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_quality_report.json`
- `expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_case_study.md`
- `SHOWCASE_METRICS.md`
- `proof/expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_proof/`
- `publish/expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref_huggingface/`

## Why This Directory Exists

The repository now has:

- a customer-support-first README
- a customer-support case study
- a support-specific evaluation story
- proof and publish commands

This directory now contains a real canonical run plus matching proof and publish bundles generated from the same dataset.

## Reproduction Flow

```bash
sdk init
sdk create --demo --num 10 --output ./showcase/customer_support_demo --showcase-summary
sdk benchmark --seeds examples/customer_support_seeds.jsonl --domain "customer support"
sdk eval ./showcase/customer_support_demo/expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref.jsonl \
  --baseline examples/customer_support_seeds.jsonl \
  --reference examples/customer_support_seeds.jsonl \
  --output ./showcase/customer_support_demo
sdk proof ./showcase/customer_support_demo/expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref.jsonl \
  --base-model llama3.1:8b \
  --trainer unsloth \
  --baseline examples/customer_support_seeds.jsonl \
  --reference examples/customer_support_seeds.jsonl \
  --output ./showcase/customer_support_demo/proof
sdk export ./showcase/customer_support_demo/expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref.jsonl \
  --format huggingface \
  --baseline examples/customer_support_seeds.jsonl \
  --reference examples/customer_support_seeds.jsonl \
  --output ./showcase/customer_support_demo/publish
```

This reproduces the full checked-in `create -> proof -> publish` showcase.
