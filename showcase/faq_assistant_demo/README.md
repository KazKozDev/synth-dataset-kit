# FAQ Assistant Showcase

This showcase skeleton exists so the product is not framed as customer-support only.

## Target Story

Turn a small FAQ/support seed set into an auditable assistant-tuning dataset.

## Suggested Inputs

- seeds: `examples/faq_assistant_seeds.jsonl`
- holdout: `examples/faq_assistant_holdout.jsonl`

## Expected Artifacts

- generated dataset
- quality report
- eval summary
- proof bundle
- Hugging Face publish bundle

## Run Template

```bash
sdk run --seeds examples/faq_assistant_seeds.jsonl --num 100 --output ./showcase/faq_assistant_demo
sdk eval <dataset.jsonl> --baseline examples/faq_assistant_seeds.jsonl --reference examples/faq_assistant_holdout.jsonl --output ./showcase/faq_assistant_demo/eval
sdk proof <dataset.jsonl> --baseline examples/faq_assistant_seeds.jsonl --reference examples/faq_assistant_holdout.jsonl --output ./showcase/faq_assistant_demo/proof
sdk publish-hf <dataset.jsonl> --repo-id your-username/faq-assistant-demo --plan-only
```
