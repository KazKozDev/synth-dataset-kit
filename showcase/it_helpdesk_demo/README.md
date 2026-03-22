# IT Helpdesk Showcase

This showcase skeleton exists to prove the product can generalize to internal support workflows.

## Target Story

Expand a small IT/helpdesk seed set into a reproducible fine-tuning dataset with audit artifacts.

## Suggested Inputs

- seeds: `examples/it_helpdesk_seeds.jsonl`
- holdout: `examples/it_helpdesk_holdout.jsonl`

## Expected Artifacts

- generated dataset
- quality report
- eval summary
- proof bundle
- Hugging Face publish bundle

## Run Template

```bash
sdk run --seeds examples/it_helpdesk_seeds.jsonl --num 100 --output ./showcase/it_helpdesk_demo
sdk eval <dataset.jsonl> --baseline examples/it_helpdesk_seeds.jsonl --reference examples/it_helpdesk_holdout.jsonl --output ./showcase/it_helpdesk_demo/eval
sdk proof <dataset.jsonl> --baseline examples/it_helpdesk_seeds.jsonl --reference examples/it_helpdesk_holdout.jsonl --output ./showcase/it_helpdesk_demo/proof
sdk publish-hf <dataset.jsonl> --repo-id your-username/it-helpdesk-demo --plan-only
```
