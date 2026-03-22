# Coding Tutor Showcase

This showcase skeleton exists to prove the tool is not limited to support chat.

## Target Story

Start with a small coding-help seed set and generate an auditable tutoring dataset.

## Suggested Inputs

- seeds: `examples/coding_tutor_seeds.jsonl`
- holdout: `examples/coding_tutor_holdout.jsonl`

## Expected Artifacts

- generated dataset
- quality report
- eval summary
- proof bundle
- Hugging Face publish bundle

## Run Template

```bash
sdk run --seeds examples/coding_tutor_seeds.jsonl --num 100 --output ./showcase/coding_tutor_demo
sdk eval <dataset.jsonl> --baseline examples/coding_tutor_seeds.jsonl --reference examples/coding_tutor_holdout.jsonl --output ./showcase/coding_tutor_demo/eval
sdk proof <dataset.jsonl> --baseline examples/coding_tutor_seeds.jsonl --reference examples/coding_tutor_holdout.jsonl --output ./showcase/coding_tutor_demo/proof
sdk publish-hf <dataset.jsonl> --repo-id your-username/coding-tutor-demo --plan-only
```
