# Customer Support Showcase Publish Runbook

This is the shortest reproducible public proof flow for the repository.

## 1. Generate the dataset

```bash
sdk run \
  --seeds examples/customer_support_seeds.jsonl \
  --num 500 \
  --format jsonl \
  --output ./output/customer_support_demo
```

Expected outputs:

- `expanded_customer_support.jsonl`
- `expanded_customer_support_quality_report.html`
- `expanded_customer_support_quality_report.json`
- `expanded_customer_support_case_study.md`

## 2. Build the proof bundle

```bash
sdk proof \
  ./output/customer_support_demo/expanded_customer_support.jsonl \
  --base-model llama3.1:8b \
  --trainer unsloth \
  --baseline examples/customer_support_seeds.jsonl \
  --reference examples/customer_support_seeds.jsonl \
  --output ./output/proof
```

Expected outputs:

- `proof_summary.json`
- `proof_summary.md`
- `run_finetune.sh`
- `run_eval.sh`

## 3. Prepare and publish the dataset bundle

```bash
export HF_TOKEN=hf_xxx

sdk publish-hf \
  ./output/customer_support_demo/expanded_customer_support.jsonl \
  --repo-id your-username/customer-support-synth-demo \
  --baseline examples/customer_support_seeds.jsonl \
  --reference examples/customer_support_seeds.jsonl \
  --output ./output/publish
```

Expected outputs:

- local Hugging Face bundle under `./output/publish/expanded_customer_support_huggingface`
- `publish_manifest.json`
- remote dataset repo at `https://huggingface.co/datasets/your-username/customer-support-synth-demo`

## 4. Optional fine-tune and uplift

```bash
sdk finetune \
  ./output/customer_support_demo/expanded_customer_support.jsonl \
  --base-model llama3.1:8b \
  --output ./output/finetune \
  --plan-only

sdk uplift \
  --base-model llama3.1:8b \
  --finetuned-model path-or-model-id \
  --holdout examples/customer_support_holdout.jsonl \
  --output ./output/uplift
```

## Public proof checklist

- publish the dataset repo
- attach the HTML quality report
- attach `proof_summary.md`
- attach holdout uplift numbers if available
- link back to the generator repository
