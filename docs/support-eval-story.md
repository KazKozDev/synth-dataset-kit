# Support Evaluation Story

## What Should Be Measured

For customer-support seed expansion, evaluation should answer four questions:

1. Does the generated dataset stay in the same support domain?
2. Does it preserve useful support behavior and style?
3. Does it avoid obvious contamination and duplication?
4. Does it improve a fine-tuned model on a support holdout set?

## Dataset-Level Signals

The repository already computes strong dataset-level signals:

- average quality score
- pass rate
- lexical and corpus diversity
- difficulty profile
- topic coverage
- contamination hits and evidence
- baseline comparison
- reference dataset comparison

These metrics help decide whether the generated data is good enough to train on.

## Reference Comparison For Support

For this vertical, the most useful reference comparisons are:

- response length alignment
- style distribution alignment
- cluster coverage distance
- difficulty profile distance
- topic overlap and novelty
- exact / near / semantic overlap against the reference set
- reference alignment score

These are now available through `sdk eval`, `sdk proof`, and publish bundles.

## Model-Level Signals

The next missing layer is model uplift.

The strongest support-specific evaluation story would compare a base model and a fine-tuned model on a fixed holdout set with metrics like:

- task success rate
- answer completeness
- escalation correctness
- refusal correctness
- hallucination rate
- empathy / tone score

## Recommended Holdout Structure

A useful customer-support holdout should contain:

- billing cases
- refund exceptions
- account lockout flows
- policy questions
- escalation scenarios
- multilingual or mixed-language edge cases if relevant

The holdout should stay separate from:

- seed examples
- generated dataset
- any reference dataset used only for profile comparison

## Minimal Proof Flow

Today the repository can already prepare the skeleton:

```bash
sdk proof ./output/zero_to_dataset/expanded_customer_support.jsonl \
  --base-model llama3.1:8b \
  --trainer unsloth \
  --baseline examples/customer_support_seeds.jsonl \
  --reference examples/customer_support_seeds.jsonl
```

This creates:

- `proof_summary.json`
- `proof_summary.md`
- `run_finetune.sh`
- `run_eval.sh`

## What “Done” Looks Like

The support evaluation story is fully done when the repo can show:

1. a small real support seed set
2. a generated audited dataset
3. a reference comparison report
4. a reproducible fine-tune script
5. a holdout-eval result showing measurable uplift

Until then, the repo already tells a strong dataset-quality story, but not yet the final model-improvement story.
