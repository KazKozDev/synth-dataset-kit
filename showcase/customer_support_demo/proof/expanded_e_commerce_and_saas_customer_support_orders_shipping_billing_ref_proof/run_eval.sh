#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="llama3.1:8b"
FINETUNED_MODEL="./artifacts/finetune/final-model"
EVAL_OUTPUT="./artifacts/eval_results.json"
HOLDOUT_PATH="./customer_support_holdout.jsonl"
RUBRIC_PATH="./support_eval_rubric.json"

# Replace this command with your actual holdout evaluation command.
echo "Evaluate $BASE_MODEL and $FINETUNED_MODEL on $HOLDOUT_PATH using $RUBRIC_PATH and write metrics to $EVAL_OUTPUT"
