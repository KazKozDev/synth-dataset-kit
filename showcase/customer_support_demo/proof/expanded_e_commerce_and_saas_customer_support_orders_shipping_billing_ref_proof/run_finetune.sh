#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="llama3.1:8b"
DATASET_PATH="../expanded_e_commerce_and_saas_customer_support_orders_shipping_billing_ref.jsonl"
OUTPUT_DIR="./artifacts/finetune"

# Replace this command with your actual trainer invocation.
echo "Run unsloth fine-tuning with $BASE_MODEL on $DATASET_PATH and write artifacts to $OUTPUT_DIR"
