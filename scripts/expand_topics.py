"""Generate support examples for additional topics not well covered in the current dataset.

Adds coverage for: returns/exchanges, account management, promo codes/discounts,
payment failures, cancellations, compliance/GDPR requests, and product questions.

Usage:
    python scripts/expand_topics.py \\
        --seed-file "support_seed (1).jsonl" \\
        --output outputs/support_expanded_topics/expanded_new_topics.jsonl \\
        --num 300 \\
        --config sdk_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.engine import DatasetEngine
from synth_dataset_kit.models import Dataset, Example, Message, Role

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Topics currently underrepresented in the shipping/delivery-heavy dataset
NEW_TOPICS = [
    "Returns and exchanges — initiating, label generation, refund timelines",
    "Account management — password reset, email change, 2FA issues, account merge",
    "Promo codes and discounts — invalid codes, stacking rules, expired offers",
    "Payment failures and billing errors — declined cards, retry logic, invoice disputes",
    "Subscription cancellations — mid-cycle cancellations, pause options, data retention",
    "GDPR and data privacy — data export requests, right to erasure, consent withdrawal",
    "Product questions and compatibility — specs, stock availability, bundle options",
    "Fraud and unauthorized access — suspicious charges, account takeover, chargeback guidance",
    "Onboarding and activation — first-time setup, plan activation, feature access",
    "SLA breaches and escalations — missed delivery windows, compensation requests",
]

# Seeds specifically for new topics (few-shot, hand-crafted)
NEW_TOPIC_SEEDS = [
    {
        "user": "I want to return an item I bought 12 days ago. It's unused, original packaging. How do I start?",
        "assistant": "Returns within 30 days are straightforward. I'll generate a prepaid return label for you — it'll go to your email within 5 minutes. Once we receive the item (usually 3-5 business days), refund processes within 2 business days to your original payment method. No restocking fee for unused items in original packaging. Want me to send the label now?",
    },
    {
        "user": "My promo code SAVE20 says 'invalid' at checkout. I got it in your email this morning.",
        "assistant": "Sorry about that — let me check the code. SAVE20 is active and valid through end of month. A few things that can cause 'invalid': the cart minimum isn't met (it requires $50+), it may not apply to already-discounted items, or there's a one-use-per-account limit. Can you tell me your cart total and whether any items are already on sale? I can verify which applies and figure out the fix.",
    },
    {
        "user": "I got charged $49.99 twice today — looks like a billing glitch. One is pending, one posted. My bank shows both.",
        "assistant": "I can see the duplicate charge on our end too. The pending one will fall off automatically within 3-5 business days — it's an authorization hold, not a capture. The posted charge is the correct one. If the pending doesn't clear by Friday, message us back and I'll escalate to our payments team to void it manually. No action needed from you unless it's still showing Monday.",
    },
    {
        "user": "I want to cancel my subscription but I'm in the middle of a billing cycle. Do I lose the rest of the month?",
        "assistant": "You won't lose access — your account stays active until the end of the current billing period. After that, no further charges. We don't offer prorated refunds for mid-cycle cancellations, but you'll have full access through [date]. Want me to proceed with cancellation, or would you prefer to pause instead? Pause locks your current rate for up to 3 months.",
    },
    {
        "user": "Under GDPR I want a copy of all data you hold on me, and then I want it deleted.",
        "assistant": "Understood. I'll submit a data subject access request (DSAR) right now — you'll receive a full export of your personal data by email within 30 days (usually faster). Once you confirm receipt, I'll initiate the erasure request. Note: we're legally required to retain transaction records for 7 years for tax purposes, but all other personal data will be deleted. I'll email you confirmation at each step. You'll also get a case reference number within 1 hour.",
    },
    {
        "user": "Someone made a purchase on my account that I didn't authorize. What do I do?",
        "assistant": "Flagging this now as a potential unauthorized transaction. First: I'm locking your account's payment methods immediately so no further charges can occur. I'll initiate a dispute for the charge — it takes 3-5 business days for provisional credit while we investigate. I'll also trigger a password reset and 2FA re-enrollment email. Please don't use the account until you've secured it. Can you confirm your email so I can send the security link right now?",
    },
    {
        "user": "I can't log in — I changed my email last week and now I can't reset my password because the code goes to the old email.",
        "assistant": "Classic catch-22 — I can break this manually. I'll verify your identity with a few questions (last 4 digits of payment on file + recent order number), then force-update your email to the new one and send a fresh password reset. This takes about 5 minutes once I verify you. Ready to go through the verification questions?",
    },
    {
        "user": "Is the XR-7 model compatible with my MacBook Pro M2? I don't want to order and have to return it.",
        "assistant": "Yes — XR-7 is fully compatible with M2 MacBook Pros running macOS 13+. It uses USB-C (the cable's included) and doesn't require any additional drivers. The only caveat: if you're on macOS 12 or below, you'd need to update first. What's your current macOS version? I can confirm before you order to be sure.",
    },
]


def save_jsonl(dataset: Dataset, output_path: str) -> None:
    """Save dataset to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in dataset.examples:
            record = {
                "messages": [{"role": m.role.value, "content": m.content} for m in ex.messages],
                "metadata": ex.metadata,
                "quality_score": ex.quality_score,
                "decontamination_flags": ex.decontamination_flags,
                "decontamination_evidence": ex.decontamination_evidence,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Saved {dataset.size} examples → {output_path}")


def build_seed_file(output_path: str) -> str:
    """Write the hand-crafted seeds to a temp JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seed in NEW_TOPIC_SEEDS:
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(NEW_TOPIC_SEEDS)} new-topic seeds → {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate examples for underrepresented support topics"
    )
    parser.add_argument(
        "--seed-file",
        default=None,
        help="Optional existing seed JSONL file to merge in (will combine with built-in seeds)",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--config", default="sdk_config.yaml", help="SDK config path")
    parser.add_argument("--num", type=int, default=300, help="Number of examples to generate")
    args = parser.parse_args()

    cfg = SDKConfig.from_yaml(args.config) if Path(args.config).exists() else SDKConfig()

    # Override domain to cover broader support topics
    cfg.generation.domain = (
        "E-commerce and SaaS customer support — returns, account management, "
        "billing, promo codes, cancellations, privacy requests, fraud, onboarding"
    )
    cfg.generation.num_examples = args.num
    cfg.generation.batch_size = 3

    # Write built-in seeds to temp file
    seed_path = "/tmp/new_topic_seeds.jsonl"
    build_seed_file(seed_path)

    engine = DatasetEngine(cfg)
    logger.info(f"Generating {args.num} examples for new topics...")
    dataset = engine.generate_from_seeds(seed_path, num_examples=args.num)

    logger.info(f"Generated {dataset.size} examples")
    save_jsonl(dataset, args.output)

    # Quick topic summary
    from collections import Counter

    topics = Counter(e.metadata.get("topic", "unknown") for e in dataset.examples)
    logger.info("\nTopic distribution:")
    for topic, count in topics.most_common(15):
        logger.info(f"  {count:3d}  {topic}")


if __name__ == "__main__":
    main()
