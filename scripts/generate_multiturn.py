"""Generate multi-turn conversations from existing single-turn support examples.

Takes a single-turn JSONL dataset and expands each example into a 3-5 turn
conversation by generating realistic follow-up exchanges.

Usage:
    python scripts/generate_multiturn.py \\
        --input outputs/support_seed_run_500/expanded_*.jsonl \\
        --output outputs/support_seed_run_500/multiturn.jsonl \\
        --num 100 \\
        --config sdk_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MULTITURN_PROMPT = """You are a synthetic data expert for customer support training.

Below is the start of a support conversation (turn 1). Your task is to extend it
into a realistic multi-turn conversation with {num_extra_turns} additional exchanges.

EXISTING CONVERSATION:
User: {user_msg}
Support: {assistant_msg}

Generate {num_extra_turns} additional turn pairs that realistically follow from this.
The user should follow up naturally — asking for clarification, providing requested info,
escalating if still unresolved, or confirming resolution.
The support agent should respond helpfully, consistently, and move toward resolution.

Return ONLY a JSON object:
{{
  "turns": [
    {{"user": "follow-up message", "assistant": "support response"}},
    ...
  ]
}}

RULES:
1. User messages must feel natural — reference prior context, not robotic
2. Support responses should build on previous turns (don't repeat info already given)
3. Move toward resolution: each turn should advance the conversation
4. Support agent should NOT ask for information already provided
5. Keep consistent tone and style with the initial exchange
6. Return strict JSON only — no markdown, no trailing commas
"""


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict[str, Any]], output_path: str) -> None:
    """Save records to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} multi-turn examples → {output_path}")


def expand_to_multiturn(
    client: LLMClient,
    record: dict[str, Any],
    num_extra_turns: int = 2,
) -> dict[str, Any] | None:
    """Expand a single-turn record into a multi-turn conversation."""
    messages = record.get("messages", [])
    user_msgs = [m for m in messages if m["role"] == "user"]
    asst_msgs = [m for m in messages if m["role"] == "assistant"]
    if not user_msgs or not asst_msgs:
        return None

    user_msg = user_msgs[0]["content"]
    assistant_msg = asst_msgs[0]["content"]

    prompt = MULTITURN_PROMPT.format(
        user_msg=user_msg,
        assistant_msg=assistant_msg,
        num_extra_turns=num_extra_turns,
    )

    try:
        result = client.complete_json(
            [{"role": "user", "content": prompt}],
            temperature=0.75,
        )
        if not isinstance(result, dict):
            return None

        turns = result.get("turns", [])
        if not turns:
            return None

        # Build the full message list
        full_messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        for turn in turns[:num_extra_turns]:
            user_follow = turn.get("user", "").strip()
            asst_follow = turn.get("assistant", "").strip()
            if user_follow and asst_follow:
                full_messages.append({"role": "user", "content": user_follow})
                full_messages.append({"role": "assistant", "content": asst_follow})

        if len(full_messages) <= 2:
            return None

        return {
            "messages": full_messages,
            "metadata": {
                **record.get("metadata", {}),
                "multiturn": True,
                "num_turns": len(full_messages) // 2,
                "source": "multiturn_expansion",
                "original_source": record.get("metadata", {}).get("source", "unknown"),
            },
            "quality_score": None,
            "decontamination_flags": [],
            "decontamination_evidence": [],
        }

    except Exception as e:
        logger.warning(f"Multi-turn expansion failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-turn support conversations")
    parser.add_argument("--input", required=True, help="Input single-turn JSONL file")
    parser.add_argument("--output", required=True, help="Output multi-turn JSONL file")
    parser.add_argument("--config", default="sdk_config.yaml", help="SDK config path")
    parser.add_argument(
        "--num", type=int, default=100, help="Number of multi-turn examples to generate"
    )
    parser.add_argument(
        "--min-turns", type=int, default=2, help="Min extra turn pairs to add (default 2)"
    )
    parser.add_argument(
        "--max-turns", type=int, default=3, help="Max extra turn pairs to add (default 3)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    cfg = SDKConfig.from_yaml(args.config) if Path(args.config).exists() else SDKConfig()
    client = LLMClient(cfg.llm)

    records = load_jsonl(args.input)
    logger.info(f"Loaded {len(records)} source examples from {args.input}")

    # Sample from generated (non-seed) examples only — richer material
    generated = [r for r in records if not r.get("metadata", {}).get("seed_example")]
    if not generated:
        generated = records
    logger.info(f"Pool of {len(generated)} generated examples for multi-turn expansion")

    rng = random.Random(args.seed)
    sample = rng.sample(generated, min(args.num, len(generated)))

    results = []
    for i, record in enumerate(sample, 1):
        num_extra = rng.randint(args.min_turns, args.max_turns)
        expanded = expand_to_multiturn(client, record, num_extra_turns=num_extra)
        if expanded:
            results.append(expanded)
            if i % 10 == 0 or i == len(sample):
                logger.info(f"  [{i}/{len(sample)}] Generated {len(results)} multi-turn so far")
        else:
            logger.warning(f"  [{i}/{len(sample)}] Expansion failed, skipping")

    logger.info(f"\nMulti-turn generation complete: {len(results)}/{len(sample)} succeeded")

    # Summary stats
    turn_counts = [r["metadata"]["num_turns"] for r in results]
    if turn_counts:
        logger.info(f"  Avg turns per conversation: {sum(turn_counts) / len(turn_counts):.1f}")
        logger.info(f"  Turn distribution: {sorted(set(turn_counts))}")

    save_jsonl(results, args.output)


if __name__ == "__main__":
    main()
