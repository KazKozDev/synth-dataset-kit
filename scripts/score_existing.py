"""Score quality for an existing JSONL dataset in-place, writing to a new file.

Supports resuming interrupted runs: if --output already exists, scored examples are
loaded from it and skipped (use --skip-scored). Progress is checkpointed every 20 examples.

Usage:
    python scripts/score_existing.py \\
        --input outputs/support_seed_run_500/expanded_*.jsonl \\
        --output outputs/support_seed_run_500/scored.jsonl \\
        --config sdk_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, Role
from synth_dataset_kit.quality import QualityJudge

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> Dataset:
    """Load a flat JSONL file (messages format) into a Dataset."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                messages = [
                    Message(role=Role(m["role"]), content=m["content"])
                    for m in data.get("messages", [])
                    if m.get("role") in ("user", "assistant", "system")
                ]
                if len(messages) >= 2:
                    examples.append(
                        Example(
                            messages=messages,
                            metadata=data.get("metadata", {}),
                            quality_score=data.get("quality_score"),
                            decontamination_flags=data.get("decontamination_flags", []),
                            decontamination_evidence=data.get("decontamination_evidence", []),
                        )
                    )
            except Exception as e:
                logger.warning(f"Skipping line {line_num}: {e}")
    name = Path(path).stem[:40]
    return Dataset(name=name, examples=examples)


def _example_to_record(ex: Example) -> dict:
    """Serialize an Example to a plain dict for JSONL output."""
    return {
        "messages": [{"role": m.role.value, "content": m.content} for m in ex.messages],
        "metadata": ex.metadata,
        "quality_score": ex.quality_score,
        "decontamination_flags": ex.decontamination_flags,
        "decontamination_evidence": ex.decontamination_evidence,
    }


def load_scored_index(output_path: str) -> dict[str, float | None]:
    """Load already-scored examples from output file keyed by user+assistant prefix."""
    index: dict[str, float | None] = {}
    p = Path(output_path)
    if not p.exists():
        return index
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                msgs = data.get("messages", [])
                user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                key = f"{user[:80]}|||{asst[:80]}"
                index[key] = data.get("quality_score")
            except Exception:
                pass
    return index


def save_jsonl(dataset: Dataset, output_path: str) -> None:
    """Write dataset back to JSONL preserving all fields including new quality_score."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Score quality for existing JSONL dataset")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--config", default="sdk_config.yaml", help="SDK config path")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="If set, filter to examples >= min_score in output",
    )
    parser.add_argument(
        "--skip-scored",
        action="store_true",
        default=False,
        help="Skip examples that already have a quality_score",
    )
    args = parser.parse_args()

    cfg = SDKConfig.from_yaml(args.config) if Path(args.config).exists() else SDKConfig()
    client = LLMClient(cfg.llm)
    judge = QualityJudge(client, cfg.quality, cfg.generation.system_prompt)

    dataset = load_jsonl(args.input)
    logger.info(f"Loaded {dataset.size} examples from {args.input}")

    # Load already-scored index for resume support
    scored_index = load_scored_index(args.output) if args.skip_scored else {}
    if scored_index:
        logger.info(f"Resume mode: {len(scored_index)} already-scored examples found in {args.output}")

    # Open output in append mode if resuming, write mode otherwise
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if (args.skip_scored and scored_index) else "w"
    out_f = open(args.output, write_mode, encoding="utf-8")

    # If resuming, flush already-scored examples first in write mode
    if write_mode == "w" and args.skip_scored and not scored_index:
        pass  # fresh start

    scored = 0
    skipped = 0
    checkpoint_interval = 20

    try:
        for i, example in enumerate(dataset.examples, 1):
            # Resume: check if already scored
            if args.skip_scored:
                user = example.user_message[:80]
                asst = example.assistant_message[:80]
                key = f"{user}|||{asst}"
                if key in scored_index:
                    example.quality_score = scored_index[key]
                    if write_mode == "w":
                        out_f.write(json.dumps(_example_to_record(example), ensure_ascii=False) + "\n")
                    skipped += 1
                    continue

            judge.score_example(example)
            out_f.write(json.dumps(_example_to_record(example), ensure_ascii=False) + "\n")
            scored += 1

            if i % checkpoint_interval == 0 or i == dataset.size:
                out_f.flush()
                scores_so_far = [
                    e.quality_score for e in dataset.examples[:i] if e.quality_score is not None
                ]
                avg = sum(scores_so_far) / max(len(scores_so_far), 1)
                logger.info(
                    f"  [{i}/{dataset.size}] scored={scored} skipped={skipped} "
                    f"avg={avg:.2f}  (checkpoint saved)"
                )
    finally:
        out_f.close()

    # Summary
    all_scores = [e.quality_score for e in dataset.examples if e.quality_score is not None]
    logger.info(f"\nScoring complete:")
    logger.info(f"  Scored: {scored}, Skipped (already scored): {skipped}")
    if all_scores:
        logger.info(
            f"  Min: {min(all_scores):.2f}, "
            f"Max: {max(all_scores):.2f}, "
            f"Avg: {sum(all_scores)/len(all_scores):.2f}"
        )
        if args.min_score is not None:
            passing = [s for s in all_scores if s >= args.min_score]
            logger.info(
                f"  Passing (>= {args.min_score}): "
                f"{len(passing)}/{len(all_scores)} ({100*len(passing)/len(all_scores):.1f}%)"
            )
        from collections import Counter
        buckets = Counter(int(s) for s in all_scores if s is not None)
        logger.info("  Score distribution:")
        for bucket in sorted(buckets):
            logger.info(f"    {bucket}-{bucket+1}: {buckets[bucket]}")

    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
