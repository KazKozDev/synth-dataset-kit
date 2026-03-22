from __future__ import annotations

import json
import logging
from pathlib import Path

from synth_dataset_kit.models import Dataset

logger = logging.getLogger(__name__)


def export_jsonl(dataset: Dataset, path: str, include_metadata: bool = False) -> str:
    """Export as JSONL with OpenAI messages format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w") as f:
        for example in dataset.examples:
            record = {
                "messages": [{"role": m.role.value, "content": m.content} for m in example.messages]
            }
            if include_metadata:
                record["metadata"] = example.metadata
                record["quality_score"] = example.quality_score
                record["decontamination_flags"] = example.decontamination_flags
                record["decontamination_evidence"] = example.decontamination_evidence
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Exported {dataset.size} examples to {p} (JSONL)")
    return str(p)


def export_alpaca(dataset: Dataset, path: str) -> str:
    """Export as Alpaca format (instruction/input/output)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for example in dataset.examples:
        records.append(
            {
                "instruction": example.user_message,
                "input": "",
                "output": example.assistant_message,
            }
        )

    with open(p, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {dataset.size} examples to {p} (Alpaca)")
    return str(p)


def export_sharegpt(dataset: Dataset, path: str) -> str:
    """Export as ShareGPT format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for example in dataset.examples:
        role_map = {"user": "human", "assistant": "gpt", "system": "system"}
        conversations = [
            {"from": role_map.get(m.role.value, m.role.value), "value": m.content}
            for m in example.messages
        ]
        records.append({"conversations": conversations})

    with open(p, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Exported {dataset.size} examples to {p} (ShareGPT)")
    return str(p)


def export_chatml(dataset: Dataset, path: str) -> str:
    """Export as ChatML format (used by many training frameworks)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w") as f:
        for example in dataset.examples:
            text = ""
            for m in example.messages:
                text += f"<|im_start|>{m.role.value}\n{m.content}<|im_end|>\n"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    logger.info(f"Exported {dataset.size} examples to {p} (ChatML)")
    return str(p)
