from __future__ import annotations

import json
import logging
from pathlib import Path

from synth_dataset_kit.exporters.formats import export_jsonl
from synth_dataset_kit.models import Dataset

logger = logging.getLogger(__name__)


def export_pipeline_artifacts(dataset: Dataset, output_dir: str) -> list[str]:
    """Export intermediate pipeline artifacts as JSONL files."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    safe_name = dataset.name.replace(" ", "_").replace("/", "_")
    output_files: list[str] = []

    for artifact_name, examples in dataset.artifacts.items():
        if not examples:
            continue
        artifact_dataset = Dataset(
            name=f"{safe_name}_{artifact_name}",
            version=dataset.version,
            created_at=dataset.created_at,
            generator=dataset.generator,
            config_snapshot=dataset.config_snapshot,
            examples=examples,
        )
        artifact_path = str(output_dir_path / f"{safe_name}_{artifact_name}.jsonl")
        export_jsonl(artifact_dataset, artifact_path, include_metadata=True)
        output_files.append(artifact_path)

    return output_files


def export_run_summary(summary: dict[str, object], path: str) -> str:
    """Save a run summary JSON artifact alongside the output bundle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Run summary saved to {p}")
    return str(p)
