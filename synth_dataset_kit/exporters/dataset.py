from __future__ import annotations

import logging
from pathlib import Path

from synth_dataset_kit.exporters.formats import (
    export_alpaca,
    export_chatml,
    export_jsonl,
    export_sharegpt,
)
from synth_dataset_kit.exporters.huggingface import export_huggingface_bundle
from synth_dataset_kit.models import Dataset, QualityReport

logger = logging.getLogger(__name__)


def export_dataset(
    dataset: Dataset,
    format: str,
    output_dir: str,
    include_metadata: bool = False,
    quality_report: QualityReport | None = None,
    baseline_dataset: Dataset | None = None,
    baseline_report: QualityReport | None = None,
    reference_dataset: Dataset | None = None,
    reference_report: QualityReport | None = None,
) -> str:
    """Export dataset to the specified format."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    safe_name = dataset.name.replace(" ", "_").replace("/", "_")

    if format == "huggingface":
        bundle_files = export_huggingface_bundle(
            dataset,
            output_dir,
            include_metadata=include_metadata,
            quality_report=quality_report,
            baseline_dataset=baseline_dataset,
            baseline_report=baseline_report,
            reference_dataset=reference_dataset,
            reference_report=reference_report,
        )
        return bundle_files[0]

    exporters = {
        "jsonl": (export_jsonl, f"{safe_name}.jsonl"),
        "openai": (export_jsonl, f"{safe_name}.jsonl"),
        "alpaca": (export_alpaca, f"{safe_name}_alpaca.json"),
        "sharegpt": (export_sharegpt, f"{safe_name}_sharegpt.jsonl"),
        "chatml": (export_chatml, f"{safe_name}_chatml.jsonl"),
    }

    if format not in exporters:
        raise ValueError(f"Unknown format '{format}'. Available: {list(exporters.keys())}")

    exporter_fn, filename = exporters[format]
    filepath = str(output_dir_path / filename)

    if format == "jsonl" or format == "openai":
        return exporter_fn(dataset, filepath, include_metadata=include_metadata)
    else:
        return exporter_fn(dataset, filepath)
