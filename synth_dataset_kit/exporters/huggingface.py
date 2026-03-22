from __future__ import annotations

import json
import logging
from pathlib import Path

from synth_dataset_kit.exporters.eval_summary import _build_eval_summary, _usable_dataset_gate
from synth_dataset_kit.exporters.formats import export_jsonl
from synth_dataset_kit.exporters.quality import (
    export_quality_report_html,
    export_quality_report_json,
)
from synth_dataset_kit.models import Dataset, QualityReport

logger = logging.getLogger(__name__)


def export_huggingface_bundle(
    dataset: Dataset,
    output_dir: str,
    include_metadata: bool = True,
    quality_report: QualityReport | None = None,
    baseline_dataset: Dataset | None = None,
    baseline_report: QualityReport | None = None,
    reference_dataset: Dataset | None = None,
    reference_report: QualityReport | None = None,
) -> list[str]:
    """Export a HuggingFace-ready dataset bundle with dataset card and metadata."""
    safe_name = dataset.name.replace(" ", "_").replace("/", "_")
    bundle_dir = Path(output_dir) / f"{safe_name}_huggingface"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    train_path = export_jsonl(
        dataset,
        str(bundle_dir / "train.jsonl"),
        include_metadata=include_metadata,
    )

    eval_summary_path = bundle_dir / "eval_summary.json"
    if quality_report:
        eval_summary = _build_eval_summary(
            dataset,
            quality_report,
            baseline_dataset=baseline_dataset,
            baseline_report=baseline_report,
            reference_dataset=reference_dataset,
            reference_report=reference_report,
        )
    else:
        eval_summary = {
            "dataset_name": dataset.name,
            "examples": dataset.size,
        }
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    quality_report_files: list[str] = []
    if quality_report:
        quality_report_files.append(
            export_quality_report_html(
                quality_report,
                str(bundle_dir / f"{dataset.name}_quality_report.html"),
            )
        )
        quality_report_files.append(
            export_quality_report_json(
                quality_report,
                str(bundle_dir / f"{dataset.name}_quality_report.json"),
            )
        )

    yaml_header = [
        "---",
        "language:",
        "- en",
        "license: mit",
        "pretty_name: " + dataset.name,
        "task_categories:",
        "- text-generation",
        "- question-answering",
        "task_ids:",
        "- conversational",
        "tags:",
        "- synthetic",
        "- customer-support",
        "- llm",
        "- instruction-tuning",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: train.jsonl",
        "size_categories:",
        f"- {_hf_size_category(dataset.size)}",
        "---",
        "",
    ]

    card_lines = [
        *yaml_header,
        f"# {dataset.name}",
        "",
        "Synthetic customer-support dataset generated with synth-dataset-kit.",
        "",
        "## What Is Included",
        "",
        f"- `{Path(train_path).name}`: training split in JSONL chat format",
        f"- `{eval_summary_path.name}`: minimal evaluation and generation summary",
    ]
    if quality_report:
        card_lines.extend(
            [
                f"- `{dataset.name}_quality_report.html`: visual quality report",
                f"- `{dataset.name}_quality_report.json`: machine-readable quality report",
            ]
        )
    card_lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- examples: {dataset.size}",
            f"- generator: {dataset.generator}",
            f"- avg quality: {quality_report.avg_quality_score:.2f}"
            if quality_report
            else "- avg quality: n/a",
            f"- passed: {quality_report.passed_examples}" if quality_report else "- passed: n/a",
            (
                "- usable dataset gate: passed"
                if quality_report and _usable_dataset_gate(quality_report)["passed"]
                else "- usable dataset gate: failed"
            ),
            f"- contamination hits: {quality_report.contamination_hits}"
            if quality_report
            else "- contamination hits: n/a",
            f"- distribution divergence: {quality_report.distribution_divergence:.4f}"
            if quality_report
            else "- distribution divergence: n/a",
            f"- rebalancing rounds: {len(quality_report.rebalancing_history)}"
            if quality_report
            else "- rebalancing rounds: n/a",
            "",
            "## Public Quality Gate",
            "",
            "- A dataset is considered usable when every example passes the quality gate, average quality is at least 7.5, contamination hits are zero, and the final export is non-empty."
            if quality_report
            else "- quality gate unavailable",
            "",
            "## Data Generation Process",
            "",
            "This dataset was created from a small customer-support seed set.",
            "Generation used staged candidate creation, quality filtering, decontamination, and distribution-aware rebalancing.",
            "",
            "## Quality Signals",
            "",
            f"- lexical diversity: {quality_report.lexical_diversity:.4f}"
            if quality_report
            else "- lexical diversity: n/a",
            f"- self-BLEU proxy: {quality_report.self_bleu_proxy:.4f}"
            if quality_report
            else "- self-BLEU proxy: n/a",
            f"- embedding diversity: {quality_report.embedding_diversity_score}"
            if quality_report
            else "- embedding diversity: n/a",
            "",
            "## Distribution Alignment",
            "",
            f"- divergence: {quality_report.distribution_divergence:.4f}"
            if quality_report
            else "- divergence: n/a",
            f"- distribution match score: {quality_report.distribution_match_score:.2f}/100"
            if quality_report
            else "- distribution match score: n/a",
            f"- semantic coverage score: {quality_report.semantic_coverage_score:.2%}"
            if quality_report
            else "- semantic coverage score: n/a",
            f"- graph coverage score: {quality_report.graph_coverage_score:.2%}"
            if quality_report
            else "- graph coverage score: n/a",
            (
                "- underrepresented clusters: "
                + ", ".join(
                    f"{cluster}={gap}"
                    for cluster, gap in quality_report.underrepresented_clusters.items()
                )
            )
            if quality_report and quality_report.underrepresented_clusters
            else "- underrepresented clusters: none",
            (
                "- semantic coverage gaps: "
                + ", ".join(
                    f"cluster_{cluster}={gap}"
                    for cluster, gap in quality_report.semantic_coverage_gaps.items()
                )
            )
            if quality_report and quality_report.semantic_coverage_gaps
            else "- semantic coverage gaps: none",
            ("- graph frontier clusters: " + ", ".join(quality_report.graph_frontier_clusters))
            if quality_report and quality_report.graph_frontier_clusters
            else "- graph frontier clusters: none",
            "",
            "## Contamination Audit",
            "",
            f"- contamination hits: {quality_report.contamination_hits}"
            if quality_report
            else "- contamination hits: n/a",
            (
                "- verdicts: "
                + ", ".join(
                    f"{verdict}={count}"
                    for verdict, count in quality_report.contamination_verdicts.items()
                )
            )
            if quality_report and quality_report.contamination_verdicts
            else "- verdicts: none",
            "",
            "## Recommended Use",
            "",
            "Use this dataset for instruction tuning or support-assistant fine-tuning experiments.",
            "Validate on your own holdout set before production use.",
            "",
            "## Reproducibility",
            "",
            "This bundle was generated by synth-dataset-kit from a small seed set.",
            "Use `eval_summary.json` plus the full HTML/JSON report to reproduce the run context.",
        ]
    )
    baseline_comparison = eval_summary.get("baseline_comparison")
    if baseline_comparison:
        card_lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                f"- baseline dataset: {baseline_comparison['baseline_dataset_name']}",
                f"- example delta: {baseline_comparison['example_delta']}",
                f"- avg quality delta: {baseline_comparison['avg_quality_delta']:+.4f}",
                f"- pass rate delta: {baseline_comparison['pass_rate_delta']:+.2%}",
                f"- lexical diversity delta: {baseline_comparison['lexical_diversity_delta']:+.4f}",
                f"- distribution divergence delta: {baseline_comparison['distribution_divergence_delta']:+.4f}",
                f"- contamination hit delta: {baseline_comparison['contamination_hit_delta']:+d}",
            ]
        )
    reference_comparison = eval_summary.get("reference_comparison")
    if reference_comparison:
        card_lines.extend(
            [
                "",
                "## Reference Dataset Comparison",
                "",
                f"- reference dataset: {reference_comparison['reference_dataset_name']}",
                f"- avg user length delta: {reference_comparison['avg_user_length_delta']:+.4f}",
                f"- avg assistant length delta: {reference_comparison['avg_assistant_length_delta']:+.4f}",
                f"- diversity delta: {reference_comparison['diversity_score_delta']:+.4f}",
                f"- lexical diversity delta: {reference_comparison['lexical_diversity_delta']:+.4f}",
                f"- style distribution distance: {reference_comparison['style_distribution_distance']:.4f}",
                f"- cluster coverage distance: {reference_comparison['cluster_coverage_distance']:.4f}",
                f"- difficulty profile distance: {reference_comparison['difficulty_profile_distance']:.4f}",
                f"- topic overlap ratio: {reference_comparison['topic_overlap_ratio']:.2%}",
                f"- topic novelty ratio: {reference_comparison['topic_novelty_ratio']:.2%}",
                f"- exact overlap ratio: {reference_comparison['exact_overlap_ratio']:.2%}",
                f"- near overlap ratio: {reference_comparison['near_overlap_ratio']:.2%}",
                (
                    f"- semantic overlap ratio: {reference_comparison['semantic_overlap_ratio']:.2%}"
                    if reference_comparison["semantic_overlap_ratio"] is not None
                    else "- semantic overlap ratio: n/a"
                ),
                f"- reference alignment score: {reference_comparison['reference_alignment_score']:.2f}/100",
            ]
        )
    card_path = bundle_dir / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("\n".join(card_lines) + "\n")

    metadata_path = bundle_dir / "dataset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": dataset.name,
                "size": dataset.size,
                "generator": dataset.generator,
                "config_snapshot": dataset.config_snapshot,
                "quality_summary": quality_report.model_dump(mode="json")
                if quality_report
                else None,
                "eval_summary_path": str(eval_summary_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"Exported HuggingFace bundle to {bundle_dir}")
    return [
        str(bundle_dir),
        str(card_path),
        str(metadata_path),
        str(train_path),
        str(eval_summary_path),
        *quality_report_files,
    ]


def _hf_size_category(size: int) -> str:
    if size < 100:
        return "n<1K"
    if size < 1_000:
        return "1K<n<10K"
    if size < 10_000:
        return "10K<n<100K"
    if size < 100_000:
        return "100K<n<1M"
    return "n>1M"
