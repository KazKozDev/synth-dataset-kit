from __future__ import annotations

import logging
from pathlib import Path

from synth_dataset_kit.models import Dataset, QualityReport

logger = logging.getLogger(__name__)


def export_case_study_bundle(
    dataset: Dataset,
    report: QualityReport,
    output_dir: str,
) -> str:
    """Write a proof-oriented case-study markdown bundle for the generated dataset."""
    safe_name = dataset.name.replace(" ", "_").replace("/", "_")
    path = Path(output_dir) / f"{safe_name}_case_study.md"
    rebalancing_rounds = len(report.rebalancing_history or [])
    lines = [
        f"# Case Study: {dataset.name}",
        "",
        "This bundle documents a single synthetic dataset generation run.",
        "",
        "## Outcome",
        "",
        f"- Final examples: {dataset.size}",
        f"- Avg quality: {report.avg_quality_score:.2f}",
        f"- Passed examples: {report.passed_examples}/{report.total_examples}",
        f"- Distribution divergence: {report.distribution_divergence:.4f}",
        f"- Distribution match score: {report.distribution_match_score:.2f}/100",
        f"- Semantic coverage score: {report.semantic_coverage_score:.2%}",
        f"- Graph coverage score: {report.graph_coverage_score:.2%}",
        f"- Rebalancing rounds: {rebalancing_rounds}",
        f"- Contamination hits: {report.contamination_hits}",
        "",
        "## Distribution",
        "",
    ]
    if report.underrepresented_clusters:
        lines.extend(
            [
                "Underrepresented clusters after generation:",
                "",
            ]
            + [f"- {cluster}: {gap}" for cluster, gap in report.underrepresented_clusters.items()]
        )
    else:
        lines.append("- No remaining underrepresented clusters.")
    lines.extend(
        [
            "",
            "## Proof Pack",
            "",
            f"- Dataset name: `{dataset.name}`",
            f"- Generator: `{dataset.generator}`",
            "- Outputs include JSONL dataset, HTML/JSON quality reports, and pipeline artifacts.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Case study bundle saved to {path}")
    return str(path)
