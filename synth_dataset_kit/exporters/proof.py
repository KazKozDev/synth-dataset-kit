from __future__ import annotations

import json
import logging
from pathlib import Path

from synth_dataset_kit.exporters.eval_summary import _build_eval_summary, _pass_rate
from synth_dataset_kit.models import Dataset, QualityReport

logger = logging.getLogger(__name__)


def export_proof_bundle(
    dataset: Dataset,
    report: QualityReport,
    output_dir: str,
    base_model: str,
    trainer: str = "unsloth",
    holdout_path: str | None = None,
    baseline_dataset: Dataset | None = None,
    baseline_report: QualityReport | None = None,
    reference_dataset: Dataset | None = None,
    reference_report: QualityReport | None = None,
) -> list[str]:
    """Write a reproducible proof bundle with training/eval scripts and summaries."""
    safe_name = dataset.name.replace(" ", "_").replace("/", "_")
    bundle_dir = Path(output_dir) / f"{safe_name}_proof"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    eval_summary = _build_eval_summary(
        dataset,
        report,
        baseline_dataset=baseline_dataset,
        baseline_report=baseline_report,
        reference_dataset=reference_dataset,
        reference_report=reference_report,
    )
    summary_path = bundle_dir / "proof_summary.json"
    summary_path.write_text(
        json.dumps(eval_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    copied_holdout_path = None
    if holdout_path:
        source_holdout = Path(holdout_path)
        if source_holdout.exists():
            copied_holdout_path = bundle_dir / source_holdout.name
            copied_holdout_path.write_text(
                source_holdout.read_text(encoding="utf-8"), encoding="utf-8"
            )

    eval_rubric_path = bundle_dir / "support_eval_rubric.json"
    eval_rubric_path.write_text(
        json.dumps(
            {
                "task": "customer_support_assistant",
                "dimensions": [
                    "task_success",
                    "policy_alignment",
                    "answer_completeness",
                    "escalation_correctness",
                    "tone_and_empathy",
                    "hallucination_avoidance",
                ],
                "scoring_scale": "1-5",
                "notes": "Use the same holdout set to compare the base model and fine-tuned model.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    finetune_script = bundle_dir / "run_finetune.sh"
    finetune_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f'BASE_MODEL="{base_model}"',
                f'DATASET_PATH="../{safe_name}.jsonl"',
                'OUTPUT_DIR="./artifacts/finetune"',
                "",
                "# Replace this command with your actual trainer invocation.",
                f'echo "Run {trainer} fine-tuning with $BASE_MODEL on $DATASET_PATH and write artifacts to $OUTPUT_DIR"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    finetune_script.chmod(0o755)

    eval_script = bundle_dir / "run_eval.sh"
    eval_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f'BASE_MODEL="{base_model}"',
                'FINETUNED_MODEL="./artifacts/finetune/final-model"',
                'EVAL_OUTPUT="./artifacts/eval_results.json"',
                f'HOLDOUT_PATH="./{copied_holdout_path.name}"'
                if copied_holdout_path
                else 'HOLDOUT_PATH="./customer_support_holdout.jsonl"',
                'RUBRIC_PATH="./support_eval_rubric.json"',
                "",
                "# Replace this command with your actual holdout evaluation command.",
                'echo "Evaluate $BASE_MODEL and $FINETUNED_MODEL on $HOLDOUT_PATH using $RUBRIC_PATH and write metrics to $EVAL_OUTPUT"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    eval_script.chmod(0o755)

    md_lines = [
        f"# Proof Bundle: {dataset.name}",
        "",
        "This bundle is a reproducible starting point for `dataset -> fine-tune -> eval uplift`.",
        "",
        "## Included Files",
        "",
        "- `proof_summary.json`",
        "- `run_finetune.sh`",
        "- `run_eval.sh`",
        "- `support_eval_rubric.json`",
        "",
        "## Recommended Flow",
        "",
        "1. Fine-tune the base model on the generated dataset.",
        "2. Evaluate the base model and fine-tuned model on the same holdout set.",
        "3. Record uplift metrics alongside the proof summary.",
        "",
        "## Current Dataset Signals",
        "",
        f"- Avg quality score: {report.avg_quality_score:.2f}",
        f"- Pass rate: {_pass_rate(report):.2%}",
        f"- Lexical diversity: {report.lexical_diversity:.4f}",
        f"- Distribution divergence: {report.distribution_divergence:.4f}",
        f"- Contamination hits: {report.contamination_hits}",
    ]
    if copied_holdout_path:
        md_lines.extend(
            [
                "",
                "## Holdout",
                "",
                f"- Included holdout file: `{copied_holdout_path.name}`",
                "- Use the same holdout for both the base model and the fine-tuned model.",
            ]
        )
    if eval_summary.get("baseline_comparison"):
        baseline = eval_summary["baseline_comparison"]
        md_lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                f"- Baseline dataset: {baseline['baseline_dataset_name']}",
                f"- Avg quality delta: {baseline['avg_quality_delta']:+.4f}",
                f"- Pass rate delta: {baseline['pass_rate_delta']:+.2%}",
            ]
        )
    if eval_summary.get("reference_comparison"):
        reference = eval_summary["reference_comparison"]
        md_lines.extend(
            [
                "",
                "## Reference Comparison",
                "",
                f"- Reference dataset: {reference['reference_dataset_name']}",
                f"- Topic overlap ratio: {reference['topic_overlap_ratio']:.2%}",
                f"- Style distribution distance: {reference['style_distribution_distance']:.4f}",
                f"- Cluster coverage distance: {reference['cluster_coverage_distance']:.4f}",
                f"- Difficulty profile distance: {reference['difficulty_profile_distance']:.4f}",
                f"- Exact overlap ratio: {reference['exact_overlap_ratio']:.2%}",
                f"- Near overlap ratio: {reference['near_overlap_ratio']:.2%}",
                (
                    f"- Semantic overlap ratio: {reference['semantic_overlap_ratio']:.2%}"
                    if reference["semantic_overlap_ratio"] is not None
                    else "- Semantic overlap ratio: n/a"
                ),
                f"- Reference alignment score: {reference['reference_alignment_score']:.2f}/100",
            ]
        )
    md_path = bundle_dir / "proof_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info(f"Proof bundle saved to {bundle_dir}")
    output_files = [
        str(bundle_dir),
        str(summary_path),
        str(finetune_script),
        str(eval_script),
        str(eval_rubric_path),
        str(md_path),
    ]
    if copied_holdout_path:
        output_files.append(str(copied_holdout_path))
    return output_files
