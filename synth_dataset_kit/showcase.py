"""Helpers for rendering showcase summaries from run artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    """Execute load json."""
    return json.loads(path.read_text(encoding="utf-8"))


def find_optional_json(run_summary_path: Path, suffix: str) -> dict | None:
    """Execute find optional json."""
    for candidate in run_summary_path.parent.glob(f"*{suffix}"):
        try:
            return load_json(candidate)
        except Exception:
            continue
    return None


def fmt_minutes(seconds: float | int | None) -> str:
    """Execute fmt minutes."""
    if seconds is None:
        return "n/a"
    return f"{float(seconds) / 60.0:.1f} min"


def build_showcase_markdown(
    run_summary: dict,
    quality_report: dict | None,
    eval_summary: dict | None,
) -> str:
    """Execute build showcase markdown."""
    stage_timings = dict(run_summary.get("stage_timings", {}))
    input_cfg = dict(run_summary.get("input", {}))
    lines = [
        "# Showcase Metrics",
        "",
        "This file is generated from `run_summary.json` and companion run artifacts.",
        "",
        "## Run Snapshot",
        "",
        f"- provider: `{run_summary.get('provider', 'n/a')}`",
        f"- model: `{run_summary.get('model', 'n/a')}`",
        f"- use case: `{input_cfg.get('domain') or 'seed-based generation'}`",
        f"- seed mode: `{'seed expansion' if input_cfg.get('seeds') else 'domain-only generation'}`",
        f"- requested examples: `{input_cfg.get('num_examples', 'n/a')}`",
        f"- retained examples: `{run_summary.get('examples_retained', 'n/a')}`",
        f"- total runtime: `{run_summary.get('runtime_minutes', 'n/a')} min`",
        "",
        "## Stage Timings",
        "",
        f"- generate: `{fmt_minutes(stage_timings.get('generate_seconds'))}`",
        f"- audit: `{fmt_minutes(stage_timings.get('audit_seconds'))}`",
        f"- filter: `{fmt_minutes(stage_timings.get('filter_seconds'))}`",
        f"- export: `{fmt_minutes(stage_timings.get('export_seconds'))}`",
        "",
        "## Quality Snapshot",
        "",
        f"- avg quality score: `{run_summary.get('avg_quality_score', 'n/a')}`",
        f"- pass rate: `{run_summary.get('pass_rate', 'n/a')}`",
        f"- contamination hits: `{run_summary.get('contamination_hits', 'n/a')}`",
    ]

    if quality_report:
        lines.extend(
            [
                f"- lexical diversity: `{quality_report.get('lexical_diversity', 'n/a')}`",
                f"- diversity score: `{quality_report.get('diversity_score', 'n/a')}`",
                f"- distribution divergence: `{quality_report.get('distribution_divergence', 'n/a')}`",
            ]
        )

    if eval_summary:
        baseline = eval_summary.get("baseline_comparison") or {}
        reference = eval_summary.get("reference_comparison") or {}
        lines.extend(
            [
                "",
                "## Comparison Snapshot",
                "",
                f"- baseline quality delta: `{baseline.get('avg_quality_delta', 'n/a')}`",
                f"- baseline pass-rate delta: `{baseline.get('pass_rate_delta', 'n/a')}`",
                f"- reference alignment score: `{reference.get('reference_alignment_score', 'n/a')}`",
            ]
        )

    return "\n".join(lines) + "\n"


def render_showcase_summary(run_summary_path: Path, output_path: Path | None = None) -> Path:
    """Execute render showcase summary."""
    run_summary = load_json(run_summary_path)
    quality_report = find_optional_json(run_summary_path, "_quality_report.json")
    eval_summary = find_optional_json(run_summary_path, "_eval_summary.json")
    markdown = build_showcase_markdown(run_summary, quality_report, eval_summary)
    target = output_path or (run_summary_path.parent / "SHOWCASE_METRICS.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    return target
