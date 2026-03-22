"""Pipeline CLI commands: generate, audit, eval, validate-match, validate-metric, run, inspect, export."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from synth_dataset_kit.cli._app import (
    _load_dataset_file,
    _record_runtime_history,
    app,
    console,
    load_config,
    setup_logging,
)
from synth_dataset_kit.cli._display import (
    _display_artifact_examples,
    _display_artifact_summary,
    _display_baseline_comparison,
    _display_reference_comparison,
    _display_report,
    _export_artifact_csv,
    _sort_artifact_examples,
)
from synth_dataset_kit.engine import DatasetEngine
from synth_dataset_kit.models import Dataset
from synth_dataset_kit.showcase import render_showcase_summary

# ─── GENERATE ────────────────────────────────────────────────────────────────


@app.command(hidden=True)
def generate(
    seeds: str | None = typer.Option(None, "--seeds", "-s", help="Path to seed JSONL file"),
    domain: str | None = typer.Option(None, "--domain", "-d", help="Domain description"),
    num: int = typer.Option(100, "--num", "-n", help="Number of examples to generate"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    format: str = typer.Option(
        "jsonl", "--format", "-f", help="Output format: jsonl, alpaca, sharegpt, chatml"
    ),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate dataset candidates only, without the full review flow."""
    setup_logging(verbose)
    cfg = load_config(config)
    cfg.generation.num_examples = num
    cfg.export.output_dir = output
    cfg.export.format = format

    if not seeds and not domain:
        console.print("[red]Error:[/red] Provide --seeds or --domain")
        raise typer.Exit(1)

    engine = DatasetEngine(cfg)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if seeds:
            progress.add_task("Generating from seeds...", total=None)
            dataset = engine.generate_from_seeds(seeds, num)
        else:
            progress.add_task(f"Generating for '{domain}'...", total=None)
            dataset = engine.generate_from_domain(domain, num)

    from synth_dataset_kit.exporters import export_dataset

    filepath = export_dataset(dataset, format, output)

    console.print(f"\n[green]✓[/green] Generated [bold]{dataset.size}[/bold] examples")
    console.print(f"  Output: [cyan]{filepath}[/cyan]")
    console.print("\n  For the full reviewed flow, run [bold]sdk create[/bold]")


# ─── AUDIT ───────────────────────────────────────────────────────────────────


@app.command(hidden=True)
def audit(
    input_file: str = typer.Argument(..., help="Path to dataset JSONL file"),
    output: str = typer.Option("./output", "--output", "-o", help="Report output directory"),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run quality scoring and decontamination as a standalone step."""
    setup_logging(verbose)
    cfg = load_config(config)
    engine = DatasetEngine(cfg)
    dataset = _load_dataset_file(input_file)
    console.print(f"Loaded [bold]{dataset.size}[/bold] examples from {input_file}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Auditing dataset...", total=None)
        report = engine.audit(dataset)

    _display_report(report)

    from synth_dataset_kit.exporters import export_quality_report_html, export_quality_report_json

    report_path = str(Path(output) / f"{dataset.name}_quality_report.html")
    export_quality_report_html(report, report_path)
    json_report_path = str(Path(output) / f"{dataset.name}_quality_report.json")
    export_quality_report_json(report, json_report_path)
    console.print(f"\n  Full report: [cyan]{report_path}[/cyan]")
    console.print(f"  JSON report: [cyan]{json_report_path}[/cyan]")


# ─── EVAL ────────────────────────────────────────────────────────────────────


@app.command(hidden=True)
def eval(
    input_file: str = typer.Argument(..., help="Path to dataset JSONL file"),
    baseline: str | None = typer.Option(
        None, "--baseline", help="Optional baseline/reference JSONL file"
    ),
    reference: str | None = typer.Option(
        None, "--reference", help="Optional reference dataset JSONL file"
    ),
    output: str = typer.Option("./output", "--output", "-o", help="Eval output directory"),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a proof-oriented evaluation stage and write eval summary artifacts."""
    setup_logging(verbose)
    cfg = load_config(config)
    engine = DatasetEngine(cfg)
    dataset = _load_dataset_file(input_file)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Evaluating dataset...", total=None)
        report = engine.audit(dataset)

    _display_report(report)

    baseline_dataset = baseline_report = reference_dataset = reference_report = None
    if baseline:
        baseline_dataset = _load_dataset_file(baseline)
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            progress.add_task("Evaluating baseline...", total=None)
            baseline_report = engine.audit(baseline_dataset)
        _display_baseline_comparison(dataset, report, baseline_dataset, baseline_report)
    if reference:
        reference_dataset = _load_dataset_file(reference)
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            progress.add_task("Evaluating reference dataset...", total=None)
            reference_report = engine.audit(reference_dataset)
        _display_reference_comparison(dataset, report, reference_dataset, reference_report)

    from synth_dataset_kit.exporters import export_eval_summary

    eval_files = export_eval_summary(
        dataset,
        report,
        output,
        baseline_dataset=baseline_dataset,
        baseline_report=baseline_report,
        reference_dataset=reference_dataset,
        reference_report=reference_report,
    )
    console.print("\n[bold]Eval outputs:[/bold]")
    for filepath in eval_files:
        console.print(f"  [cyan]{filepath}[/cyan]")


# ─── VALIDATE-MATCH ──────────────────────────────────────────────────────────


@app.command(name="validate-match", hidden=True)
def validate_match(
    generated: str = typer.Argument(..., help="Generated dataset JSONL file"),
    reference: str = typer.Argument(..., help="Reference dataset JSONL file"),
    output: str = typer.Option(
        "./output/validation", "--output", "-o", help="Validation output directory"
    ),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Calibrate distribution-match quality against a stronger reference dataset."""
    setup_logging(verbose)
    cfg = load_config(config)
    engine = DatasetEngine(cfg)
    generated_dataset = _load_dataset_file(generated)
    reference_dataset = _load_dataset_file(reference)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Auditing generated dataset...", total=None)
        generated_report = engine.audit(generated_dataset)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Auditing reference dataset...", total=None)
        reference_report = engine.audit(reference_dataset)

    _display_report(generated_report)
    _display_reference_comparison(
        generated_dataset, generated_report, reference_dataset, reference_report
    )

    from synth_dataset_kit.exporters import export_eval_summary

    files = export_eval_summary(
        generated_dataset,
        generated_report,
        output,
        reference_dataset=reference_dataset,
        reference_report=reference_report,
    )
    console.print("\n[bold]Validation outputs:[/bold]")
    for filepath in files:
        console.print(f"  [cyan]{filepath}[/cyan]")


# ─── VALIDATE-METRIC ─────────────────────────────────────────────────────────


@app.command(name="validate-metric", hidden=True)
def validate_metric(
    run_dirs: Annotated[
        list[str], typer.Option("--run-dir", help="Run directory with eval/uplift artifacts")
    ],
    output: str = typer.Option(
        "./output/metric_validation", "--output", "-o", help="Validation report output directory"
    ),
):
    """Correlate calibrated distribution-match scores with downstream uplift across runs."""
    from synth_dataset_kit.evaluation import (
        build_metric_validation_report,
        export_metric_validation_report,
    )

    runs: list[dict[str, object]] = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        if not path.exists():
            console.print(f"[yellow]Skipping missing run dir:[/yellow] {run_dir}")
            continue
        eval_candidates = sorted(path.glob("*_eval_summary.json"))
        uplift_candidates = sorted(path.glob("*_results.json"))
        if not eval_candidates or not uplift_candidates:
            console.print(f"[yellow]Skipping incomplete run dir:[/yellow] {run_dir}")
            continue
        eval_summary = json.loads(eval_candidates[0].read_text(encoding="utf-8"))
        uplift_results = json.loads(uplift_candidates[0].read_text(encoding="utf-8"))
        runs.append(
            {
                "name": path.name,
                "run_dir": str(path),
                "eval_summary": eval_summary,
                "uplift": uplift_results,
            }
        )

    report = build_metric_validation_report(runs)
    files = export_metric_validation_report(report, output)

    table = Table(title="Metric Validation", show_header=False, box=None)
    table.add_column(style="cyan", width=36)
    table.add_column()
    table.add_row("Runs analyzed", str(report["runs_analyzed"]))
    table.add_row("Avg calibration error", f"{float(report['avg_calibration_error']):.4f}")
    correlations = dict(report.get("correlations", {}))
    for label, key in [
        ("Score vs task success", "validated_match_vs_task_success_delta"),
        ("Score vs pass rate", "validated_match_vs_pass_rate_delta"),
        ("Score vs token F1", "validated_match_vs_token_f1_delta"),
    ]:
        value = correlations.get(key)
        table.add_row(label, "n/a" if value is None else f"{float(value):+.4f}")
    console.print(table)

    console.print("\n[bold]Validation outputs:[/bold]")
    for filepath in files:
        console.print(f"  [cyan]{filepath}[/cyan]")


# ─── INSPECT ─────────────────────────────────────────────────────────────────


def _artifact_base_name(path: Path) -> str:
    """Infer the dataset base name from a dataset or artifact filename."""
    stem = path.stem
    suffixes = [
        "_candidates",
        "_accepted",
        "_rejected",
        "_quality_report",
        "_alpaca",
        "_sharegpt",
        "_chatml",
    ]
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _resolve_artifact_group(path: Path) -> tuple[str, dict[str, Path | None]]:
    """Resolve artifact files from a directory or specific file path."""
    artifact_suffixes = ("_candidates.jsonl", "_accepted.jsonl", "_rejected.jsonl")

    if path.is_file():
        base_name = _artifact_base_name(path)
        search_dir = path.parent
    else:
        groups: dict[str, list[Path]] = {}
        for artifact_path in path.glob("*.jsonl"):
            if artifact_path.name.endswith(artifact_suffixes):
                base_name = _artifact_base_name(artifact_path)
                groups.setdefault(base_name, []).append(artifact_path)

        if not groups:
            return path.name, {}

        if len(groups) == 1:
            base_name = next(iter(groups))
        else:
            base_name = max(
                groups, key=lambda name: max(file.stat().st_mtime for file in groups[name])
            )
        search_dir = path

    artifact_paths = {
        "candidates": search_dir / f"{base_name}_candidates.jsonl",
        "accepted": search_dir / f"{base_name}_accepted.jsonl",
        "rejected": search_dir / f"{base_name}_rejected.jsonl",
    }
    return base_name, {
        name: artifact_path if artifact_path.exists() else None
        for name, artifact_path in artifact_paths.items()
    }


@app.command()
def inspect(
    target: str = typer.Argument(..., help="Output directory or dataset/artifact file"),
    show: str | None = typer.Option(
        None, "--show", help="Show artifact examples: candidates, accepted, or rejected"
    ),
    limit: int = typer.Option(10, "--limit", min=1, help="Maximum number of examples to print"),
    sort_by: str | None = typer.Option(
        None, "--sort-by", help="Sort shown/exported examples by score, topic, or reason"
    ),
    export_csv: str | None = typer.Option(
        None, "--export-csv", help="Write the selected artifact bucket to a CSV file"
    ),
):
    """Inspect previously saved pipeline artifacts without rerunning generation."""
    path = Path(target)
    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {target}")
        raise typer.Exit(1)

    base_name, artifact_paths = _resolve_artifact_group(path)
    if not artifact_paths:
        console.print(f"[red]Error:[/red] No pipeline artifacts found for: {target}")
        raise typer.Exit(1)

    dataset = Dataset(name=base_name)
    discovered_files: list[str] = []
    for artifact_name, artifact_path in artifact_paths.items():
        if artifact_path is None:
            continue
        artifact_dataset = _load_dataset_file(str(artifact_path))
        dataset.artifacts[artifact_name] = artifact_dataset.examples
        discovered_files.append(str(artifact_path))

    console.print(
        Panel(
            f"Artifact group: [cyan]{base_name}[/cyan]\n"
            f"Files loaded: [cyan]{len(discovered_files)}[/cyan]",
            title="🧪 Inspect",
            border_style="blue",
        )
    )
    _display_artifact_summary(dataset)

    if export_csv and not show:
        console.print("[red]Error:[/red] Use --export-csv together with --show")
        raise typer.Exit(1)

    if show:
        artifact_name = show.strip().lower()
        if artifact_name not in {"candidates", "accepted", "rejected"}:
            console.print(f"[red]Error:[/red] Unknown artifact bucket: {show}")
            raise typer.Exit(1)
        sorted_examples = _sort_artifact_examples(dataset.artifacts.get(artifact_name, []), sort_by)
        _display_artifact_examples(sorted_examples, artifact_name, limit)
        if export_csv:
            csv_path = _export_artifact_csv(sorted_examples, export_csv)
            console.print(f"\n  CSV export: [cyan]{csv_path}[/cyan]")

    console.print("\n[bold]Discovered files:[/bold]")
    for discovered_file in discovered_files:
        console.print(f"  [cyan]{discovered_file}[/cyan]")


# ─── RUN (FULL PIPELINE) ────────────────────────────────────────────────────


@app.command(hidden=True)
def run(
    seeds: str | None = typer.Option(None, "--seeds", "-s", help="Path to seed JSONL file"),
    domain: str | None = typer.Option(None, "--domain", "-d", help="Domain description"),
    num: int = typer.Option(100, "--num", "-n", help="Number of examples to generate"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Output format"),
    min_quality: float = typer.Option(7.0, "--min-quality", help="Minimum quality score"),
    showcase_summary: bool = typer.Option(
        False, "--showcase-summary", help="Write SHOWCASE_METRICS.md after the run"
    ),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the full non-interactive generation, audit, filter, and export pipeline."""
    setup_logging(verbose)
    cfg = load_config(config)

    if not seeds and not domain:
        console.print("[red]Error:[/red] Provide --seeds or --domain")
        raise typer.Exit(1)

    engine = DatasetEngine(cfg)

    console.print(
        Panel(
            f"Seeds: [cyan]{seeds or 'N/A'}[/cyan]\n"
            f"Domain: [cyan]{domain or 'auto-detect'}[/cyan]\n"
            f"Target: [cyan]{num}[/cyan] examples\n"
            f"Model: [cyan]{cfg.llm.model}[/cyan]\n"
            f"Format: [cyan]{format}[/cyan]",
            title="🧪 Pipeline Configuration",
            border_style="blue",
        )
    )

    pipeline_start = time.time()

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Running full pipeline...", total=None)
        dataset, report, output_files, stage_timings = engine.run_full_pipeline(
            seed_file=seeds,
            domain=domain,
            num_examples=num,
            format=format,
            output_dir=output,
            min_quality=min_quality,
        )
    duration_seconds = time.time() - pipeline_start
    _record_runtime_history(
        cfg,
        num_examples=num,
        has_seeds=bool(seeds),
        duration_seconds=duration_seconds,
        stage_timings=stage_timings,
    )
    from synth_dataset_kit.exporters import export_run_summary

    run_summary_path = export_run_summary(
        {
            "dataset_name": dataset.name,
            "provider": cfg.llm.provider.value,
            "model": cfg.llm.model,
            "input": {
                "seeds": seeds,
                "domain": domain,
                "num_examples": num,
                "format": format,
                "min_quality": min_quality,
            },
            "output_dir": output,
            "generated_examples_retained": int(
                dataset.config_snapshot.get("generated_examples_retained", dataset.size)
            ),
            "seed_examples_included": int(dataset.config_snapshot.get("seed_examples_included", 0)),
            "final_export_examples": dataset.size,
            "examples_retained": dataset.size,
            "avg_quality_score": round(report.avg_quality_score, 4),
            "pass_rate": round(report.passed_examples / max(report.total_examples, 1), 4),
            "contamination_hits": report.contamination_hits,
            "runtime_seconds": round(duration_seconds, 3),
            "runtime_minutes": round(duration_seconds / 60.0, 2),
            "stage_timings": {key: round(float(value), 3) for key, value in stage_timings.items()},
            "output_files": output_files,
        },
        str(Path(output) / "run_summary.json"),
    )
    output_files = [*output_files, run_summary_path]
    if showcase_summary:
        showcase_path = render_showcase_summary(Path(run_summary_path))
        output_files = [*output_files, str(showcase_path)]

    _display_report(report)
    _display_artifact_summary(dataset)
    console.print(f"\n[dim]Actual runtime: {duration_seconds / 60.0:.1f} min[/dim]")
    if stage_timings:
        stage_line = ", ".join(
            f"{label} {stage_timings[key] / 60.0:.1f}m"
            for key, label in [
                ("generate_seconds", "generate"),
                ("audit_seconds", "audit"),
                ("filter_seconds", "filter"),
                ("export_seconds", "export"),
            ]
            if key in stage_timings
        )
        if stage_line:
            console.print(f"[dim]Stage timings: {stage_line}[/dim]")

    console.print("\n[bold]Output files:[/bold]")
    for f in output_files:
        console.print(f"  [cyan]{f}[/cyan]")
    if any(
        name.endswith(("_candidates.jsonl", "_accepted.jsonl", "_rejected.jsonl"))
        for name in output_files
    ):
        console.print("\n  Intermediate artifacts saved for inspection.")
