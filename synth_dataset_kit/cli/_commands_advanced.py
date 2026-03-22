"""Advanced CLI commands: export, publish-hf, proof, finetune, uplift, benchmark, health, version."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from synth_dataset_kit import __version__
from synth_dataset_kit.cli._app import _load_dataset_file, app, console, load_config, setup_logging
from synth_dataset_kit.cli._display import (
    _display_baseline_comparison,
    _display_reference_comparison,
    _display_report,
)
from synth_dataset_kit.config import LLMProvider
from synth_dataset_kit.engine import DatasetEngine

# ─── EXPORT ──────────────────────────────────────────────────────────────────


@app.command(name="export", hidden=True)
def export_cmd(
    input_file: str = typer.Argument(..., help="Path to dataset JSONL file"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Output format"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    min_quality: float = typer.Option(0.0, "--min-quality", help="Minimum quality score filter"),
    baseline: str | None = typer.Option(None, "--baseline", help="Optional baseline JSONL file"),
    reference: str | None = typer.Option(
        None, "--reference", help="Optional reference dataset JSONL file"
    ),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Export a dataset to a fine-tuning format."""
    setup_logging(verbose)
    dataset = _load_dataset_file(input_file)
    cfg = load_config(config)
    engine = DatasetEngine(cfg)

    if min_quality > 0:
        original = dataset.size
        dataset = dataset.filter_by_quality(min_quality)
        console.print(f"Filtered: {dataset.size}/{original} examples (min score: {min_quality})")

    from synth_dataset_kit.exporters import export_dataset

    quality_report = baseline_dataset = baseline_report = reference_dataset = reference_report = (
        None
    )

    if format == "huggingface":
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            progress.add_task("Auditing dataset for publish bundle...", total=None)
            quality_report = engine.audit(dataset)
        if baseline:
            baseline_dataset = _load_dataset_file(baseline)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Auditing baseline for publish bundle...", total=None)
                baseline_report = engine.audit(baseline_dataset)
        if reference:
            reference_dataset = _load_dataset_file(reference)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Auditing reference for publish bundle...", total=None)
                reference_report = engine.audit(reference_dataset)

    filepath = export_dataset(
        dataset,
        format,
        output,
        quality_report=quality_report,
        baseline_dataset=baseline_dataset,
        baseline_report=baseline_report,
        reference_dataset=reference_dataset,
        reference_report=reference_report,
    )
    console.print(f"[green]✓[/green] Exported to [cyan]{filepath}[/cyan]")


# ─── PUBLISH-HF ──────────────────────────────────────────────────────────────


@app.command(name="publish-hf", hidden=True)
def publish_hf(
    input_file: str = typer.Argument(..., help="Path to dataset JSONL file"),
    repo_id: str = typer.Option(..., "--repo-id", help="Hugging Face dataset repo id"),
    output: str = typer.Option(
        "./output/publish", "--output", "-o", help="Local publish bundle output directory"
    ),
    min_quality: float = typer.Option(7.5, "--min-quality", help="Minimum quality score to keep"),
    baseline: str | None = typer.Option(None, "--baseline", help="Optional baseline JSONL file"),
    reference: str | None = typer.Option(
        None, "--reference", help="Optional reference dataset JSONL file"
    ),
    token: str | None = typer.Option(
        None, "--token", help="Hugging Face token; falls back to HF_TOKEN"
    ),
    private: bool = typer.Option(
        False, "--private/--public", help="Create private or public dataset repo"
    ),
    push: bool = typer.Option(
        True, "--push/--plan-only", help="Upload bundle now or only prepare it locally"
    ),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Build a Hugging Face dataset bundle and optionally publish it."""
    setup_logging(verbose)
    cfg = load_config(config)
    engine = DatasetEngine(cfg)
    dataset = _load_dataset_file(input_file)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Auditing dataset for Hugging Face publish...", total=None)
        report = engine.audit(dataset)
    if min_quality > 0:
        original_size = dataset.size
        dataset = dataset.filter_by_quality(min_quality)
        if dataset.size != original_size:
            console.print(
                f"Filtered publish dataset: {dataset.size}/{original_size} examples (min score: {min_quality})"
            )
            report = engine.audit(dataset)

    baseline_dataset = baseline_report = reference_dataset = reference_report = None
    if baseline:
        baseline_dataset = _load_dataset_file(baseline)
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            progress.add_task("Auditing baseline dataset...", total=None)
            baseline_report = engine.audit(baseline_dataset)
    if reference:
        reference_dataset = _load_dataset_file(reference)
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            progress.add_task("Auditing reference dataset...", total=None)
            reference_report = engine.audit(reference_dataset)

    from synth_dataset_kit.exporters import export_huggingface_bundle
    from synth_dataset_kit.publishing import (
        build_publish_manifest,
        publish_huggingface_bundle,
        write_publish_manifest,
    )

    bundle_files = export_huggingface_bundle(
        dataset,
        output,
        include_metadata=True,
        quality_report=report,
        baseline_dataset=baseline_dataset,
        baseline_report=baseline_report,
        reference_dataset=reference_dataset,
        reference_report=reference_report,
    )
    bundle_dir = bundle_files[0]

    try:
        if push:
            manifest = publish_huggingface_bundle(
                bundle_dir=bundle_dir, repo_id=repo_id, token=token, private=private
            )
        else:
            file_count = sum(1 for path in Path(bundle_dir).rglob("*") if path.is_file())
            manifest = build_publish_manifest(
                repo_id=repo_id,
                bundle_dir=bundle_dir,
                private=private,
                pushed=False,
                uploaded_files=file_count,
            )
        manifest_path = write_publish_manifest(bundle_dir, manifest)
    except RuntimeError as exc:
        console.print(f"[yellow]Publish skipped:[/yellow] {exc}")
        manifest = build_publish_manifest(
            repo_id=repo_id,
            bundle_dir=bundle_dir,
            private=private,
            pushed=False,
            uploaded_files=sum(1 for path in Path(bundle_dir).rglob("*") if path.is_file()),
        )
        manifest_path = write_publish_manifest(bundle_dir, manifest)

    console.print(
        Panel(
            f"Bundle: [cyan]{bundle_dir}[/cyan]\nRepo: [cyan]{repo_id}[/cyan]\n"
            f"Pushed: [cyan]{manifest['pushed']}[/cyan]\nURL: [cyan]{manifest['dataset_url']}[/cyan]\n"
            f"Manifest: [cyan]{manifest_path}[/cyan]",
            title="🧪 Hugging Face Publish",
            border_style="green" if manifest["pushed"] else "yellow",
        )
    )


# ─── PROOF ───────────────────────────────────────────────────────────────────


@app.command(hidden=True)
def proof(
    input_file: str = typer.Argument(..., help="Path to generated dataset JSONL file"),
    base_model: str = typer.Option("llama3.1:8b", "--base-model", help="Base model to fine-tune"),
    trainer: str = typer.Option("unsloth", "--trainer", help="Training stack name"),
    holdout: str | None = typer.Option(None, "--holdout", help="Optional holdout JSONL file"),
    baseline: str | None = typer.Option(None, "--baseline", help="Optional baseline JSONL file"),
    reference: str | None = typer.Option(
        None, "--reference", help="Optional reference dataset JSONL file"
    ),
    output: str = typer.Option(
        "./output/proof", "--output", "-o", help="Proof bundle output directory"
    ),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Build a reproducible fine-tune/eval proof bundle."""
    setup_logging(verbose)
    cfg = load_config(config)
    engine = DatasetEngine(cfg)
    dataset = _load_dataset_file(input_file)
    holdout_path = Path(holdout) if holdout else Path("examples/customer_support_holdout.jsonl")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Auditing generated dataset...", total=None)
        report = engine.audit(dataset)

    baseline_dataset = baseline_report = reference_dataset = reference_report = None
    if baseline:
        baseline_dataset = _load_dataset_file(baseline)
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            progress.add_task("Auditing baseline dataset...", total=None)
            baseline_report = engine.audit(baseline_dataset)
    if reference:
        reference_dataset = _load_dataset_file(reference)
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            progress.add_task("Auditing reference dataset...", total=None)
            reference_report = engine.audit(reference_dataset)

    _display_report(report)
    if baseline_dataset and baseline_report:
        _display_baseline_comparison(dataset, report, baseline_dataset, baseline_report)
    if reference_dataset and reference_report:
        _display_reference_comparison(dataset, report, reference_dataset, reference_report)

    from synth_dataset_kit.exporters import export_proof_bundle

    proof_files = export_proof_bundle(
        dataset,
        report,
        output,
        base_model=base_model,
        trainer=trainer,
        holdout_path=str(holdout_path) if holdout_path.exists() else None,
        baseline_dataset=baseline_dataset,
        baseline_report=baseline_report,
        reference_dataset=reference_dataset,
        reference_report=reference_report,
    )
    console.print("\n[bold]Proof bundle:[/bold]")
    for filepath in proof_files:
        console.print(f"  [cyan]{filepath}[/cyan]")


# ─── FINETUNE ────────────────────────────────────────────────────────────────


@app.command(hidden=True)
def finetune(
    dataset_file: str = typer.Argument(..., help="Path to training dataset JSONL file"),
    base_model: str = typer.Option(..., "--base-model", help="Base model to fine-tune"),
    trainer: str = typer.Option("unsloth", "--trainer", help="Training stack to use"),
    output: str = typer.Option(
        "./output/finetune", "--output", "-o", help="Fine-tune output directory"
    ),
    epochs: int = typer.Option(1, "--epochs", min=1, help="Number of training epochs"),
    learning_rate: float = typer.Option(2e-4, "--learning-rate", help="Learning rate"),
    batch_size: int = typer.Option(2, "--batch-size", min=1, help="Per-device batch size"),
    gradient_accumulation_steps: int = typer.Option(
        4, "--grad-accum", min=1, help="Gradient accumulation steps"
    ),
    max_seq_length: int = typer.Option(
        2048, "--max-seq-length", min=256, help="Max sequence length"
    ),
    execute: bool = typer.Option(
        True, "--execute/--plan-only", help="Run training immediately or only write the job plan"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run an optional in-repo fine-tune workflow."""
    setup_logging(verbose)
    if not Path(dataset_file).exists():
        console.print(f"[red]Error:[/red] Dataset file not found: {dataset_file}")
        raise typer.Exit(1)

    from synth_dataset_kit.training import TrainingJob, run_training_job, save_training_job

    job = TrainingJob(
        dataset_path=dataset_file,
        base_model=base_model,
        output_dir=output,
        trainer=trainer,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
    )
    job_files = save_training_job(job)

    console.print(
        Panel(
            f"Trainer: [cyan]{trainer}[/cyan]\nBase model: [cyan]{base_model}[/cyan]\n"
            f"Dataset: [cyan]{dataset_file}[/cyan]\nOutput: [cyan]{output}[/cyan]\n"
            f"Execute now: [cyan]{execute}[/cyan]",
            title="🧪 Fine-Tune Job",
            border_style="blue",
        )
    )

    if not execute:
        console.print("\n[bold]Training job files:[/bold]")
        for filepath in job_files:
            console.print(f"  [cyan]{filepath}[/cyan]")
        return

    try:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            progress.add_task("Running fine-tune job...", total=None)
            result = run_training_job(job)
    except RuntimeError as exc:
        console.print(f"[red]Training unavailable:[/red] {exc}")
        console.print("\n[bold]Training job files:[/bold]")
        for filepath in job_files:
            console.print(f"  [cyan]{filepath}[/cyan]")
        raise typer.Exit(1) from exc

    console.print(
        f"[green]✓[/green] Fine-tune complete. Model saved to [cyan]{result['model_dir']}[/cyan]"
    )
    console.print(f"  Metrics: [cyan]{result['metrics_path']}[/cyan]")
    console.print("\n[bold]Training job files:[/bold]")
    for filepath in job_files:
        console.print(f"  [cyan]{filepath}[/cyan]")


# ─── UPLIFT ──────────────────────────────────────────────────────────────────


@app.command(hidden=True)
def uplift(
    base_model: str = typer.Option(..., "--base-model", help="Base model to evaluate"),
    finetuned_model: str = typer.Option(
        ..., "--finetuned-model", help="Fine-tuned model to evaluate"
    ),
    holdout: str = typer.Option(
        "examples/customer_support_holdout.jsonl", "--holdout", help="Holdout JSONL file"
    ),
    output: str = typer.Option("./output/uplift", "--output", "-o", help="Uplift output directory"),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Evaluate base vs fine-tuned model on the same support holdout and write uplift artifacts."""
    setup_logging(verbose)
    cfg = load_config(config)
    if not Path(holdout).exists():
        console.print(f"[red]Error:[/red] Holdout file not found: {holdout}")
        raise typer.Exit(1)

    base_cfg = cfg.model_copy(deep=True)
    finetuned_cfg = cfg.model_copy(deep=True)
    base_cfg.llm.model = base_model
    finetuned_cfg.llm.model = finetuned_model

    from synth_dataset_kit.evaluation import compare_models_on_holdout, export_uplift_results

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Evaluating base and fine-tuned models on holdout...", total=None)
        results = compare_models_on_holdout(base_cfg, finetuned_cfg, holdout)

    files = export_uplift_results(
        results,
        output,
        name=f"{Path(holdout).stem}_{base_model.replace('/', '_')}_vs_{finetuned_model.replace('/', '_')}",
    )

    uplift_metrics = results["uplift"]
    table = Table(title="Uplift Results", show_header=False, box=None)
    table.add_column(style="cyan", width=28)
    table.add_column()
    table.add_row("Base Model", base_model)
    table.add_row("Fine-Tuned Model", finetuned_model)
    table.add_row("Task Success Delta", f"{uplift_metrics['task_success_rate_delta']:+.2%}")
    table.add_row("Pass Rate Delta", f"{uplift_metrics['pass_rate_delta']:+.2%}")
    table.add_row("Token F1 Delta", f"{uplift_metrics['avg_token_f1_delta']:+.4f}")
    table.add_row("Empathy Delta", f"{uplift_metrics['avg_empathy_score_delta']:+.4f}")
    console.print(table)

    console.print("\n[bold]Uplift outputs:[/bold]")
    for filepath in files:
        console.print(f"  [cyan]{filepath}[/cyan]")


# ─── BENCHMARK ───────────────────────────────────────────────────────────────


def _select_benchmark_models(
    available_models: list[dict[str, object]],
    domain: str,
    requested_models: str | None = None,
    top_n: int = 3,
    recommended_model: str | None = None,
) -> list[str]:
    """Choose which installed models to benchmark."""
    available_names = [
        str(model.get("name", "")).strip()
        for model in available_models
        if str(model.get("name", "")).strip()
    ]
    if requested_models:
        requested = [item.strip() for item in requested_models.split(",") if item.strip()]
        return [name for name in requested if name in available_names]

    selected: list[str] = []
    preferred = recommended_model
    if not preferred and available_names:
        preferred = available_names[0]
    if preferred:
        selected.append(preferred)

    domain_lower = domain.lower()
    sorted_names = sorted(
        available_names,
        key=lambda name: (
            "coder" in name.lower() and "code" not in domain_lower,
            name != preferred,
            name,
        ),
    )
    for name in sorted_names:
        if name not in selected:
            selected.append(name)
        if len(selected) >= top_n:
            break
    return selected[:top_n]


def _recommend_benchmark_result(results: list[dict[str, object]]) -> dict[str, object] | None:
    """Recommend the best model from successful benchmark runs."""
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        return None

    def _normalize(values: list[float], value: float) -> float:
        low, high = min(values), max(values)
        if abs(high - low) < 1e-9:
            return 1.0
        return (value - low) / (high - low)

    quality_values = [float(item["avg_quality_score"]) for item in successful]
    pass_rate_values = [float(item["pass_rate"]) for item in successful]
    speed_values = [float(item["examples_per_second"]) for item in successful]

    ranked: list[dict[str, object]] = []
    for item in successful:
        composite = (
            _normalize(quality_values, float(item["avg_quality_score"])) * 0.5
            + _normalize(pass_rate_values, float(item["pass_rate"])) * 0.35
            + _normalize(speed_values, float(item["examples_per_second"])) * 0.15
        )
        ranked.append({**item, "benchmark_score": round(composite, 4)})

    return max(
        ranked,
        key=lambda item: (
            float(item["benchmark_score"]),
            float(item["avg_quality_score"]),
            float(item["pass_rate"]),
        ),
    )


def _display_benchmark_results(
    results: list[dict[str, object]], recommendation: dict[str, object] | None
) -> None:
    """Display benchmark leaderboard in the terminal."""
    table = Table(title="Benchmark Leaderboard", show_header=True, header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Status")
    table.add_column("Quality", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Contam", justify="right")

    for result in results:
        if result.get("status") != "ok":
            table.add_row(
                str(result.get("model", "unknown")), "[red]error[/red]", "-", "-", "-", "-"
            )
            continue
        table.add_row(
            str(result["model"]),
            "[green]ok[/green]",
            f"{float(result['avg_quality_score']):.2f}",
            f"{float(result['pass_rate']):.2%}",
            f"{float(result['examples_per_second']):.2f}/s",
            str(result["contamination_hits"]),
        )

    console.print(table)
    if recommendation:
        console.print(
            Panel(
                f"Recommended model: [bold]{recommendation['model']}[/bold]\n"
                f"Benchmark score: [cyan]{recommendation['benchmark_score']:.4f}[/cyan]\n"
                f"Quality: [cyan]{recommendation['avg_quality_score']:.2f}[/cyan]\n"
                f"Pass rate: [cyan]{recommendation['pass_rate']:.2%}[/cyan]\n"
                f"Speed: [cyan]{recommendation['examples_per_second']:.2f}/s[/cyan]",
                title="Recommendation",
                border_style="green",
            )
        )


def _export_benchmark_summary(
    results: list[dict[str, object]],
    recommendation: dict[str, object] | None,
    output_dir: str,
    seeds: str,
    domain: str,
    num: int,
) -> list[str]:
    """Write benchmark summary files for later review."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain": domain,
        "seed_file": seeds,
        "examples_per_model": num,
        "results": results,
        "recommended_model": recommendation["model"] if recommendation else None,
        "recommendation": recommendation,
    }
    json_path = output_dir_path / "benchmark_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines = [
        "# Ollama Benchmark Summary",
        "",
        f"- Domain: {domain}",
        f"- Seed file: {seeds}",
        f"- Examples per model: {num}",
        "",
        "## Results",
        "",
    ]
    for result in results:
        if result.get("status") != "ok":
            md_lines.append(f"- {result.get('model')}: error")
            continue
        md_lines.append(
            f"- {result['model']}: quality={float(result['avg_quality_score']):.2f}, "
            f"pass_rate={float(result['pass_rate']):.2%}, "
            f"speed={float(result['examples_per_second']):.2f}/s, "
            f"contamination_hits={result['contamination_hits']}"
        )
    if recommendation:
        md_lines.extend(
            [
                "",
                "## Recommendation",
                "",
                f"- Recommended model: {recommendation['model']}",
                f"- Benchmark score: {float(recommendation['benchmark_score']):.4f}",
            ]
        )

    md_path = output_dir_path / "benchmark_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return [str(json_path), str(md_path)]


@app.command(hidden=True)
def benchmark(
    seeds: str = typer.Option(..., "--seeds", "-s", help="Path to seed JSONL file"),
    domain: str = typer.Option("customer support", "--domain", "-d", help="Use case/domain"),
    num: int = typer.Option(30, "--num", "-n", help="Examples to generate per model"),
    models: str | None = typer.Option(
        None, "--models", help="Comma-separated Ollama models to benchmark"
    ),
    top_n: int = typer.Option(
        3, "--top-n", help="How many installed models to benchmark by default"
    ),
    output: str = typer.Option(
        "./output/benchmarks", "--output", "-o", help="Benchmark output directory"
    ),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Benchmark local Ollama models on the same seed-expansion task."""
    setup_logging(verbose)
    cfg = load_config(config)
    if cfg.llm.provider != LLMProvider.OLLAMA:
        console.print(
            "[yellow]Warning:[/yellow] Benchmarking is tuned for Ollama; continuing with current provider config."
        )

    if not Path(seeds).exists():
        console.print(f"[red]Error:[/red] Seed file not found: {seeds}")
        raise typer.Exit(1)

    client = DatasetEngine(cfg).client
    selected_models = _select_benchmark_models(
        client.list_models(),
        domain=domain,
        requested_models=models,
        top_n=top_n,
        recommended_model=client.recommend_model(domain),
    )
    if not selected_models:
        console.print("[red]Error:[/red] No benchmark models available.")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"Seeds: [cyan]{seeds}[/cyan]\nDomain: [cyan]{domain}[/cyan]\n"
            f"Examples per model: [cyan]{num}[/cyan]\n"
            f"Models: [cyan]{', '.join(selected_models)}[/cyan]",
            title="🧪 Ollama Benchmark",
            border_style="blue",
        )
    )

    results: list[dict[str, object]] = []
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        for model_name in selected_models:
            task = progress.add_task(f"Benchmarking {model_name}...", total=None)
            cfg_copy = cfg.model_copy(deep=True)
            cfg_copy.llm.model = model_name
            cfg_copy.generation.domain = domain
            engine = DatasetEngine(cfg_copy)
            try:
                started = time.perf_counter()
                dataset = engine.generate_from_seeds(seeds, num)
                report = engine.audit(dataset)
                elapsed = time.perf_counter() - started
                results.append(
                    {
                        "model": model_name,
                        "status": "ok",
                        "examples": dataset.size,
                        "elapsed_seconds": round(elapsed, 3),
                        "examples_per_second": round(dataset.size / max(elapsed, 1e-6), 3),
                        "avg_quality_score": round(report.avg_quality_score, 4),
                        "pass_rate": round(
                            report.passed_examples / max(report.total_examples, 1), 4
                        ),
                        "lexical_diversity": round(report.lexical_diversity, 4),
                        "contamination_hits": report.contamination_hits,
                    }
                )
            except Exception as exc:
                results.append({"model": model_name, "status": "error", "error": str(exc)})
            finally:
                progress.remove_task(task)

    recommendation = _recommend_benchmark_result(results)
    _display_benchmark_results(results, recommendation)
    recommendation_path = None
    if recommendation:
        recommendation_path = client.save_benchmark_recommendation(domain, recommendation)
    output_files = _export_benchmark_summary(results, recommendation, output, seeds, domain, num)
    console.print("\n[bold]Benchmark outputs:[/bold]")
    for filepath in output_files:
        console.print(f"  [cyan]{filepath}[/cyan]")
    if recommendation_path:
        console.print(f"  Saved recommendation cache: [cyan]{recommendation_path}[/cyan]")


# ─── HEALTH ──────────────────────────────────────────────────────────────────


@app.command(hidden=True)
def health(
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
):
    """Check connectivity to the configured model endpoint."""
    setup_logging()
    cfg = load_config(config)
    engine = DatasetEngine(cfg)

    console.print(
        f"Checking [cyan]{cfg.llm.provider.value}[/cyan] at [cyan]{cfg.llm.api_base}[/cyan]..."
    )

    if engine.client.health_check():
        console.print(f"[green]✓[/green] Connected! Model: [bold]{cfg.llm.model}[/bold]")
        models = engine.client.list_models()
        if cfg.llm.provider.value == "ollama" and models:
            model_names = ", ".join(model["name"] for model in models[:5])
            recommendation = engine.client.recommend_model(cfg.generation.domain)
            console.print(f"  Installed models: [cyan]{model_names}[/cyan]")
            if recommendation:
                console.print(f"  Recommended model: [bold]{recommendation}[/bold]")
    else:
        console.print("[red]✗[/red] Connection failed.")
        console.print(f"  Make sure your LLM is running at {cfg.llm.api_base}")
        if cfg.llm.provider.value == "ollama":
            console.print(
                "  Try: [bold]ollama serve[/bold] and [bold]ollama pull llama3.1:8b[/bold]"
            )


# ─── VERSION ─────────────────────────────────────────────────────────────────


@app.command(hidden=True)
def version():
    """Show version."""
    console.print(f"synth-dataset-kit v{__version__}")
