"""CLI interface for synth-dataset-kit.

Commands:
  sdk init          — Create a config file with sensible defaults
  sdk generate      — Generate a synthetic dataset
  sdk audit         — Run quality + decontamination checks
  sdk export        — Export dataset to a fine-tuning format
  sdk run           — Full pipeline: generate → audit → filter → export
  sdk health        — Check LLM endpoint connectivity
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from synth_dataset_kit import __version__
from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.engine import DatasetEngine
from synth_dataset_kit.models import Dataset

app = typer.Typer(
    name="sdk",
    help="🧪 Synth Dataset Kit — Generate high-quality synthetic datasets for LLM fine-tuning",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def load_config(config_path: str = "sdk_config.yaml") -> SDKConfig:
    p = Path(config_path)
    if p.exists():
        return SDKConfig.from_yaml(p)
    return SDKConfig()


# ─── INIT ────────────────────────────────────────────────────────────────────


@app.command()
def init(
    provider: str = typer.Option(
        "ollama",
        help="LLM provider: ollama, openai, anthropic, vllm, custom",
    ),
    output: str = typer.Option("sdk_config.yaml", help="Config file path"),
):
    """Create a configuration file with sensible defaults."""
    config = SDKConfig.default_for_provider(provider)
    config.to_yaml(output)

    console.print(
        Panel(
            f"[green]✓[/green] Config created: [bold]{output}[/bold]\n\n"
            f"Provider: [cyan]{provider}[/cyan]\n"
            f"Model: [cyan]{config.llm.model}[/cyan]\n"
            f"API: [cyan]{config.llm.api_base}[/cyan]\n\n"
            f"Edit the file to customize, then run:\n"
            f"  [bold]sdk generate --domain 'customer support'[/bold]\n"
            f"  [bold]sdk generate --seeds my_data.jsonl[/bold]",
            title="🧪 Synth Dataset Kit",
            border_style="green",
        )
    )


# ─── GENERATE ────────────────────────────────────────────────────────────────


@app.command()
def generate(
    seeds: str | None = typer.Option(None, "--seeds", "-s", help="Path to seed JSONL file"),
    domain: str | None = typer.Option(None, "--domain", "-d", help="Domain description"),
    num: int = typer.Option(100, "--num", "-n", help="Number of examples to generate"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Output format: jsonl, alpaca, sharegpt, chatml"),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a synthetic dataset from seeds or domain description."""
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

    # Quick export (without audit)
    from synth_dataset_kit.exporters import export_dataset
    filepath = export_dataset(dataset, format, output)

    console.print(f"\n[green]✓[/green] Generated [bold]{dataset.size}[/bold] examples")
    console.print(f"  Output: [cyan]{filepath}[/cyan]")
    console.print(f"\n  Run [bold]sdk audit {filepath}[/bold] to check quality")


# ─── AUDIT ───────────────────────────────────────────────────────────────────


@app.command()
def audit(
    input_file: str = typer.Argument(..., help="Path to dataset JSONL file"),
    output: str = typer.Option("./output", "--output", "-o", help="Report output directory"),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run quality scoring and decontamination on a dataset."""
    setup_logging(verbose)
    cfg = load_config(config)
    engine = DatasetEngine(cfg)

    # Load dataset
    dataset = _load_dataset_file(input_file)
    console.print(f"Loaded [bold]{dataset.size}[/bold] examples from {input_file}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Auditing dataset...", total=None)
        report = engine.audit(dataset)

    # Display report
    _display_report(report)

    # Export HTML report
    from synth_dataset_kit.exporters import export_quality_report_html
    report_path = str(Path(output) / f"{dataset.name}_quality_report.html")
    export_quality_report_html(report, report_path)
    console.print(f"\n  Full report: [cyan]{report_path}[/cyan]")


# ─── EXPORT ──────────────────────────────────────────────────────────────────


@app.command(name="export")
def export_cmd(
    input_file: str = typer.Argument(..., help="Path to dataset JSONL file"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Output format"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    min_quality: float = typer.Option(0.0, "--min-quality", help="Minimum quality score filter"),
):
    """Export a dataset to a fine-tuning format."""
    setup_logging()
    dataset = _load_dataset_file(input_file)

    if min_quality > 0:
        original = dataset.size
        dataset = dataset.filter_by_quality(min_quality)
        console.print(f"Filtered: {dataset.size}/{original} examples (min score: {min_quality})")

    from synth_dataset_kit.exporters import export_dataset
    filepath = export_dataset(dataset, format, output)
    console.print(f"[green]✓[/green] Exported to [cyan]{filepath}[/cyan]")


# ─── RUN (FULL PIPELINE) ────────────────────────────────────────────────────


@app.command()
def run(
    seeds: str | None = typer.Option(None, "--seeds", "-s", help="Path to seed JSONL file"),
    domain: str | None = typer.Option(None, "--domain", "-d", help="Domain description"),
    num: int = typer.Option(100, "--num", "-n", help="Number of examples to generate"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Output format"),
    min_quality: float = typer.Option(7.0, "--min-quality", help="Minimum quality score"),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Full pipeline: generate → audit → filter → export."""
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running full pipeline...", total=None)
        dataset, report, output_files, _ = engine.run_full_pipeline(
            seed_file=seeds,
            domain=domain,
            num_examples=num,
            format=format,
            output_dir=output,
            min_quality=min_quality,
        )

    _display_report(report)

    console.print("\n[bold]Output files:[/bold]")
    for f in output_files:
        console.print(f"  [cyan]{f}[/cyan]")


# ─── HEALTH CHECK ────────────────────────────────────────────────────────────


@app.command()
def health(
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
):
    """Check LLM endpoint connectivity."""
    setup_logging()
    cfg = load_config(config)
    engine = DatasetEngine(cfg)

    console.print(f"Checking [cyan]{cfg.llm.provider.value}[/cyan] at [cyan]{cfg.llm.api_base}[/cyan]...")

    if engine.client.health_check():
        console.print(f"[green]✓[/green] Connected! Model: [bold]{cfg.llm.model}[/bold]")
    else:
        console.print("[red]✗[/red] Connection failed.")
        console.print(f"  Make sure your LLM is running at {cfg.llm.api_base}")
        if cfg.llm.provider.value == "ollama":
            console.print("  Try: [bold]ollama serve[/bold] and [bold]ollama pull llama3.1:8b[/bold]")


# ─── VERSION ─────────────────────────────────────────────────────────────────


@app.command()
def version():
    """Show version."""
    console.print(f"synth-dataset-kit v{__version__}")


# ─── HELPERS ─────────────────────────────────────────────────────────────────


def _load_dataset_file(path: str) -> Dataset:
    """Load a dataset from JSONL file."""
    from synth_dataset_kit.generators.seed_expander import load_seed_file

    examples = load_seed_file(path)
    name = Path(path).stem
    return Dataset(name=name, examples=examples)


def _display_report(report):
    """Display a quality report in the terminal."""
    table = Table(title="Quality Report", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    quality_color = "green" if report.avg_quality_score >= 7 else "yellow" if report.avg_quality_score >= 5 else "red"

    table.add_row("Total Examples", str(report.total_examples))
    table.add_row("Passed", f"[green]{report.passed_examples}[/green]")
    table.add_row("Failed", f"[red]{report.failed_examples}[/red]")
    table.add_row("Avg Quality", f"[{quality_color}]{report.avg_quality_score:.1f}[/{quality_color}]")
    table.add_row("Diversity", f"{report.diversity_score:.2f}")
    table.add_row("Avg User Length", f"{report.avg_user_length:.0f} words")
    table.add_row("Avg Response Length", f"{report.avg_assistant_length:.0f} words")

    if report.contamination_hits > 0:
        table.add_row(
            "Contamination",
            f"[red]{report.contamination_hits} hits ({', '.join(report.contaminated_benchmarks)})[/red]",
        )
    else:
        table.add_row("Contamination", "[green]Clean[/green]")

    console.print(table)


if __name__ == "__main__":
    app()
