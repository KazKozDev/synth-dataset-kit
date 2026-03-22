"""Primary CLI commands: init, create, go."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt

from synth_dataset_kit.cli._app import (
    _autoconfigure_model,
    _default_demo_seed_path,
    _estimate_create_duration_minutes,
    _slugify_label,
    app,
    console,
    load_config,
)
from synth_dataset_kit.config import SDKConfig


def _run_demo_path(
    *,
    seeds: str | None,
    domain: str | None,
    num: int,
    format: str | None,
    output: str,
    config: str,
    verbose: bool,
    showcase_summary: bool = False,
) -> None:
    """Run the non-interactive demo path used by `create --demo` and the legacy `go` alias."""
    cfg = load_config(config)
    cfg.generation.domain = domain or cfg.generation.domain or "customer support"
    selected_format = format or cfg.export.format or "jsonl"
    cfg.export.format = selected_format
    cfg = _autoconfigure_model(cfg)
    cfg.to_yaml(config)

    seed_path = Path(seeds) if seeds else None
    if seed_path is None and cfg.generation.domain.lower() in {
        "customer support",
        "customer_support",
        "support",
    }:
        seed_path = _default_demo_seed_path()
    if seed_path is not None and not seed_path.exists():
        console.print(f"[red]Error:[/red] Seed file not found: {seed_path}")
        raise typer.Exit(1)
    if seed_path is None and not domain:
        console.print(
            "[red]Error:[/red] Provide `--domain` or `--seeds`. "
            "Built-in demo seeds are only auto-used for customer support."
        )
        raise typer.Exit(1)

    console.print(
        Panel(
            f"Use case: [cyan]{cfg.generation.domain}[/cyan]\n"
            f"Seeds: [cyan]{seed_path or 'domain-only generation'}[/cyan]\n"
            f"Model: [cyan]{cfg.llm.model}[/cyan]\n"
            f"Target: [cyan]{num}[/cyan] examples\n"
            f"Format: [cyan]{selected_format}[/cyan]\n"
            f"Output: [cyan]{output}[/cyan]",
            title="🧪 Demo Dataset Run",
            border_style="green",
        )
    )

    from synth_dataset_kit.cli._commands_pipeline import run

    run(
        seeds=str(seed_path) if seed_path else None,
        domain=None if seed_path else cfg.generation.domain,
        num=num,
        output=output,
        format=selected_format,
        min_quality=cfg.quality.min_score,
        config=config,
        verbose=verbose,
        showcase_summary=showcase_summary,
    )


@app.command()
def init(
    provider: str = typer.Option(
        "ollama",
        help="LLM provider: ollama, openai, anthropic, vllm, custom",
    ),
    output: str = typer.Option("sdk_config.yaml", help="Config file path"),
):
    """Configure local or API-backed dataset creation in under a minute."""
    config = SDKConfig.default_for_provider(provider)
    ollama_message = ""
    if provider == "ollama":
        try:
            from synth_dataset_kit.llm_client import LLMClient

            client = LLMClient(config.llm)
            recommended = client.recommend_model()
            models = client.list_models()
            if recommended:
                config.llm.model = recommended
                ollama_message = f"\nDetected Ollama models: [cyan]{len(models)}[/cyan]\nRecommended: [cyan]{recommended}[/cyan]"
        except Exception:
            pass
    config.to_yaml(output)

    console.print(
        Panel(
            f"[green]✓[/green] Config created: [bold]{output}[/bold]\n\n"
            f"Provider: [cyan]{provider}[/cyan]\n"
            f"Model: [cyan]{config.llm.model}[/cyan]\n"
            f"API: [cyan]{config.llm.api_base}[/cyan]\n\n"
            f"{ollama_message}\n"
            f"Edit the file to customize, then run:\n"
            f"  [bold]sdk create[/bold]\n"
            f"  [bold]sdk create --demo[/bold]",
            title="🧪 Synth Dataset Kit",
            border_style="green",
        )
    )


@app.command()
def create(
    demo: bool = typer.Option(
        False, "--demo", help="Run the built-in fast demo path without prompts"
    ),
    seeds: str | None = typer.Option(None, "--seeds", "-s", help="Optional seed JSONL file"),
    domain: str | None = typer.Option(None, "--domain", "-d", help="Use case/domain"),
    num: int | None = typer.Option(None, "--num", "-n", help="Number of examples to generate"),
    format: str | None = typer.Option(None, "--format", "-f", help="Output format"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
    showcase_summary: bool = typer.Option(
        False, "--showcase-summary", help="Write SHOWCASE_METRICS.md after the run"
    ),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Turn a small seed set into a reviewed dataset bundle."""
    if demo:
        _run_demo_path(
            seeds=seeds,
            domain=domain,
            num=num,
            format=format,
            output=output or "./output/zero_to_dataset",
            config=config,
            verbose=verbose,
            showcase_summary=showcase_summary,
        )
        return

    cfg = load_config(config)
    cfg.generation.domain = cfg.generation.domain or "customer support"
    cfg = _autoconfigure_model(cfg)
    demo_seed = _default_demo_seed_path()
    selected_format = format or cfg.export.format or "jsonl"
    selected_num = num
    selected_domain = domain
    selected_seeds = seeds
    selected_output = output

    console.print(
        Panel(
            f"Provider: [cyan]{cfg.llm.provider.value}[/cyan]\n"
            f"Model: [cyan]{cfg.llm.model}[/cyan]\n"
            f"Format: [cyan]{selected_format}[/cyan]",
            title="🧪 Create",
            border_style="green",
        )
    )

    if not selected_domain:
        selected_domain = Prompt.ask(
            "1/3 Use case",
            default=cfg.generation.domain or "customer support",
        ).strip()

    if selected_num is None:
        selected_num = IntPrompt.ask(
            "2/3 Examples",
            default=cfg.generation.num_examples,
        )

    if selected_seeds is None:
        default_seed = str(demo_seed) if demo_seed and "support" in selected_domain.lower() else ""
        selected_seeds = Prompt.ask(
            "3/3 Seed file",
            default=default_seed,
            show_default=bool(default_seed),
        ).strip()

    if selected_seeds and not Path(selected_seeds).exists():
        console.print(f"[red]Error:[/red] Seed file not found: {selected_seeds}")
        raise typer.Exit(1)

    if not selected_output:
        output_label = (
            Path(selected_seeds).stem if selected_seeds else _slugify_label(selected_domain)
        )
        selected_output = f"./output/{output_label}_dataset"

    cfg.generation.domain = selected_domain
    cfg.generation.num_examples = selected_num
    cfg.export.format = selected_format
    cfg.export.output_dir = selected_output
    cfg.to_yaml(config)
    eta_low, eta_high, eta_basis, eta_stages = _estimate_create_duration_minutes(
        cfg,
        num_examples=selected_num,
        has_seeds=bool(selected_seeds),
    )

    console.print(
        Panel(
            f"Use case: [cyan]{selected_domain}[/cyan]\n"
            f"Seeds: [cyan]{selected_seeds or 'domain-only generation'}[/cyan]\n"
            f"Examples: [cyan]{selected_num}[/cyan]\n"
            f"Estimated: [cyan]~{eta_low}-{eta_high} min[/cyan]\n"
            f"Output: [cyan]{selected_output}[/cyan]",
            title="🧪 Run",
            border_style="blue",
        )
    )
    console.print(f"[dim]Estimate basis: {eta_basis}[/dim]")
    if eta_stages:
        stage_line = ", ".join(
            f"{label} ~{eta_stages[key]}m"
            for key, label in [
                ("generate_seconds", "generate"),
                ("audit_seconds", "audit"),
                ("filter_seconds", "filter"),
                ("export_seconds", "export"),
            ]
            if key in eta_stages
        )
        if stage_line:
            console.print(f"[dim]Stage split: {stage_line}[/dim]")

    from synth_dataset_kit.cli._commands_pipeline import run

    run(
        seeds=selected_seeds or None,
        domain=None if selected_seeds else selected_domain,
        num=selected_num,
        output=selected_output,
        format=selected_format,
        min_quality=cfg.quality.min_score,
        config=config,
        verbose=verbose,
        showcase_summary=showcase_summary,
    )


@app.command(hidden=True)
def go(
    seeds: str | None = typer.Option(
        None, "--seeds", "-s", help="Seed JSONL file; defaults to built-in demo seeds"
    ),
    domain: str | None = typer.Option(
        None, "--domain", "-d", help="Use case/domain when generating without seeds"
    ),
    num: int = typer.Option(100, "--num", "-n", help="Number of examples to generate"),
    format: str = typer.Option("jsonl", "--format", "-f", help="Output format"),
    output: str = typer.Option(
        "./output/zero_to_dataset", "--output", "-o", help="Output directory"
    ),
    showcase_summary: bool = typer.Option(
        False, "--showcase-summary", help="Write SHOWCASE_METRICS.md after the run"
    ),
    config: str = typer.Option("sdk_config.yaml", "--config", "-c", help="Config file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Legacy alias for the built-in fast demo path."""
    _run_demo_path(
        seeds=seeds,
        domain=domain,
        num=num,
        format=format,
        output=output,
        config=config,
        verbose=verbose,
        showcase_summary=showcase_summary,
    )
