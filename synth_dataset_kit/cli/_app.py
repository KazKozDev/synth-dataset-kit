"""Shared Typer app instance, console, and small helpers used across CLI sub-modules."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from synth_dataset_kit.config import LLMProvider, SDKConfig
from synth_dataset_kit.models import Dataset

app = typer.Typer(
    name="sdk",
    help="🧪 Synth Dataset Kit — Turn a small seed set into a reviewed fine-tuning dataset.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


# ─── LOGGING / CONFIG ────────────────────────────────────────────────────────


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging with a Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def load_config(config_path: str = "sdk_config.yaml") -> SDKConfig:
    """Load or create a default SDK configuration."""
    p = Path(config_path)
    if p.exists():
        return SDKConfig.from_yaml(p)
    return SDKConfig()


# ─── MODEL AUTO-CONFIG ────────────────────────────────────────────────────────


def _default_demo_seed_path() -> Path | None:
    """Return the built-in customer-support demo seed file when available."""
    candidate = Path("examples/customer_support_seeds.jsonl")
    return candidate if candidate.exists() else None


def _autoconfigure_model(cfg: SDKConfig) -> SDKConfig:
    """Prefer a recommended local model without asking the user to choose one."""
    if cfg.llm.provider != LLMProvider.OLLAMA:
        return cfg
    try:
        from synth_dataset_kit.llm_client import LLMClient

        client = LLMClient(cfg.llm)
        recommended = client.recommend_model(cfg.generation.domain or "customer support")
        if recommended:
            cfg.llm.model = recommended
    except Exception:
        pass
    return cfg


def _slugify_label(value: str) -> str:
    """Convert a free-form label to a filesystem-friendly slug."""
    slug = value.strip().lower().replace("/", "_").replace(" ", "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "dataset"


# ─── RUNTIME HISTORY ─────────────────────────────────────────────────────────


def _runtime_history_path() -> Path:
    """Path to the local create-run history cache."""
    return Path(".sdk_cache/runtime/create_history.json")


def _load_runtime_history() -> list[dict[str, object]]:
    """Load previously recorded pipeline run timings."""
    path = _runtime_history_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _save_runtime_history(entries: list[dict[str, object]]) -> None:
    """Persist the last 200 runtime history entries."""
    path = _runtime_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries[-200:], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _record_runtime_history(
    cfg: SDKConfig,
    *,
    num_examples: int,
    has_seeds: bool,
    duration_seconds: float,
    stage_timings: dict[str, float] | None = None,
) -> None:
    """Append a new runtime entry after a pipeline run."""
    entries = _load_runtime_history()
    entries.append(
        {
            "provider": cfg.llm.provider.value,
            "model": cfg.llm.model,
            "num_examples": num_examples,
            "has_seeds": has_seeds,
            "duration_seconds": round(duration_seconds, 2),
            "stage_timings": {
                key: round(float(value), 3) for key, value in (stage_timings or {}).items()
            },
            "updated_at": int(time.time()),
        }
    )
    _save_runtime_history(entries)


def _estimate_from_runtime_history(
    cfg: SDKConfig,
    *,
    num_examples: int,
    has_seeds: bool,
) -> tuple[int, int, str, dict[str, int]] | None:
    """Derive an ETA from cached past runs for the same provider/model."""
    entries = _load_runtime_history()
    relevant = [
        item
        for item in entries
        if str(item.get("provider")) == cfg.llm.provider.value
        and str(item.get("model")) == cfg.llm.model
        and bool(item.get("has_seeds")) == has_seeds
    ]
    if not relevant:
        return None

    relevant.sort(key=lambda item: abs(int(item.get("num_examples", num_examples)) - num_examples))
    sample = relevant[:5]
    projected_minutes: list[float] = []
    projected_stage_minutes: dict[str, list[float]] = {}
    for item in sample:
        past_examples = max(int(item.get("num_examples", num_examples)), 1)
        past_seconds = max(float(item.get("duration_seconds", 0.0)), 1.0)
        scale = num_examples / past_examples
        projected_minutes.append((past_seconds / 60.0) * scale)
        raw_stage_timings = item.get("stage_timings", {})
        if isinstance(raw_stage_timings, dict):
            for stage_name, stage_seconds in raw_stage_timings.items():
                try:
                    projected_value = (float(stage_seconds) / 60.0) * scale
                except (TypeError, ValueError):
                    continue
                projected_stage_minutes.setdefault(str(stage_name), []).append(projected_value)

    if not projected_minutes:
        return None

    projected_minutes.sort()
    center = projected_minutes[len(projected_minutes) // 2]
    low = max(1, round(center * 0.85))
    high = max(low + 1, round(center * 1.2))
    stage_estimate_minutes: dict[str, int] = {}
    for stage_name, values in projected_stage_minutes.items():
        values.sort()
        stage_estimate_minutes[stage_name] = max(1, round(values[len(values) // 2]))
    return low, high, f"{len(sample)} local run(s)", stage_estimate_minutes


def _estimate_create_duration_minutes(
    cfg: SDKConfig,
    *,
    num_examples: int,
    has_seeds: bool,
) -> tuple[int, int, str, dict[str, int]]:
    """Return a rough min/max duration estimate for create flows.

    The estimate is intentionally heuristic. If the user has benchmarked models
    locally, we could replace this with real throughput later, but for now the
    goal is to set expectations honestly rather than imply precision.
    """
    empirical = _estimate_from_runtime_history(
        cfg,
        num_examples=num_examples,
        has_seeds=has_seeds,
    )
    if empirical is not None:
        return empirical

    model_name = cfg.llm.model.lower()
    provider = cfg.llm.provider.value

    if provider == "ollama":
        if any(token in model_name for token in ["70b", "72b", "qwq", "mixtral"]):
            per_100 = (7.0, 12.0)
        elif any(token in model_name for token in ["32b", "34b", "27b"]):
            per_100 = (5.0, 9.0)
        elif any(token in model_name for token in ["13b", "14b"]):
            per_100 = (3.5, 6.5)
        else:
            per_100 = (2.5, 5.0)
    elif provider in {"openai", "anthropic"}:
        per_100 = (1.0, 2.5)
    else:
        per_100 = (1.5, 4.0)

    multiplier = max(num_examples, 25) / 100.0
    low = per_100[0] * multiplier
    high = per_100[1] * multiplier

    if has_seeds:
        low *= 1.1
        high *= 1.25
    else:
        low *= 0.9
        high *= 1.0

    low = max(1, round(low))
    high = max(low + 1, round(high))
    basis = f"{provider}:{cfg.llm.model} heuristic"
    return low, high, basis, {}


# ─── DATASET LOADING ─────────────────────────────────────────────────────────


def _load_dataset_file(path: str) -> Dataset:
    """Load a dataset from JSONL file."""
    from synth_dataset_kit.generators.seed_expander import load_seed_file

    examples = load_seed_file(path)
    name = Path(path).stem
    return Dataset(name=name, examples=examples)
