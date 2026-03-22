"""Terminal display helpers for quality reports, comparisons, and artifact previews."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from synth_dataset_kit.cli._app import console
from synth_dataset_kit.models import Dataset, Example

# ─── QUALITY REPORT ──────────────────────────────────────────────────────────


def _display_report(report) -> None:
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
    table.add_row("Near Duplicates", str(report.near_duplicate_examples))
    table.add_row("Lexical Diversity", f"{report.lexical_diversity:.2f}")
    table.add_row("Self-BLEU Proxy", f"{report.self_bleu_proxy:.2f}")
    table.add_row("Embedding Diversity", "n/a" if report.embedding_diversity_score is None else f"{report.embedding_diversity_score:.2f}")

    if report.difficulty_distribution:
        table.add_row(
            "Difficulty",
            ", ".join(f"{k}={v}" for k, v in report.difficulty_distribution.items()),
        )
    if report.generated_cluster_distribution:
        table.add_row("Seed Divergence", f"{report.distribution_divergence:.2f}")
        table.add_row("Distribution Match", f"{report.distribution_match_score:.1f}/100")
        table.add_row("Semantic Coverage", f"{report.semantic_coverage_score:.2%}")
        table.add_row("Graph Coverage", f"{report.graph_coverage_score:.2%}")
    if report.underrepresented_clusters:
        table.add_row(
            "Underrepresented",
            ", ".join(
                f"{cluster}={gap}"
                for cluster, gap in list(report.underrepresented_clusters.items())[:4]
            ),
        )
    if report.semantic_coverage_gaps:
        table.add_row(
            "Semantic Gaps",
            ", ".join(
                f"cluster_{cluster}={gap}"
                for cluster, gap in list(report.semantic_coverage_gaps.items())[:4]
            ),
        )
    if report.graph_frontier_clusters:
        table.add_row("Graph Frontier", ", ".join(report.graph_frontier_clusters[:4]))
    if report.rebalancing_history:
        last_round = report.rebalancing_history[-1]
        table.add_row("Rebalance Rounds", str(len(report.rebalancing_history)))
        table.add_row(
            "Last Rebalance",
            f"accepted={last_round.get('accepted_total', 0)}, "
            f"div={last_round.get('distribution_divergence', 0.0)}",
        )

    if report.contamination_hits > 0:
        table.add_row(
            "Contamination",
            f"[red]{report.contamination_hits} hits ({', '.join(report.contaminated_benchmarks)})[/red]",
        )
    else:
        table.add_row("Contamination", "[green]Clean[/green]")
    if report.contamination_verdicts:
        table.add_row(
            "Contam Verdicts",
            ", ".join(
                f"{verdict}={count}"
                for verdict, count in sorted(report.contamination_verdicts.items())
            ),
        )

    if report.benchmark_sources:
        table.add_row(
            "Benchmark Sources",
            ", ".join(
                f"{bench}={source}({report.benchmark_sample_counts.get(bench, 0)})"
                for bench, source in report.benchmark_sources.items()
            ),
        )
    if report.contamination_methods:
        table.add_row(
            "Contam Methods",
            ", ".join(
                f"{method}={count}"
                for method, count in sorted(
                    report.contamination_methods.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        )
    if report.contamination_method_benchmarks:
        lines = []
        for bench, method_counts in sorted(report.contamination_method_benchmarks.items()):
            method_text = "/".join(
                f"{method}:{count}"
                for method, count in sorted(
                    method_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            lines.append(f"{bench}={method_text}")
        table.add_row("Contam Evidence", ", ".join(lines[:5]))
    if report.benchmark_load_errors:
        table.add_row(
            "Benchmark Errors",
            ", ".join(f"{bench}={error}" for bench, error in report.benchmark_load_errors.items()),
        )

    if report.issue_counts:
        top_issues = ", ".join(
            f"{name}={count}" for name, count in list(report.issue_counts.items())[:3]
        )
        table.add_row("Top Issues", top_issues)

    console.print(table)


# ─── COMPARISONS ─────────────────────────────────────────────────────────────


def _distribution_distance(left: dict[str, int], right: dict[str, int]) -> float:
    """Compute simple L1 distance between two count distributions."""
    left_total = max(sum(left.values()), 1)
    right_total = max(sum(right.values()), 1)
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return round(
        sum(abs((left.get(key, 0) / left_total) - (right.get(key, 0) / right_total)) for key in keys) / 2,
        4,
    )


def _style_bucket(example: Example) -> str:
    """Classify an example response into a style category."""
    text = example.assistant_message.strip()
    lower = text.lower()
    word_count = len(text.split())
    if any(token in lower for token in ["step 1", "first,", "follow these steps", "1.", "2."]):
        return "procedural"
    if word_count <= 25:
        return "concise"
    if any(token in lower for token in ["i understand", "i'm sorry", "thanks for", "let me help"]):
        return "empathetic"
    if word_count >= 90:
        return "detailed"
    return "balanced"


def _dataset_cluster_distribution(dataset: Dataset) -> dict[str, int]:
    """Get cluster-id distribution from dataset metadata."""
    counts: Counter[str] = Counter()
    for example in dataset.examples:
        cluster_id = str(example.metadata.get("cluster_id") or example.metadata.get("seed_cluster_id") or "unknown")
        counts[cluster_id] += 1
    return dict(counts)


def _dataset_style_distribution(dataset: Dataset) -> dict[str, int]:
    """Get style-bucket distribution for a dataset."""
    counts: Counter[str] = Counter()
    for example in dataset.examples:
        counts[_style_bucket(example)] += 1
    return dict(counts)


def _normalized_pair(example: Example) -> tuple[str, str]:
    """Normalize user+assistant text for exact-match overlap checks."""
    return (
        " ".join(example.user_message.lower().split()),
        " ".join(example.assistant_message.lower().split()),
    )


def _exact_pair_overlap_ratio(dataset: Dataset, reference_dataset: Dataset) -> float:
    """Fraction of generated examples that exactly match a reference pair."""
    reference_pairs = {_normalized_pair(example) for example in reference_dataset.examples}
    if not dataset.examples:
        return 0.0
    matches = sum(1 for example in dataset.examples if _normalized_pair(example) in reference_pairs)
    return matches / len(dataset.examples)


def _ngram_set(text: str, n: int = 3) -> set[str]:
    """Extract word-level n-gram set from text."""
    words = " ".join(text.lower().split()).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + n]) for index in range(len(words) - n + 1)}


def _near_pair_overlap_ratio(dataset: Dataset, reference_dataset: Dataset) -> float:
    """Fraction of generated examples with n-gram Jaccard >= 0.75 to a reference."""
    if not dataset.examples:
        return 0.0
    reference_ngrams = [_ngram_set(example.assistant_message) for example in reference_dataset.examples]
    matches = 0
    for example in dataset.examples:
        example_ngrams = _ngram_set(example.assistant_message)
        best = 0.0
        for candidate in reference_ngrams:
            if not candidate:
                continue
            best = max(best, len(example_ngrams & candidate) / max(len(example_ngrams | candidate), 1))
        if best >= 0.75:
            matches += 1
    return matches / len(dataset.examples)


def _semantic_overlap_ratio(
    dataset: Dataset,
    reference_dataset: Dataset,
    threshold: float = 0.88,
) -> float | None:
    """Fraction of generated examples with embedding cosine >= threshold to a reference."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    generated_texts = [example.assistant_message.strip() for example in dataset.examples if example.assistant_message.strip()]
    reference_texts = [example.assistant_message.strip() for example in reference_dataset.examples if example.assistant_message.strip()]
    if not generated_texts or not reference_texts:
        return 0.0

    model = SentenceTransformer("all-MiniLM-L6-v2")
    generated_embeddings = model.encode(generated_texts[:100], normalize_embeddings=True)
    reference_embeddings = model.encode(reference_texts[:200], normalize_embeddings=True)

    matches = 0
    for vector in generated_embeddings:
        similarities = np.dot(reference_embeddings, vector)
        if float(similarities.max()) >= threshold:
            matches += 1
    return matches / max(len(generated_embeddings), 1)


def _reference_alignment_score(
    style_distance: float,
    cluster_distance: float,
    difficulty_distance: float,
    topic_overlap_ratio: float,
    topic_novelty_ratio: float,
    exact_overlap_ratio: float,
) -> float:
    """Composite alignment score between generated and reference datasets."""
    penalty = (
        (style_distance * 0.23)
        + (cluster_distance * 0.23)
        + (difficulty_distance * 0.18)
        + ((1 - topic_overlap_ratio) * 0.18)
        + (topic_novelty_ratio * 0.10)
        + (exact_overlap_ratio * 0.08)
    )
    return max(0.0, 1.0 - min(1.0, penalty)) * 100


def _display_baseline_comparison(dataset, report, baseline_dataset, baseline_report) -> None:
    """Display delta metrics between generated dataset and a baseline."""
    table = Table(title="Baseline Comparison", show_header=False, box=None)
    table.add_column(style="cyan", width=28)
    table.add_column()
    generated_pass_rate = report.passed_examples / max(report.total_examples, 1)
    baseline_pass_rate = baseline_report.passed_examples / max(baseline_report.total_examples, 1)
    quality_delta = report.avg_quality_score - baseline_report.avg_quality_score
    pass_rate_delta = generated_pass_rate - baseline_pass_rate
    lexical_delta = report.lexical_diversity - baseline_report.lexical_diversity
    divergence_delta = report.distribution_divergence - baseline_report.distribution_divergence
    contamination_delta = report.contamination_hits - baseline_report.contamination_hits

    table.add_row("Baseline", baseline_dataset.name)
    table.add_row("Example Delta", f"{dataset.size - baseline_dataset.size:+d}")
    table.add_row("Quality Delta", f"{quality_delta:+.2f}")
    table.add_row("Pass Rate Delta", f"{pass_rate_delta:+.2%}")
    table.add_row("Lexical Delta", f"{lexical_delta:+.4f}")
    table.add_row("Divergence Delta", f"{divergence_delta:+.4f}")
    table.add_row("Contamination Delta", f"{contamination_delta:+d}")
    console.print(table)


def _display_reference_comparison(dataset, report, reference_dataset, reference_report) -> None:
    """Display profile comparison between generated and reference datasets."""
    table = Table(title="Reference Comparison", show_header=False, box=None)
    table.add_column(style="cyan", width=28)
    table.add_column()
    difficulty_distance = _distribution_distance(
        report.difficulty_distribution,
        reference_report.difficulty_distribution,
    )
    generated_topics = set(report.topic_coverage)
    reference_topics = set(reference_report.topic_coverage)
    topic_overlap = len(generated_topics & reference_topics) / max(len(reference_topics), 1)
    topic_novelty = len(generated_topics - reference_topics) / max(len(generated_topics), 1)
    generated_styles = _dataset_style_distribution(dataset)
    reference_styles = _dataset_style_distribution(reference_dataset)
    style_distance = _distribution_distance(generated_styles, reference_styles)
    generated_cluster_distance = _distribution_distance(
        _dataset_cluster_distribution(dataset),
        _dataset_cluster_distribution(reference_dataset),
    )
    exact_overlap = _exact_pair_overlap_ratio(dataset, reference_dataset)
    near_overlap = _near_pair_overlap_ratio(dataset, reference_dataset)
    semantic_overlap = _semantic_overlap_ratio(dataset, reference_dataset)
    alignment_score = _reference_alignment_score(
        style_distance,
        generated_cluster_distance,
        difficulty_distance,
        topic_overlap,
        topic_novelty,
        exact_overlap,
    )

    table.add_row("Reference", reference_dataset.name)
    table.add_row("User Length Delta", f"{report.avg_user_length - reference_report.avg_user_length:+.2f}")
    table.add_row("Response Length Delta", f"{report.avg_assistant_length - reference_report.avg_assistant_length:+.2f}")
    table.add_row("Diversity Delta", f"{report.diversity_score - reference_report.diversity_score:+.4f}")
    table.add_row("Lexical Delta", f"{report.lexical_diversity - reference_report.lexical_diversity:+.4f}")
    table.add_row("Style Distance", f"{style_distance:.4f}")
    table.add_row("Cluster Distance", f"{generated_cluster_distance:.4f}")
    table.add_row("Difficulty Distance", f"{difficulty_distance:.4f}")
    table.add_row("Topic Overlap", f"{topic_overlap:.2%}")
    table.add_row("Topic Novelty", f"{topic_novelty:.2%}")
    table.add_row("Exact Overlap", f"{exact_overlap:.2%}")
    table.add_row("Near Overlap", f"{near_overlap:.2%}")
    table.add_row("Semantic Overlap", "n/a" if semantic_overlap is None else f"{semantic_overlap:.2%}")
    table.add_row("Alignment Score", f"{alignment_score:.2f}/100")
    console.print(table)


# ─── ARTIFACT DISPLAY ────────────────────────────────────────────────────────


def _artifact_summary(dataset: Dataset) -> dict[str, object]:
    """Aggregate pipeline artifact counts and top rejection reasons."""
    artifacts = dataset.artifacts or {}
    candidates = artifacts.get("candidates", [])
    accepted = artifacts.get("accepted", [])
    rejected = artifacts.get("rejected", [])

    rejection_counts: Counter[str] = Counter()
    for example in rejected:
        for reason in example.metadata.get("rejection_reasons", []):
            rejection_counts[reason] += 1

    return {
        "candidates": len(candidates),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "top_rejection_reasons": rejection_counts.most_common(3),
    }


def _display_artifact_summary(dataset: Dataset) -> None:
    """Display a short summary of seed-pipeline artifacts in the terminal."""
    summary = _artifact_summary(dataset)
    if summary["candidates"] == 0 and summary["accepted"] == 0 and summary["rejected"] == 0:
        return

    table = Table(title="Pipeline Artifacts", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Candidates", str(summary["candidates"]))
    table.add_row("Accepted", f"[green]{summary['accepted']}[/green]")
    table.add_row("Rejected", f"[yellow]{summary['rejected']}[/yellow]")

    top_reasons = summary["top_rejection_reasons"]
    if top_reasons:
        reason_text = ", ".join(f"{reason}={count}" for reason, count in top_reasons)
        table.add_row("Top Rejection Reasons", reason_text)
    else:
        table.add_row("Top Rejection Reasons", "[green]None[/green]")

    console.print(table)


def _artifact_example_preview(example: Example) -> dict[str, object]:
    """Prepare a compact terminal preview for an artifact example."""
    evidence = list(example.decontamination_evidence or [])
    evidence_methods: list[str] = []
    evidence_summary: list[str] = []
    for item in evidence:
        method = str(item.get("method", "unknown"))
        benchmark = str(item.get("benchmark", "unknown"))
        confidence = item.get("confidence")
        if method not in evidence_methods:
            evidence_methods.append(method)
        summary = benchmark if confidence is None else f"{benchmark}:{method}:{confidence}"
        evidence_summary.append(summary)

    preview = {
        "id": example.id,
        "quality_score": example.quality_score,
        "user": example.user_message.strip(),
        "assistant": example.assistant_message.strip(),
        "selection_decision": example.metadata.get("selection_decision"),
        "rejection_reasons": example.metadata.get("rejection_reasons", []),
        "topic": example.metadata.get("topic"),
        "persona": example.metadata.get("persona"),
        "difficulty": example.metadata.get("difficulty"),
        "decontamination_flags": list(example.decontamination_flags or []),
        "decontamination_methods": evidence_methods,
        "decontamination_evidence_summary": evidence_summary,
    }
    return preview


def _truncate(text: str, max_len: int = 140) -> str:
    """Truncate long strings for terminal preview."""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _sort_artifact_examples(examples: list[Example], sort_by: str | None) -> list[Example]:
    """Return artifact examples sorted for terminal display or CSV export."""
    items = list(examples)
    if not sort_by:
        return items

    normalized = sort_by.strip().lower()
    if normalized == "score":
        return sorted(items, key=lambda e: (e.quality_score is None, -(e.quality_score or 0.0)))
    if normalized == "topic":
        return sorted(items, key=lambda e: str(e.metadata.get("topic", "")).lower())
    if normalized == "reason":
        return sorted(
            items,
            key=lambda e: ",".join(str(r).lower() for r in e.metadata.get("rejection_reasons", [])),
        )
    return items


def _display_artifact_examples(examples: list[Example], artifact_name: str, limit: int) -> None:
    """Print a limited set of artifact examples to the terminal."""
    if not examples:
        console.print(f"\n[bold]{artifact_name.title()}:[/bold] none")
        return

    console.print(f"\n[bold]{artifact_name.title()}[/bold] (showing up to {min(limit, len(examples))})")

    for index, example in enumerate(examples[:limit], start=1):
        preview = _artifact_example_preview(example)
        body_lines = [
            f"[bold]User:[/bold] {_truncate(str(preview['user']))}",
            f"[bold]Assistant:[/bold] {_truncate(str(preview['assistant']))}",
        ]

        meta_parts: list[str] = []
        if preview["quality_score"] is not None:
            meta_parts.append(f"score={preview['quality_score']:.1f}")
        if preview["topic"]:
            meta_parts.append(f"topic={preview['topic']}")
        if preview["persona"]:
            meta_parts.append(f"persona={preview['persona']}")
        if preview["difficulty"]:
            meta_parts.append(f"difficulty={preview['difficulty']}")
        if preview["selection_decision"]:
            meta_parts.append(f"decision={preview['selection_decision']}")
        if preview["rejection_reasons"]:
            meta_parts.append(
                "reasons=" + ", ".join(str(reason) for reason in preview["rejection_reasons"])
            )
        if preview["decontamination_flags"]:
            meta_parts.append(
                "decon=" + ", ".join(str(flag) for flag in preview["decontamination_flags"])
            )
        if preview["decontamination_methods"]:
            meta_parts.append(
                "decon_methods=" + ", ".join(str(method) for method in preview["decontamination_methods"])
            )

        if meta_parts:
            body_lines.append(f"[bold]Meta:[/bold] {' | '.join(meta_parts)}")
        if preview["decontamination_evidence_summary"]:
            body_lines.append(
                "[bold]Decontamination Evidence:[/bold] "
                + "; ".join(str(item) for item in preview["decontamination_evidence_summary"][:3])
            )

        console.print(
            Panel(
                "\n".join(body_lines),
                title=f"{artifact_name[:-1].title()} #{index} ({preview['id']})",
                border_style="magenta" if artifact_name == "rejected" else "cyan",
            )
        )


def _export_artifact_csv(examples: list[Example], output_path: str) -> str:
    """Export artifact examples to CSV for manual review."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "quality_score",
        "selection_decision",
        "topic",
        "persona",
        "difficulty",
        "rejection_reasons",
        "decontamination_flags",
        "decontamination_methods",
        "decontamination_evidence_summary",
        "user",
        "assistant",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            preview = _artifact_example_preview(example)
            writer.writerow(
                {
                    "id": preview["id"],
                    "quality_score": preview["quality_score"],
                    "selection_decision": preview["selection_decision"],
                    "topic": preview["topic"],
                    "persona": preview["persona"],
                    "difficulty": preview["difficulty"],
                    "rejection_reasons": " | ".join(str(x) for x in preview["rejection_reasons"]),
                    "decontamination_flags": " | ".join(str(x) for x in preview["decontamination_flags"]),
                    "decontamination_methods": " | ".join(str(x) for x in preview["decontamination_methods"]),
                    "decontamination_evidence_summary": " | ".join(
                        str(x) for x in preview["decontamination_evidence_summary"]
                    ),
                    "user": preview["user"],
                    "assistant": preview["assistant"],
                }
            )

    return str(path)
