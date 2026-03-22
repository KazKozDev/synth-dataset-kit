from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from synth_dataset_kit.models import Dataset, Example, QualityReport

logger = logging.getLogger(__name__)


def _pass_rate(report: QualityReport) -> float:
    return report.passed_examples / max(report.total_examples, 1)


def _usable_dataset_gate(report: QualityReport) -> dict[str, object]:
    """Public product-facing definition of a usable synthetic dataset."""
    checks = {
        "all_examples_pass_quality_gate": report.failed_examples == 0,
        "avg_quality_at_least_7_5": report.avg_quality_score >= 7.5,
        "contamination_hits_equal_zero": report.contamination_hits == 0,
        "has_examples": report.total_examples > 0,
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "checks": checks,
        "thresholds": {
            "avg_quality_score_min": 7.5,
            "contamination_hits_max": 0,
            "failed_examples_max": 0,
        },
    }


def _build_baseline_comparison(
    dataset: Dataset,
    report: QualityReport,
    baseline_dataset: Dataset | None = None,
    baseline_report: QualityReport | None = None,
) -> dict[str, object] | None:
    """Build a compact proof-oriented comparison against a baseline dataset."""
    if baseline_dataset is None or baseline_report is None:
        return None

    return {
        "baseline_dataset_name": baseline_dataset.name,
        "baseline_examples": baseline_dataset.size,
        "generated_examples": dataset.size,
        "example_delta": dataset.size - baseline_dataset.size,
        "baseline_avg_quality_score": round(baseline_report.avg_quality_score, 4),
        "generated_avg_quality_score": round(report.avg_quality_score, 4),
        "avg_quality_delta": round(report.avg_quality_score - baseline_report.avg_quality_score, 4),
        "baseline_pass_rate": round(_pass_rate(baseline_report), 4),
        "generated_pass_rate": round(_pass_rate(report), 4),
        "pass_rate_delta": round(_pass_rate(report) - _pass_rate(baseline_report), 4),
        "baseline_lexical_diversity": round(baseline_report.lexical_diversity, 4),
        "generated_lexical_diversity": round(report.lexical_diversity, 4),
        "lexical_diversity_delta": round(
            report.lexical_diversity - baseline_report.lexical_diversity, 4
        ),
        "baseline_distribution_divergence": round(baseline_report.distribution_divergence, 4),
        "generated_distribution_divergence": round(report.distribution_divergence, 4),
        "distribution_divergence_delta": round(
            report.distribution_divergence - baseline_report.distribution_divergence,
            4,
        ),
        "baseline_contamination_hits": baseline_report.contamination_hits,
        "generated_contamination_hits": report.contamination_hits,
        "contamination_hit_delta": report.contamination_hits - baseline_report.contamination_hits,
    }


def _build_eval_summary(
    dataset: Dataset,
    report: QualityReport,
    baseline_dataset: Dataset | None = None,
    baseline_report: QualityReport | None = None,
    reference_dataset: Dataset | None = None,
    reference_report: QualityReport | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "dataset_name": dataset.name,
        "examples": dataset.size,
        "avg_quality_score": round(report.avg_quality_score, 4),
        "passed_examples": report.passed_examples,
        "failed_examples": report.failed_examples,
        "pass_rate": round(_pass_rate(report), 4),
        "lexical_diversity": round(report.lexical_diversity, 4),
        "distribution_divergence": report.distribution_divergence,
        "distribution_match_score": report.distribution_match_score,
        "semantic_coverage_score": report.semantic_coverage_score,
        "semantic_coverage_gaps": report.semantic_coverage_gaps,
        "graph_coverage_score": report.graph_coverage_score,
        "graph_frontier_clusters": report.graph_frontier_clusters,
        "underrepresented_clusters": report.underrepresented_clusters,
        "contamination_hits": report.contamination_hits,
        "contamination_verdicts": report.contamination_verdicts,
        "rebalancing_rounds": len(report.rebalancing_history),
        "top_issues": dict(list(report.issue_counts.items())[:5]),
        "usable_dataset_gate": _usable_dataset_gate(report),
    }
    baseline_comparison = _build_baseline_comparison(
        dataset,
        report,
        baseline_dataset=baseline_dataset,
        baseline_report=baseline_report,
    )
    if baseline_comparison:
        summary["baseline_comparison"] = baseline_comparison
    reference_comparison = _build_reference_comparison(
        dataset,
        report,
        reference_dataset=reference_dataset,
        reference_report=reference_report,
    )
    if reference_comparison:
        summary["reference_comparison"] = reference_comparison
        summary["distribution_validation"] = _build_distribution_validation(
            report,
            reference_comparison,
        )
    return summary


def _build_distribution_validation(
    report: QualityReport,
    reference_comparison: dict[str, object],
) -> dict[str, object]:
    """Calibrate the internal distribution score against a reference comparison."""
    internal_score = float(report.distribution_match_score)
    reference_alignment = float(reference_comparison.get("reference_alignment_score", 0.0))
    graph_coverage = float(report.graph_coverage_score)
    calibration_error = round(abs(internal_score - reference_alignment), 4)
    validated_score = round(
        (internal_score * 0.45) + (reference_alignment * 0.45) + (graph_coverage * 100 * 0.10),
        2,
    )
    return {
        "status": "reference_calibrated",
        "internal_distribution_match_score": internal_score,
        "reference_alignment_score": reference_alignment,
        "graph_coverage_score": round(graph_coverage * 100, 2),
        "calibration_error": calibration_error,
        "validated_distribution_match_score": validated_score,
    }


def _normalized_distribution(counts: dict[str, int]) -> dict[str, float]:
    total = max(sum(counts.values()), 1)
    return {key: value / total for key, value in counts.items()}


def _distribution_distance(left: dict[str, int], right: dict[str, int]) -> float:
    left_norm = _normalized_distribution(left)
    right_norm = _normalized_distribution(right)
    keys = set(left_norm) | set(right_norm)
    if not keys:
        return 0.0
    distance = sum(abs(left_norm.get(key, 0.0) - right_norm.get(key, 0.0)) for key in keys) / 2
    return round(distance, 4)


def _style_bucket(example: Example) -> str:
    """Infer a coarse assistant style bucket from the response text."""
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


def _style_distribution(dataset: Dataset) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in dataset.examples:
        counts[_style_bucket(example)] += 1
    return dict(counts)


def _cluster_distribution(dataset: Dataset) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in dataset.examples:
        cluster_id = str(
            example.metadata.get("cluster_id")
            or example.metadata.get("seed_cluster_id")
            or "unknown"
        )
        counts[cluster_id] += 1
    return dict(counts)


def _normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def _pair_overlap(example: Example, reference_pairs: set[tuple[str, str]]) -> bool:
    return (
        _normalized_text(example.user_message),
        _normalized_text(example.assistant_message),
    ) in reference_pairs


def _ngram_set(text: str, n: int = 3) -> set[str]:
    words = _normalized_text(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + n]) for index in range(len(words) - n + 1)}


def _max_ngram_overlap_ratio(example: Example, reference_ngrams: list[set[str]]) -> float:
    example_ngrams = _ngram_set(example.assistant_message)
    if not example_ngrams or not reference_ngrams:
        return 0.0
    best = 0.0
    for candidate in reference_ngrams:
        if not candidate:
            continue
        overlap = len(example_ngrams & candidate) / max(len(example_ngrams | candidate), 1)
        if overlap > best:
            best = overlap
    return round(best, 4)


def _semantic_overlap_ratio(
    dataset: Dataset,
    reference_dataset: Dataset,
    threshold: float = 0.88,
) -> float | None:
    """Compute semantic overlap ratio via sentence embeddings when available."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    generated_texts = [
        example.assistant_message.strip()
        for example in dataset.examples
        if example.assistant_message.strip()
    ]
    reference_texts = [
        example.assistant_message.strip()
        for example in reference_dataset.examples
        if example.assistant_message.strip()
    ]
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
    return round(matches / max(len(generated_embeddings), 1), 4)


def _reference_alignment_score(
    style_distance: float,
    cluster_distance: float,
    difficulty_distance: float,
    topic_overlap_ratio: float,
    topic_novelty_ratio: float,
    exact_overlap_ratio: float,
) -> float:
    """Combine reference-comparison signals into a simple 0-100 alignment score."""
    penalty = (
        (style_distance * 0.23)
        + (cluster_distance * 0.23)
        + (difficulty_distance * 0.18)
        + ((1 - topic_overlap_ratio) * 0.18)
        + (topic_novelty_ratio * 0.10)
        + (exact_overlap_ratio * 0.08)
    )
    return round(max(0.0, 1.0 - min(1.0, penalty)) * 100, 2)


def _build_reference_comparison(
    dataset: Dataset,
    report: QualityReport,
    reference_dataset: Dataset | None = None,
    reference_report: QualityReport | None = None,
) -> dict[str, object] | None:
    """Build a compact dataset-profile comparison against a stronger reference dataset."""
    if reference_dataset is None or reference_report is None:
        return None

    generated_topics = set(report.topic_coverage)
    reference_topics = set(reference_report.topic_coverage)
    shared_topics = generated_topics & reference_topics
    new_topics = generated_topics - reference_topics
    difficulty_distance = _distribution_distance(
        report.difficulty_distribution,
        reference_report.difficulty_distribution,
    )
    generated_styles = _style_distribution(dataset)
    reference_styles = _style_distribution(reference_dataset)
    generated_clusters = _cluster_distribution(dataset)
    reference_clusters = _cluster_distribution(reference_dataset)
    reference_pairs = {
        (_normalized_text(example.user_message), _normalized_text(example.assistant_message))
        for example in reference_dataset.examples
    }
    exact_overlap_count = sum(
        1 for example in dataset.examples if _pair_overlap(example, reference_pairs)
    )
    reference_ngrams = [
        _ngram_set(example.assistant_message) for example in reference_dataset.examples
    ]
    near_overlap_count = sum(
        1
        for example in dataset.examples
        if _max_ngram_overlap_ratio(example, reference_ngrams) >= 0.75
    )
    style_distance = _distribution_distance(generated_styles, reference_styles)
    cluster_distance = _distribution_distance(generated_clusters, reference_clusters)
    topic_overlap_ratio = round(len(shared_topics) / max(len(reference_topics), 1), 4)
    topic_novelty_ratio = round(len(new_topics) / max(len(generated_topics), 1), 4)
    exact_overlap_ratio = round(exact_overlap_count / max(dataset.size, 1), 4)
    near_overlap_ratio = round(near_overlap_count / max(dataset.size, 1), 4)
    semantic_overlap_ratio = _semantic_overlap_ratio(dataset, reference_dataset)

    return {
        "reference_dataset_name": reference_dataset.name,
        "reference_examples": reference_dataset.size,
        "generated_examples": dataset.size,
        "avg_user_length_delta": round(
            report.avg_user_length - reference_report.avg_user_length, 4
        ),
        "avg_assistant_length_delta": round(
            report.avg_assistant_length - reference_report.avg_assistant_length,
            4,
        ),
        "diversity_score_delta": round(
            report.diversity_score - reference_report.diversity_score, 4
        ),
        "lexical_diversity_delta": round(
            report.lexical_diversity - reference_report.lexical_diversity,
            4,
        ),
        "style_distribution_distance": style_distance,
        "generated_style_distribution": generated_styles,
        "reference_style_distribution": reference_styles,
        "cluster_coverage_distance": cluster_distance,
        "generated_cluster_distribution": generated_clusters,
        "reference_cluster_distribution": reference_clusters,
        "difficulty_profile_distance": difficulty_distance,
        "topic_overlap_ratio": topic_overlap_ratio,
        "topic_novelty_ratio": topic_novelty_ratio,
        "exact_overlap_ratio": exact_overlap_ratio,
        "near_overlap_ratio": near_overlap_ratio,
        "semantic_overlap_ratio": semantic_overlap_ratio,
        "reference_alignment_score": _reference_alignment_score(
            style_distance,
            cluster_distance,
            difficulty_distance,
            topic_overlap_ratio,
            topic_novelty_ratio,
            exact_overlap_ratio,
        ),
        "reference_top_topics": list(reference_report.topic_coverage.keys())[:5],
        "generated_top_topics": list(report.topic_coverage.keys())[:5],
    }


def export_eval_summary(
    dataset: Dataset,
    report: QualityReport,
    output_dir: str,
    baseline_dataset: Dataset | None = None,
    baseline_report: QualityReport | None = None,
    reference_dataset: Dataset | None = None,
    reference_report: QualityReport | None = None,
) -> list[str]:
    """Write a standalone eval summary bundle for a generated dataset."""
    safe_name = dataset.name.replace(" ", "_").replace("/", "_")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    summary = _build_eval_summary(
        dataset,
        report,
        baseline_dataset=baseline_dataset,
        baseline_report=baseline_report,
        reference_dataset=reference_dataset,
        reference_report=reference_report,
    )

    json_path = output_dir_path / f"{safe_name}_eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_path = output_dir_path / f"{safe_name}_eval_summary.md"
    markdown = [
        f"# Eval Summary: {dataset.name}",
        "",
        "## Overview",
        "",
        f"- Examples: {dataset.size}",
        f"- Avg quality score: {report.avg_quality_score:.2f}",
        f"- Passed examples: {report.passed_examples}/{report.total_examples}",
        f"- Pass rate: {_pass_rate(report):.2%}",
        f"- Distribution divergence: {report.distribution_divergence:.4f}",
        f"- Distribution match score: {report.distribution_match_score:.2f}/100",
        f"- Semantic coverage score: {report.semantic_coverage_score:.2%}",
        f"- Graph coverage score: {report.graph_coverage_score:.2%}",
        f"- Contamination hits: {report.contamination_hits}",
        f"- Rebalancing rounds: {len(report.rebalancing_history)}",
        "",
    ]
    baseline_comparison = summary.get("baseline_comparison")
    if baseline_comparison:
        markdown.extend(
            [
                "## Baseline Comparison",
                "",
                f"- Baseline dataset: {baseline_comparison['baseline_dataset_name']}",
                f"- Example delta: {baseline_comparison['example_delta']}",
                (f"- Avg quality delta: {baseline_comparison['avg_quality_delta']:+.4f}"),
                (f"- Pass rate delta: {baseline_comparison['pass_rate_delta']:+.2%}"),
                (
                    "- Lexical diversity delta: "
                    f"{baseline_comparison['lexical_diversity_delta']:+.4f}"
                ),
                (
                    "- Distribution divergence delta: "
                    f"{baseline_comparison['distribution_divergence_delta']:+.4f}"
                ),
                (f"- Contamination hit delta: {baseline_comparison['contamination_hit_delta']:+d}"),
                "",
            ]
        )
    reference_comparison = summary.get("reference_comparison")
    if reference_comparison:
        markdown.extend(
            [
                "## Reference Dataset Comparison",
                "",
                f"- Reference dataset: {reference_comparison['reference_dataset_name']}",
                f"- Avg user length delta: {reference_comparison['avg_user_length_delta']:+.4f}",
                f"- Avg assistant length delta: {reference_comparison['avg_assistant_length_delta']:+.4f}",
                f"- Diversity delta: {reference_comparison['diversity_score_delta']:+.4f}",
                f"- Lexical diversity delta: {reference_comparison['lexical_diversity_delta']:+.4f}",
                f"- Style distribution distance: {reference_comparison['style_distribution_distance']:.4f}",
                f"- Cluster coverage distance: {reference_comparison['cluster_coverage_distance']:.4f}",
                f"- Difficulty profile distance: {reference_comparison['difficulty_profile_distance']:.4f}",
                f"- Topic overlap ratio: {reference_comparison['topic_overlap_ratio']:.2%}",
                f"- Topic novelty ratio: {reference_comparison['topic_novelty_ratio']:.2%}",
                f"- Exact overlap ratio: {reference_comparison['exact_overlap_ratio']:.2%}",
                f"- Near overlap ratio: {reference_comparison['near_overlap_ratio']:.2%}",
                (
                    f"- Semantic overlap ratio: {reference_comparison['semantic_overlap_ratio']:.2%}"
                    if reference_comparison["semantic_overlap_ratio"] is not None
                    else "- Semantic overlap ratio: n/a"
                ),
                f"- Reference alignment score: {reference_comparison['reference_alignment_score']:.2f}/100",
                "",
            ]
        )
    distribution_validation = summary.get("distribution_validation")
    if distribution_validation:
        markdown.extend(
            [
                "## Distribution Validation",
                "",
                f"- Status: {distribution_validation['status']}",
                (
                    "- Internal distribution match score: "
                    f"{distribution_validation['internal_distribution_match_score']:.2f}/100"
                ),
                (
                    "- Reference alignment score: "
                    f"{distribution_validation['reference_alignment_score']:.2f}/100"
                ),
                (
                    "- Graph coverage score: "
                    f"{distribution_validation['graph_coverage_score']:.2f}/100"
                ),
                (f"- Calibration error: {distribution_validation['calibration_error']:.4f}"),
                (
                    "- Validated distribution match score: "
                    f"{distribution_validation['validated_distribution_match_score']:.2f}/100"
                ),
                "",
            ]
        )

    markdown.extend(
        [
            "## Top Issues",
            "",
        ]
    )
    if report.issue_counts:
        markdown.extend(
            [f"- {name}: {count}" for name, count in list(report.issue_counts.items())[:5]]
        )
    else:
        markdown.append("- None")
    markdown.extend(
        [
            "",
            "## Verdicts",
            "",
            ", ".join(
                f"{name}={count}" for name, count in sorted(report.contamination_verdicts.items())
            )
            or "clean=0",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown) + "\n")

    logger.info(f"Eval summary saved to {json_path} and {md_path}")
    return [str(json_path), str(md_path)]
