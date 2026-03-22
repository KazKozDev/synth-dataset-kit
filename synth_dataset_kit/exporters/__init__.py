"""Export datasets to various fine-tuning formats."""

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
        "lexical_diversity_delta": round(report.lexical_diversity - baseline_report.lexical_diversity, 4),
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
        from sentence_transformers import SentenceTransformer
        import numpy as np
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
    exact_overlap_count = sum(1 for example in dataset.examples if _pair_overlap(example, reference_pairs))
    reference_ngrams = [_ngram_set(example.assistant_message) for example in reference_dataset.examples]
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
        "avg_user_length_delta": round(report.avg_user_length - reference_report.avg_user_length, 4),
        "avg_assistant_length_delta": round(
            report.avg_assistant_length - reference_report.avg_assistant_length,
            4,
        ),
        "diversity_score_delta": round(report.diversity_score - reference_report.diversity_score, 4),
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


def export_jsonl(dataset: Dataset, path: str, include_metadata: bool = False) -> str:
    """Export as JSONL with OpenAI messages format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w") as f:
        for example in dataset.examples:
            record = {
                "messages": [
                    {"role": m.role.value, "content": m.content}
                    for m in example.messages
                ]
            }
            if include_metadata:
                record["metadata"] = example.metadata
                record["quality_score"] = example.quality_score
                record["decontamination_flags"] = example.decontamination_flags
                record["decontamination_evidence"] = example.decontamination_evidence
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Exported {dataset.size} examples to {p} (JSONL)")
    return str(p)


def export_alpaca(dataset: Dataset, path: str) -> str:
    """Export as Alpaca format (instruction/input/output)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for example in dataset.examples:
        records.append({
            "instruction": example.user_message,
            "input": "",
            "output": example.assistant_message,
        })

    with open(p, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {dataset.size} examples to {p} (Alpaca)")
    return str(p)


def export_sharegpt(dataset: Dataset, path: str) -> str:
    """Export as ShareGPT format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for example in dataset.examples:
        role_map = {"user": "human", "assistant": "gpt", "system": "system"}
        conversations = [
            {"from": role_map.get(m.role.value, m.role.value), "value": m.content}
            for m in example.messages
        ]
        records.append({"conversations": conversations})

    with open(p, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Exported {dataset.size} examples to {p} (ShareGPT)")
    return str(p)


def export_chatml(dataset: Dataset, path: str) -> str:
    """Export as ChatML format (used by many training frameworks)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w") as f:
        for example in dataset.examples:
            text = ""
            for m in example.messages:
                text += f"<|im_start|>{m.role.value}\n{m.content}<|im_end|>\n"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    logger.info(f"Exported {dataset.size} examples to {p} (ChatML)")
    return str(p)


def export_huggingface_bundle(
    dataset: Dataset,
    output_dir: str,
    include_metadata: bool = True,
    quality_report: QualityReport | None = None,
    baseline_dataset: Dataset | None = None,
    baseline_report: QualityReport | None = None,
    reference_dataset: Dataset | None = None,
    reference_report: QualityReport | None = None,
) -> list[str]:
    """Export a HuggingFace-ready dataset bundle with dataset card and metadata."""
    safe_name = dataset.name.replace(" ", "_").replace("/", "_")
    bundle_dir = Path(output_dir) / f"{safe_name}_huggingface"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    train_path = export_jsonl(
        dataset,
        str(bundle_dir / "train.jsonl"),
        include_metadata=include_metadata,
    )

    eval_summary_path = bundle_dir / "eval_summary.json"
    if quality_report:
        eval_summary = _build_eval_summary(
            dataset,
            quality_report,
            baseline_dataset=baseline_dataset,
            baseline_report=baseline_report,
            reference_dataset=reference_dataset,
            reference_report=reference_report,
        )
    else:
        eval_summary = {
            "dataset_name": dataset.name,
            "examples": dataset.size,
        }
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    quality_report_files: list[str] = []
    if quality_report:
        quality_report_files.append(
            export_quality_report_html(
                quality_report,
                str(bundle_dir / f"{dataset.name}_quality_report.html"),
            )
        )
        quality_report_files.append(
            export_quality_report_json(
                quality_report,
                str(bundle_dir / f"{dataset.name}_quality_report.json"),
            )
        )

    yaml_header = [
        "---",
        "language:",
        "- en",
        "license: mit",
        "pretty_name: " + dataset.name,
        "task_categories:",
        "- text-generation",
        "- question-answering",
        "task_ids:",
        "- conversational",
        "tags:",
        "- synthetic",
        "- customer-support",
        "- llm",
        "- instruction-tuning",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: train.jsonl",
        "size_categories:",
        f"- {_hf_size_category(dataset.size)}",
        "---",
        "",
    ]

    card_lines = [
        *yaml_header,
        f"# {dataset.name}",
        "",
        "Synthetic customer-support dataset generated with synth-dataset-kit.",
        "",
        "## What Is Included",
        "",
        f"- `{Path(train_path).name}`: training split in JSONL chat format",
        f"- `{eval_summary_path.name}`: minimal evaluation and generation summary",
    ]
    if quality_report:
        card_lines.extend(
            [
                f"- `{dataset.name}_quality_report.html`: visual quality report",
                f"- `{dataset.name}_quality_report.json`: machine-readable quality report",
            ]
        )
    card_lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- examples: {dataset.size}",
            f"- generator: {dataset.generator}",
            f"- avg quality: {quality_report.avg_quality_score:.2f}" if quality_report else "- avg quality: n/a",
            f"- passed: {quality_report.passed_examples}" if quality_report else "- passed: n/a",
            (
                "- usable dataset gate: passed"
                if quality_report and _usable_dataset_gate(quality_report)["passed"]
                else "- usable dataset gate: failed"
            ),
            f"- contamination hits: {quality_report.contamination_hits}" if quality_report else "- contamination hits: n/a",
            f"- distribution divergence: {quality_report.distribution_divergence:.4f}" if quality_report else "- distribution divergence: n/a",
            f"- rebalancing rounds: {len(quality_report.rebalancing_history)}" if quality_report else "- rebalancing rounds: n/a",
            "",
            "## Public Quality Gate",
            "",
            "- A dataset is considered usable when every example passes the quality gate, average quality is at least 7.5, contamination hits are zero, and the final export is non-empty." if quality_report else "- quality gate unavailable",
            "",
            "## Data Generation Process",
            "",
            "This dataset was created from a small customer-support seed set.",
            "Generation used staged candidate creation, quality filtering, decontamination, and distribution-aware rebalancing.",
            "",
            "## Quality Signals",
            "",
            f"- lexical diversity: {quality_report.lexical_diversity:.4f}" if quality_report else "- lexical diversity: n/a",
            f"- self-BLEU proxy: {quality_report.self_bleu_proxy:.4f}" if quality_report else "- self-BLEU proxy: n/a",
            f"- embedding diversity: {quality_report.embedding_diversity_score}" if quality_report else "- embedding diversity: n/a",
            "",
            "## Distribution Alignment",
            "",
            f"- divergence: {quality_report.distribution_divergence:.4f}" if quality_report else "- divergence: n/a",
            f"- distribution match score: {quality_report.distribution_match_score:.2f}/100" if quality_report else "- distribution match score: n/a",
            f"- semantic coverage score: {quality_report.semantic_coverage_score:.2%}" if quality_report else "- semantic coverage score: n/a",
            f"- graph coverage score: {quality_report.graph_coverage_score:.2%}" if quality_report else "- graph coverage score: n/a",
            (
                "- underrepresented clusters: "
                + ", ".join(f"{cluster}={gap}" for cluster, gap in quality_report.underrepresented_clusters.items())
            ) if quality_report and quality_report.underrepresented_clusters else "- underrepresented clusters: none",
            (
                "- semantic coverage gaps: "
                + ", ".join(
                    f"cluster_{cluster}={gap}"
                    for cluster, gap in quality_report.semantic_coverage_gaps.items()
                )
            ) if quality_report and quality_report.semantic_coverage_gaps else "- semantic coverage gaps: none",
            (
                "- graph frontier clusters: "
                + ", ".join(quality_report.graph_frontier_clusters)
            ) if quality_report and quality_report.graph_frontier_clusters else "- graph frontier clusters: none",
            "",
            "## Contamination Audit",
            "",
            f"- contamination hits: {quality_report.contamination_hits}" if quality_report else "- contamination hits: n/a",
            (
                "- verdicts: "
                + ", ".join(f"{verdict}={count}" for verdict, count in quality_report.contamination_verdicts.items())
            ) if quality_report and quality_report.contamination_verdicts else "- verdicts: none",
            "",
            "## Recommended Use",
            "",
            "Use this dataset for instruction tuning or support-assistant fine-tuning experiments.",
            "Validate on your own holdout set before production use.",
            "",
            "## Reproducibility",
            "",
            "This bundle was generated by synth-dataset-kit from a small seed set.",
            "Use `eval_summary.json` plus the full HTML/JSON report to reproduce the run context.",
        ]
    )
    baseline_comparison = eval_summary.get("baseline_comparison")
    if baseline_comparison:
        card_lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                f"- baseline dataset: {baseline_comparison['baseline_dataset_name']}",
                f"- example delta: {baseline_comparison['example_delta']}",
                f"- avg quality delta: {baseline_comparison['avg_quality_delta']:+.4f}",
                f"- pass rate delta: {baseline_comparison['pass_rate_delta']:+.2%}",
                f"- lexical diversity delta: {baseline_comparison['lexical_diversity_delta']:+.4f}",
                f"- distribution divergence delta: {baseline_comparison['distribution_divergence_delta']:+.4f}",
                f"- contamination hit delta: {baseline_comparison['contamination_hit_delta']:+d}",
            ]
        )
    reference_comparison = eval_summary.get("reference_comparison")
    if reference_comparison:
        card_lines.extend(
            [
                "",
                "## Reference Dataset Comparison",
                "",
                f"- reference dataset: {reference_comparison['reference_dataset_name']}",
                f"- avg user length delta: {reference_comparison['avg_user_length_delta']:+.4f}",
                f"- avg assistant length delta: {reference_comparison['avg_assistant_length_delta']:+.4f}",
                f"- diversity delta: {reference_comparison['diversity_score_delta']:+.4f}",
                f"- lexical diversity delta: {reference_comparison['lexical_diversity_delta']:+.4f}",
                f"- style distribution distance: {reference_comparison['style_distribution_distance']:.4f}",
                f"- cluster coverage distance: {reference_comparison['cluster_coverage_distance']:.4f}",
                f"- difficulty profile distance: {reference_comparison['difficulty_profile_distance']:.4f}",
                f"- topic overlap ratio: {reference_comparison['topic_overlap_ratio']:.2%}",
                f"- topic novelty ratio: {reference_comparison['topic_novelty_ratio']:.2%}",
                f"- exact overlap ratio: {reference_comparison['exact_overlap_ratio']:.2%}",
                f"- near overlap ratio: {reference_comparison['near_overlap_ratio']:.2%}",
                (
                    f"- semantic overlap ratio: {reference_comparison['semantic_overlap_ratio']:.2%}"
                    if reference_comparison['semantic_overlap_ratio'] is not None
                    else "- semantic overlap ratio: n/a"
                ),
                f"- reference alignment score: {reference_comparison['reference_alignment_score']:.2f}/100",
            ]
        )
    card_path = bundle_dir / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("\n".join(card_lines) + "\n")

    metadata_path = bundle_dir / "dataset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": dataset.name,
                "size": dataset.size,
                "generator": dataset.generator,
                "config_snapshot": dataset.config_snapshot,
                "quality_summary": quality_report.model_dump(mode="json") if quality_report else None,
                "eval_summary_path": str(eval_summary_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"Exported HuggingFace bundle to {bundle_dir}")
    return [str(bundle_dir), str(card_path), str(metadata_path), str(train_path), str(eval_summary_path), *quality_report_files]


def _hf_size_category(size: int) -> str:
    if size < 100:
        return "n<1K"
    if size < 1_000:
        return "1K<n<10K"
    if size < 10_000:
        return "10K<n<100K"
    if size < 100_000:
        return "100K<n<1M"
    return "n>1M"


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
                (
                    "- Avg quality delta: "
                    f"{baseline_comparison['avg_quality_delta']:+.4f}"
                ),
                (
                    "- Pass rate delta: "
                    f"{baseline_comparison['pass_rate_delta']:+.2%}"
                ),
                (
                    "- Lexical diversity delta: "
                    f"{baseline_comparison['lexical_diversity_delta']:+.4f}"
                ),
                (
                    "- Distribution divergence delta: "
                    f"{baseline_comparison['distribution_divergence_delta']:+.4f}"
                ),
                (
                    "- Contamination hit delta: "
                    f"{baseline_comparison['contamination_hit_delta']:+d}"
                ),
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
                    if reference_comparison['semantic_overlap_ratio'] is not None
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
                (
                    "- Calibration error: "
                    f"{distribution_validation['calibration_error']:.4f}"
                ),
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
        markdown.extend([f"- {name}: {count}" for name, count in list(report.issue_counts.items())[:5]])
    else:
        markdown.append("- None")
    markdown.extend(
        [
            "",
            "## Verdicts",
            "",
            ", ".join(f"{name}={count}" for name, count in sorted(report.contamination_verdicts.items()))
            or "clean=0",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown) + "\n")

    logger.info(f"Eval summary saved to {json_path} and {md_path}")
    return [str(json_path), str(md_path)]


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
    summary_path.write_text(json.dumps(eval_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    copied_holdout_path = None
    if holdout_path:
        source_holdout = Path(holdout_path)
        if source_holdout.exists():
            copied_holdout_path = bundle_dir / source_holdout.name
            copied_holdout_path.write_text(source_holdout.read_text(encoding="utf-8"), encoding="utf-8")

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
                f'HOLDOUT_PATH="./{copied_holdout_path.name}"' if copied_holdout_path else 'HOLDOUT_PATH="./customer_support_holdout.jsonl"',
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
                    if reference['semantic_overlap_ratio'] is not None
                    else "- Semantic overlap ratio: n/a"
                ),
                f"- Reference alignment score: {reference['reference_alignment_score']:.2f}/100",
            ]
        )
    md_path = bundle_dir / "proof_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info(f"Proof bundle saved to {bundle_dir}")
    output_files = [str(bundle_dir), str(summary_path), str(finetune_script), str(eval_script), str(eval_rubric_path), str(md_path)]
    if copied_holdout_path:
        output_files.append(str(copied_holdout_path))
    return output_files


def export_dataset(
    dataset: Dataset,
    format: str,
    output_dir: str,
    include_metadata: bool = False,
    quality_report: QualityReport | None = None,
    baseline_dataset: Dataset | None = None,
    baseline_report: QualityReport | None = None,
    reference_dataset: Dataset | None = None,
    reference_report: QualityReport | None = None,
) -> str:
    """Export dataset to the specified format."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    safe_name = dataset.name.replace(" ", "_").replace("/", "_")

    if format == "huggingface":
        bundle_files = export_huggingface_bundle(
            dataset,
            output_dir,
            include_metadata=include_metadata,
            quality_report=quality_report,
            baseline_dataset=baseline_dataset,
            baseline_report=baseline_report,
            reference_dataset=reference_dataset,
            reference_report=reference_report,
        )
        return bundle_files[0]

    exporters = {
        "jsonl": (export_jsonl, f"{safe_name}.jsonl"),
        "openai": (export_jsonl, f"{safe_name}.jsonl"),
        "alpaca": (export_alpaca, f"{safe_name}_alpaca.json"),
        "sharegpt": (export_sharegpt, f"{safe_name}_sharegpt.jsonl"),
        "chatml": (export_chatml, f"{safe_name}_chatml.jsonl"),
    }

    if format not in exporters:
        raise ValueError(f"Unknown format '{format}'. Available: {list(exporters.keys())}")

    exporter_fn, filename = exporters[format]
    filepath = str(output_dir_path / filename)

    if format == "jsonl" or format == "openai":
        return exporter_fn(dataset, filepath, include_metadata=include_metadata)
    else:
        return exporter_fn(dataset, filepath)


def export_pipeline_artifacts(dataset: Dataset, output_dir: str) -> list[str]:
    """Export intermediate pipeline artifacts as JSONL files."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    safe_name = dataset.name.replace(" ", "_").replace("/", "_")
    output_files: list[str] = []

    for artifact_name, examples in dataset.artifacts.items():
        if not examples:
            continue
        artifact_dataset = Dataset(
            name=f"{safe_name}_{artifact_name}",
            version=dataset.version,
            created_at=dataset.created_at,
            generator=dataset.generator,
            config_snapshot=dataset.config_snapshot,
            examples=examples,
        )
        artifact_path = str(output_dir_path / f"{safe_name}_{artifact_name}.jsonl")
        export_jsonl(artifact_dataset, artifact_path, include_metadata=True)
        output_files.append(artifact_path)

    return output_files


def export_quality_report_html(report: QualityReport, path: str) -> str:
    """Generate an HTML quality report."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Score distribution for chart
    buckets = sorted(report.score_distribution.items())
    chart_labels = [b[0] for b in buckets]
    chart_values = [b[1] for b in buckets]

    # Topic coverage
    topics = list(report.topic_coverage.items())[:15]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quality Report — {report.dataset_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 2rem; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: #fff; }}
  .subtitle {{ color: #888; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #1a1a1a; border-radius: 12px; padding: 1.5rem; border: 1px solid #2a2a2a; }}
  .card .label {{ color: #888; font-size: 0.85rem; margin-bottom: 0.5rem; }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; }}
  .card .value.green {{ color: #4ade80; }}
  .card .value.yellow {{ color: #fbbf24; }}
  .card .value.red {{ color: #f87171; }}
  .card .value.blue {{ color: #60a5fa; }}
  .section {{ background: #1a1a1a; border-radius: 12px; padding: 1.5rem; border: 1px solid #2a2a2a; margin-bottom: 1.5rem; }}
  .section h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: #fff; }}
  .bar-chart {{ display: flex; flex-direction: column; gap: 0.5rem; }}
  .bar-row {{ display: flex; align-items: center; gap: 0.5rem; }}
  .bar-label {{ width: 120px; font-size: 0.8rem; color: #aaa; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .bar-track {{ flex: 1; height: 24px; background: #2a2a2a; border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 0.75rem; color: #fff; }}
  .bar-fill.score {{ background: linear-gradient(90deg, #4ade80, #22c55e); }}
  .bar-fill.topic {{ background: linear-gradient(90deg, #60a5fa, #3b82f6); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin: 2px; }}
  .badge.warn {{ background: #fbbf2433; color: #fbbf24; }}
  .badge.ok {{ background: #4ade8033; color: #4ade80; }}
  .footer {{ text-align: center; color: #555; font-size: 0.8rem; margin-top: 2rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>📊 Quality Report</h1>
  <p class="subtitle">{report.dataset_name} — {report.total_examples} examples</p>

  <div class="grid">
    <div class="card">
      <div class="label">Total Examples</div>
      <div class="value blue">{report.total_examples}</div>
    </div>
    <div class="card">
      <div class="label">Passed (≥{7.0})</div>
      <div class="value green">{report.passed_examples}</div>
    </div>
    <div class="card">
      <div class="label">Failed</div>
      <div class="value {'red' if report.failed_examples > 0 else 'green'}">{report.failed_examples}</div>
    </div>
    <div class="card">
      <div class="label">Avg Quality Score</div>
      <div class="value {'green' if report.avg_quality_score >= 7 else 'yellow' if report.avg_quality_score >= 5 else 'red'}">{report.avg_quality_score:.1f}</div>
    </div>
    <div class="card">
      <div class="label">Diversity Score</div>
      <div class="value {'green' if report.diversity_score >= 0.7 else 'yellow'}">{report.diversity_score:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Self-BLEU Proxy</div>
      <div class="value {'green' if report.self_bleu_proxy <= 0.3 else 'yellow'}">{report.self_bleu_proxy:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Lexical Diversity</div>
      <div class="value {'green' if report.lexical_diversity >= 0.2 else 'yellow'}">{report.lexical_diversity:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Contamination Hits</div>
      <div class="value {'red' if report.contamination_hits > 0 else 'green'}">{report.contamination_hits}</div>
    </div>
    <div class="card">
      <div class="label">Near Duplicates</div>
      <div class="value {'yellow' if report.near_duplicate_examples > 0 else 'green'}">{report.near_duplicate_examples}</div>
    </div>
    <div class="card">
      <div class="label">Embedding Diversity</div>
      <div class="value {'green' if (report.embedding_diversity_score or 0) >= 0.65 else 'yellow'}">{report.embedding_diversity_score if report.embedding_diversity_score is not None else 'n/a'}</div>
    </div>
  </div>

  <div class="grid" style="grid-template-columns: 1fr 1fr;">
    <div class="card">
      <div class="label">Avg User Message</div>
      <div class="value blue">{report.avg_user_length:.0f} <span style="font-size:0.8rem;color:#888">words</span></div>
    </div>
    <div class="card">
      <div class="label">Avg Assistant Response</div>
      <div class="value blue">{report.avg_assistant_length:.0f} <span style="font-size:0.8rem;color:#888">words</span></div>
    </div>
  </div>

  <div class="section">
    <h2>Score Distribution</h2>
    <div class="bar-chart">
"""

    max_count = max(chart_values) if chart_values else 1
    for label, value in buckets:
        pct = (value / max_count) * 100
        html += f"""      <div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bar-track"><div class="bar-fill score" style="width:{pct}%">{value}</div></div>
      </div>
"""

    html += """    </div>
  </div>

  <div class="section">
    <h2>Topic Coverage</h2>
    <div class="bar-chart">
"""

    max_topic = max((t[1] for t in topics), default=1)
    for topic_name, count in topics:
        pct = (count / max_topic) * 100
        display_name = topic_name[:40] + "..." if len(topic_name) > 40 else topic_name
        html += f"""      <div class="bar-row">
        <div class="bar-label" title="{topic_name}">{display_name}</div>
        <div class="bar-track"><div class="bar-fill topic" style="width:{pct}%">{count}</div></div>
      </div>
"""

    html += """    </div>
  </div>
"""

    if report.difficulty_distribution:
        html += """  <div class="section">
    <h2>Difficulty Distribution</h2>
    <div class="bar-chart">
"""
        max_diff = max(report.difficulty_distribution.values())
        for difficulty_name, count in report.difficulty_distribution.items():
            pct = (count / max_diff) * 100
            html += f"""      <div class="bar-row">
        <div class="bar-label">{difficulty_name}</div>
        <div class="bar-track"><div class="bar-fill score" style="width:{pct}%">{count}</div></div>
      </div>
"""
        html += """    </div>
  </div>
"""

    if report.issue_counts:
        html += """  <div class="section">
    <h2>Rule-Based Issues</h2>
    <div class="bar-chart">
"""
        max_issue_count = max(report.issue_counts.values())
        for issue_name, count in report.issue_counts.items():
            pct = (count / max_issue_count) * 100
            html += f"""      <div class="bar-row">
        <div class="bar-label">{issue_name}</div>
        <div class="bar-track"><div class="bar-fill topic" style="width:{pct}%">{count}</div></div>
      </div>
"""
        html += """    </div>
  </div>
"""

    if report.topic_heatmap:
        html += """  <div class="section">
    <h2>Topic Heatmap</h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Topic</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Easy</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Medium</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Hard</th>
        </tr>
      </thead>
      <tbody>
"""
        for topic_name, difficulty_counts in list(report.topic_heatmap.items())[:15]:
            html += f"""        <tr>
          <td style="padding:8px;border-bottom:1px solid #222;">{topic_name}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{difficulty_counts.get('easy', 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{difficulty_counts.get('medium', 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{difficulty_counts.get('hard', 0)}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""

    if report.seed_cluster_distribution or report.generated_cluster_distribution:
        html += """  <div class="section">
    <h2>Seed vs Generated Distribution</h2>
    <div style="margin-bottom:12px;color:#aaa;">Divergence: """
        html += f"""{report.distribution_divergence:.4f} · Match Score: {report.distribution_match_score:.2f}/100 · Semantic Coverage: {report.semantic_coverage_score:.2%} · Graph Coverage: {report.graph_coverage_score:.2%}</div>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Cluster</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Planned</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Generated</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Gap</th>
        </tr>
      </thead>
      <tbody>
"""
        cluster_ids = sorted(set(report.seed_cluster_distribution) | set(report.generated_cluster_distribution))
        for cluster_id in cluster_ids[:20]:
            planned = report.seed_cluster_distribution.get(cluster_id, 0)
            generated = report.generated_cluster_distribution.get(cluster_id, 0)
            gap = report.underrepresented_clusters.get(cluster_id, 0)
            html += f"""        <tr>
          <td style="padding:8px;border-bottom:1px solid #222;">{cluster_id}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{planned}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{generated}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{gap}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""
    if report.graph_frontier_clusters:
        html += """  <div class="section">
    <h2>Graph Frontier</h2>
    <div style="color:#aaa;">"""
        html += ", ".join(report.graph_frontier_clusters[:8])
        html += """</div>
  </div>
"""
    if report.semantic_cluster_target_distribution or report.semantic_cluster_generated_distribution:
        html += """  <div class="section">
    <h2>Semantic Coverage</h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Semantic Cluster</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Target</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Generated</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Gap</th>
        </tr>
      </thead>
      <tbody>
"""
        semantic_cluster_ids = sorted(
            set(report.semantic_cluster_target_distribution)
            | set(report.semantic_cluster_generated_distribution)
        )
        for cluster_id in semantic_cluster_ids[:20]:
            target = report.semantic_cluster_target_distribution.get(cluster_id, 0)
            generated = report.semantic_cluster_generated_distribution.get(cluster_id, 0)
            gap = report.semantic_coverage_gaps.get(cluster_id, 0)
            html += f"""        <tr>
          <td style="padding:8px;border-bottom:1px solid #222;">cluster_{cluster_id}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{target}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{generated}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{gap}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""

    if report.rebalancing_history:
        html += """  <div class="section">
    <h2>Rebalancing History</h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Round</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Requested</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Accepted</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Rejected</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Divergence</th>
        </tr>
      </thead>
      <tbody>
"""
        for round_info in report.rebalancing_history[:20]:
            html += f"""        <tr>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get('round', 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get('requested', 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get('accepted_total', 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get('rejected_batch', 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get('distribution_divergence', 0.0)}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""

    if report.contaminated_benchmarks:
        html += """  <div class="section">
    <h2>⚠️ Contamination Detected</h2>
    <p style="margin-bottom:0.5rem;">The following benchmarks had potential overlap:</p>
"""
        for bench in report.contaminated_benchmarks:
            html += f'    <span class="badge warn">{bench}</span>\n'
        html += "  </div>\n"
    else:
        html += """  <div class="section">
    <h2>✅ No Contamination Detected</h2>
    <p>Dataset passed decontamination checks against all configured benchmarks.</p>
  </div>
"""

    if report.contamination_verdicts:
        html += """  <div class="section">
    <h2>Contamination Verdicts</h2>
"""
        for verdict, count in sorted(report.contamination_verdicts.items()):
            html += f'    <span class="badge {"warn" if verdict != "clean" else "ok"}">{verdict}: {count}</span>\n'
        html += "  </div>\n"

    if report.contamination_methods:
        html += """  <div class="section">
    <h2>Contamination Methods</h2>
    <div class="bar-chart">
"""
        max_method_count = max(report.contamination_methods.values())
        for method_name, count in sorted(
            report.contamination_methods.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            pct = (count / max_method_count) * 100
            html += f"""      <div class="bar-row">
        <div class="bar-label">{method_name}</div>
        <div class="bar-track"><div class="bar-fill topic" style="width:{pct}%">{count}</div></div>
      </div>
"""
        html += """    </div>
  </div>
"""

    if report.contamination_method_benchmarks:
        html += """  <div class="section">
    <h2>Contamination Evidence By Benchmark</h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Benchmark</th>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Methods</th>
        </tr>
      </thead>
      <tbody>
"""
        for bench_name, method_counts in sorted(report.contamination_method_benchmarks.items()):
            method_text = ", ".join(
                f"{method}: {count}"
                for method, count in sorted(
                    method_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            html += f"""        <tr>
          <td style="padding:8px;border-bottom:1px solid #222;">{bench_name}</td>
          <td style="padding:8px;border-bottom:1px solid #222;">{method_text}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""

    if report.contamination_evidence_samples:
        html += """  <div class="section">
    <h2>Contamination Evidence Samples</h2>
"""
        for sample in report.contamination_evidence_samples[:10]:
            matched_text = str(sample.get("matched_text", ""))[:180]
            confidence = sample.get("confidence")
            confidence_text = f" ({confidence})" if confidence is not None else ""
            html += (
                f'    <div style="padding:10px 0;border-bottom:1px solid #222;">'
                f'<strong>{sample.get("benchmark", "unknown")}</strong> via '
                f'<span class="badge warn">{sample.get("method", "unknown")}{confidence_text}</span>'
                f'<div style="color:#aaa;margin-top:6px;">{matched_text}</div>'
                f"</div>\n"
            )
        html += "  </div>\n"

    if report.benchmark_sources:
        html += """  <div class="section">
    <h2>Benchmark Sources</h2>
"""
        for bench_name, source in report.benchmark_sources.items():
            sample_count = report.benchmark_sample_counts.get(bench_name, 0)
            html += f'    <span class="badge {"ok" if source == "datasets" else "warn"}">{bench_name}: {source} ({sample_count})</span>\n'
        html += "  </div>\n"

    if report.benchmark_load_errors:
        html += """  <div class="section">
    <h2>Benchmark Load Errors</h2>
"""
        for bench_name, error in report.benchmark_load_errors.items():
            html += f'    <span class="badge warn">{bench_name}: {error}</span>\n'
        html += "  </div>\n"

    html += f"""
  <div class="footer">
    Generated by synth-dataset-kit v0.1.0
  </div>
</div>
</body>
</html>"""

    with open(p, "w") as f:
        f.write(html)

    logger.info(f"Quality report saved to {p}")
    return str(p)


def export_quality_report_json(report: QualityReport, path: str) -> str:
    """Save the quality report as machine-readable JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
    logger.info(f"Quality report JSON saved to {p}")
    return str(p)


def export_run_summary(summary: dict[str, object], path: str) -> str:
    """Save a run summary JSON artifact alongside the output bundle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Run summary saved to {p}")
    return str(p)
