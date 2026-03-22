"""Holdout evaluation utilities for before/after model comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.generators.seed_expander import load_seed_file
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, Role


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(
        min(pred_counts.get(token, 0), ref_counts.get(token, 0))
        for token in set(pred_counts) | set(ref_counts)
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return (2 * precision * recall) / max(precision + recall, 1e-9)


def _empathy_score(text: str) -> float:
    lower = _normalize(text)
    markers = ["sorry", "understand", "help", "thanks", "i can", "let me"]
    return min(1.0, sum(1 for marker in markers if marker in lower) / 3)


def _task_success(prediction: str, reference: str) -> float:
    f1 = _token_f1(prediction, reference)
    if f1 >= 0.6:
        return 1.0
    if f1 >= 0.4:
        return 0.5
    return 0.0


def _score_prediction(prediction: str, reference: str) -> dict[str, float]:
    f1 = _token_f1(prediction, reference)
    pred_len = len(prediction.split())
    ref_len = max(len(reference.split()), 1)
    length_ratio = min(pred_len / ref_len, ref_len / max(pred_len, 1))
    empathy = _empathy_score(prediction)
    task_success = _task_success(prediction, reference)
    return {
        "token_f1": round(f1, 4),
        "length_ratio": round(length_ratio, 4),
        "empathy_score": round(empathy, 4),
        "task_success": round(task_success, 4),
        "pass": 1.0 if (f1 >= 0.45 and length_ratio >= 0.55) else 0.0,
    }


def holdout_dataset(path: str) -> Dataset:
    """Execute holdout dataset."""
    return Dataset(name=Path(path).stem, examples=load_seed_file(path))


def generate_holdout_predictions(
    client: LLMClient,
    holdout: Dataset,
    system_prompt: str = "",
) -> Dataset:
    """Execute generate holdout predictions."""
    predictions = Dataset(name=f"{holdout.name}_predictions")
    for example in holdout.examples:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": example.user_message})
        response = client.complete(messages, temperature=0.2)
        predictions.add(
            Example(
                messages=[
                    Message(role=Role.USER, content=example.user_message),
                    Message(role=Role.ASSISTANT, content=response.strip()),
                ],
                metadata={"reference_assistant": example.assistant_message, **example.metadata},
            )
        )
    return predictions


def evaluate_prediction_dataset(predictions: Dataset, holdout: Dataset) -> dict[str, Any]:
    """Execute evaluate prediction dataset."""
    metrics = []
    for predicted, reference in zip(predictions.examples, holdout.examples):
        row = _score_prediction(predicted.assistant_message, reference.assistant_message)
        row["id"] = predicted.id
        row["topic"] = reference.metadata.get("topic", "unknown")
        metrics.append(row)

    def _avg(key: str) -> float:
        return round(sum(float(item[key]) for item in metrics) / max(len(metrics), 1), 4)

    topic_scores: dict[str, list[float]] = {}
    for item in metrics:
        topic_scores.setdefault(str(item["topic"]), []).append(float(item["task_success"]))

    return {
        "examples": len(metrics),
        "task_success_rate": _avg("task_success"),
        "pass_rate": _avg("pass"),
        "avg_token_f1": _avg("token_f1"),
        "avg_length_ratio": _avg("length_ratio"),
        "avg_empathy_score": _avg("empathy_score"),
        "topic_task_success": {
            topic: round(sum(values) / max(len(values), 1), 4)
            for topic, values in sorted(topic_scores.items())
        },
        "per_example": metrics,
    }


def compare_models_on_holdout(
    base_config: SDKConfig,
    finetuned_config: SDKConfig,
    holdout_path: str,
) -> dict[str, Any]:
    """Execute compare models on holdout."""
    holdout = holdout_dataset(holdout_path)
    base_predictions = generate_holdout_predictions(
        LLMClient(base_config.llm),
        holdout,
        system_prompt=base_config.generation.system_prompt,
    )
    finetuned_predictions = generate_holdout_predictions(
        LLMClient(finetuned_config.llm),
        holdout,
        system_prompt=finetuned_config.generation.system_prompt,
    )
    base_metrics = evaluate_prediction_dataset(base_predictions, holdout)
    finetuned_metrics = evaluate_prediction_dataset(finetuned_predictions, holdout)
    uplift = {
        "task_success_rate_delta": round(
            float(finetuned_metrics["task_success_rate"])
            - float(base_metrics["task_success_rate"]),
            4,
        ),
        "pass_rate_delta": round(
            float(finetuned_metrics["pass_rate"]) - float(base_metrics["pass_rate"]),
            4,
        ),
        "avg_token_f1_delta": round(
            float(finetuned_metrics["avg_token_f1"]) - float(base_metrics["avg_token_f1"]),
            4,
        ),
        "avg_empathy_score_delta": round(
            float(finetuned_metrics["avg_empathy_score"])
            - float(base_metrics["avg_empathy_score"]),
            4,
        ),
    }
    return {
        "holdout_path": holdout_path,
        "base_model": base_config.llm.model,
        "finetuned_model": finetuned_config.llm.model,
        "base_metrics": base_metrics,
        "finetuned_metrics": finetuned_metrics,
        "uplift": uplift,
    }


def export_uplift_results(
    results: dict[str, Any], output_dir: str, name: str = "uplift"
) -> list[str]:
    """Execute export uplift results."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    json_path = output_dir_path / f"{name}_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    base = results["base_metrics"]
    finetuned = results["finetuned_metrics"]
    uplift = results["uplift"]
    md_lines = [
        f"# Uplift Results: {results['base_model']} -> {results['finetuned_model']}",
        "",
        f"- Holdout: {results['holdout_path']}",
        "",
        "## Base Model",
        "",
        f"- Task success rate: {base['task_success_rate']:.2%}",
        f"- Pass rate: {base['pass_rate']:.2%}",
        f"- Avg token F1: {base['avg_token_f1']:.4f}",
        f"- Avg empathy score: {base['avg_empathy_score']:.4f}",
        "",
        "## Fine-Tuned Model",
        "",
        f"- Task success rate: {finetuned['task_success_rate']:.2%}",
        f"- Pass rate: {finetuned['pass_rate']:.2%}",
        f"- Avg token F1: {finetuned['avg_token_f1']:.4f}",
        f"- Avg empathy score: {finetuned['avg_empathy_score']:.4f}",
        "",
        "## Uplift",
        "",
        f"- Task success delta: {uplift['task_success_rate_delta']:+.2%}",
        f"- Pass rate delta: {uplift['pass_rate_delta']:+.2%}",
        f"- Token F1 delta: {uplift['avg_token_f1_delta']:+.4f}",
        f"- Empathy score delta: {uplift['avg_empathy_score_delta']:+.4f}",
    ]
    md_path = output_dir_path / f"{name}_results.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return [str(json_path), str(md_path)]


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=False))
    x_variance = sum((x - x_mean) ** 2 for x in xs)
    y_variance = sum((y - y_mean) ** 2 for y in ys)
    denominator = (x_variance * y_variance) ** 0.5
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def build_metric_validation_report(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a calibration report from eval summaries and uplift results."""
    task_success_pairs: list[tuple[float, float]] = []
    pass_rate_pairs: list[tuple[float, float]] = []
    token_f1_pairs: list[tuple[float, float]] = []
    calibration_errors: list[float] = []
    validated_scores: list[float] = []
    reference_alignment_scores: list[float] = []

    for run in runs:
        eval_summary = dict(run.get("eval_summary", {}))
        uplift = dict(run.get("uplift", {}))
        validation = dict(eval_summary.get("distribution_validation", {}))
        if not validation or not uplift:
            continue

        validated_score = float(validation.get("validated_distribution_match_score", 0.0))
        internal_score = float(validation.get("internal_distribution_match_score", 0.0))
        reference_alignment = float(validation.get("reference_alignment_score", 0.0))
        calibration_error = float(validation.get("calibration_error", 0.0))
        uplift_metrics = dict(uplift.get("uplift", {}))
        task_success_delta = float(uplift_metrics.get("task_success_rate_delta", 0.0))
        pass_rate_delta = float(uplift_metrics.get("pass_rate_delta", 0.0))
        token_f1_delta = float(uplift_metrics.get("avg_token_f1_delta", 0.0))

        task_success_pairs.append((validated_score, task_success_delta))
        pass_rate_pairs.append((validated_score, pass_rate_delta))
        token_f1_pairs.append((validated_score, token_f1_delta))
        calibration_errors.append(calibration_error)
        validated_scores.append(validated_score)
        reference_alignment_scores.append(reference_alignment)
        run["validation_summary"] = {
            "validated_distribution_match_score": validated_score,
            "internal_distribution_match_score": internal_score,
            "reference_alignment_score": reference_alignment,
            "calibration_error": calibration_error,
            "task_success_rate_delta": task_success_delta,
            "pass_rate_delta": pass_rate_delta,
            "avg_token_f1_delta": token_f1_delta,
        }

    validated_task_corr = _pearson_correlation(
        [score for score, _ in task_success_pairs],
        [delta for _, delta in task_success_pairs],
    )
    validated_pass_corr = _pearson_correlation(
        [score for score, _ in pass_rate_pairs],
        [delta for _, delta in pass_rate_pairs],
    )
    validated_token_corr = _pearson_correlation(
        [score for score, _ in token_f1_pairs],
        [delta for _, delta in token_f1_pairs],
    )

    return {
        "runs_analyzed": len(task_success_pairs),
        "avg_calibration_error": round(
            sum(calibration_errors) / max(len(calibration_errors), 1), 4
        ),
        "avg_validated_distribution_match_score": round(
            sum(validated_scores) / max(len(validated_scores), 1),
            4,
        )
        if validated_scores
        else 0.0,
        "avg_reference_alignment_score": round(
            sum(reference_alignment_scores) / max(len(reference_alignment_scores), 1),
            4,
        )
        if reference_alignment_scores
        else 0.0,
        "correlations": {
            "validated_match_vs_task_success_delta": validated_task_corr,
            "validated_match_vs_pass_rate_delta": validated_pass_corr,
            "validated_match_vs_token_f1_delta": validated_token_corr,
        },
        "runs": runs,
    }


def export_metric_validation_report(
    report: dict[str, Any],
    output_dir: str,
    name: str = "metric_validation",
) -> list[str]:
    """Write a metric-validation report to JSON and Markdown."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    json_path = output_dir_path / f"{name}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    correlations = dict(report.get("correlations", {}))
    lines = [
        "# Metric Validation Report",
        "",
        f"- Runs analyzed: {report.get('runs_analyzed', 0)}",
        f"- Avg calibration error: {float(report.get('avg_calibration_error', 0.0)):.4f}",
        (
            "- Avg validated distribution match score: "
            f"{float(report.get('avg_validated_distribution_match_score', 0.0)):.2f}/100"
        ),
        (
            "- Avg reference alignment score: "
            f"{float(report.get('avg_reference_alignment_score', 0.0)):.2f}/100"
        ),
        "",
        "## Correlations",
        "",
        (
            "- Validated match vs task success delta: "
            + (
                "n/a"
                if correlations.get("validated_match_vs_task_success_delta") is None
                else f"{float(correlations['validated_match_vs_task_success_delta']):+.4f}"
            )
        ),
        (
            "- Validated match vs pass rate delta: "
            + (
                "n/a"
                if correlations.get("validated_match_vs_pass_rate_delta") is None
                else f"{float(correlations['validated_match_vs_pass_rate_delta']):+.4f}"
            )
        ),
        (
            "- Validated match vs token F1 delta: "
            + (
                "n/a"
                if correlations.get("validated_match_vs_token_f1_delta") is None
                else f"{float(correlations['validated_match_vs_token_f1_delta']):+.4f}"
            )
        ),
        "",
        "## Run Summaries",
        "",
    ]
    runs = list(report.get("runs", []))
    if runs:
        for run in runs:
            summary = dict(run.get("validation_summary", {}))
            lines.extend(
                [
                    f"- {run.get('name', 'run')}: "
                    f"score={summary.get('validated_distribution_match_score', 0.0):.2f}, "
                    f"task_delta={summary.get('task_success_rate_delta', 0.0):+.4f}, "
                    f"pass_delta={summary.get('pass_rate_delta', 0.0):+.4f}, "
                    f"f1_delta={summary.get('avg_token_f1_delta', 0.0):+.4f}",
                ]
            )
    else:
        lines.append("- No valid runs found.")

    md_path = output_dir_path / f"{name}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(json_path), str(md_path)]
