"""Quality assessment using LLM scoring plus transparent rule-based checks."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from jinja2 import Template

from synth_dataset_kit.config import QualityConfig
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, QualityReport
from synth_dataset_kit.prompts import TEMPLATES

logger = logging.getLogger(__name__)


class QualityJudge:
    """Evaluate and score dataset examples using LLM-as-a-judge."""

    def __init__(self, client: LLMClient, config: QualityConfig, system_prompt: str = ""):
        self.client = client
        self.config = config
        self.system_prompt = system_prompt

    def score_example(self, example: Example) -> float:
        """Score a single example, returning overall score 1-10."""
        rule_issues = self._rule_issues(example)
        example.metadata["quality_rule_issues"] = rule_issues

        template = Template(TEMPLATES["quality_judge"])
        prompt = template.render(
            system_prompt=self.system_prompt,
            user_message=example.user_message,
            assistant_message=example.assistant_message,
        )

        try:
            result = self.client.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            if isinstance(result, dict):
                overall = float(result.get("overall", 5))
                overall = max(1.0, overall - self._rule_penalty(rule_issues))
                example.quality_score = overall

                # Combine LLM judge issues with deterministic rule flags.
                issues = list(rule_issues)
                if result.get("has_pii"):
                    issues.append("pii_detected")
                if result.get("has_toxic_content"):
                    issues.append("toxic_content")
                if result.get("issues"):
                    issues.extend(result["issues"])
                example.metadata["quality_issues"] = sorted(set(issues))
                example.metadata["quality_details"] = {
                    k: result.get(k)
                    for k in ["relevance", "accuracy", "completeness", "naturalness", "helpfulness"]
                    if k in result
                }

                return overall

        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")
            example.quality_score = max(1.0, 5.0 - self._rule_penalty(rule_issues))
            example.metadata["quality_issues"] = sorted(set(rule_issues + ["llm_scoring_failed"]))

        return example.quality_score or 5.0

    def score_dataset(self, dataset: Dataset) -> Dataset:
        """Score all examples in a dataset."""
        total = dataset.size
        logger.info(f"Scoring {total} examples...")

        for i, example in enumerate(dataset.examples):
            self.score_example(example)
            if (i + 1) % 10 == 0:
                logger.info(f"Scored {i + 1}/{total}")

        return dataset

    def generate_report(self, dataset: Dataset) -> QualityReport:
        """Generate a comprehensive quality report."""
        scores = [e.quality_score or 0 for e in dataset.examples]
        passing = [s for s in scores if s >= self.config.min_score]

        # Score distribution buckets
        distribution = Counter()
        for s in scores:
            bucket = f"{int(s)}-{int(s)+1}"
            distribution[bucket] += 1

        # Length stats
        user_lengths = [len(e.user_message.split()) for e in dataset.examples]
        asst_lengths = [len(e.assistant_message.split()) for e in dataset.examples]

        # Diversity metrics
        diversity = self._compute_diversity(dataset)
        self_bleu_proxy = round(max(0.0, 1.0 - diversity), 4)
        lexical_diversity = self._compute_lexical_diversity(dataset)
        embedding_diversity = self._compute_embedding_diversity(dataset)

        # Topic coverage
        topic_counts: Counter = Counter()
        difficulty_counts: Counter = Counter()
        topic_heatmap: dict[str, dict[str, int]] = {}
        for e in dataset.examples:
            topic = e.metadata.get("topic", "unknown")
            topic_counts[topic] += 1
            difficulty = self._infer_difficulty(e)
            difficulty_counts[difficulty] += 1
            topic_heatmap.setdefault(topic, {"easy": 0, "medium": 0, "hard": 0})
            topic_heatmap[topic][difficulty] += 1

        # Rule-based transparency
        issue_counts: Counter = Counter()
        for e in dataset.examples:
            for issue in e.metadata.get("quality_issues", []):
                issue_counts[issue] += 1

        seed_cluster_distribution, generated_cluster_distribution, distribution_divergence, underrepresented_clusters = self._distribution_alignment(dataset)
        final_distribution_status = dict(dataset.config_snapshot.get("final_distribution_status", {}))

        duplicate_groups, near_duplicate_examples = self._duplicate_stats(dataset)

        # Contamination stats
        contaminated = [e for e in dataset.examples if e.decontamination_flags]
        contaminated_benchmarks = set()
        contamination_verdicts: Counter = Counter()
        contamination_methods: Counter = Counter()
        contamination_method_benchmarks: dict[str, Counter] = defaultdict(Counter)
        contamination_evidence_samples: list[dict[str, object]] = []
        for e in dataset.examples:
            contamination_verdicts[str(e.metadata.get("contamination_verdict", "clean"))] += 1
        for e in contaminated:
            contaminated_benchmarks.update(e.decontamination_flags)
            for evidence in e.decontamination_evidence:
                method = str(evidence.get("method", "unknown"))
                benchmark = str(evidence.get("benchmark", "unknown"))
                contamination_methods[method] += 1
                contamination_method_benchmarks[benchmark][method] += 1
                if len(contamination_evidence_samples) < 10:
                    contamination_evidence_samples.append(
                        {
                            "example_id": e.id,
                            "benchmark": benchmark,
                            "method": method,
                            "confidence": evidence.get("confidence"),
                            "matched_text": evidence.get("matched_text", ""),
                        }
                    )

        return QualityReport(
            dataset_name=dataset.name,
            total_examples=dataset.size,
            passed_examples=len(passing),
            failed_examples=dataset.size - len(passing),
            avg_quality_score=sum(scores) / max(len(scores), 1),
            score_distribution=dict(distribution),
            avg_user_length=sum(user_lengths) / max(len(user_lengths), 1),
            avg_assistant_length=sum(asst_lengths) / max(len(asst_lengths), 1),
            diversity_score=diversity,
            self_bleu_proxy=self_bleu_proxy,
            lexical_diversity=lexical_diversity,
            embedding_diversity_score=embedding_diversity,
            diversity_method="embedding_cosine" if embedding_diversity is not None else "ngram_jaccard",
            difficulty_distribution=dict(difficulty_counts),
            topic_coverage=dict(topic_counts.most_common(20)),
            topic_heatmap=topic_heatmap,
            seed_cluster_distribution=seed_cluster_distribution,
            generated_cluster_distribution=generated_cluster_distribution,
            distribution_divergence=distribution_divergence,
            distribution_match_score=float(final_distribution_status.get("distribution_match_score", 0.0)),
            semantic_cluster_target_distribution=dict(final_distribution_status.get("semantic_cluster_target_distribution", {})),
            semantic_cluster_generated_distribution=dict(final_distribution_status.get("semantic_cluster_generated_distribution", {})),
            semantic_coverage_score=float(final_distribution_status.get("semantic_coverage_score", 0.0)),
            semantic_coverage_gaps=dict(final_distribution_status.get("semantic_coverage_gaps", {})),
            graph_coverage_score=float(final_distribution_status.get("graph_coverage_score", 0.0)),
            graph_frontier_clusters=list(final_distribution_status.get("graph_frontier_clusters", [])),
            underrepresented_clusters=underrepresented_clusters,
            rebalancing_history=list(dataset.config_snapshot.get("rebalancing_history", [])),
            final_distribution_status=final_distribution_status,
            issue_counts=dict(issue_counts.most_common()),
            duplicate_groups=duplicate_groups,
            near_duplicate_examples=near_duplicate_examples,
            contamination_hits=len(contaminated),
            contaminated_benchmarks=list(contaminated_benchmarks),
            contamination_verdicts=dict(contamination_verdicts),
            contamination_methods=dict(contamination_methods),
            contamination_method_benchmarks={
                benchmark: dict(method_counts)
                for benchmark, method_counts in contamination_method_benchmarks.items()
            },
            contamination_evidence_samples=contamination_evidence_samples,
            benchmark_sources=dict(dataset.config_snapshot.get("benchmark_sources", {})),
            benchmark_sample_counts=dict(dataset.config_snapshot.get("benchmark_sample_counts", {})),
            benchmark_load_errors=dict(dataset.config_snapshot.get("benchmark_load_errors", {})),
            audit_method="llm_judge+rules",
        )

    def _distribution_alignment(
        self,
        dataset: Dataset,
    ) -> tuple[dict[str, int], dict[str, int], float, dict[str, int]]:
        """Compare planned seed cluster distribution against generated output."""
        profile = dataset.config_snapshot.get("seed_distribution_profile", {}) or {}
        clusters = profile.get("clusters", []) if isinstance(profile, dict) else []
        seed_cluster_distribution = {
            str(cluster.get("cluster_id", "unknown")): int(cluster.get("target_examples", 0))
            for cluster in clusters
        }
        generated_counter: Counter = Counter()
        for example in dataset.examples:
            cluster_id = str(example.metadata.get("cluster_id", "unknown"))
            generated_counter[cluster_id] += 1

        all_cluster_ids = sorted(set(seed_cluster_distribution) | set(generated_counter))
        if not all_cluster_ids:
            return {}, {}, 0.0, {}

        planned_total = max(sum(seed_cluster_distribution.values()), 1)
        generated_total = max(sum(generated_counter.values()), 1)
        divergence = 0.0
        underrepresented: dict[str, int] = {}

        for cluster_id in all_cluster_ids:
            planned = seed_cluster_distribution.get(cluster_id, 0)
            actual = generated_counter.get(cluster_id, 0)
            divergence += abs((planned / planned_total) - (actual / generated_total))
            if actual < planned:
                underrepresented[cluster_id] = planned - actual

        return (
            seed_cluster_distribution,
            dict(generated_counter),
            round(divergence / 2, 4),
            underrepresented,
        )

    def _compute_diversity(self, dataset: Dataset, n: int = 3) -> float:
        """Compute self-BLEU-like diversity score (0-1, higher = more diverse)."""
        if dataset.size < 2:
            return 1.0

        # Use trigram overlap as a proxy for diversity
        all_ngrams: list[set[str]] = []
        for example in dataset.examples:
            text = example.assistant_message.lower()
            words = text.split()
            ngrams = set()
            for i in range(len(words) - n + 1):
                ngrams.add(" ".join(words[i : i + n]))
            all_ngrams.append(ngrams)

        # Average pairwise Jaccard distance (sample for efficiency)
        import random

        sample_size = min(50, len(all_ngrams))
        sample = random.sample(range(len(all_ngrams)), sample_size)

        total_distance = 0.0
        pairs = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                set_a = all_ngrams[sample[i]]
                set_b = all_ngrams[sample[j]]
                if not set_a and not set_b:
                    continue
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                jaccard = intersection / max(union, 1)
                total_distance += 1 - jaccard
                pairs += 1

        return round(total_distance / max(pairs, 1), 4)

    def _compute_lexical_diversity(self, dataset: Dataset) -> float:
        """Compute corpus-level type-token ratio on assistant responses."""
        tokens: list[str] = []
        for example in dataset.examples:
            tokens.extend(example.assistant_message.lower().split())
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / len(tokens), 4)

    def _compute_embedding_diversity(self, dataset: Dataset) -> float | None:
        """Optionally compute semantic diversity with sentence embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            return None

        texts = [example.assistant_message for example in dataset.examples if example.assistant_message.strip()]
        if len(texts) < 2:
            return 1.0

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts[:50], normalize_embeddings=True)
        if len(embeddings) < 2:
            return 1.0

        total_distance = 0.0
        pairs = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = float(np.dot(embeddings[i], embeddings[j]))
                total_distance += 1 - similarity
                pairs += 1
        return round(total_distance / max(pairs, 1), 4)

    def _rule_issues(self, example: Example) -> list[str]:
        """Cheap, deterministic checks that explain common failures."""
        issues: list[str] = []
        user_words = example.user_message.split()
        assistant_words = example.assistant_message.split()
        assistant_lower = example.assistant_message.lower()

        if len(user_words) < 3:
            issues.append("user_too_short")
        if len(assistant_words) < 12:
            issues.append("assistant_too_short")
        if len(assistant_words) > 450:
            issues.append("assistant_too_long")
        if example.user_message.strip() == example.assistant_message.strip():
            issues.append("answer_copies_prompt")
        if any(token in assistant_lower for token in ["lorem ipsum", "todo", "placeholder"]):
            issues.append("placeholder_content")
        if assistant_lower.count("i don't know") + assistant_lower.count("cannot help") > 0:
            issues.append("weak_answer")
        return issues

    def _rule_penalty(self, issues: list[str]) -> float:
        penalty = 0.0
        weights = {
            "user_too_short": 0.5,
            "assistant_too_short": 1.5,
            "assistant_too_long": 0.5,
            "answer_copies_prompt": 2.5,
            "placeholder_content": 3.0,
            "weak_answer": 1.5,
        }
        for issue in issues:
            penalty += weights.get(issue, 0.5)
        return penalty

    def _duplicate_stats(self, dataset: Dataset) -> tuple[int, int]:
        """Count duplicate groups using normalized user/assistant pairs."""
        counts: Counter = Counter()
        for example in dataset.examples:
            key = (
                " ".join(example.user_message.lower().split()),
                " ".join(example.assistant_message.lower().split()),
            )
            counts[key] += 1

        duplicate_groups = sum(1 for count in counts.values() if count > 1)
        duplicate_examples = sum(count for count in counts.values() if count > 1)
        return duplicate_groups, duplicate_examples

    def _infer_difficulty(self, example: Example) -> str:
        """Use metadata first, then fall back to a simple heuristic."""
        metadata_difficulty = str(example.metadata.get("difficulty", "")).lower()
        if metadata_difficulty in {"easy", "medium", "hard"}:
            return metadata_difficulty

        user_len = len(example.user_message.split())
        assistant_len = len(example.assistant_message.split())
        score = example.quality_score or 0.0

        if user_len <= 8 and assistant_len <= 40 and score < 7.5:
            return "easy"
        if user_len >= 20 or assistant_len >= 120 or score >= 8.5:
            return "hard"
        return "medium"
