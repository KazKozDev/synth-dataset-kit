"""Seed-to-dataset generator: amplify small seed sets into full datasets."""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from jinja2 import Template

from synth_dataset_kit.config import GenerationConfig
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, Role
from synth_dataset_kit.prompts import TEMPLATES
from synth_dataset_kit.utils import safe_slug

logger = logging.getLogger(__name__)


def _normalize_pair(user_text: str, assistant_text: str) -> tuple[str, str]:
    """Normalize a user/assistant pair for duplicate detection."""
    return (" ".join(user_text.lower().split()), " ".join(assistant_text.lower().split()))


def _slugify(value: str) -> str:
    """Convert free text to a stable identifier fragment."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "general"


def load_seed_file(path: str) -> list[Example]:
    """Load seed examples from JSONL file.

    Supports formats:
      - {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
      - {"user": "...", "assistant": "..."}
      - {"instruction": "...", "output": "..."}  (Alpaca)
      - {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}  (ShareGPT)
    """
    examples = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    with open(p) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                example = _parse_example(data)
                if example:
                    examples.append(example)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping line {line_num}: {e}")

    logger.info(f"Loaded {len(examples)} seed examples from {path}")
    return examples


def _parse_example(data: dict) -> Example | None:
    """Parse a single example from various formats."""
    messages = []

    # OpenAI / ChatML format
    if "messages" in data:
        for m in data["messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in ("user", "assistant", "system"):
                messages.append(Message(role=Role(role), content=content))

    # Simple user/assistant format
    elif "user" in data and "assistant" in data:
        messages = [
            Message(role=Role.USER, content=data["user"]),
            Message(role=Role.ASSISTANT, content=data["assistant"]),
        ]

    # Alpaca format
    elif "instruction" in data and "output" in data:
        user_msg = data["instruction"]
        if data.get("input"):
            user_msg += f"\n\n{data['input']}"
        messages = [
            Message(role=Role.USER, content=user_msg),
            Message(role=Role.ASSISTANT, content=data["output"]),
        ]

    # ShareGPT format
    elif "conversations" in data:
        role_map = {"human": Role.USER, "gpt": Role.ASSISTANT, "system": Role.SYSTEM}
        for turn in data["conversations"]:
            role = role_map.get(turn.get("from", ""), Role.USER)
            messages.append(Message(role=role, content=turn.get("value", "")))

    if len(messages) >= 2:
        return Example(messages=messages, metadata=data.get("metadata", {}))
    return None


class SeedExpander:
    """Analyze seed examples and generate diverse expansions."""

    def __init__(
        self,
        client: LLMClient,
        config: GenerationConfig,
        quality_judge=None,
        min_quality: float = 0.0,
        max_rounds: int = 3,
    ):
        self.client = client
        self.config = config
        self.quality_judge = quality_judge
        self.min_quality = min_quality
        self.max_rounds = max_rounds
        self.last_artifacts: dict[str, list[Example]] = {
            "candidates": [],
            "accepted": [],
            "rejected": [],
        }
        self.seed_embedding_model_name = "all-MiniLM-L6-v2"

    def analyze_seeds(self, seeds: list[Example]) -> dict:
        """Analyze seed examples to understand patterns."""
        template = Template(TEMPLATES["analyze_seeds"])
        prompt = template.render(seeds=seeds[:20])  # Limit to 20 for context

        result = self.client.complete_json(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        if isinstance(result, dict):
            result["seed_distribution_profile"] = self.build_seed_distribution_profile(
                seeds,
                result,
            )
            logger.info(f"Seed analysis — domain: {result.get('domain', 'unknown')}")
            return result

        fallback = {
            "domain": self.config.domain or "general",
            "tone": "neutral",
            "common_patterns": [],
            "topics_covered": [],
            "suggested_new_topics": [],
        }
        fallback["seed_distribution_profile"] = self.build_seed_distribution_profile(
            seeds,
            fallback,
        )
        return fallback

    def build_seed_distribution_profile(
        self,
        seeds: list[Example],
        analysis: dict,
    ) -> dict[str, object]:
        """Build a lightweight seed distribution profile for coverage-balanced generation."""
        embedding_cluster_ids = self._embedding_cluster_assignments(seeds)
        cluster_seed_texts: dict[str, list[str]] = {}
        topic_hints = [
            str(topic).strip()
            for topic in analysis.get("topics_covered", [])
            if str(topic).strip()
        ]
        clusters: dict[str, dict[str, object]] = {}
        for index, seed in enumerate(seeds):
            topic = self._infer_seed_topic(seed, topic_hints, analysis)
            complexity = self._infer_complexity_bucket(seed)
            style = self._infer_style_bucket(seed)
            semantic_cluster = embedding_cluster_ids[index]
            cluster_id = f"{_slugify(topic)}__c{semantic_cluster}__{complexity}__{style}"
            seed.metadata.setdefault("seed_cluster_id", cluster_id)
            seed.metadata.setdefault("seed_topic", topic)
            seed.metadata.setdefault("seed_complexity", complexity)
            seed.metadata.setdefault("seed_style", style)
            seed.metadata.setdefault("seed_semantic_cluster", semantic_cluster)

            cluster = clusters.setdefault(
                cluster_id,
                {
                    "cluster_id": cluster_id,
                    "semantic_cluster": semantic_cluster,
                    "topic": topic,
                    "complexity": complexity,
                    "style": style,
                    "difficulty": complexity,
                    "seed_count": 0,
                    "target_examples": 0,
                },
            )
            cluster["seed_count"] = int(cluster["seed_count"]) + 1
            cluster_seed_texts.setdefault(cluster_id, []).append(
                f"{seed.user_message}\n{seed.assistant_message}"
            )

        total_seed_count = max(len(seeds), 1)
        ordered_clusters = sorted(
            clusters.values(),
            key=lambda item: (-int(item["seed_count"]), str(item["cluster_id"])),
        )
        remaining = self.config.num_examples
        for index, cluster in enumerate(ordered_clusters):
            seed_count = int(cluster["seed_count"])
            if index == len(ordered_clusters) - 1:
                target_examples = remaining
            else:
                target_examples = max(
                    1,
                    round(self.config.num_examples * (seed_count / total_seed_count)),
                )
                target_examples = min(target_examples, remaining)
            cluster["target_examples"] = target_examples
            remaining -= target_examples

        if ordered_clusters and remaining > 0:
            ordered_clusters[0]["target_examples"] = int(ordered_clusters[0]["target_examples"]) + remaining

        semantic_graph = self._build_semantic_coverage_graph(
            ordered_clusters,
            cluster_seed_texts,
        )

        return {
            "clusters": ordered_clusters,
            "cluster_count": len(ordered_clusters),
            "embedding_clustering": any(cluster_id > 0 for cluster_id in embedding_cluster_ids) or len(set(embedding_cluster_ids)) > 1,
            "embedding_model": self.seed_embedding_model_name if len(set(embedding_cluster_ids)) > 1 else None,
            "semantic_graph": semantic_graph,
        }

    def _build_semantic_coverage_graph(
        self,
        clusters: list[dict[str, object]],
        cluster_seed_texts: dict[str, list[str]],
    ) -> dict[str, object]:
        """Build a lightweight cluster graph for distance-based coverage allocation."""
        if not clusters:
            return {"allocator": "none", "neighbors": {}, "cluster_count": 0}

        cluster_ids = [str(cluster.get("cluster_id", "unknown")) for cluster in clusters]
        neighbor_k = max(1, int(getattr(self.config, "graph_neighbor_k", 3)))
        centroid_vectors: dict[str, list[float]] = {}
        allocator = "lexical_fallback"

        try:
            from sentence_transformers import SentenceTransformer

            texts = [
                " ".join(cluster_seed_texts.get(cluster_id, [cluster_id]))
                for cluster_id in cluster_ids
            ]
            model = SentenceTransformer(self.seed_embedding_model_name)
            embeddings = model.encode(texts, normalize_embeddings=True)
            for cluster_id, vector in zip(cluster_ids, embeddings, strict=False):
                centroid_vectors[cluster_id] = [float(value) for value in vector]
            allocator = "embedding_centroid"
        except ImportError:
            for cluster in clusters:
                cluster_id = str(cluster.get("cluster_id", "unknown"))
                signature = " ".join(
                    [
                        str(cluster.get("topic", "")),
                        str(cluster.get("style", "")),
                        str(cluster.get("difficulty", "")),
                        " ".join(cluster_seed_texts.get(cluster_id, [])),
                    ]
                ).lower()
                tokens = sorted(set(re.findall(r"\w+", signature)))
                centroid_vectors[cluster_id] = [float(len(token)) for token in tokens[:24]]

        neighbors: dict[str, list[dict[str, float | str]]] = {}
        for cluster in clusters:
            cluster_id = str(cluster.get("cluster_id", "unknown"))
            source_vector = centroid_vectors.get(cluster_id, [])
            scored_neighbors: list[dict[str, float | str]] = []
            for other in clusters:
                other_id = str(other.get("cluster_id", "unknown"))
                if other_id == cluster_id:
                    continue
                similarity = self._vector_similarity(
                    source_vector,
                    centroid_vectors.get(other_id, []),
                )
                scored_neighbors.append(
                    {
                        "cluster_id": other_id,
                        "similarity": round(similarity, 4),
                        "distance": round(max(0.0, 1.0 - similarity), 4),
                    }
                )
            scored_neighbors.sort(
                key=lambda item: (-float(item["similarity"]), str(item["cluster_id"]))
            )
            neighbors[cluster_id] = scored_neighbors[:neighbor_k]

        return {
            "allocator": allocator,
            "neighbors": neighbors,
            "cluster_count": len(clusters),
        }

    def _vector_similarity(
        self,
        left: list[float],
        right: list[float],
    ) -> float:
        """Compute a bounded similarity score between two vectors."""
        if not left or not right:
            return 0.0
        overlap = min(len(left), len(right))
        if overlap == 0:
            return 0.0
        left = left[:overlap]
        right = right[:overlap]
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if not left_norm or not right_norm:
            return 0.0
        similarity = dot / (left_norm * right_norm)
        return max(0.0, min(1.0, float(similarity)))

    def _embedding_cluster_assignments(self, seeds: list[Example]) -> list[int]:
        """Assign seed examples to semantic clusters using embeddings when available."""
        if len(seeds) <= 1:
            return [0] * len(seeds)

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            return list(range(len(seeds)))

        texts = [f"{seed.user_message}\n{seed.assistant_message}" for seed in seeds]
        model = SentenceTransformer(self.seed_embedding_model_name)
        embeddings = model.encode(texts, normalize_embeddings=True)

        centroids: list[Any] = []
        cluster_counts: list[int] = []
        assignments: list[int] = []
        threshold = 0.82

        for vector in embeddings:
            if not centroids:
                centroids.append(vector)
                cluster_counts.append(1)
                assignments.append(0)
                continue

            scores = [float(np.dot(vector, centroid)) for centroid in centroids]
            best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
            best_score = scores[best_idx]
            if best_score >= threshold:
                count = cluster_counts[best_idx]
                centroids[best_idx] = ((centroids[best_idx] * count) + vector) / (count + 1)
                norm = np.linalg.norm(centroids[best_idx])
                if norm:
                    centroids[best_idx] = centroids[best_idx] / norm
                cluster_counts[best_idx] += 1
                assignments.append(best_idx)
            else:
                centroids.append(vector)
                cluster_counts.append(1)
                assignments.append(len(centroids) - 1)

        return assignments

    def _infer_seed_topic(
        self,
        example: Example,
        topic_hints: list[str],
        analysis: dict,
    ) -> str:
        """Infer a coarse topic label for a seed example."""
        existing = example.metadata.get("topic") or example.metadata.get("seed_topic")
        if existing:
            return str(existing)

        combined = f"{example.user_message} {example.assistant_message}".lower()
        for topic in topic_hints:
            topic_tokens = [token for token in re.findall(r"\w+", topic.lower()) if len(token) > 3]
            if topic_tokens and any(token in combined for token in topic_tokens):
                return topic

        domain = str(analysis.get("domain", self.config.domain or "general"))
        keyword_topics = {
            "billing": ["bill", "invoice", "refund", "payment", "subscription", "charge"],
            "account": ["account", "login", "password", "sign in", "reset", "profile"],
            "support": ["support", "help", "issue", "problem", "ticket"],
            "technical": ["api", "error", "bug", "timeout", "install", "configure"],
        }
        for topic, keywords in keyword_topics.items():
            if any(keyword in combined for keyword in keywords):
                return topic
        return domain

    def _infer_complexity_bucket(self, example: Example) -> str:
        """Infer a simple complexity bucket from message lengths."""
        user_words = len(example.user_message.split())
        assistant_words = len(example.assistant_message.split())
        combined = user_words + assistant_words
        if combined >= 80 or user_words >= 22 or assistant_words >= 55:
            return "hard"
        if combined >= 35 or user_words >= 10 or assistant_words >= 24:
            return "medium"
        return "easy"

    def _infer_style_bucket(self, example: Example) -> str:
        """Infer a coarse response style bucket."""
        assistant = example.assistant_message.strip()
        assistant_lower = assistant.lower()
        if any(token in assistant_lower for token in ["step", "first", "then", "next", "finally"]):
            return "procedural"
        if len(assistant.split()) >= 50 or assistant.count(".") >= 3:
            return "detailed"
        return "concise"

    def build_generation_plan(
        self,
        analysis: dict,
        accepted_examples: list[Example] | None = None,
        focus_cluster_ids: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Create a coverage-balanced generation plan from analysis output."""
        distribution = analysis.get("seed_distribution_profile", {})
        clusters = list(distribution.get("clusters", [])) if isinstance(distribution, dict) else []
        semantic_graph = dict(distribution.get("semantic_graph", {})) if isinstance(distribution, dict) else {}
        accepted_counts = self._accepted_counts_by_cluster(accepted_examples or [])
        focus_ids = {str(cluster_id) for cluster_id in (focus_cluster_ids or []) if cluster_id}
        if clusters:
            prioritized_clusters = self._prioritize_clusters(
                clusters,
                accepted_counts,
                focus_ids=focus_ids,
                semantic_graph=semantic_graph,
            )
            allocations = self._optimize_cluster_allocations(
                prioritized_clusters,
                accepted_counts,
            )
            plans: list[dict[str, str]] = []
            for cluster in prioritized_clusters:
                cluster_id = str(cluster.get("cluster_id", "general__medium__concise"))
                topic = str(cluster.get("topic", analysis.get("domain", self.config.domain or "general")))
                style = str(cluster.get("style", "concise"))
                difficulty = str(cluster.get("difficulty", cluster.get("complexity", "medium")))
                allocation = max(1, allocations.get(cluster_id, 0))
                for _ in range(allocation):
                    plans.append(
                        {
                            "cluster_id": cluster_id,
                            "topic": topic,
                            "style": style,
                            "persona": self._persona_for_cluster(style, difficulty),
                            "difficulty": difficulty,
                        }
                    )
                cluster["_planned_allocation"] = allocation
            return plans

        topics = list(analysis.get("suggested_new_topics", []))
        if not topics:
            topics = list(analysis.get("topics_covered", []))
        if not topics:
            topics = [analysis.get("domain", self.config.domain or "general")]
        if self.config.domain and self.config.domain not in topics:
            topics.append(self.config.domain)

        personas = self.config.personas or ["general"]
        difficulties = self.config.difficulty_levels or ["medium"]
        plans: list[dict[str, str]] = []
        for topic in topics:
            for persona in personas:
                for difficulty in difficulties:
                    plans.append(
                        {
                            "topic": topic,
                            "persona": persona,
                            "difficulty": difficulty,
                        }
                    )

        random.shuffle(plans)
        return plans or [{"topic": "general", "persona": "general", "difficulty": "medium"}]

    def _accepted_counts_by_cluster(self, examples: list[Example]) -> dict[str, int]:
        """Count accepted examples by generated cluster id."""
        counts: dict[str, int] = {}
        for example in examples:
            cluster_id = str(
                example.metadata.get("cluster_id")
                or example.metadata.get("seed_cluster_id")
                or "general__medium__concise"
            )
            counts[cluster_id] = counts.get(cluster_id, 0) + 1
        return counts

    def _semantic_cluster_status(
        self,
        clusters: list[dict[str, object]],
        accepted_counts: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Aggregate target/generated counts by semantic cluster."""
        target_counts: dict[str, int] = {}
        generated_counts: dict[str, int] = {}
        for cluster in clusters:
            cluster_id = str(cluster.get("cluster_id", "unknown"))
            semantic_cluster = str(cluster.get("semantic_cluster", "0"))
            target_counts[semantic_cluster] = (
                target_counts.get(semantic_cluster, 0)
                + int(cluster.get("target_examples", 0))
            )
            generated_counts[semantic_cluster] = (
                generated_counts.get(semantic_cluster, 0)
                + accepted_counts.get(cluster_id, 0)
            )
        gaps: dict[str, int] = {}
        for semantic_cluster in sorted(set(target_counts) | set(generated_counts)):
            target = target_counts.get(semantic_cluster, 0)
            actual = generated_counts.get(semantic_cluster, 0)
            if actual < target:
                gaps[semantic_cluster] = target - actual
        return target_counts, generated_counts, gaps

    def _prioritize_clusters(
        self,
        clusters: list[dict[str, object]],
        accepted_counts: dict[str, int],
        focus_ids: set[str] | None = None,
        semantic_graph: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Prioritize clusters using semantic gaps, saturation control, and long-tail balancing."""
        focus_ids = focus_ids or set()
        semantic_targets, semantic_generated, semantic_gaps = self._semantic_cluster_status(
            clusters,
            accepted_counts,
        )
        semantic_mode = any("semantic_cluster" in cluster for cluster in clusters)
        graph_neighbors = (
            dict(semantic_graph.get("neighbors", {}))
            if isinstance(semantic_graph, dict)
            else {}
        )
        seed_counts = [int(cluster.get("seed_count", 1)) for cluster in clusters]
        median_seed_count = sorted(seed_counts)[len(seed_counts) // 2] if seed_counts else 1
        max_active = max(
            1,
            int(getattr(self.config, "max_active_clusters_per_round", 12)),
        )
        saturation_threshold = float(getattr(self.config, "saturation_threshold", 1.05))
        long_tail_boost = float(getattr(self.config, "long_tail_boost", 0.35))
        semantic_focus_top_k = max(1, int(getattr(self.config, "semantic_focus_top_k", 3)))
        distance_allocator_weight = float(
            getattr(self.config, "distance_allocator_weight", 0.6)
        )

        ranked: list[dict[str, object]] = []
        for cluster in clusters:
            cluster_id = str(cluster.get("cluster_id", "unknown"))
            if focus_ids and cluster_id not in focus_ids:
                continue

            target = int(cluster.get("target_examples", 0))
            accepted = accepted_counts.get(cluster_id, 0)
            gap = max(0, target - accepted)
            if not focus_ids and gap <= 0:
                continue
            semantic_cluster = str(cluster.get("semantic_cluster", "0"))
            semantic_target = semantic_targets.get(semantic_cluster, 0)
            semantic_actual = semantic_generated.get(semantic_cluster, 0)
            semantic_gap = semantic_gaps.get(semantic_cluster, 0) if semantic_mode else 0
            saturation_ratio = accepted / max(target, 1)
            long_tail = int(cluster.get("seed_count", 1)) <= max(1, median_seed_count)
            long_tail_score = long_tail_boost if long_tail and semantic_mode else 0.0
            semantic_gap_ratio = (
                semantic_gap / max(semantic_target, 1)
                if semantic_mode
                else 0.0
            )
            graph_frontier_score = 0.0
            for neighbor in graph_neighbors.get(cluster_id, []):
                neighbor_id = str(neighbor.get("cluster_id", "unknown"))
                neighbor_cluster = next(
                    (
                        item for item in clusters
                        if str(item.get("cluster_id", "unknown")) == neighbor_id
                    ),
                    None,
                )
                if neighbor_cluster is None:
                    continue
                neighbor_target = int(neighbor_cluster.get("target_examples", 0))
                neighbor_gap = max(0, neighbor_target - accepted_counts.get(neighbor_id, 0))
                neighbor_gap_ratio = neighbor_gap / max(neighbor_target, 1)
                graph_frontier_score += float(neighbor.get("similarity", 0.0)) * neighbor_gap_ratio
            saturation_penalty = max(0.0, saturation_ratio - saturation_threshold) * 2.0
            priority = (
                float(gap)
                + semantic_gap_ratio
                + (graph_frontier_score * distance_allocator_weight)
                + long_tail_score
                - saturation_penalty
            )
            adaptive_bonus = 0
            if (
                semantic_mode
                and semantic_gap > 0
                and gap > 0
                and saturation_ratio < saturation_threshold
            ):
                adaptive_bonus = min(
                    2,
                    max(
                        0,
                        round(
                            (semantic_gap_ratio * semantic_focus_top_k)
                            + (graph_frontier_score * distance_allocator_weight)
                            + long_tail_score
                        ),
                    ),
                )
            enriched = dict(cluster)
            enriched["_priority_score"] = round(priority, 4)
            enriched["_adaptive_bonus"] = adaptive_bonus
            enriched["_semantic_gap"] = semantic_gap
            enriched["_semantic_gap_ratio"] = round(semantic_gap_ratio, 4)
            enriched["_graph_frontier_score"] = round(graph_frontier_score, 4)
            enriched["_saturation_ratio"] = round(saturation_ratio, 4)
            ranked.append(enriched)

        ranked.sort(
            key=lambda cluster: (
                -float(cluster.get("_priority_score", 0.0)),
                -int(cluster.get("target_examples", 0)),
                str(cluster.get("cluster_id", "")),
            )
        )
        if focus_ids:
            return ranked
        return ranked[:max_active]

    def _optimize_cluster_allocations(
        self,
        prioritized_clusters: list[dict[str, object]],
        accepted_counts: dict[str, int],
    ) -> dict[str, int]:
        """Allocate generation budget across clusters using iterative marginal utility."""
        if not prioritized_clusters:
            return {}

        remaining_gaps = {
            str(cluster.get("cluster_id", "unknown")): max(
                0,
                int(cluster.get("target_examples", 0))
                - accepted_counts.get(str(cluster.get("cluster_id", "unknown")), 0),
            )
            for cluster in prioritized_clusters
        }
        base_budget = sum(remaining_gaps.values())
        if base_budget <= 0:
            return {}

        allocations = {cluster_id: 0 for cluster_id in remaining_gaps}
        cluster_lookup = {
            str(cluster.get("cluster_id", "unknown")): cluster
            for cluster in prioritized_clusters
        }

        for _ in range(base_budget):
            best_cluster_id = None
            best_score = float("-inf")
            for cluster_id, gap in remaining_gaps.items():
                if gap <= 0:
                    continue
                cluster = cluster_lookup[cluster_id]
                target = max(1, int(cluster.get("target_examples", 1)))
                graph_score = float(cluster.get("_graph_frontier_score", 0.0))
                long_tail_score = float(cluster.get("_priority_score", 0.0)) - float(gap)
                current_fill = allocations[cluster_id] / target
                marginal_utility = (
                    float(gap)
                    + graph_score
                    + (long_tail_score * 0.15)
                    - current_fill
                )
                if marginal_utility > best_score:
                    best_score = marginal_utility
                    best_cluster_id = cluster_id
            if best_cluster_id is None:
                break
            allocations[best_cluster_id] += 1
            remaining_gaps[best_cluster_id] -= 1

        for cluster in prioritized_clusters:
            cluster_id = str(cluster.get("cluster_id", "unknown"))
            allocations[cluster_id] += int(cluster.get("_adaptive_bonus", 0))

        return allocations

    def _graph_coverage_score(
        self,
        clusters: list[dict[str, object]],
        accepted_counts: dict[str, int],
        semantic_graph: dict[str, object],
    ) -> float:
        """Estimate how well generated examples cover the semantic graph."""
        if not clusters:
            return 0.0
        graph_neighbors = (
            dict(semantic_graph.get("neighbors", {}))
            if isinstance(semantic_graph, dict)
            else {}
        )
        cluster_lookup = {
            str(cluster.get("cluster_id", "unknown")): cluster
            for cluster in clusters
        }
        weighted_total = 0.0
        weighted_covered = 0.0
        for cluster in clusters:
            cluster_id = str(cluster.get("cluster_id", "unknown"))
            target = int(cluster.get("target_examples", 0))
            if target <= 0:
                continue
            actual = accepted_counts.get(cluster_id, 0)
            direct_coverage = min(1.0, actual / target)
            neighbor_support = 0.0
            for neighbor in graph_neighbors.get(cluster_id, []):
                neighbor_id = str(neighbor.get("cluster_id", "unknown"))
                neighbor_cluster = cluster_lookup.get(neighbor_id)
                if neighbor_cluster is None:
                    continue
                neighbor_target = int(neighbor_cluster.get("target_examples", 0))
                neighbor_actual = accepted_counts.get(neighbor_id, 0)
                neighbor_coverage = min(1.0, neighbor_actual / max(neighbor_target, 1))
                neighbor_support = max(
                    neighbor_support,
                    float(neighbor.get("similarity", 0.0)) * neighbor_coverage,
                )
            blended = min(1.0, (direct_coverage * 0.8) + (neighbor_support * 0.2))
            weighted_total += target
            weighted_covered += target * blended
        if weighted_total <= 0:
            return 0.0
        return round(weighted_covered / weighted_total, 4)

    def _persona_for_cluster(self, style: str, difficulty: str) -> str:
        """Choose a persona that roughly matches the cluster profile."""
        if difficulty == "hard":
            return "expert"
        if style == "concise":
            return "beginner"
        return "skeptic" if style == "detailed" else "general"

    def _select_seeds_for_cluster(
        self,
        seeds: list[Example],
        cluster_id: str | None,
    ) -> list[Example]:
        """Return seeds that belong to the requested cluster."""
        if not cluster_id:
            return list(seeds)
        selected = [
            seed
            for seed in seeds
            if str(seed.metadata.get("seed_cluster_id")) == str(cluster_id)
        ]
        return selected or list(seeds)

    def generate_candidates(
        self,
        seeds: list[Example],
        analysis: dict,
        target_count: int | None = None,
        accepted_examples: list[Example] | None = None,
        focus_cluster_ids: list[str] | None = None,
    ) -> list[Example]:
        """Generate raw candidate examples before dedup/finalization."""
        target = target_count or self.config.num_examples
        generated = 0
        candidates: list[Example] = []
        plan = self.build_generation_plan(
            analysis,
            accepted_examples=accepted_examples,
            focus_cluster_ids=focus_cluster_ids,
        )
        plan_index = 0

        logger.info(
            f"Expanding {len(seeds)} seeds → {target} examples "
            f"across {len(plan)} planned topic/persona/difficulty combinations"
        )

        while generated < target:
            stage = plan[plan_index % len(plan)]
            plan_index += 1

            topic = stage["topic"]
            persona = stage["persona"]
            difficulty = stage["difficulty"]
            cluster_id = stage.get("cluster_id")
            style = stage.get("style", "concise")
            cluster_seeds = self._select_seeds_for_cluster(seeds, cluster_id)
            seed_pool = cluster_seeds or seeds
            sample_seeds = random.sample(seed_pool, min(3, len(seed_pool)))
            remaining = min(self.config.batch_size, target - generated)

            template = Template(TEMPLATES["seed_expand"])
            prompt = template.render(
                domain=analysis.get("domain", self.config.domain),
                tone=analysis.get("tone", "neutral"),
                patterns=", ".join(analysis.get("common_patterns", [])),
                difficulty=difficulty,
                persona=persona,
                topic=topic,
                style=style,
                language=self.config.language,
                system_prompt=self.config.system_prompt,
                sample_seeds=sample_seeds,
                batch_size=remaining,
            )

            try:
                result = self.client.complete_json(
                    [{"role": "user", "content": prompt}],
                    temperature=self.config.generation_temperature
                    if hasattr(self.config, "generation_temperature")
                    else 0.8,
                )

                examples_data = result.get("examples", []) if isinstance(result, dict) else []
                for ex_data in examples_data:
                    if generated >= target:
                        break
                    user_msg = ex_data.get("user", "")
                    asst_msg = ex_data.get("assistant", "")
                    if user_msg and asst_msg:
                        messages = [
                            Message(role=Role.USER, content=user_msg),
                            Message(role=Role.ASSISTANT, content=asst_msg),
                        ]
                        if self.config.system_prompt:
                            messages.insert(
                                0,
                                Message(role=Role.SYSTEM, content=self.config.system_prompt),
                            )
                        example = Example(
                            messages=messages,
                            metadata={
                                "source": "seed_expansion_candidate",
                                "pipeline_stage": "generate_candidates",
                                "topic": topic,
                                "persona": persona,
                                "difficulty": difficulty,
                                "style": style,
                                "cluster_id": cluster_id or "general__medium__concise",
                                "analysis_domain": analysis.get("domain", self.config.domain or "general"),
                            },
                        )
                        candidates.append(example)
                        generated += 1

                logger.info(f"Generated {generated}/{target} examples")

            except Exception as e:
                logger.error(f"Generation batch failed: {e}")
                continue

        return candidates

    def _distribution_status(
        self,
        analysis: dict,
        accepted_examples: list[Example],
    ) -> dict[str, object]:
        """Summarize current cluster gaps and divergence against the seed target profile."""
        distribution = analysis.get("seed_distribution_profile", {})
        clusters = list(distribution.get("clusters", [])) if isinstance(distribution, dict) else []
        semantic_graph = dict(distribution.get("semantic_graph", {})) if isinstance(distribution, dict) else {}
        accepted_counts = self._accepted_counts_by_cluster(accepted_examples)
        target_counts = {
            str(cluster.get("cluster_id", "unknown")): int(cluster.get("target_examples", 0))
            for cluster in clusters
        }
        semantic_target_counts, semantic_generated_counts, semantic_coverage_gaps = (
            self._semantic_cluster_status(clusters, accepted_counts)
        )
        total_target = max(sum(target_counts.values()), 1)
        total_accepted = max(sum(accepted_counts.values()), 1)
        gaps: dict[str, int] = {}
        oversaturated_clusters: dict[str, int] = {}
        divergence = 0.0
        saturation_threshold = float(getattr(self.config, "saturation_threshold", 1.05))

        for cluster_id in sorted(set(target_counts) | set(accepted_counts)):
            target = target_counts.get(cluster_id, 0)
            actual = accepted_counts.get(cluster_id, 0)
            if actual < target:
                gaps[cluster_id] = target - actual
            if target > 0 and (actual / target) > saturation_threshold:
                oversaturated_clusters[cluster_id] = actual - target
            divergence += abs((target / total_target) - (actual / total_accepted))

        prioritized = [
            cluster_id
            for cluster_id, _gap in sorted(gaps.items(), key=lambda item: (-item[1], item[0]))
        ]
        semantic_prioritized_cluster_ids = [
            str(cluster.get("cluster_id", "unknown"))
            for cluster in self._prioritize_clusters(
                clusters,
                accepted_counts,
                semantic_graph=semantic_graph,
            )
        ]
        graph_coverage_score = self._graph_coverage_score(
            clusters,
            accepted_counts,
            semantic_graph,
        )
        graph_frontier_clusters = [
            str(cluster.get("cluster_id", "unknown"))
            for cluster in self._prioritize_clusters(
                clusters,
                accepted_counts,
                semantic_graph=semantic_graph,
            )[:5]
        ]

        total_semantic_target = max(sum(semantic_target_counts.values()), 1)
        matched_semantic = 0
        for semantic_cluster, target in semantic_target_counts.items():
            matched_semantic += min(target, semantic_generated_counts.get(semantic_cluster, 0))
        semantic_coverage_score = round(matched_semantic / total_semantic_target, 4)
        gap_ratio = round(sum(gaps.values()) / total_target, 4)
        distribution_match_score = round(
            max(
                0.0,
                1.0
                - min(
                    1.0,
                    (round(divergence / 2, 4) * 0.5)
                    + (gap_ratio * 0.3)
                    + ((1.0 - semantic_coverage_score) * 0.1)
                    + ((1.0 - graph_coverage_score) * 0.1),
                ),
            )
            * 100,
            2,
        )
        return {
            "accepted_counts": accepted_counts,
            "target_counts": target_counts,
            "gaps": gaps,
            "prioritized_cluster_ids": prioritized,
            "semantic_prioritized_cluster_ids": semantic_prioritized_cluster_ids,
            "oversaturated_clusters": oversaturated_clusters,
            "distribution_divergence": round(divergence / 2, 4),
            "distribution_match_score": distribution_match_score,
            "semantic_cluster_target_distribution": semantic_target_counts,
            "semantic_cluster_generated_distribution": semantic_generated_counts,
            "semantic_coverage_score": semantic_coverage_score,
            "semantic_coverage_gaps": semantic_coverage_gaps,
            "graph_coverage_score": graph_coverage_score,
            "graph_frontier_clusters": graph_frontier_clusters,
        }

    def dedup_candidates(
        self,
        seeds: list[Example],
        candidates: list[Example],
        existing_examples: list[Example] | None = None,
    ) -> list[Example]:
        """Remove exact duplicates against seeds and previously accepted candidates."""
        seen_pairs = {
            _normalize_pair(seed.user_message, seed.assistant_message)
            for seed in seeds
        }
        for example in existing_examples or []:
            seen_pairs.add(_normalize_pair(example.user_message, example.assistant_message))
        unique_candidates: list[Example] = []
        duplicates_removed = 0

        for example in candidates:
            pair = _normalize_pair(example.user_message, example.assistant_message)
            if pair in seen_pairs:
                duplicates_removed += 1
                continue
            seen_pairs.add(pair)
            example.metadata["pipeline_stage"] = "deduped"
            unique_candidates.append(example)

        logger.info(
            f"Deduped candidates: kept {len(unique_candidates)}/{len(candidates)}, "
            f"removed {duplicates_removed} duplicates"
        )
        return unique_candidates

    def audit_candidates(self, candidates: list[Example]) -> list[Example]:
        """Score candidates before finalization."""
        if not self.quality_judge:
            for example in candidates:
                example.metadata["pipeline_stage"] = "audited"
            return candidates

        for example in candidates:
            self.quality_judge.score_example(example)
            example.metadata["pipeline_stage"] = "audited"
        return candidates

    def keep_best(
        self,
        candidates: list[Example],
        limit: int,
    ) -> list[Example]:
        """Keep the strongest candidates above the quality threshold."""
        if not candidates:
            return []

        if self.quality_judge:
            kept = [
                example
                for example in candidates
                if (example.quality_score or 0.0) >= self.min_quality
            ]
            kept.sort(key=lambda example: example.quality_score or 0.0, reverse=True)
        else:
            kept = list(candidates)

        selected = kept[:limit]
        for example in selected:
            example.metadata["pipeline_stage"] = "keep_best"
        logger.info(
            f"Selected {len(selected)}/{len(candidates)} audited candidates "
            f"(min_quality={self.min_quality:.1f})"
        )
        return selected

    def split_selection(
        self,
        candidates: list[Example],
        selected: list[Example],
    ) -> tuple[list[Example], list[Example]]:
        """Split audited candidates into accepted and rejected lists with reasons."""
        selected_ids = {example.id for example in selected}
        accepted: list[Example] = []
        rejected: list[Example] = []

        for example in candidates:
            if example.id in selected_ids:
                example.metadata["selection_decision"] = "accepted"
                accepted.append(example)
                continue

            reasons = list(example.metadata.get("quality_issues", []))
            if self.quality_judge and (example.quality_score or 0.0) < self.min_quality:
                reasons.append(f"below_min_quality:{self.min_quality:.1f}")
            example.metadata["selection_decision"] = "rejected"
            example.metadata["rejection_reasons"] = sorted(set(reasons or ["not_selected"]))
            rejected.append(example)

        return accepted, rejected

    def finalize_dataset(self, analysis: dict, candidates: list[Example]) -> Dataset:
        """Create the final dataset from accepted candidates."""
        safe_domain = safe_slug(str(analysis.get("domain", "dataset")))
        dataset = Dataset(name=f"expanded_{safe_domain}")
        for example in candidates[: self.config.num_examples]:
            example.metadata["pipeline_stage"] = "finalized"
            dataset.add(example)
        logger.info(f"Finalized dataset with {dataset.size} examples")
        return dataset

    def expand(self, seeds: list[Example], analysis: dict) -> Dataset:
        """Run the seed expansion pipeline through explicit stages."""
        accepted: list[Example] = []
        rejected: list[Example] = []
        all_candidates: list[Example] = []
        rebalancing_history: list[dict[str, object]] = []
        rounds = 0
        target = self.config.num_examples
        divergence_threshold = float(getattr(self.config, "divergence_threshold", 0.15))
        focus_top_k_clusters = max(1, int(getattr(self.config, "focus_top_k_clusters", 2)))
        rebalancing_strategy = str(getattr(self.config, "rebalancing_strategy", "strict"))
        focus_cluster_ids: list[str] | None = None

        while len(accepted) < target and rounds < self.max_rounds:
            rounds += 1
            remaining = target - len(accepted)
            request_count = max(remaining, self.config.batch_size)
            logger.info(
                f"Seed expansion round {rounds}/{self.max_rounds}: "
                f"need {remaining}, requesting {request_count} candidates"
            )

            raw_candidates = self.generate_candidates(
                seeds,
                analysis,
                target_count=request_count,
                accepted_examples=accepted,
                focus_cluster_ids=focus_cluster_ids,
            )
            all_candidates.extend(raw_candidates)
            unique_candidates = self.dedup_candidates(
                seeds,
                raw_candidates,
                existing_examples=accepted,
            )
            audited_candidates = self.audit_candidates(unique_candidates)
            kept_candidates = self.keep_best(audited_candidates, limit=remaining)
            accepted_batch, rejected_batch = self.split_selection(
                audited_candidates,
                kept_candidates,
            )
            accepted.extend(accepted_batch)
            rejected.extend(rejected_batch)

            distribution_status = self._distribution_status(analysis, accepted)
            rebalancing_history.append(
                {
                    "round": rounds,
                    "requested": request_count,
                    "accepted_total": len(accepted),
                    "accepted_batch": len(accepted_batch),
                    "rejected_batch": len(rejected_batch),
                    "distribution_divergence": distribution_status["distribution_divergence"],
                    "top_cluster_gaps": list(distribution_status["gaps"].items())[:5],
                    "semantic_coverage_score": distribution_status["semantic_coverage_score"],
                    "top_semantic_gaps": list(distribution_status["semantic_coverage_gaps"].items())[:5],
                    "focus_cluster_ids": list(focus_cluster_ids or []),
                }
            )
            prioritized = list(
                distribution_status.get("semantic_prioritized_cluster_ids")
                or distribution_status["prioritized_cluster_ids"]
            )
            if rebalancing_strategy == "soft" and prioritized:
                focus_cluster_ids = prioritized[:focus_top_k_clusters] + prioritized[focus_top_k_clusters:focus_top_k_clusters + 1]
            else:
                focus_cluster_ids = prioritized[:focus_top_k_clusters] or None

            if not kept_candidates:
                logger.warning("No acceptable candidates in this round; stopping early")
                break
            if len(accepted) >= target:
                break
            if (
                len(accepted) > 0
                and not distribution_status["gaps"]
                and float(distribution_status["distribution_divergence"]) <= divergence_threshold
            ):
                logger.info(
                    "Stopping early: distribution divergence is within threshold "
                    f"({distribution_status['distribution_divergence']:.4f})"
                )
                break

        self.last_artifacts = {
            "candidates": list(all_candidates),
            "accepted": list(accepted),
            "rejected": list(rejected),
        }
        dataset = self.finalize_dataset(analysis, accepted)
        dataset.artifacts = self.last_artifacts
        dataset.config_snapshot["seed_distribution_profile"] = analysis.get(
            "seed_distribution_profile",
            {},
        )
        dataset.config_snapshot["accepted_cluster_counts"] = self._accepted_counts_by_cluster(
            accepted,
        )
        dataset.config_snapshot["rebalancing_history"] = rebalancing_history
        dataset.config_snapshot["final_distribution_status"] = self._distribution_status(
            analysis,
            accepted,
        )
        return dataset
