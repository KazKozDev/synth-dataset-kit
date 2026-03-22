"""Topic tree generator for diverse dataset creation from domain descriptions."""

from __future__ import annotations

import logging
import random

from jinja2 import Template

from synth_dataset_kit.config import GenerationConfig
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, Example, Message, Role
from synth_dataset_kit.prompts import TEMPLATES
from synth_dataset_kit.utils import safe_slug

logger = logging.getLogger(__name__)


class TopicTreeGenerator:
    """Generate datasets using a topic tree for comprehensive domain coverage."""

    def __init__(self, client: LLMClient, config: GenerationConfig):
        self.client = client
        self.config = config

    def build_topic_tree(self, domain: str) -> dict:
        """Generate a hierarchical topic tree for a domain."""
        template = Template(TEMPLATES["topic_tree"])
        prompt = template.render(domain=domain, existing_topics="")

        result = self.client.complete_json(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        if isinstance(result, dict) and "branches" in result:
            total_topics = sum(len(b.get("leaves", [])) for b in result["branches"])
            logger.info(
                f"Topic tree built: {len(result['branches'])} branches, "
                f"{total_topics} leaf topics"
            )
            return result

        # Fallback
        return {
            "root": domain,
            "branches": [
                {"name": domain, "leaves": [f"{domain} basics", f"{domain} advanced"]}
            ],
        }

    def flatten_topics(self, tree: dict) -> list[str]:
        """Flatten a topic tree into a list of specific topics."""
        topics = []
        for branch in tree.get("branches", []):
            branch_name = branch.get("name", "")
            for leaf in branch.get("leaves", []):
                topics.append(f"{branch_name} — {leaf}")
        return topics

    def generate(self, domain: str) -> Dataset:
        """Generate a dataset covering all topics in the tree."""
        tree = self.build_topic_tree(domain)
        topics = self.flatten_topics(tree)
        dataset = Dataset(name=f"topictree_{safe_slug(domain)}")
        target = self.config.num_examples
        generated = 0

        personas = self.config.personas
        difficulties = self.config.difficulty_levels

        logger.info(
            f"Generating {target} examples across {len(topics)} topics for '{domain}'"
        )

        # Distribute examples evenly across topics
        examples_per_topic = max(1, target // max(len(topics), 1))

        for topic in topics:
            if generated >= target:
                break

            batch_count = min(examples_per_topic, self.config.batch_size, target - generated)
            persona = random.choice(personas)
            difficulty = random.choice(difficulties)

            template = Template(TEMPLATES["domain_generate"])
            prompt = template.render(
                domain=domain,
                difficulty=difficulty,
                persona=persona,
                topic=topic,
                language=self.config.language,
                system_prompt=self.config.system_prompt,
                batch_size=batch_count,
            )

            try:
                result = self.client.complete_json(
                    [{"role": "user", "content": prompt}],
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
                                "source": "topic_tree",
                                "topic": topic,
                                "persona": persona,
                                "difficulty": difficulty,
                            },
                        )
                        dataset.add(example)
                        generated += 1

                logger.info(f"Generated {generated}/{target} — topic: {topic[:50]}")

            except Exception as e:
                logger.error(f"Failed on topic '{topic}': {e}")
                continue

        return dataset
