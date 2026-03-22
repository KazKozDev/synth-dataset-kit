"""Main orchestration engine — ties all components together."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from synth_dataset_kit.config import SDKConfig
from synth_dataset_kit.decontamination import Decontaminator
from synth_dataset_kit.exporters import (
    export_case_study_bundle,
    export_dataset,
    export_pipeline_artifacts,
    export_quality_report_html,
    export_quality_report_json,
)
from synth_dataset_kit.generators import SeedExpander, TopicTreeGenerator, load_seed_file
from synth_dataset_kit.llm_client import LLMClient
from synth_dataset_kit.models import Dataset, QualityReport
from synth_dataset_kit.quality import QualityJudge

logger = logging.getLogger(__name__)


class DatasetEngine:
    """High-level API for generating, auditing, and exporting datasets.

    Usage:
        config = SDKConfig.default_for_provider("ollama")
        engine = DatasetEngine(config)

        # From seed examples
        dataset = engine.generate_from_seeds("my_seeds.jsonl", num_examples=500)

        # From domain description
        dataset = engine.generate_from_domain("customer support chatbot", num_examples=200)

        # Audit
        report = engine.audit(dataset)

        # Export
        engine.export(dataset, format="jsonl", output_dir="./output")
    """

    def __init__(self, config: SDKConfig):
        self.config = config
        self.client = LLMClient(config.llm)

    def generate_from_seeds(
        self,
        seed_file: str,
        num_examples: int | None = None,
    ) -> Dataset:
        """Generate a dataset by expanding seed examples.

        This is the primary and most powerful generation method.
        """
        start = time.time()

        # Load seeds
        seeds = load_seed_file(seed_file)
        if not seeds:
            raise ValueError(f"No valid examples found in {seed_file}")

        logger.info(f"Loaded {len(seeds)} seed examples")

        # Configure
        gen_config = self.config.generation.model_copy()
        if num_examples:
            gen_config.num_examples = num_examples
        gen_config.seed_file = seed_file

        # Analyze and expand
        quality_judge = None
        if self.config.quality.enabled:
            quality_judge = QualityJudge(
                self.client,
                self.config.quality,
                self.config.generation.system_prompt,
            )
        expander = SeedExpander(
            self.client,
            gen_config,
            quality_judge=quality_judge,
            min_quality=self.config.quality.min_score,
        )
        analysis = expander.analyze_seeds(seeds)
        dataset = expander.expand(seeds, analysis)

        elapsed = time.time() - start
        dataset.config_snapshot = self.config.model_dump()
        logger.info(
            f"Generated {dataset.size} examples from {len(seeds)} seeds "
            f"in {elapsed:.1f}s"
        )

        return dataset

    def generate_from_domain(
        self,
        domain: str,
        num_examples: int | None = None,
    ) -> Dataset:
        """Generate a dataset from a domain description using topic trees."""
        start = time.time()

        gen_config = self.config.generation.model_copy()
        if num_examples:
            gen_config.num_examples = num_examples
        gen_config.domain = domain

        generator = TopicTreeGenerator(self.client, gen_config)
        dataset = generator.generate(domain)

        elapsed = time.time() - start
        dataset.config_snapshot = self.config.model_dump()
        logger.info(
            f"Generated {dataset.size} examples for '{domain}' in {elapsed:.1f}s"
        )

        return dataset

    def _final_export_dataset(
        self,
        dataset: Dataset,
        seed_file: str | None = None,
    ) -> tuple[Dataset, int]:
        """Build the final exported dataset.

        By default, the primary exported dataset includes the original seed
        examples plus retained generated examples. Audit and pipeline artifacts
        remain generated-only so quality reporting stays honest.
        """
        if not seed_file or not self.config.export.include_seed_examples:
            final_dataset = dataset.model_copy(deep=True)
            for example in final_dataset.examples:
                example.metadata.setdefault("source", "generated")
            return final_dataset, 0

        seed_examples = load_seed_file(seed_file)
        final_examples = []

        for example in seed_examples:
            copied = example.model_copy(deep=True)
            copied.metadata["source"] = "seed"
            copied.metadata["seed_example"] = True
            final_examples.append(copied)

        for example in dataset.examples:
            copied = example.model_copy(deep=True)
            original_source = str(copied.metadata.get("source", "generated"))
            copied.metadata["generation_source"] = original_source
            copied.metadata["source"] = "generated"
            copied.metadata["seed_example"] = False
            final_examples.append(copied)

        final_dataset = Dataset(
            name=dataset.name,
            version=dataset.version,
            created_at=dataset.created_at,
            generator=dataset.generator,
            config_snapshot=dataset.config_snapshot,
            examples=final_examples,
            artifacts=dataset.artifacts,
        )
        final_dataset.config_snapshot["seed_examples_included"] = len(seed_examples)
        final_dataset.config_snapshot["generated_examples_retained"] = dataset.size
        final_dataset.config_snapshot["final_export_examples"] = final_dataset.size
        return final_dataset, len(seed_examples)

    def audit(self, dataset: Dataset) -> QualityReport:
        """Run quality scoring + decontamination on a dataset.

        Returns a QualityReport and modifies examples in-place with scores/flags.
        """
        start = time.time()

        # Quality scoring
        if self.config.quality.enabled:
            judge = QualityJudge(
                self.client,
                self.config.quality,
                self.config.generation.system_prompt,
            )
            dataset = judge.score_dataset(dataset)

        # Decontamination
        if self.config.decontamination.enabled:
            decontaminator = Decontaminator(
                benchmarks=self.config.decontamination.benchmarks,
                similarity_threshold=self.config.decontamination.similarity_threshold,
                method=self.config.decontamination.method,
                use_benchmark_datasets=self.config.decontamination.use_benchmark_datasets,
                load_full_benchmark_corpus=self.config.decontamination.load_full_benchmark_corpus,
                benchmark_sample_limit=self.config.decontamination.benchmark_sample_limit,
                cache_dir=self.config.decontamination.cache_dir,
                embedding_model=self.config.decontamination.embedding_model,
                embedding_index_backend=self.config.decontamination.embedding_index_backend,
                embedding_top_k=self.config.decontamination.embedding_top_k,
                review_threshold=self.config.decontamination.review_threshold,
                hard_fail_methods=self.config.decontamination.hard_fail_methods,
                review_methods=self.config.decontamination.review_methods,
            )
            dataset = decontaminator.check_dataset(dataset)
            dataset.config_snapshot["benchmark_sources"] = dict(decontaminator.benchmark_sources)
            dataset.config_snapshot["benchmark_sample_counts"] = dict(decontaminator.benchmark_sample_counts)
            dataset.config_snapshot["benchmark_load_errors"] = dict(decontaminator.benchmark_load_errors)

        # Generate report
        judge = QualityJudge(
            self.client, self.config.quality, self.config.generation.system_prompt
        )
        report = judge.generate_report(dataset)
        report.generation_time_seconds = time.time() - start

        logger.info(
            f"Audit complete: {report.passed_examples}/{report.total_examples} passed, "
            f"avg score: {report.avg_quality_score:.1f}, "
            f"contamination: {report.contamination_hits}"
        )

        return report

    def export(
        self,
        dataset: Dataset,
        format: str | None = None,
        output_dir: str | None = None,
        include_metadata: bool | None = None,
        quality_report: QualityReport | None = None,
    ) -> list[str]:
        """Export dataset and optionally quality report.

        Returns list of output file paths.
        """
        fmt = format or self.config.export.format
        out_dir = output_dir or self.config.export.output_dir
        meta = include_metadata if include_metadata is not None else self.config.export.include_metadata
        if fmt in {"jsonl", "openai"} and dataset.config_snapshot.get("seed_examples_included"):
            meta = True

        output_files = []

        # Export dataset
        filepath = export_dataset(
            dataset,
            fmt,
            out_dir,
            include_metadata=meta,
            quality_report=quality_report,
        )
        output_files.append(filepath)
        output_files.extend(export_pipeline_artifacts(dataset, out_dir))

        # Export quality report as HTML
        if quality_report and self.config.export.include_quality_report:
            report_path = str(Path(out_dir) / f"{dataset.name}_quality_report.html")
            export_quality_report_html(quality_report, report_path)
            output_files.append(report_path)
            json_report_path = str(Path(out_dir) / f"{dataset.name}_quality_report.json")
            export_quality_report_json(quality_report, json_report_path)
            output_files.append(json_report_path)
            case_study_path = export_case_study_bundle(dataset, quality_report, out_dir)
            output_files.append(case_study_path)

        return output_files

    def run_full_pipeline(
        self,
        seed_file: str | None = None,
        domain: str | None = None,
        num_examples: int = 100,
        format: str = "jsonl",
        output_dir: str = "./output",
        min_quality: float | None = None,
    ) -> tuple[Dataset, QualityReport, list[str], dict[str, float]]:
        """Run the complete pipeline: generate → audit → filter → export.

        Returns (dataset, report, output_files, stage_timings).
        """
        stage_timings: dict[str, float] = {}

        # Generate
        started = time.time()
        if seed_file:
            dataset = self.generate_from_seeds(seed_file, num_examples)
        elif domain:
            dataset = self.generate_from_domain(domain, num_examples)
        else:
            raise ValueError("Provide either seed_file or domain")
        stage_timings["generate_seconds"] = round(time.time() - started, 3)

        # Audit
        started = time.time()
        report = self.audit(dataset)
        stage_timings["audit_seconds"] = round(time.time() - started, 3)

        # Filter
        min_q = min_quality or self.config.quality.min_score
        started = time.time()
        clean_dataset = dataset.remove_contaminated().filter_by_quality(min_q)
        logger.info(
            f"After filtering: {clean_dataset.size}/{dataset.size} examples retained"
        )

        # Re-generate report for clean dataset
        judge = QualityJudge(
            self.client, self.config.quality, self.config.generation.system_prompt
        )
        clean_report = judge.generate_report(clean_dataset)
        stage_timings["filter_seconds"] = round(time.time() - started, 3)

        final_export_dataset, seed_examples_included = self._final_export_dataset(
            clean_dataset,
            seed_file=seed_file,
        )

        # Export
        started = time.time()
        output_files = self.export(
            final_export_dataset,
            format=format,
            output_dir=output_dir,
            quality_report=clean_report,
        )
        stage_timings["export_seconds"] = round(time.time() - started, 3)

        final_export_dataset.config_snapshot["seed_examples_included"] = seed_examples_included
        final_export_dataset.config_snapshot["generated_examples_retained"] = clean_dataset.size
        final_export_dataset.config_snapshot["final_export_examples"] = final_export_dataset.size

        return final_export_dataset, clean_report, output_files, stage_timings
