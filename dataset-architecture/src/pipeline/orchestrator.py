import json
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime

import yaml

from .ingestion import DataIngestor, DatasetExample
from .preprocessing import Preprocessor
from .cleaning import DataCleaner, PIIRemover
from .validation import CodeValidator, MarkdownValidator, SchemaValidator
from .export import DataExporter
from ..quality.scoring import QualityScorer, HardExampleMiner
from ..quality.deduplication import Deduplicator
from ..quality.filtering import QualityFilter
from ..quality.ranking import ResponseRanker
from ..synthetic.generator import SyntheticDataGenerator
from ..synthetic.augmenter import DataAugmenter
from ..curriculum.difficulty import DifficultyScorer
from ..curriculum.scheduler import CurriculumScheduler


class DatasetPipeline:
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        pipeline_config = self.config.get("pipeline", {})
        self.seed = pipeline_config.get("seed", 42)
        self.num_workers = pipeline_config.get("num_workers", 8)
        self.batch_size = pipeline_config.get("batch_size", 1000)
        self.stages = pipeline_config.get("stages", [])
        paths = pipeline_config.get("paths", {})
        self.raw_path = paths.get("raw_data", "data/raw")
        self.processed_path = paths.get("processed_data", "data/processed")
        self.curated_path = paths.get("curated_data", "data/curated")
        self.logs_path = paths.get("logs", "logs")
        self.cache_path = paths.get("cache", ".cache")
        self.exports_path = paths.get("exports", "exports")

        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.curated_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)

        self._setup_logging()

        self.ingestor = DataIngestor(config_path)
        self.preprocessor = Preprocessor()
        self.cleaner = DataCleaner()
        self.code_validator = CodeValidator()
        self.markdown_validator = MarkdownValidator()
        quality_config_path = "config/quality.yaml"
        self.scorer = QualityScorer(quality_config_path)
        self.deduplicator = Deduplicator(quality_config_path)
        self.filter = QualityFilter(quality_config_path)
        self.ranker = ResponseRanker()
        self.synthetic_generator = SyntheticDataGenerator(config_path)
        self.augmenter = DataAugmenter(seed=self.seed)
        self.difficulty_scorer = DifficultyScorer()
        self.exporter = DataExporter(config_path)
        self.hard_miner = HardExampleMiner()
        self.scheduler = CurriculumScheduler(seed=self.seed)

        self.stats = {"total_examples": 0, "stage_stats": {}}

    def _setup_logging(self):
        log_config = self.config.get("pipeline", {}).get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"), logging.INFO)
        log_file = Path(self.logs_path) / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=level,
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def run(self, **kwargs) -> Dict:
        self.logger.info("Starting dataset pipeline")
        start_time = time.time()

        examples = []

        for stage in self.stages:
            stage_start = time.time()
            stage_func = getattr(self, f"_stage_{stage}", None)
            if stage_func:
                self.logger.info(f"Running stage: {stage}")
                examples = stage_func(examples, **kwargs)
                elapsed = time.time() - stage_start
                self.stats["stage_stats"][stage] = {
                    "count": len(examples),
                    "time_seconds": round(elapsed, 2),
                }
                self.logger.info(f"Stage '{stage}' complete: {len(examples)} examples in {elapsed:.2f}s")
            else:
                self.logger.warning(f"Unknown stage: {stage}")

        self.stats["total_examples"] = len(examples)
        self.stats["total_time"] = round(time.time() - start_time, 2)
        self.stats["completed_at"] = datetime.utcnow().isoformat()

        self.logger.info(f"Pipeline complete: {len(examples)} total examples in {self.stats['total_time']}s")
        return {"examples": examples, "stats": self.stats}

    def _stage_ingestion(self, examples: List, **kwargs) -> List[DatasetExample]:
        all_examples = []
        categories = self.config.get("categories", {})
        cats = kwargs.get("categories", list(categories.keys()))

        if kwargs.get("huggingface", False):
            hf_loader = __import__("src.pipeline.ingestion", fromlist=["HuggingFaceLoader"])
            pass

        for cat in cats:
            raw_cat_path = Path(self.raw_path) / cat
            if raw_cat_path.exists():
                cat_examples = list(self.ingestor.ingest_from_path(str(raw_cat_path), category=cat))
                all_examples.extend(cat_examples)
                self.logger.info(f"Ingested {len(cat_examples)} examples for category '{cat}'")

        if kwargs.get("load_huggingface", False):
            try:
                from .ingestion import HuggingFaceLoader
                hf_loader = HuggingFaceLoader(self.config)
                for ds_name in ["competitive_programming", "algorithms_dsa", "debugging",
                                "general_reasoning", "math_logic", "conversational_ai", "tool_usage"]:
                    loader = getattr(hf_loader, f"load_{ds_name}", None)
                    if loader:
                        ds_examples = list(loader())
                        for ex in ds_examples:
                            ex.category = ds_name
                        all_examples.extend(ds_examples)
                        self.logger.info(f"Loaded {len(ds_examples)} examples from HuggingFace '{ds_name}'")
            except ImportError:
                self.logger.warning("HuggingFace datasets not available. Install: pip install datasets")

        return all_examples

    def _stage_preprocessing(self, examples: List[DatasetExample], **kwargs) -> List[DatasetExample]:
        return self.preprocessor.process(examples, num_workers=self.num_workers)

    def _stage_validation(self, examples: List[DatasetExample], **kwargs) -> List[DatasetExample]:
        valid_config = self.config.get("quality", {}).get("code_validation", {})
        if valid_config.get("enabled", True):
            examples = self.code_validator.validate(examples, num_workers=self.num_workers)

        md_config = self.config.get("quality", {}).get("markdown_validation", {})
        if md_config.get("enabled", True):
            examples = self.markdown_validator.validate(examples, num_workers=self.num_workers)

        return examples

    def _stage_quality_scoring(self, examples: List[DatasetExample], **kwargs) -> List[DatasetExample]:
        return self.scorer.score(examples, num_workers=self.num_workers)

    def _stage_deduplication(self, examples: List[DatasetExample], **kwargs) -> List[DatasetExample]:
        return self.deduplicator.deduplicate(examples, num_workers=self.num_workers)

    def _stage_filtering(self, examples: List[DatasetExample], **kwargs) -> List[DatasetExample]:
        return self.filter.filter(examples, num_workers=self.num_workers)

    def _stage_synthetic_augmentation(self, examples: List[DatasetExample], **kwargs) -> List[DatasetExample]:
        augmented = []

        for cat in set(ex.category for ex in examples):
            gen_count = max(100, len([e for e in examples if e.category == cat]) // 2)
            synthetic = self.synthetic_generator.generate(cat, count=gen_count)
            augmented.extend(synthetic)

        augmented_examples = self.augmenter.augment(examples, num_workers=self.num_workers)
        augmented_examples.extend(augmented)

        return augmented_examples

    def _stage_curriculum_scoring(self, examples: List[DatasetExample], **kwargs) -> List[DatasetExample]:
        examples = self.difficulty_scorer.score_difficulty(examples)
        examples = self.ranker.rank(examples)
        return examples

    def _stage_export(self, examples: List[DatasetExample], **kwargs) -> List[DatasetExample]:
        output_dir = kwargs.get("output_dir", self.exports_path)
        export_format = kwargs.get("format", "jsonl")
        export_framework = kwargs.get("framework", "transformers")

        self.exporter.export_for_training(examples, output_dir, framework=export_framework)
        self.exporter.export_metadata(examples, output_dir)

        self.logger.info(f"Exported {len(examples)} examples to {output_dir}")

        return examples

    def get_report(self) -> Dict:
        return {
            "pipeline": self.config.get("pipeline", {}).get("name"),
            "version": self.config.get("pipeline", {}).get("version"),
            "stats": self.stats,
            "component_stats": {
                "ingestor": self.ingestor.stats,
                "preprocessor": self.preprocessor.stats,
                "cleaner": self.cleaner.stats,
                "code_validator": self.code_validator.stats,
                "markdown_validator": self.markdown_validator.stats,
                "scorer": self.scorer.stats,
                "deduplicator": self.deduplicator.stats,
                "filter": self.filter.stats,
                "synthetic_generator": self.synthetic_generator.stats,
                "augmenter": self.augmenter.stats,
                "difficulty_scorer": self.difficulty_scorer.stats,
            },
        }
