import json
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .schema import ReasoningExample, ReasoningTask, ContradictionPair, REASONING_CONFIG
from .generators.multi_step import MultiStepReasoningGenerator
from .generators.contradiction_engine import ContradictionEngine
from .verifiers.reasoning_verifier import ReasoningVerifier, LogicalFallacyDetector
from .annotation.strategy import AnnotationStrategy
from .exporters.format_converter import ReasoningFormatConverter


class ReasoningPipeline:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or REASONING_CONFIG
        self.generator = MultiStepReasoningGenerator()
        self.contradiction_engine = ContradictionEngine()
        self.verifier = ReasoningVerifier()
        self.fallacy_detector = LogicalFallacyDetector()
        self.annotation = AnnotationStrategy()
        self.converter = ReasoningFormatConverter()

        self.logger = logging.getLogger("reasoning_pipeline")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.stats = {
            "generated": 0,
            "verified_valid": 0,
            "verified_invalid": 0,
            "contradiction_pairs": 0,
            "training_examples": 0,
            "pipeline_time": 0.0,
        }

    def run(
        self,
        total_examples: int = 10000,
        output_dir: str = "exports/reasoning_dataset",
        framework: str = "transformers",
        verify: bool = True,
    ) -> Dict[str, Any]:
        start = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Starting Reasoning Dataset Pipeline")
        self.logger.info("=" * 60)

        type_dist = self.config.get("type_distribution", {})
        counts = {}
        for task_name, ratio in type_dist.items():
            counts[task_name] = int(total_examples * ratio)

        if sum(counts.values()) < total_examples:
            diff = total_examples - sum(counts.values())
            if counts:
                largest = max(counts, key=counts.get)
                counts[largest] += diff

        all_examples = []

        self.logger.info(f"Generating reasoning examples across {len(counts)} task types...")
        for task_name, count in counts.items():
            if count <= 0:
                continue
            try:
                task = ReasoningTask(task_name)
            except ValueError:
                self.logger.warning(f"Unknown task: {task_name}")
                continue

            examples = self.generator.generate(task, count=count)
            all_examples.extend(examples)
            self.logger.info(f"  {task.value}: {len(examples)} examples")

        self.stats["generated"] = len(all_examples)
        self.logger.info(f"Total generated: {len(all_examples)}")

        contradiction_pairs = self.contradiction_engine.generate(count=int(total_examples * 0.05))
        self.stats["contradiction_pairs"] = len(contradiction_pairs)
        self.logger.info(f"Contradiction pairs: {len(contradiction_pairs)}")

        if verify:
            self.logger.info("Verifying reasoning examples...")
            valid_examples = self.verifier.verify_batch(all_examples)
            self.stats["verified_valid"] = len(valid_examples)
            self.stats["verified_invalid"] = len(all_examples) - len(valid_examples)
            self.logger.info(f"Valid: {len(valid_examples)}, Invalid: {len(all_examples) - len(valid_examples)}")
            all_examples = valid_examples

        self.logger.info("Checking for logical fallacies...")
        fallacy_count = 0
        for ex in all_examples:
            fallacies = self.fallacy_detector.detect(ex.question + " " + ex.context)
            if fallacies:
                ex.metadata["fallacies"] = fallacies
                fallacy_count += 1
        self.logger.info(f"Examples with fallacies detected: {fallacy_count}")

        self.logger.info("Converting to training format...")
        records = self.converter.convert_to_instructions(
            all_examples, include_cot=True, include_verification=True,
        )

        contradiction_records = self.converter.convert_contradiction_pairs(contradiction_pairs)
        all_records = records + contradiction_records

        self.stats["training_examples"] = len(all_records)

        self.converter.export_for_finetuning(all_records, output_dir, framework=framework)
        self.logger.info(f"Exported {len(all_records)} examples to {output_dir}")

        self.stats["pipeline_time"] = round(time.time() - start, 2)

        report = {
            "status": "success",
            "config": {
                "total_examples": total_examples,
                "framework": framework,
                "verify": verify,
            },
            "stats": self.stats,
            "type_distribution": {k: v for k, v in self.converter.stats.get("by_task", {}).items()},
            "output_dir": output_dir,
            "completed_at": datetime.utcnow().isoformat(),
        }

        report_path = Path(output_dir) / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Pipeline complete: {len(all_records)} examples in {self.stats['pipeline_time']}s")
        self.logger.info(f"Report saved: {report_path}")

        return report

    def run_annotation_mode(
        self,
        output_dir: str = "exports/reasoning_annotation",
    ) -> Dict[str, Any]:
        guidelines = self.annotation.get_all_guidelines()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for task_name, guideline in guidelines.items():
            prompt = self.annotation.build_annotation_prompt(ReasoningTask(task_name))
            filepath = output_path / f"annotation_guide_{task_name}.md"
            with open(filepath, "w") as f:
                f.write(prompt)

        return {
            "status": "success",
            "guidelines": len(guidelines),
            "output_dir": output_dir,
        }

    def get_stats(self) -> Dict:
        return self.stats
