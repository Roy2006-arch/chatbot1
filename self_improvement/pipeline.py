import json
import os
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from .schema import (
    SelfImprovementExample, CorrectionRecord, HardExample,
    ModelEvalResult, EvalCase, ImprovementReport,
)
from .correction_generator import CorrectionGenerator
from .quality_curator import QualityCurator
from .hard_example_miner import HardExampleMiner
from .dataset_builder import DatasetBuilder
from .model_evaluator import ModelEvaluator


class SelfImprovementPipeline:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.correction_gen = CorrectionGenerator(self.config.get("correction_generator", {}))
        self.quality_curator = QualityCurator(self.config.get("quality_curator", {}))
        self.hard_miner = HardExampleMiner(self.config.get("hard_example_miner", {}))
        self.dataset_builder = DatasetBuilder(self.config.get("dataset_builder", {}))
        self.model_evaluator = ModelEvaluator(self.config.get("model_evaluator", {}))
        self.report = ImprovementReport(
            run_id="",
            timestamp=datetime.utcnow().isoformat(),
        )

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        pkg_dir = Path(__file__).parent
        default = pkg_dir / "config.yaml"
        if default.exists():
            with open(default, "r") as f:
                return yaml.safe_load(f)
        return {}

    def run(
        self,
        response_fn: Optional[Callable] = None,
        eval_cases: Optional[List[EvalCase]] = None,
        output_dir: Optional[str] = None,
        max_failed: int = 500,
        max_corrections: int = 200,
        max_quality: int = 300,
        max_hard: int = 100,
        max_total: int = 1000,
    ) -> ImprovementReport:
        start = time.time()

        run_id = f"{self.config.get('pipeline', {}).get('run_id_prefix', 'si')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_dir = output_dir or self.config.get("pipeline", {}).get("dataset_output", "self_improvement_data")
        os.makedirs(output_dir, exist_ok=True)

        self.report.run_id = run_id

        eval_before = None
        if response_fn and self.model_evaluator.enabled:
            eval_before = self._eval_model_before(response_fn, eval_cases)
            self.report.model_before = eval_before

        corrections = self._run_corrections(max_failed, max_corrections)
        quality_examples = self._run_quality_curation(max_quality)
        hard_examples = self._run_hard_example_mining(max_hard)

        all_examples = corrections + quality_examples + hard_examples
        dataset = self.dataset_builder.build_dataset(
            correction_examples=corrections,
            quality_examples=quality_examples,
            hard_examples=hard_examples,
            max_total=max_total,
        )

        train, val, test = self.dataset_builder.split_dataset(dataset)
        exported = self.dataset_builder.export_all_formats(train, val, test, output_dir)
        self.dataset_builder.save_metadata(dataset, output_dir)
        self._save_report(output_dir)

        eval_after = None
        if response_fn and self.model_evaluator.enabled and eval_before:
            eval_after = self._eval_model_after(response_fn, eval_cases)
            self.report.model_after = eval_after
            comparison = self.model_evaluator.compare_models(eval_before, eval_after)
            self.report.score_improvement = comparison.get("score_improvement", 0.0)
            self._save_comparison(comparison, output_dir)

        self.report.total_failed_queries = self.correction_gen.stats.get("loaded", 0)
        self.report.corrections_generated = len(corrections)
        self.report.high_quality_examples = len(quality_examples)
        self.report.hard_examples_mined = len(hard_examples)
        self.report.dataset_examples = len(dataset)
        self.report.metadata["processing_time"] = round(time.time() - start, 2)
        self.report.metadata["exported_files"] = exported

        print(f"\n{'='*60}")
        print(f"Self-Improvement Pipeline Complete (run: {run_id})")
        print(f"{'='*60}")
        print(f"  Failed queries loaded:   {self.report.total_failed_queries}")
        print(f"  Corrections generated:    {self.report.corrections_generated}")
        print(f"  High-quality extracted:   {self.report.high_quality_examples}")
        print(f"  Hard examples mined:      {self.report.hard_examples_mined}")
        print(f"  Dataset examples built:   {self.report.dataset_examples}")
        print(f"  Processing time:          {self.report.metadata['processing_time']}s")
        if eval_before and eval_after:
            print(f"  Score improvement:        {self.report.score_improvement:+.4f}")
        print(f"  Output:                   {output_dir}")
        print(f"{'='*60}")

        return self.report

    def _run_corrections(
        self, max_failed: int, max_corrections: int
    ) -> List[SelfImprovementExample]:
        if not self.correction_gen.enabled:
            return []

        failed = self.correction_gen.load_failed_queries(limit=max_failed)
        if not failed:
            return []

        records = self.correction_gen.generate_batch(failed[:max_failed])
        records = records[:max_corrections]
        return self.correction_gen.to_examples(records)

    def _run_quality_curation(self, max_quality: int) -> List[SelfImprovementExample]:
        if not self.quality_curator.enabled:
            return []
        return self.quality_curator.curate(limit=max_quality)

    def _run_hard_example_mining(self, max_hard: int) -> List[SelfImprovementExample]:
        if not self.hard_miner.enabled:
            return []
        hard = self.hard_miner.mine()
        hard = hard[:max_hard]
        return self.hard_miner.to_examples(hard)

    def _eval_model_before(self, response_fn, eval_cases):
        print("Evaluating model BEFORE retraining...")
        result = self.model_evaluator.evaluate_model(
            model_name="current", response_fn=response_fn, cases=eval_cases
        )
        print(f"  Composite: {result.avg_composite:.4f}, Pass rate: {result.pass_rate:.2%}")
        return result

    def _eval_model_after(self, response_fn, eval_cases):
        print("Evaluating model AFTER retraining...")
        result = self.model_evaluator.evaluate_model(
            model_name="retrained", response_fn=response_fn, cases=eval_cases
        )
        print(f"  Composite: {result.avg_composite:.4f}, Pass rate: {result.pass_rate:.2%}")
        return result

    def export_dpo_for_retraining(
        self, output_path: str, max_examples: int = 500
    ) -> str:
        corrections = self._run_corrections(max_failed=max_examples, max_corrections=max_examples)
        return self.dataset_builder.export_dpo(corrections, output_path)

    def analyze_failed_queries(self) -> Dict:
        failed = self.correction_gen.load_failed_queries(limit=1000, unresolved_only=False)
        if not failed:
            return {"total": 0}

        from collections import Counter
        reason_counts: Counter = Counter()
        category_counts: Counter = Counter()
        scores = []

        for item in failed:
            reasons_str = item.get("failure_reasons", "[]")
            if isinstance(reasons_str, str):
                try:
                    reasons = json.loads(reasons_str)
                except (json.JSONDecodeError, TypeError):
                    reasons = [str(reasons_str)]
            else:
                reasons = list(reasons_str) if reasons_str else []
            for r in reasons:
                reason_counts[str(r).strip()] += 1

            prompt = item.get("prompt", "")
            cat = self._categorize(prompt)
            category_counts[cat] += 1

            score = item.get("composite_score", 0)
            try:
                scores.append(float(score))
            except (ValueError, TypeError):
                pass

        return {
            "total": len(failed),
            "by_failure_reason": dict(reason_counts.most_common(15)),
            "by_category": dict(category_counts.most_common()),
            "avg_score": round(sum(scores) / max(len(scores), 1), 4) if scores else 0,
        }

    def _categorize(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(k in prompt_lower for k in ["code", "function", "debug", "error", "implement"]):
            return "code"
        if any(k in prompt_lower for k in ["why", "explain", "reason", "analyze"]):
            return "reasoning"
        if any(k in prompt_lower for k in ["how to", "setup", "configure"]):
            return "technical"
        if any(k in prompt_lower for k in ["what is", "who is", "definition"]):
            return "factual"
        return "general"

    def _save_report(self, output_dir: str):
        path = os.path.join(output_dir, "improvement_report.json")
        with open(path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2)

    def _save_comparison(self, comparison: Dict, output_dir: str):
        path = os.path.join(output_dir, "model_comparison.json")
        with open(path, "w") as f:
            json.dump(comparison, f, indent=2)

    def get_report(self) -> ImprovementReport:
        return self.report
