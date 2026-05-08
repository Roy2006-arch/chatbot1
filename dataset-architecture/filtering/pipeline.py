import json
import os
import re
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from .models import FilterResult, FilterIssue, FilterConfig, FilterReport, Severity
from .hallucination_detector import HallucinationDetector
from .repetition_detector import RepetitionDetector
from .reasoning_validator import ReasoningValidator
from .semantic_filter import SemanticSimilarityFilter
from .toxicity_filter import ToxicityFilter
from .code_validator import AdvancedCodeValidator
from .markdown_validator import AdvancedMarkdownValidator
from .quality_scorer import EnhancedQualityScorer
from .response_ranker import EnhancedResponseRanker


class FilteringPipeline:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        fc = self.config.get("filtering", {})
        self.filter_config = FilterConfig.from_dict(fc)

        self.hallucination_detector = HallucinationDetector(self.config.get("hallucination", {}))
        self.repetition_detector = RepetitionDetector(self.config.get("repetition", {}))
        self.reasoning_validator = ReasoningValidator(self.config.get("reasoning", {}))
        self.semantic_filter = SemanticSimilarityFilter(self.config.get("semantic_similarity", {}))
        self.toxicity_filter = ToxicityFilter(self.config.get("toxicity", {}))
        self.code_validator = AdvancedCodeValidator(self.config.get("code_validation", {}))
        self.markdown_validator = AdvancedMarkdownValidator(self.config.get("markdown_validation", {}))
        raw_dims = self.config.get("quality_scoring", {}).get("dimensions", None)
        if raw_dims and isinstance(next(iter(raw_dims.values())), dict):
            dimensions = {k: v["weight"] for k, v in raw_dims.items()}
        else:
            dimensions = raw_dims
        self.quality_scorer = EnhancedQualityScorer(
            dimensions=dimensions,
            config=self.config,
        )
        self.ranker = EnhancedResponseRanker(scorer=self.quality_scorer)
        self.report = FilterReport()

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        pkg_dir = Path(__file__).parent
        default_path = pkg_dir / "config.yaml"
        if default_path.exists():
            with open(default_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def run(
        self,
        examples: List[Any],
        stages: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> List[Any]:
        start = time.time()
        total = len(examples)
        self.report = FilterReport(total_examples=total)
        stages = stages or ["toxicity", "hallucination", "repetition", "reasoning", "code", "markdown", "quality", "semantic", "rank"]

        passed = examples[:]
        stage_order = [
            ("toxicity", self._filter_toxicity),
            ("hallucination", self._filter_hallucination),
            ("repetition", self._filter_repetition),
            ("reasoning", self._filter_reasoning),
            ("code", self._filter_code),
            ("markdown", self._filter_markdown),
            ("quality", self._filter_quality),
            ("semantic", self._filter_semantic),
            ("rank", self._filter_rank),
        ]

        for stage_name, stage_fn in stage_order:
            if stage_name not in stages:
                continue
            before = len(passed)
            passed = stage_fn(passed)
            after = len(passed)
            removed = before - after
            self.report.rejection_breakdown[stage_name] = removed
            self.report.rejected += removed

        self.report.passed = len(passed)
        self.report.processing_time = time.time() - start

        if output_dir:
            self._save_report(output_dir)
            self._save_filtered(passed, output_dir)

        return passed

    def run_on_texts(
        self,
        instructions: List[str],
        outputs: List[str],
        inputs: Optional[List[str]] = None,
    ) -> List[FilterResult]:
        inputs = inputs or [""] * len(instructions)
        results = self.quality_scorer.score_batch(instructions, outputs, inputs)
        return results

    def analyze_dataset(
        self, examples: List[Any]
    ) -> Dict[str, Any]:
        report = {
            "total": len(examples),
            "avg_quality": 0.0,
            "dimension_averages": {},
            "category_counts": {},
            "quality_distribution": {"0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0},
            "issues_summary": {},
        }

        if not examples:
            return report

        scores = [ex.quality_score for ex in examples]
        report["avg_quality"] = round(sum(scores) / len(scores), 4) if scores else 0.0

        for ex in examples:
            qs = ex.quality_score
            if qs < 0.2:
                report["quality_distribution"]["0-0.2"] += 1
            elif qs < 0.4:
                report["quality_distribution"]["0.2-0.4"] += 1
            elif qs < 0.6:
                report["quality_distribution"]["0.4-0.6"] += 1
            elif qs < 0.8:
                report["quality_distribution"]["0.6-0.8"] += 1
            else:
                report["quality_distribution"]["0.8-1.0"] += 1

            cat = ex.category or "uncategorized"
            report["category_counts"][cat] = report["category_counts"].get(cat, 0) + 1

            dims = ex.metadata.get("quality_scores", {}) or ex.metadata.get("quality_dimensions", {})
            for dim, score in dims.items():
                if dim not in report["dimension_averages"]:
                    report["dimension_averages"][dim] = []
                report["dimension_averages"][dim].append(score)

        for dim in report["dimension_averages"]:
            vals = report["dimension_averages"][dim]
            report["dimension_averages"][dim] = round(sum(vals) / len(vals), 4) if vals else 0.0

        return report

    def _filter_toxicity(self, examples: List[Any]) -> List[Any]:
        threshold = self.filter_config.max_toxicity_score
        kept = []
        for ex in examples:
            result = self.toxicity_filter.check(ex.output)
            ex.metadata["toxicity_result"] = result.to_dict()
            if result.score >= 1.0 - threshold:
                kept.append(ex)
        return kept

    def _filter_hallucination(self, examples: List[Any]) -> List[Any]:
        threshold = self.filter_config.max_hallucination_score
        kept = []
        for ex in examples:
            result = self.hallucination_detector.check(ex.output)
            ex.metadata["hallucination_result"] = result.to_dict()
            if result.score >= threshold:
                kept.append(ex)
        return kept

    def _filter_repetition(self, examples: List[Any]) -> List[Any]:
        threshold = self.filter_config.max_repetition_score
        kept = []
        for ex in examples:
            result = self.repetition_detector.check(ex.output)
            ex.metadata["repetition_result"] = result.to_dict()
            if result.score >= threshold:
                kept.append(ex)
        return kept

    def _filter_reasoning(self, examples: List[Any]) -> List[Any]:
        threshold = self.filter_config.min_reasoning_score
        kept = []
        for ex in examples:
            result = self.reasoning_validator.check(ex.output)
            ex.metadata["reasoning_result"] = result.to_dict()
            if result.score >= threshold:
                kept.append(ex)
        return kept

    def _filter_code(self, examples: List[Any]) -> List[Any]:
        if not self.filter_config.enable_code_validation:
            return examples
        kept = []
        for ex in examples:
            code_blocks = re.findall(r"```(\w+)?\n(.*?)```", ex.output, re.DOTALL)
            if not code_blocks:
                kept.append(ex)
                continue

            all_valid = True
            block_results = []
            for lang, code in code_blocks:
                lang = lang.strip() or "unknown"
                code = code.strip()
                if code:
                    result = self.code_validator.check(code, lang)
                    block_results.append(result.to_dict())
                    if not result.passed:
                        all_valid = False

            ex.metadata["code_validation_results"] = block_results
            if all_valid:
                kept.append(ex)
        return kept

    def _filter_markdown(self, examples: List[Any]) -> List[Any]:
        if not self.filter_config.enable_markdown_validation:
            return examples
        kept = []
        for ex in examples:
            result = self.markdown_validator.check(ex.output)
            ex.metadata["markdown_result"] = result.to_dict()
            if result.score >= 0.5:
                kept.append(ex)
        return kept

    def _filter_quality(self, examples: List[Any]) -> List[Any]:
        threshold = self.filter_config.min_quality_score
        kept = []
        instructions = [ex.instruction for ex in examples]
        outputs = [ex.output for ex in examples]
        inputs = [ex.input for ex in examples]
        results = self.quality_scorer.score_batch(instructions, outputs, inputs)
        for ex, result in zip(examples, results):
            ex.quality_score = result.score
            ex.metadata["quality_result"] = result.to_dict()
            if result.score >= threshold:
                kept.append(ex)
        return kept

    def _filter_semantic(self, examples: List[Any]) -> List[Any]:
        if not self.filter_config.enable_semantic_dedup or len(examples) < 2:
            return examples
        try:
            texts = [f"{ex.instruction} {ex.input} {ex.output}" for ex in examples]
            dup_indices = self.semantic_filter.cluster_duplicates(texts)
            if not dup_indices:
                return examples
            seen = set()
            for cluster in dup_indices:
                cluster.sort(key=lambda i: examples[i].quality_score, reverse=True)
                seen.update(cluster[1:])
            kept = [ex for i, ex in enumerate(examples) if i not in seen]
            return kept
        except ImportError:
            return examples

    def _filter_rank(self, examples: List[Any]) -> List[Any]:
        return self.ranker.rank_by_quality(examples)

    def _save_report(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "filter_report.json")
        with open(path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2)

    def _save_filtered(self, examples: List[Any], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "filtered_dataset.jsonl")
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

    def get_report(self) -> FilterReport:
        return self.report
