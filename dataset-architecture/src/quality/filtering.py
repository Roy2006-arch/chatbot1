from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from ..pipeline.ingestion import DatasetExample


class QualityFilter:
    def __init__(self, config_path: str = "config/quality.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        filter_config = config.get("filtering", {})
        self.min_instruction_length = filter_config.get("min_instruction_length", 5)
        self.max_instruction_length = filter_config.get("max_instruction_length", 4096)
        self.min_output_length = filter_config.get("min_output_length", 1)
        self.max_output_length = filter_config.get("max_output_length", 32768)
        self.composite_threshold = config.get("scoring", {}).get("composite_threshold", 0.65)
        self.stats = {"kept": 0, "removed": 0}

    def filter(self, examples: List[DatasetExample], num_workers: int = 8) -> List[DatasetExample]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._passes_filter, ex) for ex in examples]
            results = []
            for future, ex in zip(as_completed(futures), examples):
                if future.result():
                    results.append(ex)
                else:
                    self.stats["removed"] += 1

        self.stats["kept"] = len(results)
        return results

    def _passes_filter(self, example: DatasetExample) -> bool:
        if len(example.instruction) < self.min_instruction_length:
            return False
        if len(example.instruction) > self.max_instruction_length:
            return False
        if len(example.output) < self.min_output_length:
            return False
        if len(example.output) > self.max_output_length:
            return False
        if example.quality_score < self.composite_threshold:
            return False
        return True

    def filter_by_category(self, examples: List[DatasetExample], categories: List[str]) -> List[DatasetExample]:
        return [ex for ex in examples if ex.category in categories]

    def sample_balanced(self, examples: List[DatasetExample], target_per_category: int) -> List[DatasetExample]:
        from collections import defaultdict
        buckets = defaultdict(list)
        for ex in examples:
            buckets[ex.category or "uncategorized"].append(ex)

        balanced = []
        for cat, cat_examples in buckets.items():
            cat_examples.sort(key=lambda x: x.quality_score, reverse=True)
            balanced.extend(cat_examples[:target_per_category])

        return balanced

    def get_stats(self) -> Dict:
        return self.stats
