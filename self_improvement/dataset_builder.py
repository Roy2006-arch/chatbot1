import json
import os
import re
import random
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

from .schema import SelfImprovementExample, ExampleSource
from .correction_generator import CorrectionGenerator
from .quality_curator import QualityCurator
from .hard_example_miner import HardExampleMiner


class DatasetBuilder:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.train_split = self.config.get("train_split", 0.8)
        self.val_split = self.config.get("val_split", 0.1)
        self.test_split = self.config.get("test_split", 0.1)
        self.curriculum_order = self.config.get("curriculum_order", True)
        self.include_metadata = self.config.get("include_metadata", True)
        self.stats = {"built": 0, "exported": 0}

    def build_dataset(
        self,
        correction_examples: List[SelfImprovementExample],
        quality_examples: List[SelfImprovementExample],
        hard_examples: List[SelfImprovementExample],
        max_total: int = 1000,
    ) -> List[SelfImprovementExample]:
        if not self.enabled:
            return []

        combined: Dict[str, SelfImprovementExample] = {}

        for ex in correction_examples:
            if ex.corrected_response and len(ex.corrected_response.strip()) > 10:
                key = ex.prompt[:100]
                if key not in combined or ex.quality_score > combined[key].quality_score:
                    combined[key] = ex

        for ex in quality_examples:
            key = ex.prompt[:100]
            if key not in combined or ex.quality_score > combined[key].quality_score:
                combined[key] = ex

        for ex in hard_examples:
            key = ex.prompt[:100]
            if key not in combined:
                combined[key] = ex

        examples = list(combined.values())

        if len(examples) > max_total:
            examples.sort(key=lambda x: self._priority_score(x), reverse=True)
            examples = examples[:max_total]

        if self.curriculum_order:
            examples = self._curriculum_sort(examples)

        self.stats["built"] = len(examples)
        return examples

    def split_dataset(
        self, examples: List[SelfImprovementExample]
    ) -> Tuple[List[SelfImprovementExample], List[SelfImprovementExample], List[SelfImprovementExample]]:
        n = len(examples)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)

        shuffled = examples[:]
        random.shuffle(shuffled)

        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]

        return train, val, test

    def export_transformers(
        self, examples: List[SelfImprovementExample], output_path: str
    ) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        path = output_path if output_path.endswith(".jsonl") else output_path + ".jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "instruction": ex.prompt,
                    "output": ex.corrected_response or ex.original_response,
                    "source": ex.source.value,
                    "category": ex.category,
                    "difficulty": ex.difficulty,
                    "quality_score": ex.quality_score,
                }
                if self.include_metadata:
                    record["metadata"] = {
                        "failure_reasons": ex.failure_reasons,
                        "correction_method": ex.correction_method.value,
                    }
                f.write(json.dumps(record) + "\n")

        self.stats["exported"] += len(examples)
        return path

    def export_openai(
        self, examples: List[SelfImprovementExample], output_path: str
    ) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        path = output_path if output_path.endswith(".jsonl") else "_openai.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                messages = [
                    {"role": "user", "content": ex.prompt},
                    {"role": "assistant", "content": ex.corrected_response or ex.original_response},
                ]
                record = {"messages": messages, "source": ex.source.value}
                if self.include_metadata:
                    record["metadata"] = {
                        "category": ex.category,
                        "difficulty": ex.difficulty,
                        "quality_score": ex.quality_score,
                    }
                f.write(json.dumps(record) + "\n")

        self.stats["exported"] += len(examples)
        return path

    def export_dpo(
        self, examples: List[SelfImprovementExample], output_path: str
    ) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        path = output_path if output_path.endswith(".jsonl") else "_dpo.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                if not ex.original_response or not ex.corrected_response:
                    continue
                record = {
                    "prompt": ex.prompt,
                    "chosen": ex.corrected_response,
                    "rejected": ex.original_response,
                    "source": ex.source.value,
                    "score": ex.quality_score,
                }
                if self.include_metadata:
                    record["metadata"] = {
                        "failure_reasons": ex.failure_reasons,
                        "difficulty": ex.difficulty,
                    }
                f.write(json.dumps(record) + "\n")

        self.stats["exported"] += len(examples)
        return path

    def export_all_formats(
        self,
        train: List[SelfImprovementExample],
        val: List[SelfImprovementExample],
        test: List[SelfImprovementExample],
        output_dir: str,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        formats = formats or self.config.get("formats", ["transformers", "openai"])
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        for fmt in formats:
            for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
                if fmt == "transformers":
                    path = self.export_transformers(
                        split_data, os.path.join(output_dir, f"{split_name}.jsonl")
                    )
                elif fmt == "openai":
                    path = self.export_openai(
                        split_data, os.path.join(output_dir, f"{split_name}")
                    )
                elif fmt == "dpo":
                    path = self.export_dpo(
                        split_data, os.path.join(output_dir, f"{split_name}")
                    )
                else:
                    continue
                paths[f"{split_name}_{fmt}"] = path

        return paths

    def _priority_score(self, ex: SelfImprovementExample) -> float:
        score = 0.0
        if ex.source == ExampleSource.HARD_EXAMPLE:
            score += 3.0
        elif ex.source == ExampleSource.FAILED_QUERY:
            score += 2.0
        elif ex.source == ExampleSource.CORRECTION:
            score += 1.5

        score += ex.quality_score * 2
        score += ex.difficulty * 0.5

        if "refusal" in str(ex.failure_reasons).lower():
            score += 1.0
        if "hallucination" in str(ex.failure_reasons).lower():
            score += 1.0

        return score

    def _curriculum_sort(self, examples: List[SelfImprovementExample]) -> List[SelfImprovementExample]:
        return sorted(examples, key=lambda x: (x.difficulty, -x.quality_score))

    def save_metadata(self, examples: List[SelfImprovementExample], output_dir: str):
        metadata = {
            "total_examples": len(examples),
            "by_source": dict(self._count_by(examples, lambda x: x.source.value)),
            "by_category": dict(self._count_by(examples, lambda x: x.category or "uncategorized")),
            "avg_quality": round(sum(e.quality_score for e in examples) / max(len(examples), 1), 4),
            "avg_difficulty": round(sum(e.difficulty for e in examples) / max(len(examples), 1), 2),
            "created_at": datetime.utcnow().isoformat(),
        }
        path = os.path.join(output_dir, "dataset_metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        return path

    @staticmethod
    def _count_by(items: List, key_fn) -> Dict:
        counts = defaultdict(int)
        for item in items:
            counts[key_fn(item)] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_stats(self) -> Dict:
        return self.stats
