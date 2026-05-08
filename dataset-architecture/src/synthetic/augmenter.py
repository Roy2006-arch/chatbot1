import random
import copy
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..pipeline.ingestion import DatasetExample


class DataAugmenter:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.stats = {"augmented": 0, "techniques_used": {}}

    def augment(self, examples: List[DatasetExample], num_workers: int = 8) -> List[DatasetExample]:
        augmented = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            all_futures = []

            techniques = [
                self._paraphrase,
                self._add_cot,
                self._scale_difficulty,
            ]

            for ex in examples:
                for technique in techniques:
                    future = executor.submit(technique, ex)
                    all_futures.append((technique.__name__, future))

            for tech_name, future in all_futures:
                result = future.result()
                if result:
                    augmented.append(result)
                    self.stats["augmented"] += 1
                    self.stats["techniques_used"][tech_name] = \
                        self.stats["techniques_used"].get(tech_name, 0) + 1

        return examples + augmented

    def _paraphrase(self, example: DatasetExample) -> Optional[DatasetExample]:
        paraphrased = copy.deepcopy(example)
        paraphrased.instruction = self._simple_paraphrase(example.instruction)
        paraphrased.source = "augmented_paraphrase"
        paraphrased.id = ""
        paraphrased.metadata["augmentation"] = "paraphrase"
        return paraphrased

    def _add_cot(self, example: DatasetExample) -> Optional[DatasetExample]:
        if "step" in example.output.lower() or "first," in example.output.lower():
            return None

        cot = copy.deepcopy(example)
        cot.output = f"Let's approach this systematically.\n\n**Step 1:** Analyze the problem.\n\n{cot.output}\n\n**Conclusion:** {cot.output}"
        cot.source = "augmented_cot"
        cot.metadata["augmentation"] = "chain_of_thought"
        return cot

    def _scale_difficulty(self, example: DatasetExample) -> Optional[DatasetExample]:
        if example.difficulty >= 5:
            return None

        harder = copy.deepcopy(example)
        harder.difficulty = min(5, example.difficulty + 1)
        harder.instruction = f"[Harder version] {harder.instruction}\n\nConsider edge cases and optimize your solution."
        harder.source = "augmented_harder"
        harder.metadata["augmentation"] = "difficulty_scale"
        return harder

    def _simple_paraphrase(self, text: str) -> str:
        replacements = [
            ("solve", "find a solution for"),
            ("explain", "describe"),
            ("implement", "write code for"),
            ("debug", "find and fix issues in"),
            ("design", "architect"),
            ("optimize", "improve the performance of"),
            ("analyze", "evaluate"),
            ("what is", "can you explain"),
            ("how to", "what's the approach to"),
            ("why", "what is the reason"),
        ]

        result = text
        for old, new in replacements:
            if old in result.lower() and random.random() < 0.3:
                idx = result.lower().find(old)
                result = result[:idx] + new + result[idx + len(old):]
                break

        return result

    def get_stats(self) -> Dict:
        return self.stats
