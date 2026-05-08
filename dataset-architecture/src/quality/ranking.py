from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..pipeline.ingestion import DatasetExample


class ResponseRanker:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.stats = {"ranked_examples": 0}

    def rank(self, examples: List[DatasetExample], num_workers: int = 8) -> List[DatasetExample]:
        if not examples:
            return examples
        ranked = sorted(examples, key=lambda x: x.quality_score, reverse=True)
        self.stats["ranked_examples"] = len(ranked)
        return ranked

    def rank_by_difficulty(self, examples: List[DatasetExample], ascending: bool = True) -> List[DatasetExample]:
        return sorted(examples, key=lambda x: x.difficulty, reverse=not ascending)

    def get_top_k_per_category(self, examples: List[DatasetExample]) -> Dict[str, List[DatasetExample]]:
        categories = {}
        for ex in examples:
            cat = ex.category or "uncategorized"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ex)

        top_k = {}
        for cat, cat_examples in categories.items():
            cat_examples.sort(key=lambda x: x.quality_score, reverse=True)
            top_k[cat] = cat_examples[:self.top_k]

        return top_k

    def rank_for_curriculum(self, examples: List[DatasetExample]) -> List[DatasetExample]:
        def curriculum_key(ex: DatasetExample) -> Tuple:
            return (ex.difficulty, -ex.quality_score)
        return sorted(examples, key=curriculum_key)

    def compute_elo_scores(self, examples: List[DatasetExample]) -> List[DatasetExample]:
        if len(examples) < 2:
            return examples

        elo = {ex.id: 1500 for ex in examples}
        K = 32

        for i in range(len(examples)):
            for j in range(i + 1, len(examples)):
                if i == j:
                    continue
                expected_i = 1.0 / (1.0 + 10 ** ((elo[examples[j].id] - elo[examples[i].id]) / 400.0))
                expected_j = 1.0 - expected_i

                if examples[i].quality_score > examples[j].quality_score:
                    elo[examples[i].id] += K * (1.0 - expected_i)
                    elo[examples[j].id] += K * (0.0 - expected_j)
                elif examples[j].quality_score > examples[i].quality_score:
                    elo[examples[i].id] += K * (0.0 - expected_i)
                    elo[examples[j].id] += K * (1.0 - expected_j)
                else:
                    elo[examples[i].id] += K * (0.5 - expected_i)
                    elo[examples[j].id] += K * (0.5 - expected_j)

        for ex in examples:
            ex.metadata["elo_score"] = round(elo[ex.id], 1)

        examples.sort(key=lambda x: x.metadata.get("elo_score", 0), reverse=True)
        return examples

    def get_stats(self) -> Dict:
        return self.stats
