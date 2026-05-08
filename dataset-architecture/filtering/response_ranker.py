from typing import Dict, List, Optional, Tuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .models import FilterResult
from .quality_scorer import EnhancedQualityScorer


@dataclass
class RankedItem:
    index: int
    score: float
    dimensions: Dict[str, float]
    metadata: Dict


class EnhancedResponseRanker:
    def __init__(self, scorer: Optional[EnhancedQualityScorer] = None, top_k: int = 3):
        self.scorer = scorer or EnhancedQualityScorer()
        self.top_k = top_k
        self.stats = {"ranked": 0}

    def _get_example_class(self):
        from src.pipeline.ingestion import DatasetExample
        return DatasetExample

    def rank_by_quality(self, examples: List[Any]) -> List[Any]:
        examples = sorted(examples, key=lambda x: x.quality_score, reverse=True)
        self.stats["ranked"] = len(examples)
        return examples

    def rank_by_difficulty(self, examples: List[Any], ascending: bool = True) -> List[Any]:
        return sorted(examples, key=lambda x: x.difficulty, reverse=not ascending)

    def rank_by_multimodal(
        self, examples: List[Any],
        primary: str = "quality_score",
        secondary: str = "difficulty",
        primary_weight: float = 0.7,
    ) -> List[Any]:
        def composite_key(ex: Any) -> float:
            p_val = getattr(ex, primary, 0) if hasattr(ex, primary) else ex.metadata.get(primary, 0)
            s_val = getattr(ex, secondary, 0) if hasattr(ex, secondary) else ex.metadata.get(secondary, 0)
            return primary_weight * p_val + (1 - primary_weight) * s_val

        return sorted(examples, key=composite_key, reverse=True)

    def compute_elo_scores(self, examples: List[Any]) -> List[Any]:
        if len(examples) < 2:
            return examples

        elo = {ex.id: 1500 for ex in examples}
        K = 32

        for i in range(len(examples)):
            for j in range(i + 1, len(examples)):
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

    def rank_texts(
        self, instructions: List[str], outputs: List[str], inputs: Optional[List[str]] = None
    ) -> List[RankedItem]:
        inputs = inputs or [""] * len(instructions)
        results = self.scorer.score_batch(instructions, outputs, inputs)

        ranked = []
        for idx, result in enumerate(results):
            ranked.append(RankedItem(
                index=idx,
                score=result.score,
                dimensions=result.dimension_scores,
                metadata={
                    "has_issues": len(result.issues) > 0,
                    "critical_issues": sum(1 for i in result.issues if i.severity.name in ("HIGH", "CRITICAL")),
                },
            ))

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    def get_top_k(self, items: List[RankedItem]) -> List[RankedItem]:
        return items[:self.top_k]

    def get_bottom_k(self, items: List[RankedItem]) -> List[RankedItem]:
        return items[-self.top_k:]

    def rank_for_curriculum(self, examples: List[Any]) -> List[Any]:
        def curriculum_key(ex: Any) -> Tuple:
            return (ex.difficulty, -ex.quality_score)
        return sorted(examples, key=curriculum_key)

    def get_percentile_rank(self, items: List[RankedItem], percentile: float = 0.9) -> List[RankedItem]:
        threshold_idx = int(len(items) * percentile)
        return items[:max(1, threshold_idx)]

    def get_stats(self) -> Dict:
        return self.stats
