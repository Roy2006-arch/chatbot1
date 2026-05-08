import random
from typing import Dict, List, Optional, Generator, Tuple
from collections import defaultdict

from ..pipeline.ingestion import DatasetExample


class CurriculumScheduler:
    def __init__(
        self,
        strategy: str = "linear",
        total_steps: int = 10000,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.seed = seed
        random.seed(seed)
        self.stats = {"scheduled_batches": 0, "strategy": strategy}

    def schedule(self, examples: List[DatasetExample]) -> Generator[List[DatasetExample], None, None]:
        if self.strategy == "linear":
            yield from self._linear_schedule(examples)
        elif self.strategy == "exponential":
            yield from self._exponential_schedule(examples)
        elif self.strategy == "pacing":
            yield from self._pacing_schedule(examples)
        elif self.strategy == "self_paced":
            yield from self._self_paced_schedule(examples)
        else:
            yield from self._linear_schedule(examples)

    def _linear_schedule(self, examples: List[DatasetExample]) -> Generator[List[DatasetExample], None, None]:
        sorted_examples = sorted(examples, key=lambda x: x.difficulty)
        steps_per_difficulty = self.total_steps // 5

        for difficulty in range(1, 6):
            pool = [ex for ex in sorted_examples if ex.difficulty == difficulty]
            if not pool:
                continue

            steps = min(steps_per_difficulty, len(pool) * 3)
            for _ in range(0, steps, self.batch_size):
                batch = random.sample(pool, min(self.batch_size, len(pool)))
                yield batch
                self.stats["scheduled_batches"] += 1

    def _exponential_schedule(self, examples: List[DatasetExample]) -> Generator[List[DatasetExample], None, None]:
        sorted_examples = sorted(examples, key=lambda x: x.difficulty)
        n = len(sorted_examples)
        batch_count = self.total_steps // self.batch_size

        import math
        for i in range(batch_count):
            progress = i / batch_count
            threshold = 1 + 4 * (1 - math.exp(-3 * progress))
            eligible = [ex for ex in sorted_examples if ex.difficulty <= threshold]
            if not eligible:
                eligible = sorted_examples[-self.batch_size:]

            batch = random.sample(eligible, min(self.batch_size, len(eligible)))
            yield batch
            self.stats["scheduled_batches"] += 1

    def _pacing_schedule(self, examples: List[DatasetExample]) -> Generator[List[DatasetExample], None, None]:
        sorted_examples = sorted(examples, key=lambda x: x.difficulty)
        by_difficulty = defaultdict(list)
        for ex in sorted_examples:
            by_difficulty[ex.difficulty].append(ex)

        batch_count = self.total_steps // self.batch_size
        for i in range(batch_count):
            phase = i / batch_count
            batch = []
            for d in range(1, 6):
                proportion = max(0, min(1, (phase - (d - 1) * 0.2) / 0.2))
                pool = by_difficulty.get(d, [])
                n_from_pool = int(self.batch_size * proportion * 0.3)
                if pool and n_from_pool > 0:
                    batch.extend(random.sample(pool, min(n_from_pool, len(pool))))

            if len(batch) < self.batch_size:
                all_pool = by_difficulty.get(5, []) or sorted_examples
                batch.extend(random.sample(all_pool, min(self.batch_size - len(batch), len(all_pool))))

            random.shuffle(batch)
            yield batch[:self.batch_size]
            self.stats["scheduled_batches"] += 1

    def _self_paced_schedule(self, examples: List[DatasetExample]) -> Generator[List[DatasetExample], None, None]:
        sorted_by_quality = sorted(examples, key=lambda x: x.quality_score)
        batch_count = self.total_steps // self.batch_size

        for i in range(batch_count):
            progress = i / batch_count
            threshold_idx = int(len(sorted_by_quality) * progress)
            eligible = sorted_by_quality[:threshold_idx + 1] or sorted_by_quality[:self.batch_size]
            batch = random.sample(eligible, min(self.batch_size, len(eligible)))
            yield batch
            self.stats["scheduled_batches"] += 1

    def create_training_plan(self, examples: List[DatasetExample]) -> Tuple[List[DatasetExample], List[DatasetExample]]:
        sorted_examples = sorted(examples, key=lambda x: (x.difficulty, -x.quality_score))
        split_idx = len(sorted_examples) * 80 // 100
        easy_hard = sorted_examples[:split_idx]
        hard_mining = sorted_examples[split_idx:]
        return easy_hard, hard_mining

    def get_stats(self) -> Dict:
        return self.stats
