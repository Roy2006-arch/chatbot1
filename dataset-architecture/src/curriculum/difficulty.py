import re
import math
from typing import Dict, List, Optional, Tuple

from ..pipeline.ingestion import DatasetExample


class DifficultyScorer:
    FEATURES = {
        "code_length": (0.15, 0, 1000),
        "vocabulary_complexity": (0.15, 0, 1),
        "concept_density": (0.20, 0, 1),
        "abstraction_level": (0.15, 1, 5),
        "multi_step_reasoning": (0.20, 1, 10),
        "domain_specificity": (0.15, 1, 5),
    }

    def __init__(self):
        self.stats = {"scored": 0, "distribution": {}}

    def score_difficulty(self, examples: List[DatasetExample]) -> List[DatasetExample]:
        for ex in examples:
            ex.difficulty = self._compute_difficulty(ex)
        self.stats["scored"] = len(examples)

        dist = {}
        for d in range(1, 6):
            dist[d] = sum(1 for ex in examples if ex.difficulty == d)
        self.stats["distribution"] = dist

        return examples

    def _compute_difficulty(self, example: DatasetExample) -> int:
        text = f"{example.instruction} {example.input} {example.output}"

        features = {
            "code_length": self._score_code_length(text),
            "vocabulary_complexity": self._score_vocabulary(text),
            "concept_density": self._score_concept_density(text),
            "abstraction_level": self._score_abstraction(text),
            "multi_step_reasoning": self._score_reasoning_steps(text),
            "domain_specificity": self._score_domain_specificity(text),
        }

        weighted_sum = sum(
            features[feat] * config[0]
            for feat, config in self.FEATURES.items()
        )

        difficulty_mapping = {1: (0, 0.2), 2: (0.2, 0.4), 3: (0.4, 0.6), 4: (0.6, 0.8), 5: (0.8, 1.0)}
        for level, (lo, hi) in difficulty_mapping.items():
            if lo <= weighted_sum < hi:
                return level

        return 5

    def _score_code_length(self, text: str) -> float:
        code_blocks = re.findall(r"```\w*\n(.*?)```", text, re.DOTALL)
        if not code_blocks:
            return 0.0
        total_lines = sum(len(block.split("\n")) for block in code_blocks)
        avg_lines = total_lines / len(code_blocks)
        return min(1.0, avg_lines / 100)

    def _score_vocabulary(self, text: str) -> float:
        advanced_terms = [
            r'\b(?:recursion|backtracking|dynamic.programming|memoization|divide.and.conquer)',
            r'\b(?:asymptotic|amortized|topological|polynomial|exponential|logarithmic)',
            r'\b(?:complexity|optimization|concurrency|parallelism|distributed|asynchronous)',
            r'\b(?:polymorphism|inheritance|encapsulation|abstraction|composition)',
        ]
        matches = sum(1 for p in advanced_terms if re.search(p, text, re.IGNORECASE))
        return min(1.0, matches / 5)

    def _score_concept_density(self, text: str) -> float:
        technical_concepts = [
            r'\b(?:algorithm|data.structure|graph|tree|heap|stack|queue|array|linked.list)',
            r'\b(?:sorting|searching|traversal|iteration|recursion|optimization)',
            r'\b(?:API|SDK|HTTP|TCP|UDP|REST|GraphQL|gRPC|WebSocket)',
            r'\b(?:database|cache|load.balancer|microservice|container|orchestration)',
        ]
        words = text.split()
        concept_count = sum(1 for p in technical_concepts if re.search(p, text, re.IGNORECASE))
        if len(words) == 0:
            return 0.0
        density = concept_count / max(len(words) / 100, 1)
        return min(1.0, density / 10)

    def _score_abstraction(self, text: str) -> float:
        abstraction_indicators = [
            r'\b(?:abstract|interface|generic|template|polymorphic|virtual)',
            r'\b(?:architecture|pattern|design|framework|paradigm)',
            r'\b(?:encapsulation|modularity|dependency|injection|inversion)',
        ]
        matches = sum(1 for p in abstraction_indicators if re.search(p, text, re.IGNORECASE))
        return min(1.0, matches / 3)

    def _score_reasoning_steps(self, text: str) -> float:
        step_patterns = [
            r'\b(?:first|second|third|finally|next|then|after|before)',
            r'\b(?:step\s+\d+|phase|stage|iteration\s+\d+)',
            r'\b(?:because|therefore|hence|thus|since|consequently)',
            r'\b(?:if|then|else|otherwise|assuming|given that)',
            r'\b(?:1\.|2\.|3\.|\d+\.\s)',
        ]
        matches = sum(1 for p in step_patterns if re.search(p, text, re.IGNORECASE))
        return min(1.0, matches / 5)

    def _score_domain_specificity(self, text: str) -> float:
        domain_terms = [
            r'\b(?:competitive programming|codeforces|leetcode|hackerrank)',
            r'\b(?:distributed.system|microservice|kubernetes|docker)',
            r'\b(?:machine.learning|deep.learning|neural|transformer|attention)',
            r'\b(?:formal.verification|proof|theorem|axiom|lemma)',
        ]
        matches = sum(1 for p in domain_terms if re.search(p, text, re.IGNORECASE))
        return min(1.0, matches / 2)

    def get_stats(self) -> Dict:
        return self.stats
