import re
import math
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..schema import Problem, DifficultyLevel, DSAPattern, Solution, Language


class CPDifficultyScorer:
    def __init__(self):
        self.stats = {"scored": 0}

    def score_problem(self, problem: Problem) -> Problem:
        scores = self._compute_scores(problem)
        composite = sum(scores.values()) / max(len(scores), 1)
        problem.difficulty = self._composite_to_level(composite)
        problem.tags["difficulty_scores"] = scores
        problem.tags["composite_difficulty"] = round(composite, 3)
        self.stats["scored"] += 1
        return problem

    def score_batch(self, problems: List[Problem], num_workers: int = 8) -> List[Problem]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.score_problem, p) for p in problems]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _compute_scores(self, problem: Problem) -> Dict[str, float]:
        return {
            "pattern_complexity": self._score_pattern_complexity(problem),
            "solution_complexity": self._score_solution_complexity(problem),
            "constraint_severity": self._score_constraints(problem),
            "scope": self._score_scope(problem),
            "reasoning_depth": self._score_reasoning(problem),
        }

    def _score_pattern_complexity(self, problem: Problem) -> float:
        pattern_difficulty = {
            DSAPattern.ARRAY: 0.1, DSAPattern.STRING: 0.15,
            DSAPattern.MATH: 0.15, DSAPattern.SORTING: 0.2,
            DSAPattern.HASH_TABLE: 0.2, DSAPattern.TWO_POINTERS: 0.3,
            DSAPattern.STACK: 0.3, DSAPattern.BINARY_SEARCH: 0.4,
            DSAPattern.SLIDING_WINDOW: 0.4, DSAPattern.PREFIX_SUM: 0.35,
            DSAPattern.GREEDY: 0.4, DSAPattern.RECURSION: 0.4,
            DSAPattern.DIVIDE_AND_CONQUER: 0.45,
            DSAPattern.DEPTH_FIRST_SEARCH: 0.4,
            DSAPattern.BREADTH_FIRST_SEARCH: 0.4,
            DSAPattern.TREE: 0.35, DSAPattern.MATRIX: 0.35,
            DSAPattern.BACKTRACKING: 0.5,
            DSAPattern.DYNAMIC_PROGRAMMING: 0.7,
            DSAPattern.GRAPH: 0.55, DSAPattern.HEAP: 0.4,
            DSAPattern.UNION_FIND: 0.5,
            DSAPattern.TOPOLOGICAL_SORT: 0.55,
            DSAPattern.SHORTEST_PATH: 0.6,
            DSAPattern.MINIMUM_SPANNING_TREE: 0.55,
            DSAPattern.SEGMENT_TREE: 0.75,
            DSAPattern.TRIE: 0.5, DSAPattern.BIT_MANIPULATION: 0.4,
            DSAPattern.MONOTONIC_STACK: 0.5,
            DSAPattern.INTERVAL: 0.4, DSAPattern.DESIGN: 0.5,
            DSAPattern.NUMBER_THEORY: 0.5,
            DSAPattern.COMBINATORICS: 0.45,
            DSAPattern.GAME_THEORY: 0.65,
            DSAPattern.SIMULATION: 0.3,
        }
        if not problem.dsa_patterns:
            return 0.3
        scores = [pattern_difficulty.get(p, 0.3) for p in problem.dsa_patterns]
        return sum(scores) / len(scores)

    def _score_solution_complexity(self, problem: Problem) -> float:
        complexity_map = {
            "O(1)": 0.1, "O(log n)": 0.2, "O(√n)": 0.3,
            "O(n)": 0.3, "O(n log n)": 0.4, "O(n√n)": 0.5,
            "O(n^2)": 0.6, "O(n^3)": 0.7, "O(2^n)": 0.85,
            "O(n!)": 0.95, "O(n^n)": 1.0,
        }
        scores = []
        for sol in problem.solutions.values():
            for complexity, score in complexity_map.items():
                if complexity.lower() in sol.time_complexity.lower():
                    scores.append(score)
                    break
            else:
                scores.append(0.5)
        return sum(scores) / max(len(scores), 1) if scores else 0.5

    def _score_constraints(self, problem: Problem) -> float:
        sizes = []
        for constraint in problem.constraints:
            nums = re.findall(r'\d+', constraint)
            for n in nums:
                val = int(n)
                if val > 1000:
                    sizes.append(min(1.0, math.log10(val) / 9))
                else:
                    sizes.append(val / 1000.0)
        if not sizes:
            return 0.3
        return sum(sizes) / len(sizes)

    def _score_scope(self, problem: Problem) -> float:
        patterns = len(problem.dsa_patterns)
        solutions = len(problem.solutions)
        test_cases = len(problem.sample_test_cases)

        scope = 0.3
        scope += min(0.3, patterns * 0.1)
        scope += min(0.2, solutions * 0.05)
        scope += min(0.2, test_cases * 0.04)
        return min(1.0, scope)

    def _score_reasoning(self, problem: Problem) -> float:
        text = f"{problem.problem_statement} {problem.solution_approach}".lower()

        reasoning_signals = {
            r"step|phase|stage|iteration": 0.05,
            r"prove|proof|theorem|lemma|claim": 0.2,
            r"optimize|optimal|optimization|improve": 0.1,
            r"trade.off|compare|alternative|approach": 0.15,
            r"complexity|efficient|performance|scal": 0.1,
            r"edge.case|corner.case|boundary": 0.1,
            r"recurrence|induction|deduction": 0.2,
        }
        score = 0.2
        for pattern, delta in reasoning_signals.items():
            if re.search(pattern, text):
                score += delta
        return min(1.0, score)

    def _composite_to_level(self, score: float) -> DifficultyLevel:
        if score < 0.25: return DifficultyLevel.EASY
        elif score < 0.50: return DifficultyLevel.EASY
        elif score < 0.65: return DifficultyLevel.MEDIUM
        elif score < 0.80: return DifficultyLevel.HARD
        return DifficultyLevel.EXPERT

    def get_stats(self) -> Dict:
        return self.stats


class ProblemSetBuilder:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def build_curriculum(
        self,
        problems: List[Problem],
        strategy: str = "progressive",
    ) -> Dict[DifficultyLevel, List[Problem]]:
        if strategy == "progressive":
            return self._progressive_curriculum(problems)
        elif strategy == "spaced_repetition":
            return self._spaced_curriculum(problems)
        return self._progressive_curriculum(problems)

    def _progressive_curriculum(self, problems: List[Problem]) -> Dict[DifficultyLevel, List[Problem]]:
        buckets = {dl: [] for dl in DifficultyLevel}
        for p in problems:
            buckets[p.difficulty].append(p)
        return buckets

    def _spaced_curriculum(self, problems: List[Problem]) -> Dict[DifficultyLevel, List[Problem]]:
        import random
        random.seed(self.seed)
        buckets = self._progressive_curriculum(problems)
        for dl in DifficultyLevel:
            random.shuffle(buckets[dl])
            interleaved = []
            half = len(buckets[dl]) // 2
            for i in range(half):
                interleaved.append(buckets[dl][i])
                if i + half < len(buckets[dl]):
                    interleaved.append(buckets[dl][i + half])
            buckets[dl] = interleaved or buckets[dl]
        return buckets

    def build_training_plan(
        self,
        problems: List[Problem],
        total_examples: int = 10000,
        easy_ratio: float = 0.20,
        medium_ratio: float = 0.40,
        hard_ratio: float = 0.30,
        expert_ratio: float = 0.10,
    ) -> List[Problem]:
        import random
        random.seed(self.seed)
        buckets = self.build_curriculum(problems)
        ratios = {
            DifficultyLevel.EASY: easy_ratio,
            DifficultyLevel.MEDIUM: medium_ratio,
            DifficultyLevel.HARD: hard_ratio,
            DifficultyLevel.EXPERT: expert_ratio,
        }
        plan = []
        for dl, ratio in ratios.items():
            pool = buckets[dl]
            count = int(total_examples * ratio)
            if pool:
                sampled = random.sample(pool, min(count, len(pool)))
                plan.extend(sampled)
        random.shuffle(plan)
        return plan

    def mine_hard_problems(
        self,
        problems: List[Problem],
        top_k: int = 100,
    ) -> List[Problem]:
        scored = sorted(
            problems,
            key=lambda p: (
                p.difficulty.value,
                p.tags.get("composite_difficulty", 0),
                -(p.acceptance_rate if p.acceptance_rate > 0 else 0.5),
            ),
            reverse=True,
        )
        return scored[:top_k]

    def get_pattern_distribution(self, problems: List[Problem]) -> Dict[str, int]:
        from collections import Counter
        all_patterns = []
        for p in problems:
            all_patterns.extend([pat.value for pat in p.dsa_patterns])
        return dict(Counter(all_patterns).most_common())

    def get_missing_patterns(self, problems: List[Problem], target_coverage: Dict[str, float]) -> List[str]:
        current = self.get_pattern_distribution(problems)
        total = sum(current.values()) or 1
        missing = []
        for pattern, target in target_coverage.items():
            current_ratio = current.get(pattern, 0) / total
            if current_ratio < target * 0.5:
                missing.append(pattern)
        return missing
