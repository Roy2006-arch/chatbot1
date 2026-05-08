import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cp_pipeline.schema import Problem, Platform, DifficultyLevel, DSAPattern, Language, Solution
from cp_pipeline.curriculum.difficulty import CPDifficultyScorer, ProblemSetBuilder


class TestCPDifficultyScorer:
    def setup_method(self):
        self.scorer = CPDifficultyScorer()

    def test_dp_is_harder_than_array(self):
        dp = Problem(
            title="DP Problem",
            dsa_patterns=[DSAPattern.DYNAMIC_PROGRAMMING],
            platform=Platform.LEETCODE,
        )
        arr = Problem(
            title="Array Problem",
            dsa_patterns=[DSAPattern.ARRAY],
            platform=Platform.LEETCODE,
        )
        dp = self.scorer.score_problem(dp)
        arr = self.scorer.score_problem(arr)
        assert dp.difficulty.value >= arr.difficulty.value

    def test_solution_complexity_matters(self):
        easy = Problem(
            title="Easy",
            dsa_patterns=[DSAPattern.ARRAY],
            platform=Platform.LEETCODE,
        )
        easy.solutions["python"] = Solution(
            language=Language.PYTHON, code="pass",
            time_complexity="O(1)", space_complexity="O(1)",
            approach="Simple",
        )
        hard = Problem(
            title="Hard",
            dsa_patterns=[DSAPattern.DYNAMIC_PROGRAMMING],
            platform=Platform.LEETCODE,
        )
        hard.solutions["python"] = Solution(
            language=Language.PYTHON, code="pass",
            time_complexity="O(n^3)", space_complexity="O(n^2)",
            approach="Complex DP",
        )
        diff_easy = self.scorer._composite_to_level(self.scorer._compute_scores(easy)["pattern_complexity"])
        diff_hard = self.scorer._composite_to_level(self.scorer._compute_scores(hard)["pattern_complexity"])
        assert diff_hard.value >= diff_easy.value

    def test_batch_scoring(self):
        problems = [
            Problem(title="P1", dsa_patterns=[DSAPattern.ARRAY], platform=Platform.LEETCODE),
            Problem(title="P2", dsa_patterns=[DSAPattern.DYNAMIC_PROGRAMMING], platform=Platform.LEETCODE),
        ]
        results = self.scorer.score_batch(problems)
        assert len(results) == 2


class TestProblemSetBuilder:
    def setup_method(self):
        self.builder = ProblemSetBuilder()

    def test_build_curriculum(self):
        problems = [
            Problem(title="Easy", difficulty=DifficultyLevel.EASY, platform=Platform.LEETCODE),
            Problem(title="Medium", difficulty=DifficultyLevel.MEDIUM, platform=Platform.LEETCODE),
            Problem(title="Hard", difficulty=DifficultyLevel.HARD, platform=Platform.LEETCODE),
        ]
        curriculum = self.builder.build_curriculum(problems)
        assert len(curriculum[DifficultyLevel.EASY]) == 1
        assert len(curriculum[DifficultyLevel.MEDIUM]) == 1
        assert len(curriculum[DifficultyLevel.HARD]) == 1

    def test_mine_hard_problems(self):
        problems = [
            Problem(title="Easy", difficulty=DifficultyLevel.EASY, acceptance_rate=80, platform=Platform.LEETCODE, tags={"composite_difficulty": 0.2}),
            Problem(title="Hard", difficulty=DifficultyLevel.HARD, acceptance_rate=25, platform=Platform.LEETCODE, tags={"composite_difficulty": 0.85}),
            Problem(title="Medium", difficulty=DifficultyLevel.MEDIUM, acceptance_rate=50, platform=Platform.LEETCODE, tags={"composite_difficulty": 0.5}),
        ]
        hard = self.builder.mine_hard_problems(problems, top_k=2)
        assert len(hard) == 2
        assert hard[0].difficulty == DifficultyLevel.HARD

    def test_training_plan(self):
        problems = [
            Problem(title=f"P{i}", difficulty=DifficultyLevel((i % 4) + 1), platform=Platform.LEETCODE)
            for i in range(20)
        ]
        plan = self.builder.build_training_plan(problems, total_examples=10)
        assert len(plan) <= 20

    def test_pattern_distribution(self):
        problems = [
            Problem(title="P1", dsa_patterns=[DSAPattern.ARRAY], platform=Platform.LEETCODE),
            Problem(title="P2", dsa_patterns=[DSAPattern.ARRAY, DSAPattern.HASH_TABLE], platform=Platform.LEETCODE),
        ]
        dist = self.builder.get_pattern_distribution(problems)
        assert dist.get("array", 0) == 2

    def test_missing_patterns(self):
        problems = [
            Problem(title="P1", dsa_patterns=[DSAPattern.ARRAY], platform=Platform.LEETCODE),
        ]
        missing = self.builder.get_missing_patterns(problems, {"dynamic_programming": 0.1})
        assert "dynamic_programming" in missing
