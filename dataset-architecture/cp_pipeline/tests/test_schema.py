import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cp_pipeline.schema import (
    Problem, TestCase, Solution, Language, Platform,
    DifficultyLevel, DSAPattern, CPInstructionExample,
)


class TestSchema:
    def test_problem_creation(self):
        p = Problem(
            title="Two Sum",
            platform=Platform.LEETCODE,
            platform_id="1",
            difficulty=DifficultyLevel.EASY,
            dsa_patterns=[DSAPattern.ARRAY, DSAPattern.HASH_TABLE],
        )
        assert p.title == "Two Sum"
        assert p.platform == Platform.LEETCODE
        assert p.id != ""
        assert p.difficulty == DifficultyLevel.EASY
        assert DSAPattern.ARRAY in p.dsa_patterns

    def test_solution_creation(self):
        sol = Solution(
            language=Language.PYTHON,
            code="def solve(): pass",
            time_complexity="O(n)",
            space_complexity="O(1)",
            approach="Hash map approach",
        )
        assert sol.language == Language.PYTHON
        assert sol.time_complexity == "O(n)"
        assert sol.line_count == 1

    def test_test_case(self):
        tc = TestCase(input="[1,2,3]", expected_output="6", explanation="Simple case", is_edge_case=False)
        assert tc.expected_output == "6"
        assert tc.is_edge_case is False

    def test_problem_to_dict(self):
        p = Problem(title="Test", platform=Platform.LEETCODE)
        d = p.to_dict()
        assert d["title"] == "Test"
        assert d["platform"] == "leetcode"

    def test_problem_from_dict(self):
        d = {
            "title": "Test",
            "platform": "leetcode",
            "difficulty": 2,
            "dsa_patterns": ["array"],
            "algorithm_categories": [],
            "sample_test_cases": [],
            "hidden_test_cases": [],
            "edge_test_cases": [],
            "solutions": {},
            "hints": [], "topics": [], "companies": [], "similar_problems": [],
            "common_mistakes": [], "debugging_examples": [],
            "acceptance_rate": 0.0, "frequency": 0.0, "rating": 0,
            "tags": {}, "problem_statement": "", "constraints": [],
            "input_format": "", "output_format": "", "solution_approach": "",
            "complexity_analysis": "", "platform_id": "", "id": "",
        }
        p = Problem.from_dict(d)
        assert p.title == "Test"
        assert p.platform == Platform.LEETCODE
        assert p.difficulty == DifficultyLevel.MEDIUM

    def test_instruction_example(self):
        ex = CPInstructionExample(
            instruction="Solve this problem",
            input="Problem: Two Sum",
            output="Solution code here",
            metadata={"type": "solve"},
        )
        assert ex.instruction == "Solve this problem"
        assert ex.metadata["type"] == "solve"
