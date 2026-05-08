import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cp_pipeline.schema import Problem, Platform, DifficultyLevel, DSAPattern, Language, Solution, TestCase
from cp_pipeline.exporters.format_converter import CPFormatConverter


class TestCPFormatConverter:
    def setup_method(self):
        self.converter = CPFormatConverter()

    def _make_test_problem(self) -> Problem:
        p = Problem(
            title="Two Sum",
            platform=Platform.LEETCODE,
            problem_statement="Given an array of integers and a target, return indices of two numbers that add up to target.",
            constraints=["2 ≤ nums.length ≤ 10^4", "-10^9 ≤ nums[i] ≤ 10^9"],
            difficulty=DifficultyLevel.EASY,
            dsa_patterns=[DSAPattern.ARRAY, DSAPattern.HASH_TABLE],
        )
        p.sample_test_cases.append(TestCase(
            input="nums = [2,7,11,15], target = 9",
            expected_output="[0, 1]",
            explanation="nums[0] + nums[1] == 9",
        ))
        p.solutions["python"] = Solution(
            language=Language.PYTHON,
            code="def two_sum(nums, target):\n    seen = {}\n    for i, v in enumerate(nums):\n        if target - v in seen:\n            return [seen[target - v], i]\n        seen[v] = i",
            time_complexity="O(n)",
            space_complexity="O(n)",
            approach="Hash map for O(1) lookups",
        )
        return p

    def test_convert_solve_example(self):
        p = self._make_test_problem()
        ex = self.converter._make_solve_example(p, cot=True)
        assert ex.instruction is not None
        assert ex.input is not None
        assert ex.output is not None
        assert "Two Sum" in ex.input

    def test_convert_explain_example(self):
        p = self._make_test_problem()
        ex = self.converter._make_explain_example(p)
        assert "Hash Map" in ex.output or "hash" in ex.output.lower() or "approach" in ex.output.lower()

    def test_convert_debug_examples(self):
        p = self._make_test_problem()
        examples = self.converter._make_debug_examples(p)
        assert len(examples) > 0
        for ex in examples:
            assert ex.metadata["type"] == "debug"

    def test_convert_edge_case_example(self):
        p = self._make_test_problem()
        ex = self.converter._make_edge_case_example(p)
        assert "Edge Case" in ex.output

    def test_convert_complexity_example(self):
        p = self._make_test_problem()
        ex = self.converter._make_complexity_example(p)
        assert "Time Complexity" in ex.output

    def test_convert_pattern_example(self):
        p = self._make_test_problem()
        ex = self.converter._make_pattern_example(p)
        assert "Pattern" in ex.output

    def test_batch_convert(self):
        p = self._make_test_problem()
        examples = self.converter.convert_to_instructions([p])
        assert len(examples) >= 5
        types = [ex.metadata["type"] for ex in examples]
        assert "solve" in types
        assert "explain" in types
        assert "debug" in types

    def test_export_jsonl(self):
        p = self._make_test_problem()
        examples = self.converter.convert_to_instructions([p])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.jsonl")
            self.converter.export_jsonl(examples, path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == len(examples)
            for line in lines:
                record = json.loads(line)
                assert "instruction" in record
                assert "output" in record

    def test_export_finetuning(self):
        p = self._make_test_problem()
        examples = self.converter.convert_to_instructions([p])
        with tempfile.TemporaryDirectory() as tmp:
            self.converter.export_for_finetuning(examples, tmp, framework="transformers")
            files = os.listdir(tmp)
            assert "train.jsonl" in files
            assert "validation.jsonl" in files
            assert "metadata.json" in files
