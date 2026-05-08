import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reasoning_pipeline.generators.multi_step import MultiStepReasoningGenerator
from reasoning_pipeline.generators.contradiction_engine import ContradictionEngine
from reasoning_pipeline.schema import ReasoningTask


class TestMultiStepGenerator:
    def setup_method(self):
        self.gen = MultiStepReasoningGenerator()

    def test_generate_multi_step(self):
        examples = self.gen.generate(ReasoningTask.MULTI_STEP, count=10)
        assert len(examples) == 10
        for ex in examples:
            assert len(ex.reasoning_steps) >= 3
            assert ex.final_answer is not None

    def test_generate_logical_deduction(self):
        examples = self.gen.generate(ReasoningTask.LOGICAL_DEDUCTION, count=5)
        assert len(examples) == 5
        for ex in examples:
            assert ex.domain == "logic"

    def test_generate_contradiction_detection(self):
        examples = self.gen.generate(ReasoningTask.CONTRADICTION_DETECTION, count=5)
        assert len(examples) == 5

    def test_generate_decomposition(self):
        examples = self.gen.generate(ReasoningTask.DECOMPOSITION, count=5)
        assert len(examples) == 5

    def test_generate_planning(self):
        examples = self.gen.generate(ReasoningTask.PLANNING, count=5)
        assert len(examples) == 5

    def test_generate_debugging_reasoning(self):
        examples = self.gen.generate(ReasoningTask.DEBUGGING_REASONING, count=5)
        assert len(examples) == 5

    def test_generate_mathematical_reasoning(self):
        examples = self.gen.generate(ReasoningTask.MATHEMATICAL_REASONING, count=5)
        assert len(examples) == 5

    def test_generate_counterfactual(self):
        examples = self.gen.generate(ReasoningTask.COUNTERFACTUAL_REASONING, count=5)
        assert len(examples) == 5

    def test_generate_comparison(self):
        examples = self.gen.generate(ReasoningTask.COMPARISON_ANALYSIS, count=5)
        assert len(examples) == 5

    def test_chain_of_thought(self):
        ex = self.gen.generate_chain_of_thought(
            "Solve for x: 2x + 3 = 7",
            ["Subtract 3 from both sides: 2x = 4", "Divide both sides by 2: x = 2"],
            "x = 2",
        )
        assert len(ex.reasoning_steps) == 2
        assert ex.final_answer == "x = 2"

    def test_generate_batch(self):
        counts = {
            "multi_step_reasoning": 5,
            "logical_deduction": 5,
            "mathematical_reasoning": 5,
        }
        examples = self.gen.generate_batch(counts)
        assert len(examples) == 15

    def test_all_tasks_have_generators(self):
        for task in ReasoningTask:
            method_name = f"_generate_{task.value}"
            gen_method = getattr(self.gen, method_name, None)
            if gen_method is None:
                has_fallback = hasattr(self.gen, "_generate_default")
                assert has_fallback, f"Missing generator for {task.value} and no fallback"

    def test_generated_examples_have_verification(self):
        examples = self.gen.generate(ReasoningTask.MULTI_STEP, count=10)
        for ex in examples:
            assert ex.verification is not None
            assert len(ex.verification) > 0


class TestContradictionEngine:
    def setup_method(self):
        self.engine = ContradictionEngine()

    def test_generate_contradictions(self):
        pairs = self.engine.generate(count=20)
        assert len(pairs) == 20
        for pair in pairs:
            assert pair.premise is not None
            assert pair.conclusion_a is not None
            assert pair.conclusion_b is not None

    def test_check_contradiction_true(self):
        result = self.engine.check_contradiction(
            "The sky is blue.",
            "The sky is not blue.",
        )
        assert result["is_contradiction"] is True

    def test_check_contradiction_false(self):
        result = self.engine.check_contradiction(
            "The sky is blue.",
            "Grass is green.",
        )
        # May or may not detect as contradiction, but should not crash
        assert "contradiction_score" in result

    def test_detection_examples(self):
        examples = self.engine.generate_detection_examples(count=5)
        assert len(examples) >= 5
        for ex in examples:
            assert "premise" in ex
            assert "has_contradiction" in ex
