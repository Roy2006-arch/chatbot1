import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reasoning_pipeline.verifiers.reasoning_verifier import ReasoningVerifier, LogicalFallacyDetector
from reasoning_pipeline.schema import ReasoningExample, ReasoningStep, ReasoningType, ReasoningTask


class TestReasoningVerifier:
    def setup_method(self):
        self.verifier = ReasoningVerifier()

    def test_valid_example(self):
        ex = ReasoningExample(
            question="Solve for x: 2x + 3 = 7",
            final_answer="x = 2",
            reasoning_type=ReasoningType.DEDUCTIVE,
            reasoning_task=ReasoningTask.MULTI_STEP,
            domain="mathematics",
            verification="Steps are logically sound.",
        )
        ex.reasoning_steps = [
            ReasoningStep(1, "Subtract 3 from both sides", ReasoningType.DEDUCTIVE, "Algebraic manipulation"),
            ReasoningStep(2, "Divide both sides by 2", ReasoningType.DEDUCTIVE, "Algebraic manipulation"),
        ]
        valid, issues = self.verifier.verify(ex)
        assert valid is True
        assert len(issues) == 0

    def test_missing_steps(self):
        ex = ReasoningExample(
            question="Test?",
            final_answer="Answer",
            reasoning_type=ReasoningType.DEDUCTIVE,
            reasoning_task=ReasoningTask.MULTI_STEP,
            domain="logic",
        )
        valid, issues = self.verifier.verify(ex)
        assert valid is False

    def test_missing_justification(self):
        ex = ReasoningExample(
            question="Test?",
            final_answer="Answer",
            reasoning_type=ReasoningType.DEDUCTIVE,
            reasoning_task=ReasoningTask.MULTI_STEP,
            domain="logic",
            verification="OK",
        )
        ex.reasoning_steps = [
            ReasoningStep(1, "Step 1", ReasoningType.DEDUCTIVE, ""),
            ReasoningStep(2, "Step 2", ReasoningType.DEDUCTIVE, ""),
        ]
        valid, issues = self.verifier.verify(ex)
        assert len(issues) > 0

    def test_quality_analysis(self):
        ex = ReasoningExample(
            question="Test?",
            final_answer="Answer",
            reasoning_type=ReasoningType.DEDUCTIVE,
            reasoning_task=ReasoningTask.MULTI_STEP,
            domain="logic",
            verification="Verified step by step.",
        )
        ex.reasoning_steps = [
            ReasoningStep(1, "Step 1", ReasoningType.DEDUCTIVE, "Because of X"),
            ReasoningStep(2, "Step 2", ReasoningType.DEDUCTIVE, "Because of Y"),
        ]
        scores = self.verifier.analyze_reasoning_quality(ex)
        assert "composite" in scores
        assert 0 <= scores["composite"] <= 1

    def test_verify_batch(self):
        valid = ReasoningExample(
            question="How to solve this equation?", final_answer="x = 2",
            reasoning_type=ReasoningType.DEDUCTIVE,
            reasoning_task=ReasoningTask.MULTI_STEP, domain="mathematics",
            verification="Each step follows from algebraic principles.",
            reasoning_steps=[ReasoningStep(1, "Subtract 3 from both sides", ReasoningType.DEDUCTIVE, "Algebraic manipulation")],
        )
        invalid = ReasoningExample(
            question="Test?", final_answer="A",
            reasoning_type=ReasoningType.DEDUCTIVE,
            reasoning_task=ReasoningTask.MULTI_STEP, domain="logic",
        )
        results = self.verifier.verify_batch([valid, invalid])
        assert len(results) == 1


class TestLogicalFallacyDetector:
    def setup_method(self):
        self.detector = LogicalFallacyDetector()

    def test_detect_ad_hominem(self):
        result = self.detector.detect("You're wrong because you're stupid.")
        assert any(r["fallacy"] == "ad_hominem" for r in result)

    def test_detect_false_dilemma(self):
        result = self.detector.detect("Either you're with us or against us. There are only two options.")
        assert any(r["fallacy"] == "false_dilemma" for r in result)

    def test_no_fallacy_clean_text(self):
        result = self.detector.detect("The evidence supports the conclusion through valid reasoning.")
        assert len(result) == 0

    def test_multiple_fallacies(self):
        result = self.detector.detect("You're wrong because you're stupid. Also, if we allow this small change, then next they'll want bigger changes and eventually everything will be destroyed.")
        assert len(result) >= 2
