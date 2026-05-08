import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reasoning_pipeline.schema import (
    ReasoningExample, ReasoningStep, ContradictionPair,
    ReasoningType, ReasoningTask, Difficulty,
)


class TestSchema:
    def test_reasoning_example_creation(self):
        ex = ReasoningExample(
            question="What is 2+2?",
            final_answer="4",
            reasoning_type=ReasoningType.DEDUCTIVE,
            reasoning_task=ReasoningTask.MULTI_STEP,
            domain="mathematics",
        )
        assert ex.question == "What is 2+2?"
        assert ex.final_answer == "4"
        assert ex.id != ""
        assert ex.difficulty == Difficulty.INTERMEDIATE

    def test_reasoning_step(self):
        step = ReasoningStep(
            index=1,
            content="Apply the addition principle",
            reasoning_type=ReasoningType.DEDUCTIVE,
            justification="Addition is commutative",
        )
        assert step.index == 1
        assert step.is_valid is True
        assert len(step.sub_steps) == 0

    def test_step_with_substeps(self):
        step = ReasoningStep(
            index=1,
            content="Main analysis",
            reasoning_type=ReasoningType.COMPOSITIONAL,
            sub_steps=[
                ReasoningStep(1, "Sub-analysis A", ReasoningType.DEDUCTIVE),
                ReasoningStep(2, "Sub-analysis B", ReasoningType.INDUCTIVE),
            ],
        )
        assert len(step.sub_steps) == 2

    def test_contradiction_pair(self):
        pair = ContradictionPair(
            premise="All A are B.",
            conclusion_a="Therefore, X is B.",
            conclusion_b="Therefore, X is not B.",
            consistent_a=True,
            consistent_b=False,
        )
        assert pair.consistent_a is True
        assert pair.consistent_b is False

    def test_example_to_dict_roundtrip(self):
        ex = ReasoningExample(
            question="Test?",
            final_answer="Answer",
            reasoning_type=ReasoningType.DEDUCTIVE,
            reasoning_task=ReasoningTask.MULTI_STEP,
            domain="logic",
        )
        ex.reasoning_steps.append(ReasoningStep(1, "Step 1", ReasoningType.DEDUCTIVE))
        d = ex.to_dict()
        restored = ReasoningExample.from_dict(d)
        assert restored.question == "Test?"
        assert restored.final_answer == "Answer"

    def test_all_reasoning_types(self):
        for rt in ReasoningType:
            ex = ReasoningExample(
                question="Test",
                final_answer="Answer",
                reasoning_type=rt,
                reasoning_task=ReasoningTask.MULTI_STEP,
                domain="general",
            )
            assert ex.reasoning_type.value == rt.value

    def test_all_tasks(self):
        for task in ReasoningTask:
            assert task.value is not None

    def test_step_alternatives(self):
        step = ReasoningStep(
            index=1,
            content="Step content",
            reasoning_type=ReasoningType.DEDUCTIVE,
            alternatives=["Alternative approach A", "Alternative approach B"],
        )
        assert len(step.alternatives) == 2
