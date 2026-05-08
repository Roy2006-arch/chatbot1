import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.reasoning_validator import ReasoningValidator


class TestReasoningValidator:
    def setup_method(self):
        self.validator = ReasoningValidator()

    def test_clean_reasoning_passes(self):
        text = (
            "First, we need to identify the key variables. "
            "Second, we calculate the sum of all inputs. "
            "Third, we divide by the count to get the average. "
            "In conclusion, the result is 42."
        )
        result = self.validator.check(text)
        assert result.passed

    def test_truncated_text_detected(self):
        result = self.validator.check("This is an incomplete response that ends abruptly")
        assert len([i for i in result.issues if i.code == "REASONING_ABRUPT_END"]) > 0

    def test_unclosed_code_fence_detected(self):
        text = "Here is some code:\n```python\nprint('hello')"
        result = self.validator.check(text)
        assert len([i for i in result.issues if i.code == "REASONING_UNCLOSED_TAG"]) > 0

    def test_missing_conclusion_detected(self):
        text = "First, we do this. Second, we do that. Third, we do something else."
        result = self.validator.check(text)
        if self.validator.conclusion_required:
            assert len([i for i in result.issues if i.code == "REASONING_NO_CONCLUSION"]) > 0

    def test_chain_of_thought_detected(self):
        text = (
            "<thought>Let me think about this step by step.</thought>\n"
            "<thought>First, we need to parse the input.</thought>\n"
            "<thought>Then, we process each element.</thought>\n"
            "Therefore, the answer is correct."
        )
        result = self.validator.check(text)
        assert result.dimension_scores.get("has_cot", 0) >= 0.5

    def test_empty_text(self):
        result = self.validator.check("")
        assert not result.passed

    def test_sufficient_reasoning_steps(self):
        text = "First, step one. Second, step two. Third, step three. Therefore, done."
        result = self.validator.check(text)
        assert result.dimension_scores.get("step_count", 0) >= 0.5

    def test_batch_processing(self):
        texts = [
            "First, step one. Finally, conclusion.",
            "Abrupt end here",
            "",
        ]
        results = self.validator.check_batch(texts, num_workers=2)
        assert len(results) == 3
