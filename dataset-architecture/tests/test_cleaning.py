import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.cleaning import DataCleaner, PIIRemover
from src.pipeline.ingestion import DatasetExample


class TestDataCleaner:
    def setup_method(self):
        self.cleaner = DataCleaner()

    def test_empty_output_removed(self):
        ex = DatasetExample(instruction="Test", output="")
        assert self.cleaner._is_empty(ex) is True

    def test_non_empty_output_kept(self):
        ex = DatasetExample(instruction="Test", output="Hello")
        assert self.cleaner._is_empty(ex) is False

    def test_too_short_instruction(self):
        ex = DatasetExample(instruction="Hi", output="Hello world")
        assert self.cleaner._is_too_short(ex) is True

    def test_refusal_detected(self):
        ex = DatasetExample(instruction="Do X", output="I'm sorry, but I cannot assist with that request.")
        assert self.cleaner._is_refusal(ex) is True

    def test_refusal_not_detected(self):
        ex = DatasetExample(instruction="Do X", output="Here is how you do X step by step.")
        assert self.cleaner._is_refusal(ex) is False

    def test_template_hallucination_detected(self):
        ex = DatasetExample(instruction="Hi", output="{{prompt}} is the answer.")
        assert self.cleaner._is_template_hallucination(ex) is True

    def test_sanitize_script_tags(self):
        text = "Hello <script>alert('xss')</script> world"
        cleaned = self.cleaner.sanitize_text(text)
        assert "<script>" not in cleaned

    def test_filter_single_keeps_good(self):
        ex = DatasetExample(instruction="Write a function", output="def foo(): pass")
        result = self.cleaner._filter_single(ex)
        assert result is not None

    def test_filter_single_removes_empty(self):
        ex = DatasetExample(instruction="Write", output="")
        result = self.cleaner._filter_single(ex)
        assert result is None


class TestPIIRemover:
    def test_remove_email(self):
        text = "Contact me at user@example.com"
        cleaned = PIIRemover.remove_pii(text)
        assert "user@example.com" not in cleaned
        assert "[REDACTED]" in cleaned

    def test_remove_phone(self):
        text = "Call me at 555-123-4567"
        cleaned = PIIRemover.remove_pii(text)
        assert "555-123-4567" not in cleaned

    def test_no_pii(self):
        text = "This is a normal message without any sensitive data."
        cleaned = PIIRemover.remove_pii(text)
        assert cleaned == text

    def test_remove_multiple_pii(self):
        text = "Email: a@b.com, Phone: 123-456-7890"
        cleaned = PIIRemover.remove_pii(text)
        assert "[REDACTED]" in cleaned
