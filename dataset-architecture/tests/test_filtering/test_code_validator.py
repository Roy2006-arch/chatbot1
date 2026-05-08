import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.code_validator import AdvancedCodeValidator


class TestAdvancedCodeValidator:
    def setup_method(self):
        self.validator = AdvancedCodeValidator()

    def test_valid_python_passes(self):
        result = self.validator.check("def add(a, b):\n    return a + b", "python")
        assert result.passed
        assert result.dimension_scores.get("syntax", 0) == 1.0

    def test_invalid_python_fails(self):
        result = self.validator.check("def add(a, b):\n    return a +", "python")
        assert not result.passed

    def test_valid_javascript_passes(self):
        result = self.validator.check("function add(a, b) { return a + b; }", "javascript")
        assert result.passed

    def test_unmatched_brackets_detected(self):
        result = self.validator.check("function add(a, b) { return a + b;", "javascript")
        assert not result.passed

    def test_code_smells_detected(self):
        code = (
            "def process():\n"
            "    password = 'secret123'\n"
            "    # TODO: implement proper validation\n"
            "    pass\n"
        )
        result = self.validator.check(code, "python")
        smells = [i for i in result.issues if i.code.startswith("CODE_SMELL_")]
        assert len(smells) > 0

    def test_empty_code_fails(self):
        result = self.validator.check("", "python")
        assert not result.passed

    def test_code_too_long(self):
        code = "x = 1\n" * 1000
        self.validator.max_code_length = 100
        result = self.validator.check(code, "python")
        assert len([i for i in result.issues if i.code == "CODE_TOO_LONG"]) > 0

    def test_language_detection(self):
        assert self.validator.detect_language("def hello():\n    print('hi')") == "python"
        assert self.validator.detect_language("function hello() { return 1; }") == "javascript"

    def test_validate_code_blocks(self):
        text = "Here is code:\n```python\nprint('hello')\n```"
        results = self.validator.validate_code_blocks(text)
        assert len(results) == 1
        assert results[0].passed

    def test_batch_processing(self):
        pairs = [
            ("def f(): return 1", "python"),
            ("function f() { return 1; }", "javascript"),
            ("def broken(", "python"),
        ]
        results = self.validator.check_batch(pairs, num_workers=2)
        assert len(results) == 3
