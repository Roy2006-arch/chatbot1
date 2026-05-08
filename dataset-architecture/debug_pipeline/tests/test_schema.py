import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from debug_pipeline.schema import (
    BuggyExample, SourceCode, ErrorInfo, TestCase,
    Language, BugCategory, ErrorType, Severity,
)


class TestSchema:
    def test_source_code(self):
        sc = SourceCode(language=Language.PYTHON, code="print('hello')")
        assert sc.language == Language.PYTHON
        assert sc.line_count() == 1

    def test_error_info(self):
        err = ErrorInfo(error_type=ErrorType.COMPILE_TIME, message="Syntax error", line_number=5)
        assert err.error_type == ErrorType.COMPILE_TIME
        assert err.line_number == 5

    def test_buggy_example_minimal(self):
        ex = BuggyExample(
            buggy_code=SourceCode(Language.PYTHON, "print('bug')"),
            corrected_code=SourceCode(Language.PYTHON, "print('fix')"),
            language=Language.PYTHON,
            category=BugCategory.SYNTAX,
        )
        assert ex.id != ""
        assert ex.severity == Severity.MEDIUM

    def test_buggy_example_full(self):
        ex = BuggyExample(
            buggy_code=SourceCode(Language.PYTHON, "x = 1 / 0"),
            corrected_code=SourceCode(Language.PYTHON, "x = 0\nif x != 0:\n    y = 1 / x"),
            language=Language.PYTHON,
            category=BugCategory.DIVISION_BY_ZERO,
            error_info=ErrorInfo(ErrorType.RUNTIME, "division by zero", line_number=1),
            severity=Severity.HIGH,
            title="Division by zero",
            description="Missing zero check",
            explanation="The divisor can be zero",
            fix_strategy="Add zero check",
            test_cases=[TestCase(input_data="", expected_output="0")],
            difficulty=2,
            tags=["math", "zero"],
        )
        assert ex.title == "Division by zero"
        assert len(ex.test_cases) == 1
        assert ex.difficulty == 2

    def test_to_dict_roundtrip(self):
        ex = BuggyExample(
            buggy_code=SourceCode(Language.PYTHON, "print('bug')"),
            corrected_code=SourceCode(Language.PYTHON, "print('fix')"),
            language=Language.PYTHON,
            category=BugCategory.SYNTAX,
            error_info=ErrorInfo(ErrorType.COMPILE_TIME, "syntax error"),
        )
        d = ex.to_dict()
        restored = BuggyExample.from_dict(d)
        assert restored.language == Language.PYTHON
        assert restored.category == BugCategory.SYNTAX
        assert restored.error_info.error_type == ErrorType.COMPILE_TIME

    def test_all_categories_have_difficulty(self):
        from debug_pipeline.schema import BUG_CATEGORY_DIFFICULTY
        for cat in BugCategory:
            assert cat in BUG_CATEGORY_DIFFICULTY, f"Missing difficulty for {cat}"

    def test_bug_category_values(self):
        for cat in BugCategory:
            assert cat.value is not None
