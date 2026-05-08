import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.code_validator import CodeSyntaxValidator
from src.utils.markdown_validator import MarkdownValidator
from src.pipeline.validation import CodeValidator


class TestCodeSyntaxValidator:
    def test_valid_python(self):
        valid, err = CodeSyntaxValidator.validate_python("print('hello')")
        assert valid is True
        assert err is None

    def test_invalid_python(self):
        valid, err = CodeSyntaxValidator.validate_python("print 'hello'")
        assert valid is False
        assert err is not None

    def test_valid_function(self):
        code = "def add(a, b):\n    return a + b"
        valid, err = CodeSyntaxValidator.validate_python(code)
        assert valid is True

    def test_class_definition(self):
        code = "class Foo:\n    def __init__(self):\n        pass"
        valid, err = CodeSyntaxValidator.validate_python(code)
        assert valid is True

    def test_detect_language_python(self):
        assert CodeSyntaxValidator.detect_language("def foo(): print('hi')") == "python"

    def test_detect_language_javascript(self):
        assert CodeSyntaxValidator.detect_language("function foo() { return 1; }") == "javascript"

    def test_detect_language_java(self):
        assert CodeSyntaxValidator.detect_language("public class Foo { }") == "java"

    def test_bracket_validation_valid(self):
        valid, err = CodeSyntaxValidator.validate_javascript("function f() { return [1, 2]; }")
        assert valid is True

    def test_bracket_validation_invalid(self):
        valid, err = CodeSyntaxValidator._check_brackets("function f() { return [1, 2; }")
        assert valid is False


class TestMarkdownValidator:
    def test_valid_markdown(self):
        text = "# Heading\n\nSome text\n\n```python\nprint('hello')\n```"
        valid, issues = MarkdownValidator.validate(text)
        assert valid is True
        assert len(issues) == 0

    def test_unmatched_code_fences(self):
        text = "# Heading\n\n```python\nprint('hello')\n"
        valid, issues = MarkdownValidator.validate(text)
        assert len(issues) > 0
        assert any("fence" in i.lower() for i in issues)

    def test_deep_headings(self):
        text = "####### Too deep"
        valid, issues = MarkdownValidator.validate(text)
        assert len(issues) > 0

    def test_large_table(self):
        cols = "|" + " col |" * 25
        valid, issues = MarkdownValidator.validate(cols)
        assert len(issues) > 0

    def test_fix_common_issues(self):
        text = "```python\n\nprint('hello')\n\n```"
        fixed = MarkdownValidator.fix_common_issues(text)
        assert "\n\n" not in fixed.replace("```python\n", "").rsplit("\n", 1)[0]

    def test_deep_list_nesting(self):
        text = "              - deeply nested item (14 spaces = level 7)"
        valid, issues = MarkdownValidator.validate(text)
        assert len(issues) > 0


class TestCodeValidator:
    def test_extract_code_blocks(self):
        validator = CodeValidator()
        text = "Some text\n```python\nprint('hello')\n```\nmore text"
        blocks = validator._extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert blocks[0]["code"] == "print('hello')"

    def test_extract_multiple_blocks(self):
        validator = CodeValidator()
        text = "```py\na=1\n```\n```js\nb=2\n```"
        blocks = validator._extract_code_blocks(text)
        assert len(blocks) == 2
