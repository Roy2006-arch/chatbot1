import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.markdown_validator import AdvancedMarkdownValidator


class TestAdvancedMarkdownValidator:
    def setup_method(self):
        self.validator = AdvancedMarkdownValidator()

    def test_clean_markdown_passes(self):
        text = "# Heading\n\nThis is a paragraph.\n\n- List item 1\n- List item 2"
        result = self.validator.check(text)
        assert result.passed

    def test_unmatched_code_fences_detected(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nMore text\n```\nunclosed"
        result = self.validator.check(text)
        assert len([i for i in result.issues if i.code == "MD_UNMATCHED_FENCES"]) > 0

    def test_excessive_heading_depth_detected(self):
        text = "# Level 1\n## Level 2\n####### Level 7 is too deep"
        result = self.validator.check(text)
        assert len([i for i in result.issues if i.code == "MD_HEADING_DEPTH"]) > 0

    def test_empty_heading_detected(self):
        text = "# \n\nContent after empty heading"
        result = self.validator.check(text)
        assert len([i for i in result.issues if i.code == "MD_EMPTY_HEADING"]) > 0

    def test_heading_skip_detected(self):
        text = "# H1\n### H3 (skipped H2)"
        result = self.validator.check(text)
        assert len([i for i in result.issues if i.code == "MD_HEADING_SKIP"]) > 0

    def test_table_validation(self):
        text = "| Col1 | Col2 |\n| --- | --- |\n| Data | Here |"
        result = self.validator.check(text)
        assert result.passed

    def test_empty_link_text_detected(self):
        text = "Click here: [](https://example.com)"
        result = self.validator.check(text)
        assert len([i for i in result.issues if i.code == "MD_EMPTY_LINK_TEXT"]) > 0

    def test_image_alt_text(self):
        text = "![alt text](image.png) and ![](broken.png)"
        result = self.validator.check(text)
        assert len([i for i in result.issues if i.code == "MD_EMPTY_ALT"]) > 0

    def test_unclosed_html_detected(self):
        text = "<div><span>unclosed content"
        result = self.validator.check(text)
        assert len([i for i in result.issues if i.code == "MD_UNCLOSED_HTML"]) > 0

    def test_list_nesting_detected(self):
        text = "- L1\n  - L2\n    - L3\n      - L4\n        - L5\n          - L6\n            - L7\n              - L8 too deep"
        result = self.validator.check(text)
        assert len([i for i in result.issues if i.code == "MD_LIST_NESTING"]) > 0

    def test_fix_common_issues(self):
        text = "```python\n\nprint('hello')\n\n```"
        fixed = self.validator.fix_common_issues(text)
        assert "\n\n" not in fixed.replace("```", "").strip()

    def test_batch_processing(self):
        texts = [
            "# Valid heading\n\nContent",
            "# H1\n####### too deep",
            "![](noalt.png)",
        ]
        results = self.validator.check_batch(texts, num_workers=2)
        assert len(results) == 3
