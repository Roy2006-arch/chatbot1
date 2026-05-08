import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from debug_pipeline.bugs.injector import BugInjector
from debug_pipeline.schema import Language, BugCategory


class TestBugInjector:
    def setup_method(self):
        self.injector = BugInjector()

    def _correct_python(self):
        return """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

    def test_inject_off_by_one(self):
        examples = self.injector.inject(self._correct_python(), Language.PYTHON, category=BugCategory.OFF_BY_ONE)
        assert len(examples) <= 1
        if examples:
            assert examples[0].category == BugCategory.OFF_BY_ONE
            assert examples[0].buggy_code.code != self._correct_python()
            assert examples[0].corrected_code.code == self._correct_python().strip()
            assert examples[0].explanation is not None

    def test_inject_syntax(self):
        examples = self.injector.inject(self._correct_python(), Language.PYTHON, category=BugCategory.SYNTAX)
        if examples:
            assert examples[0].category == BugCategory.SYNTAX

    def test_inject_multiple_bugs(self):
        examples = self.injector.inject(self._correct_python(), Language.PYTHON, count=3)
        assert len(examples) <= 3

    def test_random_categories(self):
        examples = self.injector.inject(self._correct_python(), Language.PYTHON, count=5)
        assert len(examples) <= 5
        categories = set(ex.category for ex in examples)
        assert len(categories) >= 1

    def test_javascript_code(self):
        code = """
function add(a, b) {
    return a + b;
}
"""
        examples = self.injector.inject(code, Language.JAVASCRIPT, count=2)
        assert len(examples) <= 2

    def test_all_languages(self):
        sources = {
            Language.PYTHON: "def f(): return 1",
            Language.JAVASCRIPT: "function f() { return 1; }",
            Language.JAVA: "public class A { public int f() { return 1; } }",
            Language.CPP: "int f() { return 1; }",
        }
        for lang, code in sources.items():
            examples = self.injector.inject(code, lang, count=1)
            assert len(examples) <= 1

    def test_inject_batch(self):
        pairs = [
            (self._correct_python(), Language.PYTHON),
            ("function f() { return 1; }", Language.JAVASCRIPT),
        ]
        examples = self.injector.inject_batch(pairs, examples_per_pair=2)
        assert len(examples) <= 4
