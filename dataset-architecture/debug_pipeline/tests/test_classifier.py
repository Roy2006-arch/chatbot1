import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from debug_pipeline.classifiers.bug_classifier import BugClassifier
from debug_pipeline.schema import BuggyExample, SourceCode, Language, BugCategory


class TestBugClassifier:
    def setup_method(self):
        self.classifier = BugClassifier()

    def test_classify_off_by_one(self):
        ex = BuggyExample(
            buggy_code=SourceCode(Language.PYTHON, "for i in range(n+1):"),
            corrected_code=SourceCode(Language.PYTHON, "for i in range(n):"),
            language=Language.PYTHON,
            category=BugCategory.LOGIC_ERROR,
            title="Off by one error in loop",
            description="The loop boundary is off by one",
            explanation="range(n+1) iterates n+1 times instead of n",
            fix_strategy="Change to range(n)",
        )
        classified = self.classifier.classify(ex)
        assert "off_by_one" in classified.tags or classified.category == BugCategory.OFF_BY_ONE

    def test_classify_null_pointer(self):
        ex = BuggyExample(
            buggy_code=SourceCode(Language.PYTHON, ""),
            corrected_code=SourceCode(Language.PYTHON, ""),
            language=Language.PYTHON,
            category=BugCategory.LOGIC_ERROR,
            title="Null pointer dereference",
            description="None type error when accessing attribute",
            explanation="Variable can be None",
            fix_strategy="Add None check",
        )
        classified = self.classifier.classify(ex)
        assert "null_pointer" in classified.tags

    def test_classify_batch(self):
        examples = [
            BuggyExample(
                buggy_code=SourceCode(Language.PYTHON, "x = 1/0"),
                corrected_code=SourceCode(Language.PYTHON, "if x: y=1/x"),
                language=Language.PYTHON,
                category=BugCategory.DIVISION_BY_ZERO,
                title="Div by zero",
                description="Division by zero",
                explanation="Zero divisor",
                fix_strategy="Add check",
            ),
            BuggyExample(
                buggy_code=SourceCode(Language.PYTHON, "for i in range(n+1):"),
                corrected_code=SourceCode(Language.PYTHON, "for i in range(n):"),
                language=Language.PYTHON,
                category=BugCategory.OFF_BY_ONE,
                title="Off by one",
                description="Loop boundary",
                explanation="Off by one",
                fix_strategy="Fix range",
            ),
        ]
        results = self.classifier.classify_batch(examples)
        assert len(results) == 2
