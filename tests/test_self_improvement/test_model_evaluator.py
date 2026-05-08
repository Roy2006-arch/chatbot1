import pytest
from self_improvement.model_evaluator import ModelEvaluator, DEFAULT_EVAL_CASES
from self_improvement.schema import EvalCase, ModelEvalResult


@pytest.fixture
def evaluator():
    return ModelEvaluator(config={
        "enabled": True,
        "score_threshold": 0.55,
        "min_eval_cases": 10,
    })


@pytest.fixture
def good_response_fn():
    def fn(prompt):
        return {
            "What is 2+2?": "The answer is 4.",
            "What is the capital of France?": "The capital of France is Paris.",
            "Explain what a variable is in programming.": "A variable is a storage location in programming that holds a value which can change during execution.",
            "Write a function that adds two numbers.": "def add(a, b): return a + b",
            "What is the difference between TCP and UDP?": "TCP is connection-oriented and reliable, while UDP is connectionless and faster.",
            "Explain how garbage collection works.": "Garbage collection automatically manages memory by reclaiming unused objects.",
            "What is the time complexity of binary search?": "The time complexity of binary search is O(log n).",
            "Describe the water cycle.": "The water cycle involves evaporation, condensation, and precipitation.",
            "What is an API?": "An API is an application programming interface.",
            "Explain recursion with an example.": "Recursion is when a function calls itself. A base case stops the recursion.",
        }.get(prompt, "I don't know the answer to that question.")
    return fn


@pytest.fixture
def poor_response_fn():
    def fn(prompt):
        return "I don't know."
    return fn


class TestModelEvaluatorInit:
    def test_default_config(self):
        me = ModelEvaluator()
        assert me.enabled is True
        assert me.score_threshold == 0.55

    def test_custom_config(self):
        me = ModelEvaluator(config={"enabled": False, "score_threshold": 0.7})
        assert me.enabled is False
        assert me.score_threshold == 0.7


class TestScoreAccuracy:
    def test_all_keywords_found(self, evaluator):
        score = evaluator._score_accuracy("Python is a programming language.", ["python", "programming"])
        assert score == 1.0

    def test_some_keywords_found(self, evaluator):
        score = evaluator._score_accuracy("Python is great.", ["python", "java", "c++"])
        assert score == pytest.approx(1 / 3)

    def test_none_found(self, evaluator):
        score = evaluator._score_accuracy("Hello world.", ["python", "java"])
        assert score == 0.0

    def test_case_insensitive(self, evaluator):
        score = evaluator._score_accuracy("PYTHON IS GREAT", ["python"])
        assert score == 1.0

    def test_no_keywords(self, evaluator):
        score = evaluator._score_accuracy("Hello", [])
        assert score is None


class TestScoreRelevance:
    def test_fallback_with_overlap(self, evaluator):
        score = evaluator._score_relevance("Python programming", "Python is a programming language")
        assert isinstance(score, float)

    def test_no_overlap(self, evaluator):
        score = evaluator._score_relevance("abcdef", "ghijkl")
        assert score <= 1.0

    def test_empty_prompt(self, evaluator):
        score = evaluator._score_relevance("", "response")
        assert -1.0 <= score <= 1.0


class TestScoreCoherence:
    def test_empty(self, evaluator):
        assert evaluator._score_coherence("") == 0.0

    def test_short(self, evaluator):
        score = evaluator._score_coherence("Hello")
        assert score == 0.3

    def test_long_with_punctuation(self, evaluator):
        score = evaluator._score_coherence("This is a complete sentence with enough words to be coherent. It even has multiple sentences!")
        assert score > 0.5

    def test_very_short_penalty(self, evaluator):
        score = evaluator._score_coherence("a b c")
        assert score < 0.4


class TestEvaluateResponse:
    def test_with_keywords(self, evaluator):
        result = evaluator.evaluate_response("What is 2+2?", "The answer is 4.", expected_keywords=["4"])
        assert result["accuracy"] > 0
        assert isinstance(result["relevance"], float)
        assert result["coherence"] > 0
        assert -1 <= result["composite"] <= 1

    def test_without_keywords(self, evaluator):
        result = evaluator.evaluate_response("Tell me something", "This is a response without keywords.")
        assert result["accuracy"] is None
        assert result["composite"] > 0

    def test_composite_high_for_good_response(self, evaluator):
        result = evaluator.evaluate_response(
            "What is the capital of France?",
            "The capital of France is Paris.",
            expected_keywords=["Paris"],
        )
        assert result["composite"] > 0.5

    def test_composite_low_for_poor_response(self, evaluator):
        result = evaluator.evaluate_response(
            "What is 2+2?",
            "I don't know the answer to that.",
            expected_keywords=["4", "four"],
        )
        assert result["composite"] < 0.5


class TestEvaluateCase:
    def test_returns_scores_with_metadata(self, evaluator):
        case = EvalCase(prompt="What is 2+2?", expected_keywords=["4"], category="math", difficulty=1)
        result = evaluator.evaluate_case(case, "The answer is 4.")
        assert "passed" in result
        assert result["category"] == "math"
        assert result["difficulty"] == 1


class TestEvaluateModel:
    def test_empty_when_disabled(self, evaluator):
        evaluator.enabled = False
        result = evaluator.evaluate_model("test", lambda x: "")
        assert result.total_cases == 0

    def test_uses_default_cases(self, evaluator, good_response_fn):
        result = evaluator.evaluate_model("test-model", good_response_fn)
        assert result.total_cases > 0
        assert result.model_name == "test-model"
        assert result.pass_rate > 0

    def test_with_poor_responses(self, evaluator, poor_response_fn):
        result = evaluator.evaluate_model("poor-model", poor_response_fn)
        assert result.pass_rate < 0.5

    def test_grades_distribution(self, evaluator, good_response_fn):
        result = evaluator.evaluate_model("test", good_response_fn)
        total_grades = sum(result.grade_distribution.values())
        assert total_grades == result.total_cases

    def test_per_category(self, evaluator, good_response_fn):
        result = evaluator.evaluate_model("test", good_response_fn)
        assert len(result.per_category) > 0
        for cat, data in result.per_category.items():
            assert "avg_composite" in data
            assert "pass_rate" in data
            assert "count" in data

    def test_custom_cases(self, evaluator):
        custom_cases = [
            EvalCase(prompt="What is 2+2?", expected_keywords=["4"], category="math", difficulty=1),
            EvalCase(prompt="What is the capital of France?", expected_keywords=["Paris"], category="factual", difficulty=1),
        ]
        result = evaluator.evaluate_model("test", lambda x: "I don't know", cases=custom_cases)
        assert result.total_cases == 2


class TestCompareModels:
    def test_compares_metrics(self, evaluator):
        before = ModelEvalResult(model_name="before", run_id="r1", timestamp="t1",
                                 avg_accuracy=0.5, avg_relevance=0.5, avg_coherence=0.5,
                                 avg_composite=0.5, pass_rate=0.5)
        after = ModelEvalResult(model_name="after", run_id="r2", timestamp="t2",
                                avg_accuracy=0.8, avg_relevance=0.7, avg_coherence=0.6,
                                avg_composite=0.7, pass_rate=0.8)

        comparison = evaluator.compare_models(before, after)
        assert comparison["overall_improved"] is True
        assert comparison["score_improvement"] > 0
        assert comparison["comparison"]["avg_accuracy"]["before"] == 0.5
        assert comparison["comparison"]["avg_accuracy"]["after"] == 0.8

    def test_no_improvement(self, evaluator):
        before = ModelEvalResult(model_name="before", run_id="r1", timestamp="t1",
                                 avg_composite=0.7, pass_rate=0.8)
        after = ModelEvalResult(model_name="after", run_id="r2", timestamp="t2",
                                avg_composite=0.6, pass_rate=0.7)
        comparison = evaluator.compare_models(before, after)
        assert comparison["overall_improved"] is False
        assert comparison["score_improvement"] < 0

    def test_grade_changes(self, evaluator):
        before = ModelEvalResult(model_name="before", run_id="r1", timestamp="t1",
                                 grade_distribution={"A": 2, "B": 3, "F": 1})
        after = ModelEvalResult(model_name="after", run_id="r2", timestamp="t2",
                                grade_distribution={"A": 5, "B": 1, "F": 0})
        comparison = evaluator.compare_models(before, after)
        assert comparison["grade_changes"]["A"]["delta"] == 3
        assert comparison["grade_changes"]["F"]["delta"] == -1


class TestGrade:
    def test_a_grade(self, evaluator):
        assert evaluator._grade(0.90) == "A"
        assert evaluator._grade(0.85) == "A"

    def test_b_grade(self, evaluator):
        assert evaluator._grade(0.75) == "B"
        assert evaluator._grade(0.70) == "B"

    def test_c_grade(self, evaluator):
        assert evaluator._grade(0.60) == "C"
        assert evaluator._grade(0.55) == "C"

    def test_d_grade(self, evaluator):
        assert evaluator._grade(0.45) == "D"
        assert evaluator._grade(0.40) == "D"

    def test_f_grade(self, evaluator):
        assert evaluator._grade(0.30) == "F"
        assert evaluator._grade(0.0) == "F"


class TestStats:
    def test_initial(self, evaluator):
        assert evaluator.get_stats()["evaluations"] == 0
        assert evaluator.get_stats()["cases_run"] == 0

    def test_updates_after_eval(self, evaluator, good_response_fn):
        evaluator.evaluate_model("test", good_response_fn, cases=DEFAULT_EVAL_CASES[:3])
        assert evaluator.get_stats()["cases_run"] == 3
        assert evaluator.get_stats()["evaluations"] == 1
