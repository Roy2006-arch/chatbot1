from enum import Enum
import pytest
from self_improvement.schema import (
    SelfImprovementExample, CorrectionRecord, HardExample,
    EvalCase, ModelEvalResult, ImprovementReport,
    CorrectionMethod, ExampleSource,
)


class TestCorrectionMethod:
    def test_values(self):
        assert CorrectionMethod.AUTO_GENERATED.value == "auto_generated"
        assert CorrectionMethod.HUMAN_ANNOTATED.value == "human_annotated"
        assert CorrectionMethod.MODEL_REGEN.value == "model_regen"
        assert CorrectionMethod.TEMPLATE_FIXED.value == "template_fixed"


class TestExampleSource:
    def test_values(self):
        assert ExampleSource.FAILED_QUERY.value == "failed_query"
        assert ExampleSource.HIGH_QUALITY.value == "high_quality"
        assert ExampleSource.HARD_EXAMPLE.value == "hard_example"
        assert ExampleSource.CORRECTION.value == "correction"
        assert ExampleSource.CURATED.value == "curated"


class TestSelfImprovementExample:
    def test_minimal_creation(self):
        ex = SelfImprovementExample(prompt="What is Python?")
        assert ex.prompt == "What is Python?"
        assert ex.original_response == ""
        assert ex.source == ExampleSource.FAILED_QUERY
        assert ex.difficulty == 1
        assert ex.quality_score == 0.0
        assert ex.failure_reasons == []
        assert ex.correction_method == CorrectionMethod.AUTO_GENERATED

    def test_full_creation(self):
        ex = SelfImprovementExample(
            prompt="Write a function",
            original_response="def foo(): pass",
            corrected_response="def foo(x): return x + 1",
            source=ExampleSource.CORRECTION,
            category="code",
            difficulty=2,
            quality_score=0.85,
            failure_reasons=["incomplete"],
            correction_method=CorrectionMethod.TEMPLATE_FIXED,
            metadata={"key": "value"},
            id="test-123",
        )
        assert ex.prompt == "Write a function"
        assert ex.corrected_response == "def foo(x): return x + 1"
        assert ex.source == ExampleSource.CORRECTION
        assert ex.difficulty == 2
        assert ex.id == "test-123"

    def test_to_dict_roundtrip(self):
        ex = SelfImprovementExample(
            prompt="Explain quantum computing",
            corrected_response="Quantum computing uses qubits.",
            source=ExampleSource.CORRECTION,
            category="science",
            difficulty=3,
            quality_score=0.9,
        )
        d = ex.to_dict()
        assert d["prompt"] == "Explain quantum computing"
        assert d["source"] == ExampleSource.CORRECTION
        assert d["difficulty"] == 3

        serialized = {k: (v.value if isinstance(v, Enum) else v) for k, v in d.items()}
        restored = SelfImprovementExample.from_dict(serialized)
        assert restored.prompt == ex.prompt
        assert restored.source == ex.source
        assert restored.difficulty == ex.difficulty

    def test_from_dict_with_string_enums(self):
        d = {
            "prompt": "Hello",
            "source": "high_quality",
            "correction_method": "human_annotated",
            "difficulty": 2,
        }
        ex = SelfImprovementExample.from_dict(d)
        assert ex.source == ExampleSource.HIGH_QUALITY
        assert ex.correction_method == CorrectionMethod.HUMAN_ANNOTATED

    def test_from_dict_ignores_extra_fields(self):
        d = {"prompt": "Hi", "source": "failed_query", "unknown_field": "should_ignore"}
        ex = SelfImprovementExample.from_dict(d)
        assert ex.prompt == "Hi"
        assert not hasattr(ex, "unknown_field")


class TestCorrectionRecord:
    def test_creation(self):
        r = CorrectionRecord(
            failed_query_id=1,
            prompt="What is AI?",
            original_response="I don't know",
            corrected_response="AI is the simulation of human intelligence.",
        )
        assert r.failed_query_id == 1
        assert r.correction_method == CorrectionMethod.AUTO_GENERATED
        assert r.validator_issues == []

    def test_to_dict(self):
        r = CorrectionRecord(
            failed_query_id=42,
            prompt="test",
            original_response="old",
            corrected_response="new",
            score_before=0.3,
            score_after=0.8,
        )
        d = r.to_dict()
        assert d["failed_query_id"] == 42
        assert d["score_before"] == 0.3
        assert d["correction_method"] == CorrectionMethod.AUTO_GENERATED


class TestHardExample:
    def test_creation(self):
        he = HardExample(
            prompt="Hard problem",
            response="tough response",
            category="code",
            difficulty=5,
            failure_reasons=["timeout", "wrong answer"],
            occurrence_count=10,
            cluster_id=3,
        )
        assert he.prompt == "Hard problem"
        assert he.difficulty == 5
        assert he.occurrence_count == 10
        assert he.cluster_id == 3

    def test_to_dict_truncates_embedding(self):
        he = HardExample(
            prompt="test", embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        )
        d = he.to_dict()
        assert len(d["embedding"]) == 10


class TestEvalCase:
    def test_creation(self):
        case = EvalCase(
            prompt="What is 2+2?",
            expected_keywords=["4", "four"],
            category="math",
            difficulty=1,
        )
        assert case.prompt == "What is 2+2?"
        assert "4" in case.expected_keywords

    def test_defaults(self):
        case = EvalCase(prompt="Test")
        assert case.expected_keywords == []
        assert case.category == ""
        assert case.difficulty == 1


class TestModelEvalResult:
    def test_creation(self):
        result = ModelEvalResult(
            model_name="test-model",
            run_id="run-001",
            timestamp="2025-01-01T00:00:00",
            total_cases=10,
            avg_accuracy=0.85,
            avg_relevance=0.80,
            avg_coherence=0.90,
            avg_composite=0.85,
            pass_rate=0.8,
            grade_distribution={"A": 5, "B": 3, "C": 2},
            failure_breakdown={"missing_keyword": 2},
        )
        assert result.model_name == "test-model"
        assert result.pass_rate == 0.8
        assert result.grade_distribution["A"] == 5

    def test_to_dict(self):
        result = ModelEvalResult(model_name="m", run_id="r", timestamp="t")
        d = result.to_dict()
        assert d["model_name"] == "m"
        assert d["total_cases"] == 0


class TestImprovementReport:
    def test_creation(self):
        report = ImprovementReport(run_id="si-test", timestamp="2025-01-01T00:00:00")
        assert report.run_id == "si-test"
        assert report.total_failed_queries == 0
        assert report.corrections_generated == 0
        assert report.score_improvement == 0.0
        assert report.model_before is None
        assert report.model_after is None

    def test_to_dict(self):
        before = ModelEvalResult(model_name="before", run_id="r1", timestamp="t1")
        after = ModelEvalResult(model_name="after", run_id="r2", timestamp="t2")
        report = ImprovementReport(
            run_id="si-001",
            timestamp="2025-01-01T00:00:00",
            total_failed_queries=100,
            corrections_generated=50,
            dataset_examples=200,
            model_before=before,
            model_after=after,
            score_improvement=0.05,
        )
        d = report.to_dict()
        assert d["run_id"] == "si-001"
        assert d["total_failed_queries"] == 100
        assert d["model_before"]["model_name"] == "before"
        assert d["model_after"]["model_name"] == "after"
        assert d["score_improvement"] == 0.05
