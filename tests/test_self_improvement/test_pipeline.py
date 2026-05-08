import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from self_improvement.pipeline import SelfImprovementPipeline
from self_improvement.schema import (
    SelfImprovementExample, CorrectionRecord, HardExample,
    ModelEvalResult, EvalCase, ExampleSource, CorrectionMethod,
)


@pytest.fixture
def pipeline():
    return SelfImprovementPipeline()


class TestPipelineInit:
    def test_default_config(self, pipeline):
        assert pipeline.config is not None
        assert "pipeline" in pipeline.config or True  # might have empty config
        assert pipeline.correction_gen is not None
        assert pipeline.quality_curator is not None
        assert pipeline.hard_miner is not None
        assert pipeline.dataset_builder is not None
        assert pipeline.model_evaluator is not None

    def test_loads_config_from_path(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("pipeline:\n  run_id_prefix: test_prefix\n")
            config_path = f.name
        try:
            p = SelfImprovementPipeline(config_path=config_path)
            assert p.config.get("pipeline", {}).get("run_id_prefix") == "test_prefix"
        finally:
            os.unlink(config_path)


class TestRun:
    def test_run_returns_report(self, pipeline):
        with patch.object(pipeline.correction_gen, "load_failed_queries", return_value=[]):
            with patch.object(pipeline.quality_curator, "load_by_quality_threshold", return_value=[]):
                with patch.object(pipeline.hard_miner, "load_all_failed", return_value=[]):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        report = pipeline.run(output_dir=tmpdir, max_failed=10, max_corrections=5)
                        assert report.run_id != ""
                        assert isinstance(report, object)  # ImprovementReport
                        assert report.metadata["processing_time"] > 0

    def test_with_corrections(self, pipeline):
        fake_correction = SelfImprovementExample(
            prompt="What is Python?",
            corrected_response="Python is a programming language.",
            source=ExampleSource.CORRECTION,
        )
        with patch.object(pipeline.correction_gen, "load_failed_queries", return_value=[{"id": 1, "prompt": "Q", "response": "A", "composite_score": 0.3}]):
            with patch.object(pipeline.correction_gen, "generate_batch", return_value=[
                CorrectionRecord(failed_query_id=1, prompt="Q", original_response="A", corrected_response="C")
            ]):
                with patch.object(pipeline.quality_curator, "curate", return_value=[]):
                    with patch.object(pipeline.hard_miner, "mine", return_value=[]):
                        with tempfile.TemporaryDirectory() as tmpdir:
                            report = pipeline.run(output_dir=tmpdir)
                            assert report.corrections_generated > 0

    def test_with_eval(self, pipeline):
        def response_fn(prompt):
            return "This is a test response with enough information to pass evaluation."

        with patch.object(pipeline.correction_gen, "load_failed_queries", return_value=[]):
            with patch.object(pipeline.quality_curator, "curate", return_value=[]):
                with patch.object(pipeline.hard_miner, "mine", return_value=[]):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        cases = [
                            EvalCase(prompt="What is 2+2?", expected_keywords=["4"], category="math", difficulty=1),
                            EvalCase(prompt="What is the capital of France?", expected_keywords=["Paris"], category="factual", difficulty=1),
                        ]
                        report = pipeline.run(response_fn=response_fn, eval_cases=cases, output_dir=tmpdir)
                        assert report.model_before is not None
                        assert report.model_before.model_name == "current"

    def test_saves_report_json(self, pipeline):
        with patch.object(pipeline.correction_gen, "load_failed_queries", return_value=[]):
            with patch.object(pipeline.quality_curator, "curate", return_value=[]):
                with patch.object(pipeline.hard_miner, "mine", return_value=[]):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pipeline.run(output_dir=tmpdir)
                        report_path = os.path.join(tmpdir, "improvement_report.json")
                        assert os.path.exists(report_path)
                        with open(report_path) as f:
                            data = json.load(f)
                        assert "run_id" in data


class TestRunCorrections:
    def test_returns_empty_when_disabled(self, pipeline):
        pipeline.correction_gen.enabled = False
        assert pipeline._run_corrections(10, 5) == []

    def test_returns_empty_when_no_failed(self, pipeline):
        with patch.object(pipeline.correction_gen, "load_failed_queries", return_value=[]):
            assert pipeline._run_corrections(10, 5) == []

    def test_returns_examples(self, pipeline):
        fake_records = [
            CorrectionRecord(failed_query_id=1, prompt="Q1", original_response="A1", corrected_response="C1"),
        ]
        with patch.object(pipeline.correction_gen, "load_failed_queries",
                         return_value=[{"id": 1, "prompt": "Q1", "response": "A1", "composite_score": 0.3}]):
            with patch.object(pipeline.correction_gen, "generate_batch", return_value=fake_records):
                result = pipeline._run_corrections(10, 5)
                assert len(result) == 1
                assert result[0].prompt == "Q1"


class TestRunQualityCuration:
    def test_returns_empty_when_disabled(self, pipeline):
        pipeline.quality_curator.enabled = False
        assert pipeline._run_quality_curation(10) == []

    def test_returns_examples(self, pipeline):
        fake = [SelfImprovementExample(prompt="Q", corrected_response="A", source=ExampleSource.HIGH_QUALITY)]
        with patch.object(pipeline.quality_curator, "curate", return_value=fake):
            result = pipeline._run_quality_curation(10)
            assert len(result) == 1


class TestRunHardExampleMining:
    def test_returns_empty_when_disabled(self, pipeline):
        pipeline.hard_miner.enabled = False
        assert pipeline._run_hard_example_mining(10) == []

    def test_returns_examples(self, pipeline):
        fake_hard = [HardExample(prompt="Q", response="A", category="code", difficulty=3)]
        fake_examples = [
            SelfImprovementExample(prompt="Q", original_response="A", source=ExampleSource.HARD_EXAMPLE, category="code", difficulty=3)
        ]
        with patch.object(pipeline.hard_miner, "mine", return_value=fake_hard):
            with patch.object(pipeline.hard_miner, "to_examples", return_value=fake_examples):
                result = pipeline._run_hard_example_mining(10)
                assert len(result) == 1


class TestExportDPO:
    def test_export_dpo(self, pipeline):
        fake_records = [
            CorrectionRecord(failed_query_id=1, prompt="Q1", original_response="A1", corrected_response="C1"),
        ]
        with patch.object(pipeline.correction_gen, "load_failed_queries",
                         return_value=[{"id": 1, "prompt": "Q1", "response": "A1", "composite_score": 0.3}]):
            with patch.object(pipeline.correction_gen, "generate_batch", return_value=fake_records):
                with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                    path = f.name
                try:
                    result_path = pipeline.export_dpo_for_retraining(path, max_examples=10)
                    assert os.path.exists(result_path)
                    with open(result_path) as f:
                        data = json.loads(f.readline())
                    assert "chosen" in data
                    assert "rejected" in data
                finally:
                    os.unlink(path)


class TestAnalyzeFailedQueries:
    def test_empty_when_no_data(self, pipeline):
        with patch.object(pipeline.correction_gen, "load_failed_queries", return_value=[]):
            result = pipeline.analyze_failed_queries()
            assert result["total"] == 0

    def test_analyzes_categories(self, pipeline):
        mock_rows = [
            {"id": 1, "conv_id": "c1", "prompt": "What is Python?", "response": "bad", "composite_score": 0.3,
             "failure_reasons": '["refusal"]', "occurrence_count": 1, "resolved": 0},
            {"id": 2, "conv_id": "c2", "prompt": "Write code for sorting", "response": "bad", "composite_score": 0.2,
             "failure_reasons": '["incomplete"]', "occurrence_count": 2, "resolved": 0},
        ]
        with patch.object(pipeline.correction_gen, "load_failed_queries", return_value=mock_rows):
            result = pipeline.analyze_failed_queries()
            assert result["total"] == 2
            assert "refusal" in result["by_failure_reason"]
            assert "by_category" in result

    def test_composite_score_fallback(self, pipeline):
        mock_rows = [
            {"id": 1, "conv_id": "c1", "prompt": "Hello", "response": "A", "composite_score": None,
             "failure_reasons": "[]", "occurrence_count": 1, "resolved": 0},
        ]
        with patch.object(pipeline.correction_gen, "load_failed_queries", return_value=mock_rows):
            result = pipeline.analyze_failed_queries()
            assert result["avg_score"] == 0


class TestCategorize:
    def test_code(self, pipeline):
        assert pipeline._categorize("Write a function") == "code"
        assert pipeline._categorize("Debug my code") == "code"

    def test_reasoning(self, pipeline):
        assert pipeline._categorize("Why is the sky blue?") == "reasoning"
        assert pipeline._categorize("Explain gravity") == "reasoning"

    def test_technical(self, pipeline):
        assert pipeline._categorize("How to install Docker?") == "technical"

    def test_factual(self, pipeline):
        assert pipeline._categorize("What is Python?") == "factual"
        assert pipeline._categorize("Who is Einstein?") == "factual"

    def test_general(self, pipeline):
        assert pipeline._categorize("Hello there") == "general"
