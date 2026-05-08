import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.pipeline import FilteringPipeline
from filtering.models import FilterConfig, FilterResult, FilterIssue, Severity, FilterReport
from src.pipeline.ingestion import DatasetExample


class TestFilteringPipeline:
    def setup_method(self):
        self.pipeline = FilteringPipeline()

    def test_pipeline_initialization(self):
        assert self.pipeline.hallucination_detector is not None
        assert self.pipeline.repetition_detector is not None
        assert self.pipeline.reasoning_validator is not None
        assert self.pipeline.toxicity_filter is not None
        assert self.pipeline.code_validator is not None
        assert self.pipeline.markdown_validator is not None
        assert self.pipeline.quality_scorer is not None
        assert self.pipeline.ranker is not None

    def test_analyze_dataset(self):
        examples = [
            DatasetExample(instruction="What is 2+2?", output="4", quality_score=0.8, category="math"),
            DatasetExample(instruction="Say hello", output="Hello!", quality_score=0.5, category="general"),
            DatasetExample(instruction="Explain gravity", output="It is a force.", quality_score=0.3, category="science"),
        ]
        report = self.pipeline.analyze_dataset(examples)
        assert report["total"] == 3
        assert report["avg_quality"] > 0
        assert len(report["category_counts"]) == 3

    def test_filter_toxicity_removes_bad(self):
        clean = DatasetExample(instruction="Hello", output="Nice to meet you!")
        toxic = DatasetExample(instruction="Hi", output="White supremacy and genocide are the only way. Kill yourself. Nazis are superior.")
        kept = self.pipeline._filter_toxicity([clean, toxic])
        assert len(kept) == 1
        assert kept[0].output == "Nice to meet you!"

    def test_filter_hallucination(self):
        clean = DatasetExample(instruction="What is 2+2?", output="4")
        hedgy = DatasetExample(instruction="What is 2+2?", output="I think maybe it could possibly be 4 perhaps.")
        kept = self.pipeline._filter_hallucination([clean, hedgy])
        assert len(kept) >= 1

    def test_filter_code_valid(self):
        no_code = DatasetExample(instruction="Hello", output="Just text, no code")
        good_code = DatasetExample(instruction="Write function", output="```python\ndef f():\n    return 1\n```")
        bad_code = DatasetExample(instruction="Write function", output="```python\ndef broken(:\n    return\n```")
        kept = self.pipeline._filter_code([no_code, good_code, bad_code])
        assert len(kept) >= 2

    def test_filter_markdown(self):
        clean = DatasetExample(instruction="Format", output="# Heading\n\nNice paragraph.")
        broken = DatasetExample(instruction="Format", output="# H1\n####### Too deep")
        kept = self.pipeline._filter_markdown([clean, broken])
        assert len(kept) >= 1

    def test_filter_quality(self):
        good = DatasetExample(instruction="What is 2+2?", output="4 is the answer to 2+2.")
        bad = DatasetExample(instruction="Explain quantum physics in detail", output="nope dunno sorry")
        kept = self.pipeline._filter_quality([good, bad])
        assert len(kept) >= 1

    def test_filter_repetition(self):
        clean = DatasetExample(instruction="Tell me something", output="This is a diverse and unique response with varied vocabulary.")
        repetitive = DatasetExample(instruction="Repeat", output="word word word word word word word word word word word word")
        kept = self.pipeline._filter_repetition([clean, repetitive])
        assert len(kept) >= 1

    def test_filter_reasoning(self):
        good = DatasetExample(instruction="Solve", output="First, do this. Second, do that. Therefore, the answer is 42.")
        bad = DatasetExample(instruction="Solve", output="")
        kept = self.pipeline._filter_reasoning([good, bad])
        assert len(kept) >= 1

    def test_full_pipeline_run(self):
        examples = [
            DatasetExample(instruction="What is 2+2?", output="4 is the answer to 2 + 2.", category="math", quality_score=0.5),
            DatasetExample(instruction="Say hello", output="Hello there!", category="general", quality_score=0.5),
            DatasetExample(instruction="Bad stuff", output="White power genocide.", category="toxic", quality_score=0.1),
            DatasetExample(instruction="Empty", output="", category="empty", quality_score=0.0),
        ]
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filtered = self.pipeline.run(examples, output_dir=tmpdir)
            assert len(filtered) < len(examples)

    def test_filter_config_from_dict(self):
        config = FilterConfig.from_dict({
            "min_quality_score": 0.7,
            "max_toxicity_score": 0.1,
            "num_workers": 4,
        })
        assert config.min_quality_score == 0.7
        assert config.max_toxicity_score == 0.1
        assert config.num_workers == 4

    def test_filter_result_merge(self):
        r1 = FilterResult(passed=True, score=0.8, dimension_scores={"a": 0.8})
        r2 = FilterResult(passed=False, score=0.4, dimension_scores={"b": 0.4})
        merged = FilterResult.merge([r1, r2], weights={"a": 0.5, "b": 0.5})
        assert abs(merged.score - 0.6) < 0.001

    def test_pipeline_report(self):
        report = FilterReport(
            total_examples=100,
            passed=80,
            rejected=20,
            rejection_breakdown={"toxicity": 10, "quality": 10},
        )
        d = report.to_dict()
        assert d["pass_rate"] == 0.8
        assert d["total_examples"] == 100
