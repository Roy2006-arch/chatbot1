import sys
import os
import tempfile
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.ingestion import DataIngestor, DatasetExample
from src.pipeline.preprocessing import Preprocessor
from src.pipeline.cleaning import DataCleaner
from src.pipeline.validation import CodeValidator, MarkdownValidator
from src.pipeline.export import DataExporter
from src.quality.scoring import QualityScorer
from src.quality.deduplication import Deduplicator
from src.quality.filtering import QualityFilter
from src.quality.ranking import ResponseRanker
from src.curriculum.difficulty import DifficultyScorer
from src.curriculum.scheduler import CurriculumScheduler


class TestDataIngestor:
    def test_normalize_item_instruction(self):
        ingestor = DataIngestor("config/pipeline.yaml")
        ex = ingestor._normalize_item({"instruction": "Do X", "output": "Result"}, "test", "source.json")
        assert ex.instruction == "Do X"
        assert ex.output == "Result"
        assert ex.category == "test"

    def test_normalize_item_alternative_fields(self):
        ingestor = DataIngestor("config/pipeline.yaml")
        ex = ingestor._normalize_item({"prompt": "Do X", "response": "Result"}, "test", "source.json")
        assert ex.instruction == "Do X"
        assert ex.output == "Result"


class TestPreprocessor:
    def test_clean_text(self):
        pre = Preprocessor()
        cleaned = pre._clean_text("Hello\r\nWorld\rThird")
        assert "\r\n" not in cleaned
        assert "\r" not in cleaned

    def test_extract_code_blocks(self):
        pre = Preprocessor()
        text = "```python\nprint('hello')\n```"
        blocks = pre.extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"


class TestQualityScorer:
    def test_score_relevance(self):
        scorer = QualityScorer("config/quality.yaml")
        ex = DatasetExample(instruction="Write a Python function", output="def foo(): pass")
        score = scorer._score_relevance(ex)
        assert 0.0 <= score <= 1.0

    def test_score_completeness(self):
        scorer = QualityScorer("config/quality.yaml")
        ex = DatasetExample(instruction="Short", output="A" * 500)
        score = scorer._score_completeness(ex)
        assert score > 0.5


class TestDeduplicator:
    def test_exact_dedup(self):
        dedup = Deduplicator("config/quality.yaml")
        ex1 = DatasetExample(instruction="Hello", output="World")
        ex2 = DatasetExample(instruction="Hello", output="World")
        ex3 = DatasetExample(instruction="Different", output="Data")
        result, removed = dedup._exact_dedup([ex1, ex2, ex3])
        assert len(result) == 2
        assert removed == 1


class TestQualityFilter:
    def test_filter_by_length(self):
        filt = QualityFilter("config/quality.yaml")
        ex = DatasetExample(instruction="Hi", output="")
        assert filt._passes_filter(ex) is False


class TestResponseRanker:
    def test_rank_by_quality(self):
        ranker = ResponseRanker()
        ex1 = DatasetExample(instruction="A", output="B", quality_score=0.5)
        ex2 = DatasetExample(instruction="C", output="D", quality_score=0.9)
        ranked = ranker.rank([ex1, ex2])
        assert ranked[0].quality_score == 0.9


class TestDifficultyScorer:
    def test_difficulty_computation(self):
        scorer = DifficultyScorer()
        ex = DatasetExample(instruction="Solve this hard DP problem with recursion", output="def solve(): pass")
        diff = scorer._compute_difficulty(ex)
        assert 1 <= diff <= 5


class TestCurriculumScheduler:
    def test_linear_schedule(self):
        examples = [
            DatasetExample(instruction="A", output="B", difficulty=1),
            DatasetExample(instruction="C", output="D", difficulty=5),
            DatasetExample(instruction="E", output="F", difficulty=3),
        ]
        scheduler = CurriculumScheduler(strategy="linear", total_steps=100, batch_size=10)
        plan, hard = scheduler.create_training_plan(examples)
        assert len(plan) + len(hard) == len(examples)


class TestDataExporter:
    def test_export_jsonl(self):
        exporter = DataExporter("config/pipeline.yaml")
        examples = [
            DatasetExample(instruction="Test", output="Data"),
            DatasetExample(instruction="Hello", output="World"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            exporter.export(examples, tmp, format="jsonl", split=False)
            files = os.listdir(tmp)
            assert "full.jsonl" in files
            with open(os.path.join(tmp, "full.jsonl")) as f:
                lines = f.readlines()
                assert len(lines) == 2


class TestEndToEnd:
    def test_small_pipeline(self):
        examples = [
            DatasetExample(instruction="Write a Python function to add two numbers", output="def add(a, b):\n    return a + b", category="algorithms_dsa", difficulty=1),
            DatasetExample(instruction="Fix this bug: print('hello'", output="The bug is a missing closing parenthesis.\n\n```python\nprint('hello')\n```", category="debugging", difficulty=2),
            DatasetExample(instruction="", output="Empty instruction test", category="algorithms_dsa", difficulty=1),
        ]

        cleaner = DataCleaner()
        examples = cleaner.clean(examples)
        assert len(examples) == 2

        scorer = QualityScorer()
        examples = scorer.score(examples)
        assert all(ex.quality_score > 0 for ex in examples)

        filt = QualityFilter()
        examples = filt.filter(examples)
        assert len(examples) >= 1

        dedup = Deduplicator()
        examples = dedup.deduplicate(examples)
        assert len(examples) >= 1

        print(f"End-to-end test passed: {len(examples)} examples remain")
