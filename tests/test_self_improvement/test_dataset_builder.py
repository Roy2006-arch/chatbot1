import json
import os
import tempfile
import pytest
from self_improvement.dataset_builder import DatasetBuilder
from self_improvement.schema import SelfImprovementExample, ExampleSource


@pytest.fixture
def builder():
    return DatasetBuilder(config={"enabled": True, "formats": ["transformers", "openai"]})


@pytest.fixture
def sample_corrections():
    return [
        SelfImprovementExample(
            prompt="What is AI?",
            original_response="I don't know",
            corrected_response="AI is the simulation of human intelligence.",
            source=ExampleSource.CORRECTION,
            quality_score=0.7,
            difficulty=1,
        ),
        SelfImprovementExample(
            prompt="Write a function",
            original_response="def f(): pass",
            corrected_response="def add(a, b): return a + b",
            source=ExampleSource.CORRECTION,
            quality_score=0.8,
            difficulty=2,
        ),
    ]


@pytest.fixture
def sample_quality():
    return [
        SelfImprovementExample(
            prompt="Explain Python",
            corrected_response="Python is a programming language.",
            source=ExampleSource.HIGH_QUALITY,
            quality_score=0.9,
            difficulty=1,
        ),
    ]


@pytest.fixture
def sample_hard():
    return [
        SelfImprovementExample(
            prompt="Hard math problem",
            original_response="Wrong answer",
            source=ExampleSource.HARD_EXAMPLE,
            quality_score=0.3,
            difficulty=5,
        ),
    ]


class TestDatasetBuilderInit:
    def test_default_config(self):
        db = DatasetBuilder()
        assert db.enabled is True
        assert db.train_split == 0.8

    def test_custom_config(self):
        db = DatasetBuilder(config={"enabled": False, "train_split": 0.7})
        assert db.enabled is False
        assert db.train_split == 0.7


class TestBuildDataset:
    def test_empty_when_disabled(self, builder):
        builder.enabled = False
        assert builder.build_dataset([], [], []) == []

    def test_empty_input(self, builder):
        assert builder.build_dataset([], [], []) == []

    def test_builds_combined(self, builder, sample_corrections, sample_quality, sample_hard):
        dataset = builder.build_dataset(sample_corrections, sample_quality, sample_hard, max_total=100)
        assert len(dataset) > 0
        assert len(dataset) <= 4

    def test_deduplicates_by_prompt(self, builder):
        examples = [
            SelfImprovementExample(prompt="Same prompt here", quality_score=0.5, source=ExampleSource.CORRECTION,
                                   corrected_response="A" * 20),
            SelfImprovementExample(prompt="Same prompt here", quality_score=0.9, source=ExampleSource.CORRECTION,
                                   corrected_response="B" * 20),
        ]
        dataset = builder.build_dataset(examples, [], [], max_total=10)
        assert len(dataset) == 1
        assert dataset[0].quality_score == 0.9
        assert dataset[0].quality_score == 0.9

    def test_respects_max_total(self, builder):
        examples = [
            SelfImprovementExample(prompt=f"Prompt {i}", quality_score=0.5, source=ExampleSource.CORRECTION,
                                   corrected_response=f"Response {i}" * 5)
            for i in range(20)
        ]
        dataset = builder.build_dataset(examples, [], [], max_total=5)
        assert len(dataset) == 5

    def test_hard_examples_take_priority(self, builder, sample_corrections, sample_hard):
        dataset = builder.build_dataset(sample_corrections, [], sample_hard, max_total=10)
        assert len(dataset) == 3
        hard = [ex for ex in dataset if ex.source == ExampleSource.HARD_EXAMPLE]
        assert len(hard) > 0

    def test_curriculum_sort(self, builder):
        examples = []
        for i in range(5):
            for diff in range(1, 4):
                examples.append(SelfImprovementExample(
                    prompt=f"P{diff}_{i}", difficulty=diff, quality_score=0.5 + i * 0.1,
                    source=ExampleSource.CORRECTION,
                ))
        builder.curriculum_order = True
        dataset = builder.build_dataset(examples, [], [], max_total=100)
        difficulties = [ex.difficulty for ex in dataset]
        assert difficulties == sorted(difficulties)


class TestSplitDataset:
    def test_split_maintains_order(self, builder):
        examples = [SelfImprovementExample(prompt=f"P{i}", source=ExampleSource.CORRECTION) for i in range(100)]
        train, val, test = builder.split_dataset(examples)
        assert len(train) + len(val) + len(test) == 100

    def test_split_ratios(self, builder):
        examples = [SelfImprovementExample(prompt=f"P{i}", source=ExampleSource.CORRECTION) for i in range(100)]
        train, val, test = builder.split_dataset(examples)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_shuffles(self, builder):
        examples = [SelfImprovementExample(prompt=f"P{i}", source=ExampleSource.CORRECTION) for i in range(100)]
        train, _, _ = builder.split_dataset(examples)
        train_prompts = [ex.prompt for ex in train]
        assert train_prompts != [f"P{i}" for i in range(80)]


class TestExportTransformers:
    def test_creates_jsonl(self, builder, sample_corrections):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            result = builder.export_transformers(sample_corrections, path)
            assert os.path.exists(result)
            with open(result) as f:
                lines = f.readlines()
            assert len(lines) == 2
            data = json.loads(lines[0])
            assert "instruction" in data
            assert "output" in data
            assert "source" in data
        finally:
            os.unlink(path)

    def test_includes_metadata(self, builder):
        ex = SelfImprovementExample(
            prompt="test", corrected_response="answer",
            source=ExampleSource.CORRECTION, category="code", difficulty=2,
            failure_reasons=["error"],
        )
        builder.include_metadata = True
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            result = builder.export_transformers([ex], path)
            with open(result) as f:
                data = json.loads(f.readline())
            assert "metadata" in data
            assert data["metadata"]["failure_reasons"] == ["error"]
        finally:
            os.unlink(path)


class TestExportOpenAI:
    def test_creates_jsonl(self, builder, sample_corrections):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            result = builder.export_openai(sample_corrections, path)
            assert os.path.exists(result)
            with open(result) as f:
                lines = f.readlines()
            assert len(lines) == 2
            data = json.loads(lines[0])
            assert "messages" in data
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][1]["role"] == "assistant"
        finally:
            os.unlink(path)

    def test_uses_corrected_response(self, builder):
        ex = SelfImprovementExample(
            prompt="test", original_response="bad", corrected_response="good",
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            result = builder.export_openai([ex], path)
            with open(result) as f:
                data = json.loads(f.readline())
            assert data["messages"][1]["content"] == "good"
        finally:
            os.unlink(path)


class TestExportDPO:
    def test_creates_jsonl(self, builder):
        ex = SelfImprovementExample(
            prompt="test", original_response="bad", corrected_response="good",
            source=ExampleSource.CORRECTION, quality_score=0.8,
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            result = builder.export_dpo([ex], path)
            with open(result) as f:
                data = json.loads(f.readline())
            assert data["prompt"] == "test"
            assert data["chosen"] == "good"
            assert data["rejected"] == "bad"
            assert data["score"] == 0.8
        finally:
            os.unlink(path)

    def test_skips_without_both_responses(self, builder):
        ex = SelfImprovementExample(prompt="test", original_response="", corrected_response="")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            result = builder.export_dpo([ex], path)
            with open(result) as f:
                lines = f.readlines()
            assert len(lines) == 0
        finally:
            os.unlink(path)


class TestExportAllFormats:
    def test_exports_requested_formats(self, builder):
        examples = [SelfImprovementExample(prompt=f"P{i}", source=ExampleSource.CORRECTION) for i in range(10)]
        train, val, test = builder.split_dataset(examples)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = builder.export_all_formats(train, val, test, tmpdir, formats=["transformers"])
            assert any("train" in k for k in paths)
            assert any("val" in k for k in paths)
            assert any("test" in k for k in paths)


class TestSaveMetadata:
    def test_creates_json(self, builder):
        examples = [
            SelfImprovementExample(prompt="P1", source=ExampleSource.CORRECTION, quality_score=0.8, difficulty=2),
            SelfImprovementExample(prompt="P2", source=ExampleSource.HIGH_QUALITY, quality_score=0.9, difficulty=1),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = builder.save_metadata(examples, tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                meta = json.load(f)
            assert meta["total_examples"] == 2
            assert meta["by_source"]["correction"] == 1
            assert meta["by_source"]["high_quality"] == 1


class TestPriorityScore:
    def test_hard_examples_high_priority(self, builder):
        hard = SelfImprovementExample(prompt="test", source=ExampleSource.HARD_EXAMPLE, quality_score=0.5, difficulty=1)
        corr = SelfImprovementExample(prompt="test", source=ExampleSource.CORRECTION, quality_score=0.5, difficulty=1)
        assert builder._priority_score(hard) > builder._priority_score(corr)


class TestStats:
    def test_initial(self, builder):
        assert builder.get_stats()["built"] == 0
        assert builder.get_stats()["exported"] == 0
