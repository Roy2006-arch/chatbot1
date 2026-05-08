import json
import os
import tempfile
import sqlite3
import pytest
from self_improvement.hard_example_miner import HardExampleMiner


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS failed_queries (
            id INTEGER PRIMARY KEY,
            conv_id TEXT,
            prompt TEXT,
            response TEXT,
            composite_score REAL DEFAULT 0,
            occurrence_count INTEGER DEFAULT 1,
            failure_reasons TEXT DEFAULT '[]'
        )
    """)
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def miner(db_path):
    return HardExampleMiner(config={
        "enabled": True,
        "min_occurrence_count": 1,
        "cluster_threshold": 0.3,
        "max_examples_per_cluster": 5,
    }, db_path=db_path)


class TestHardExampleMinerInit:
    def test_default_config(self):
        miner = HardExampleMiner()
        assert miner.enabled is True
        assert miner.min_occurrence_count == 2

    def test_custom_config(self):
        miner = HardExampleMiner(config={"enabled": False, "min_occurrence_count": 3})
        assert miner.enabled is False
        assert miner.min_occurrence_count == 3


class TestLoadFailedQueries:
    def test_no_db_returns_empty(self):
        miner = HardExampleMiner(db_path="nonexistent.db")
        assert miner.load_failed_queries() == []

    def test_filters_by_occurrence_count(self, miner, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO failed_queries (id, prompt, response, occurrence_count) VALUES (?, ?, ?, ?)",
            (1, "Q1", "A1", 1),
        )
        conn.execute(
            "INSERT INTO failed_queries (id, prompt, response, occurrence_count) VALUES (?, ?, ?, ?)",
            (2, "Q2", "A2", 5),
        )
        conn.commit()
        conn.close()

        miner.min_occurrence_count = 3
        rows = miner.load_failed_queries(limit=10)
        assert len(rows) == 1
        assert rows[0]["id"] == 2

    def test_limit(self, miner, db_path):
        conn = sqlite3.connect(db_path)
        for i in range(5):
            conn.execute(
                "INSERT INTO failed_queries (id, prompt, response, occurrence_count) VALUES (?, ?, ?, ?)",
                (i, f"Q{i}", f"A{i}", 2),
            )
        conn.commit()
        conn.close()

        rows = miner.load_failed_queries(limit=3)
        assert len(rows) == 3


class TestLoadAllFailed:
    def test_returns_all(self, miner, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO failed_queries (id, prompt, response, occurrence_count) VALUES (?, ?, ?, ?)",
                     (1, "Q1", "A1", 1))
        conn.execute("INSERT INTO failed_queries (id, prompt, response, occurrence_count) VALUES (?, ?, ?, ?)",
                     (2, "Q2", "A2", 5))
        conn.commit()
        conn.close()

        rows = miner.load_all_failed(limit=10)
        assert len(rows) == 2


class TestTokenSimilarity:
    def test_identical_texts(self, miner):
        sim = miner._token_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_different_texts(self, miner):
        sim = miner._token_similarity("hello world", "goodbye moon")
        assert sim < 1.0

    def test_empty_text(self, miner):
        assert miner._token_similarity("", "hello") == 0.0
        assert miner._token_similarity("hello", "") == 0.0
        assert miner._token_similarity("", "") == 0.0

    def test_partial_overlap(self, miner):
        sim = miner._token_similarity("hello world foo", "hello world bar")
        assert 0.4 < sim <= 0.6


class TestFailureTypesOverlap:
    def test_overlap(self, miner):
        assert miner._failure_types_overlap(["refusal", "incomplete"], ["refusal"]) is True

    def test_no_overlap(self, miner):
        assert miner._failure_types_overlap(["refusal"], ["hallucination"]) is False

    def test_empty_lists(self, miner):
        assert miner._failure_types_overlap([], []) is False

    def test_case_insensitive(self, miner):
        assert miner._failure_types_overlap(["REFUSAL"], ["refusal"]) is True


class TestClusterByFailureType:
    def test_basic_clustering(self, miner, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO failed_queries (id, prompt, failure_reasons) VALUES (?, ?, ?)",
                     (1, "What is Python?", json.dumps(["refusal"])))
        conn.execute("INSERT INTO failed_queries (id, prompt, failure_reasons) VALUES (?, ?, ?)",
                     (2, "What is Python used for?", json.dumps(["refusal"])))
        conn.execute("INSERT INTO failed_queries (id, prompt, failure_reasons) VALUES (?, ?, ?)",
                     (3, "How to cook pasta?", json.dumps(["hallucination"])))
        conn.commit()
        conn.close()

        records = miner.load_all_failed()
        clusters = miner._cluster_by_failure_type(records)
        assert len(clusters) >= 1

    def test_all_similar_texts_same_cluster(self, miner):
        records = [
            {"prompt": "What is Python?", "failure_reasons": json.dumps(["refusal"])},
            {"prompt": "What is Python used for?", "failure_reasons": json.dumps(["refusal"])},
        ]
        clusters = miner._cluster_by_failure_type(records)
        assert any(len(members) > 1 for members in clusters.values())


class TestCategorizePrompt:
    def test_code_category(self, miner):
        assert miner._categorize_prompt("Write a function to sort") == "code"
        assert miner._categorize_prompt("Debug this error") == "code"

    def test_reasoning_category(self, miner):
        assert miner._categorize_prompt("Why is the sky blue?") == "reasoning"
        assert miner._categorize_prompt("Explain gravity") == "reasoning"

    def test_math_category(self, miner):
        assert miner._categorize_prompt("Calculate 2+2") == "math"
        assert miner._categorize_prompt("Solve equation x+1=2") == "math"

    def test_technical_category(self, miner):
        assert miner._categorize_prompt("How to install Docker?") == "technical"
        assert miner._categorize_prompt("Setup a web server") == "technical"

    def test_factual_category(self, miner):
        assert miner._categorize_prompt("What is the capital of France?") == "factual"
        assert miner._categorize_prompt("Who is Albert Einstein?") == "factual"

    def test_fallback_general(self, miner):
        assert miner._categorize_prompt("Hello there") == "general"


class TestMine:
    def test_empty_when_disabled(self, miner):
        miner.enabled = False
        assert miner.mine() == []

    def test_empty_when_no_data(self, miner):
        assert miner.mine() == []

    def test_returns_hard_examples(self, miner, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO failed_queries (id, prompt, response, composite_score, occurrence_count, failure_reasons) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (1, "Hard problem here", "Bad response", 0.2, 5, json.dumps(["refusal"])),
        )
        conn.commit()
        conn.close()

        results = miner.mine()
        assert len(results) > 0
        assert results[0].prompt == "Hard problem here"
        assert results[0].difficulty > 0
        assert results[0].occurrence_count >= 1

    def test_difficulty_scales_with_occurrence(self, miner, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO failed_queries (id, prompt, response, occurrence_count) VALUES (?, ?, ?, ?)",
            (1, "Easy problem", "A", 1),
        )
        conn.execute(
            "INSERT INTO failed_queries (id, prompt, response, occurrence_count) VALUES (?, ?, ?, ?)",
            (2, "Hard problem", "B", 10),
        )
        conn.commit()
        conn.close()

        results = miner.mine()
        difficulties = {r.prompt: r.difficulty for r in results}
        assert difficulties.get("Hard problem", 0) >= difficulties.get("Easy problem", 0)


class TestToExamples:
    def test_converts_hard_examples(self, miner):
        from self_improvement.schema import HardExample
        he = HardExample(prompt="test", response="bad", category="code", difficulty=3, failure_reasons=["error"])
        examples = miner.to_examples([he])
        assert len(examples) == 1
        assert examples[0].source.value == "hard_example"
        assert examples[0].original_response == "bad"


class TestComputeStatistics:
    def test_empty_input(self, miner):
        assert miner.compute_statistics([]) == {}

    def test_returns_counts(self, miner):
        from self_improvement.schema import HardExample
        examples = [
            HardExample(prompt="P1", category="code", failure_reasons=["refusal"]),
            HardExample(prompt="P2", category="code", failure_reasons=["incomplete"]),
            HardExample(prompt="P3", category="math", failure_reasons=["refusal"]),
        ]
        stats = miner.compute_statistics(examples)
        assert stats["total_mined"] == 3
        assert stats["by_category"]["code"] == 2
        assert stats["by_category"]["math"] == 1
        assert stats["top_failure_reasons"]["refusal"] == 2


class TestStats:
    def test_initial(self, miner):
        assert miner.get_stats()["loaded"] == 0
        assert miner.get_stats()["mined"] == 0
