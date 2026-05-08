import os
import tempfile
import sqlite3
import pytest
from self_improvement.quality_curator import QualityCurator


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            conv_id TEXT,
            turn_index INTEGER,
            role TEXT,
            content TEXT,
            composite_score REAL DEFAULT 0,
            timestamp_utc TEXT DEFAULT '',
            grade TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            conv_id TEXT,
            turn_index INTEGER,
            vote INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS failed_queries (
            id INTEGER PRIMARY KEY,
            conv_id TEXT,
            prompt TEXT,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def curator(db_path):
    return QualityCurator(config={
        "enabled": True,
        "min_quality_score": 0.8,
        "min_response_length": 10,
    }, db_path=db_path)


class TestQualityCuratorInit:
    def test_default_config(self):
        qc = QualityCurator()
        assert qc.enabled is True
        assert qc.min_quality_score == 0.8

    def test_custom_config(self):
        qc = QualityCurator(config={"enabled": False, "min_quality_score": 0.9})
        assert qc.enabled is False
        assert qc.min_quality_score == 0.9


class TestLoadHighQuality:
    def test_no_db_returns_empty(self):
        qc = QualityCurator(db_path="nonexistent.db")
        assert qc.load_high_quality_conversations() == []

    def test_returns_assistant_only(self, curator, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content, composite_score) VALUES (?, ?, ?, ?, ?)",
            ("c1", 0, "user", "Hello", 0.0),
        )
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content, composite_score) VALUES (?, ?, ?, ?, ?)",
            ("c1", 1, "assistant", "Hi there! How can I help?", 0.9),
        )
        conn.commit()
        conn.close()

        rows = curator.load_high_quality_conversations(limit=10)
        assert len(rows) == 1
        assert rows[0]["role"] == "assistant"

    def test_filters_by_min_length(self, curator, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content, composite_score) VALUES (?, ?, ?, ?, ?)",
            ("c1", 1, "assistant", "Short", 0.9),
        )
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content, composite_score) VALUES (?, ?, ?, ?, ?)",
            ("c2", 1, "assistant", "A much longer response that should pass the filter", 0.9),
        )
        conn.commit()
        conn.close()

        rows = curator.load_high_quality_conversations(limit=10)
        assert len(rows) == 1
        assert len(rows[0]["content"]) >= 20


class TestLoadByQualityThreshold:
    def test_returns_above_threshold(self, curator, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content, composite_score) VALUES (?, ?, ?, ?, ?)",
            ("c1", 1, "assistant", "Good response", 0.9),
        )
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content, composite_score) VALUES (?, ?, ?, ?, ?)",
            ("c2", 1, "assistant", "Poor response", 0.3),
        )
        conn.commit()
        conn.close()

        rows = curator.load_by_quality_threshold(threshold=0.8, limit=10)
        assert len(rows) == 1
        assert rows[0]["composite_score"] >= 0.8

    def test_no_db_returns_empty(self):
        qc = QualityCurator(db_path="nonexistent.db")
        assert qc.load_by_quality_threshold() == []


class TestScoreResponseQuality:
    def test_empty_response(self, curator):
        scores = curator.score_response_quality("Hello", "")
        assert scores["composite"] == 0.0

    def test_high_quality_response(self, curator):
        response = (
            "Here is a detailed explanation of the concept that covers multiple aspects.\n"
            "1. First key point to understand is the basics.\n"
            "2. Second key point builds on the first.\n"
            "```python\nprint('hello')\n```\n"
            "In conclusion, this is a complete and thorough answer."
        )
        scores = curator.score_response_quality("Explain concept", response)
        assert scores["length"] > 0.5
        assert scores["structure"] > 0.5
        assert scores["completeness"] > 0.5
        assert scores["composite"] > 0.5

    def test_low_quality_response(self, curator):
        response = "ok"
        scores = curator.score_response_quality("Hi", response)
        assert scores["composite"] < 0.5

    def test_length_score_scaling(self, curator):
        short = curator.score_response_quality("Hi", "Hello world")
        long = curator.score_response_quality("Hi", " ".join(["word"] * 100))
        assert long["length"] > short["length"]

    def test_structure_with_code_blocks(self, curator):
        response = "Some text\n```python\nx = 1\n```\n# Heading\n- list"
        scores = curator.score_response_quality("Test", response)
        assert scores["structure"] > 0.5


class TestCurate:
    def test_empty_when_disabled(self, curator):
        curator.enabled = False
        assert curator.curate(limit=10) == []

    def test_empty_when_no_data(self, curator):
        assert curator.curate(limit=10) == []

    def test_returns_examples(self, curator, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content, composite_score) VALUES (?, ?, ?, ?, ?)",
            ("c1", 1, "assistant",
             "A very long response that should definitely pass the quality threshold and be curated.",
             0.95),
        )
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content) VALUES (?, ?, ?, ?)",
            ("c1", 0, "user", "Tell me about AI"),
        )
        conn.commit()
        conn.close()

        examples = curator.curate(limit=10)
        assert len(examples) > 0
        assert examples[0].source.value == "high_quality"
        assert examples[0].prompt == "Tell me about AI"
        assert examples[0].corrected_response != ""

    def test_respects_max_per_category(self, curator, db_path):
        conn = sqlite3.connect(db_path)
        for i in range(5):
            conn.execute(
                "INSERT INTO conversations (conv_id, turn_index, role, content, composite_score) VALUES (?, ?, ?, ?, ?)",
                (f"c{i}", 1, "assistant",
                 f"A high quality response that is long enough to pass the filter number {i}.",
                 0.95),
            )
            conn.execute(
                "INSERT INTO conversations (conv_id, turn_index, role, content) VALUES (?, ?, ?, ?)",
                (f"c{i}", 0, "user", f"Question {i}"),
            )
        conn.commit()
        conn.close()

        curator.max_examples_per_category = 3
        examples = curator.curate(limit=10)
        assert len(examples) <= 3


class TestGetPromptForTurn:
    def test_returns_empty_for_no_db(self, curator):
        curator.db_path = "nonexistent.db"
        result = curator._get_prompt_for_turn("c1", 1)
        assert result == ""

    def test_returns_previous_turn(self, curator, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content) VALUES (?, ?, ?, ?)",
            ("c1", 0, "user", "What is AI?"),
        )
        conn.execute(
            "INSERT INTO conversations (conv_id, turn_index, role, content) VALUES (?, ?, ?, ?)",
            ("c1", 1, "assistant", "AI is..."),
        )
        conn.commit()
        conn.close()

        prompt = curator._get_prompt_for_turn("c1", 1)
        assert prompt == "What is AI?"

    def test_returns_empty_for_missing_turn(self, curator, db_path):
        prompt = curator._get_prompt_for_turn("nonexistent", 0)
        assert prompt == ""


class TestStats:
    def test_initial(self, curator):
        assert curator.get_stats()["scanned"] == 0
        assert curator.get_stats()["extracted"] == 0
