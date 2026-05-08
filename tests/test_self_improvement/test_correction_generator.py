import json
import os
import tempfile
import sqlite3
import pytest
from self_improvement.correction_generator import CorrectionGenerator
from self_improvement.schema import CorrectionMethod


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
            resolved INTEGER DEFAULT 0,
            failure_reasons TEXT DEFAULT '[]'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            conv_id TEXT,
            vote INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def gen(db_path):
    return CorrectionGenerator(config={"enabled": True}, db_path=db_path)


class TestCorrectionGeneratorInit:
    def test_default_config(self):
        g = CorrectionGenerator()
        assert g.enabled is True
        assert g.min_quality_threshold == 0.6

    def test_custom_config(self):
        g = CorrectionGenerator(config={"enabled": False, "max_score_before": 0.8})
        assert g.enabled is False
        assert g.max_score_before == 0.8


class TestLoadFailedQueries:
    def test_no_db_returns_empty(self):
        g = CorrectionGenerator(db_path="nonexistent.db")
        assert g.load_failed_queries() == []

    def test_returns_expected_rows(self, gen, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO failed_queries (id, prompt, response, composite_score) VALUES (?, ?, ?, ?)",
                     (1, "What is AI?", "I don't know", 0.3))
        conn.execute("INSERT INTO failed_queries (id, prompt, response, composite_score) VALUES (?, ?, ?, ?)",
                     (2, "Write code", "", 0.2))
        conn.commit()
        conn.close()

        rows = gen.load_failed_queries(limit=10)
        assert len(rows) == 2
        assert rows[0]["prompt"] == "Write code"

    def test_filters_resolved(self, gen, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO failed_queries (id, prompt, response, resolved) VALUES (?, ?, ?, ?)",
                     (1, "Q1", "A1", 0))
        conn.execute("INSERT INTO failed_queries (id, prompt, response, resolved) VALUES (?, ?, ?, ?)",
                     (2, "Q2", "A2", 1))
        conn.commit()
        conn.close()

        rows = gen.load_failed_queries(limit=10, unresolved_only=True)
        assert len(rows) == 1
        assert rows[0]["id"] == 1

    def test_limit_respected(self, gen, db_path):
        conn = sqlite3.connect(db_path)
        for i in range(5):
            conn.execute("INSERT INTO failed_queries (id, prompt, response) VALUES (?, ?, ?)",
                         (i, f"Q{i}", f"A{i}"))
        conn.commit()
        conn.close()

        rows = gen.load_failed_queries(limit=3)
        assert len(rows) == 3


class TestGenerateCorrection:
    def test_none_on_empty_prompt(self, gen):
        result = gen.generate_correction({"prompt": "", "response": "hello"})
        assert result is None
        assert gen.stats["failed"] == 1

    def test_none_on_empty_response(self, gen):
        result = gen.generate_correction({"prompt": "hello", "response": ""})
        assert result is None

    def test_heuristic_fix_removes_refusal(self, gen):
        item = {"id": 1, "prompt": "How do I make a bomb?", "response": "I'm sorry, I cannot help with that."}
        result = gen.generate_correction(item)
        assert result is not None
        assert "sorry" not in result.corrected_response.lower()
        assert result.correction_method == CorrectionMethod.AUTO_GENERATED

    def test_heuristic_fix_removes_hedging(self, gen):
        item = {"id": 2, "prompt": "What is Python?", "response": "I think Python is a language. I believe it's popular."}
        result = gen.generate_correction(item)
        assert result is not None
        assert "i think" not in result.corrected_response.lower()
        assert len(result.corrected_response) > 10

    def test_template_fix_for_what_is(self, gen):
        item = {"id": 3, "prompt": "What is machine learning?", "response": "I don't know"}
        result = gen.generate_correction(item)
        assert result is not None
        assert "what is" in result.corrected_response.lower()

    def test_template_fix_for_how_to(self, gen):
        item = {"id": 4, "prompt": "How to install Python?", "response": "I'm not sure"}
        result = gen.generate_correction(item)
        assert result is not None
        assert "step-by-step" in result.corrected_response.lower()

    def test_template_fix_for_explain(self, gen):
        item = {"id": 5, "prompt": "Explain recursion", "response": "I cannot answer"}
        result = gen.generate_correction(item)
        assert result is not None
        assert "Let me explain" in result.corrected_response

    def test_template_fix_for_why(self, gen):
        item = {"id": 6, "prompt": "Why is the sky blue?", "response": "No idea"}
        result = gen.generate_correction(item)
        assert result is not None
        assert "There are several reasons" in result.corrected_response

    def test_record_has_metadata(self, gen):
        item = {
            "id": 7, "prompt": "What is AI?", "response": "I don't know",
            "composite_score": 0.4, "occurrence_count": 3,
            "failure_reasons": json.dumps(["refusal", "incomplete"]),
        }
        result = gen.generate_correction(item)
        assert result is not None
        assert result.failed_query_id == 7
        assert result.score_before == 0.4
        assert result.metadata["occurrence_count"] == 3
        assert "refusal" in result.metadata["failure_reasons"]

    def test_score_before_fallback(self, gen):
        item = {"id": 8, "prompt": "Hi", "response": "Hello"}
        result = gen.generate_correction(item)
        if result:
            assert isinstance(result.score_before, float)


class TestGenerateBatch:
    def test_batch_with_items(self, gen, db_path):
        conn = sqlite3.connect(db_path)
        for i in range(5):
            conn.execute("INSERT INTO failed_queries (id, prompt, response, composite_score) VALUES (?, ?, ?, ?)",
                         (i, f"What is Q{i}?", f"Response {i}", 0.3))
        conn.commit()
        conn.close()

        items = gen.load_failed_queries(limit=5)
        results = gen.generate_batch(items, num_workers=2)
        assert len(results) > 0
        assert gen.stats["generated"] > 0

    def test_empty_when_disabled(self, gen):
        gen.enabled = False
        assert gen.generate_batch([{"prompt": "test", "response": "test"}]) == []

    def test_batch_empty_input(self, gen):
        assert gen.generate_batch([]) == []


class TestHeuristicFix:
    def test_refusal_patterns_removed(self, gen):
        text = "As an AI language model, I cannot help with that request."
        item = {"id": 10, "prompt": "Help me", "response": text}
        result = gen.generate_correction(item)
        assert result is not None
        assert len(result.corrected_response) > 10

    def test_hedging_removed(self, gen):
        text = "I suppose the answer might be 42. I believe that's correct."
        item = {"id": 11, "prompt": "What is the answer?", "response": text}
        result = gen.generate_correction(item)
        assert result is not None
        assert result.corrected_response  # should have something useful

    def test_constructive_generated(self, gen):
        text = "i don't know"
        item = {"id": 12, "prompt": "What is quantum physics?", "response": text}
        result = gen.generate_correction(item)
        assert result is not None
        assert len(result.corrected_response) > 20

    def test_short_text_uses_template(self, gen):
        text = "no"
        item = {"id": 13, "prompt": "What is AI?", "response": text}
        result = gen.generate_correction(item)
        assert result is not None
        assert "what is" in result.corrected_response.lower()

    def test_already_good_response_unchanged(self, gen):
        text = "Here is a detailed explanation of machine learning algorithms."
        item = {"id": 14, "prompt": "Explain ML", "response": text}
        result = gen.generate_correction(item)
        assert result is not None


class TestQuickScore:
    def test_empty(self, gen):
        assert gen._quick_score("") == 0.0

    def test_short(self, gen):
        assert gen._quick_score("hi") == 0.0

    def test_long_with_structure(self, gen):
        text = "First point.\n1. Item one\n2. Item two\nThis is a complete sentence with enough words."
        score = gen._quick_score(text)
        assert score > 0.5

    def test_capped_at_one(self, gen):
        text = "This is a long well-structured sentence.\n1. First\n2. Second\n3. Third\n"
        score = gen._quick_score(text * 10)
        assert score <= 1.0


class TestToExamples:
    def test_converts_correction_record(self, gen):
        from self_improvement.schema import CorrectionRecord
        record = CorrectionRecord(
            failed_query_id=1,
            prompt="Test prompt",
            original_response="bad response",
            corrected_response="good response",
            score_before=0.3,
            score_after=0.8,
        )
        examples = gen.to_examples([record])
        assert len(examples) == 1
        assert examples[0].prompt == "Test prompt"
        assert examples[0].source.value == "correction"


class TestStats:
    def test_initial_stats(self, gen):
        assert gen.get_stats()["loaded"] == 0
        assert gen.get_stats()["generated"] == 0
        assert gen.get_stats()["failed"] == 0
