import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.response_ranker import EnhancedResponseRanker
from filtering.quality_scorer import EnhancedQualityScorer
from src.pipeline.ingestion import DatasetExample


class TestEnhancedResponseRanker:
    def setup_method(self):
        self.ranker = EnhancedResponseRanker(top_k=2)

    def _make_example(self, quality: float, difficulty: int = 1, idx: str = "") -> DatasetExample:
        return DatasetExample(
            instruction=f"Instruction {idx}",
            output=f"Output {idx}" * 10,
            difficulty=difficulty,
            quality_score=quality,
            id=idx or f"ex_{quality}_{difficulty}",
        )

    def test_rank_by_quality(self):
        examples = [
            self._make_example(0.3, 1, "a"),
            self._make_example(0.9, 2, "b"),
            self._make_example(0.6, 3, "c"),
        ]
        ranked = self.ranker.rank_by_quality(examples)
        assert ranked[0].quality_score == 0.9
        assert ranked[-1].quality_score == 0.3

    def test_rank_by_difficulty_ascending(self):
        examples = [
            self._make_example(0.5, 3, "x"),
            self._make_example(0.5, 1, "y"),
            self._make_example(0.5, 2, "z"),
        ]
        ranked = self.ranker.rank_by_difficulty(examples, ascending=True)
        assert ranked[0].difficulty == 1
        assert ranked[-1].difficulty == 3

    def test_rank_by_difficulty_descending(self):
        examples = [
            self._make_example(0.5, 1, "a"),
            self._make_example(0.5, 3, "b"),
            self._make_example(0.5, 2, "c"),
        ]
        ranked = self.ranker.rank_by_difficulty(examples, ascending=False)
        assert ranked[0].difficulty == 3
        assert ranked[-1].difficulty == 1

    def test_compute_elo_scores(self):
        examples = [
            self._make_example(0.3, 1, "a"),
            self._make_example(0.7, 1, "b"),
            self._make_example(0.9, 1, "c"),
        ]
        ranked = self.ranker.compute_elo_scores(examples)
        assert "elo_score" in ranked[0].metadata
        assert ranked[0].metadata["elo_score"] > ranked[-1].metadata["elo_score"]

    def test_rank_texts(self):
        instructions = ["What is 2+2?", "Say hello", "Explain gravity"]
        outputs = ["4", "Hello!", "Gravity is a force that attracts objects with mass."]
        ranked = self.ranker.rank_texts(instructions, outputs)
        assert len(ranked) == 3
        assert all(isinstance(r.score, float) for r in ranked)
        assert ranked[0].score >= ranked[-1].score

    def test_get_top_k(self):
        items = self.ranker.rank_texts(
            ["Q1", "Q2", "Q3", "Q4", "Q5"],
            ["A1", "A2", "A3", "A4", "A5"],
        )
        top = self.ranker.get_top_k(items)
        assert len(top) == 2

    def test_get_bottom_k(self):
        items = self.ranker.rank_texts(
            ["Q1", "Q2", "Q3"],
            ["A1 answer here", "A2", "A3 response text"],
        )
        bottom = self.ranker.get_bottom_k(items)
        assert len(bottom) == 2
