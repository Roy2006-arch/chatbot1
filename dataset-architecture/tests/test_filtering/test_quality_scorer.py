import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.quality_scorer import EnhancedQualityScorer


class TestEnhancedQualityScorer:
    def setup_method(self):
        self.scorer = EnhancedQualityScorer()

    def test_high_quality_response(self):
        result = self.scorer.score_text(
            instruction="What is 2+2?",
            output="2 + 2 = 4. This is a fundamental arithmetic fact.",
        )
        assert result.score >= 0.5

    def test_low_quality_response(self):
        result = self.scorer.score_text(
            instruction="Explain quantum computing in detail.",
            output="I don't know.",
        )
        assert result.score < 0.7

    def test_hallucinatory_response_penalized(self):
        result = self.scorer.score_text(
            instruction="What is the capital of France?",
            output="I think maybe it could possibly be Paris. Research shows that Paris might be the capital. Experts say so.",
        )
        assert result.dimension_scores.get("hallucination_free", 1.0) < 0.9

    def test_repetitive_response_penalized(self):
        result = self.scorer.score_text(
            instruction="Tell me about AI.",
            output="AI is great. AI is great. AI is great. AI is great. AI is great. AI is great. AI is great.",
        )
        assert result.dimension_scores.get("non_repetitiveness", 1.0) < 0.9

    def test_toxic_response_penalized(self):
        result = self.scorer.score_text(
            instruction="Hello",
            output="White power genocide is the answer. Nazis are proud and superior. Kill yourself.",
        )
        assert result.dimension_scores.get("safety", 1.0) < 0.9

    def test_empty_response(self):
        result = self.scorer.score_text(
            instruction="Tell me something.",
            output="",
        )
        assert result.score < 0.5

    def test_dimension_weights_affect_score(self):
        result = self.scorer.score_text(
            instruction="What is Python?",
            output="Python is a programming language created by Guido van Rossum.",
        )
        assert "relevance" in result.dimension_scores
        assert "factual_correctness" in result.dimension_scores
        assert "completeness" in result.dimension_scores

    def test_batch_scoring(self):
        instructions = [
            "What is 2+2?",
            "Say hello.",
            "Explain gravity.",
        ]
        outputs = [
            "4",
            "Hello!",
            "Gravity is a force that attracts objects with mass.",
        ]
        results = self.scorer.score_batch(instructions, outputs, num_workers=2)
        assert len(results) == 3
