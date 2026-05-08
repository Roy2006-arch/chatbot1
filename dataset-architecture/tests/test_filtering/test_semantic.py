import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from filtering.semantic_filter import SemanticSimilarityFilter
from filtering.models import FilterResult


class TestSemanticSimilarityFilter:
    def setup_method(self):
        self.filter = SemanticSimilarityFilter({"threshold": 0.85, "batch_size": 32})

    def test_requires_sentence_transformers(self):
        try:
            import sentence_transformers
        except ImportError:
            import pytest
            pytest.skip("sentence-transformers not installed")

    def test_initialization(self):
        assert self.filter.threshold == 0.85
        assert self.filter.batch_size == 32

    def test_cluster_no_duplicates(self):
        try:
            import sentence_transformers
        except ImportError:
            import pytest
            pytest.skip("sentence-transformers not installed")

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a programming language.",
            "The weather today is sunny and warm.",
        ]
        clusters = self.filter.cluster_duplicates(texts)
        assert isinstance(clusters, list)

    def test_find_near_duplicates(self):
        try:
            import sentence_transformers
        except ImportError:
            import pytest
            pytest.skip("sentence-transformers not installed")

        pairs = self.filter.find_near_duplicates([
            "What is the capital of France?",
            "What is the capital of France?",
            "The sky is blue.",
        ])
        assert isinstance(pairs, list)
