from unittest.mock import patch, MagicMock
import numpy as np
import pytest


def _make_embedding():
    v = np.random.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    """Mock sentence_transformers globally for all self-improvement tests."""
    mock_instance = MagicMock()

    def encode_side_effect(sentences, **kwargs):
        if isinstance(sentences, str):
            return _make_embedding()
        return np.stack([_make_embedding() for _ in sentences])

    mock_instance.encode.side_effect = encode_side_effect
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance):
        yield
