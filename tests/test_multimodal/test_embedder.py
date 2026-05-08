import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
from multimodal.embedder import MultimodalEmbedder
from multimodal.schema import MultimodalExample, MediaType


class TestMultimodalEmbedderInit:
    def test_default_config(self):
        e = MultimodalEmbedder()
        assert e.enabled is True
        assert e.model_name == "clip-ViT-B-32"
        assert e.text_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert e.normalize_embeddings is True

    def test_custom_config(self):
        e = MultimodalEmbedder(config={
            "enabled": False, "model_name": "custom-model",
            "normalize_embeddings": False,
        })
        assert e.enabled is False
        assert e.model_name == "custom-model"
        assert e.normalize_embeddings is False


class TestClipAvailable:
    def test_false_no_model(self):
        e = MultimodalEmbedder()
        with patch.object(e, "_load_clip", return_value=None):
            assert e.clip_available is False

    def test_true_when_loaded(self):
        e = MultimodalEmbedder()
        with patch.object(e, "_load_clip", return_value=MagicMock()):
            assert e.clip_available is True


class TestTextModelAvailable:
    def test_false_no_model(self):
        e = MultimodalEmbedder()
        with patch.object(e, "_load_sentence_model", return_value=None):
            with patch.object(e, "_load_text_model", return_value=None):
                assert e.text_model_available is False

    def test_true_when_loaded(self):
        e = MultimodalEmbedder()
        with patch.object(e, "_load_sentence_model", return_value=MagicMock()):
            assert e.text_model_available is True


class TestEmbedImage:
    def test_disabled_returns_none(self, sample_image_path):
        e = MultimodalEmbedder(config={"enabled": False})
        assert e.embed_image(sample_image_path) is None

    def test_nonexistent_path(self):
        e = MultimodalEmbedder()
        assert e.embed_image("nonexistent.png") is None

    def test_uses_clip_when_available(self, sample_image_path):
        e = MultimodalEmbedder()
        fake_emb = np.array([0.1, 0.2, 0.3])
        with patch.object(e, "_load_clip", return_value=MagicMock()) as mock_load:
            mock_load.return_value.encode.return_value = fake_emb
            result = e.embed_image(sample_image_path)
            assert result is not None
            assert len(result) == 3

    def test_fallback_when_no_clip(self, sample_image_path):
        e = MultimodalEmbedder()
        fake_emb = [0.5] * 512
        with patch.object(e, "_load_clip", return_value=None):
            with patch.object(e, "_fallback_image_embedding", return_value=fake_emb) as mock_fb:
                result = e.embed_image(sample_image_path)
                mock_fb.assert_called_once()
                assert len(result) == 512

    def test_tracks_stats(self, sample_image_path):
        e = MultimodalEmbedder()
        fake_emb = np.array([0.1, 0.2, 0.3])
        with patch.object(e, "_load_clip", return_value=MagicMock()) as mock_load:
            mock_load.return_value.encode.return_value = fake_emb
            e.embed_image(sample_image_path)
            assert e.stats["images_embedded"] == 1


class TestEmbedText:
    def test_disabled_returns_none(self):
        e = MultimodalEmbedder(config={"enabled": False})
        assert e.embed_text("hello") is None

    def test_empty_text(self):
        e = MultimodalEmbedder()
        assert e.embed_text("") is None

    def test_uses_model_when_available(self):
        e = MultimodalEmbedder()
        fake_emb = np.array([0.1, 0.2, 0.3])
        with patch.object(e, "_load_sentence_model", return_value=MagicMock()) as mock_load:
            mock_load.return_value.encode.return_value = fake_emb
            result = e.embed_text("Hello world")
            assert result is not None
            assert len(result) == 3

    def test_fallback_when_no_model(self):
        e = MultimodalEmbedder()
        fake_emb = [0.5] * 384
        with patch.object(e, "_load_sentence_model", return_value=None):
            with patch.object(e, "_fallback_text_embedding", return_value=fake_emb) as mock_fb:
                result = e.embed_text("Hello")
                mock_fb.assert_called_once()
                assert len(result) == 384


class TestEmbedExample:
    def test_disabled_returns_unchanged(self):
        e = MultimodalEmbedder(config={"enabled": False})
        ex = MultimodalExample(prompt="Q", response="A")
        result = e.embed_example(ex)
        assert result is ex

    def test_embeds_image_and_text(self, sample_image_path):
        e = MultimodalEmbedder()
        ex = MultimodalExample(
            prompt="Q", response="A", image_path=sample_image_path
        )
        fake_emb = np.array([0.1] * 384)
        with patch.object(e, "_load_clip", return_value=MagicMock()) as mock_clip:
            mock_clip.return_value.encode.return_value = fake_emb
            with patch.object(e, "_load_sentence_model", return_value=MagicMock()) as mock_txt:
                mock_txt.return_value.encode.return_value = fake_emb
                result = e.embed_example(ex)
                assert result.image_embedding is not None
                assert "text_embedding" in result.metadata


class TestEmbedBatch:
    def test_returns_embeddings(self, sample_image_path):
        e = MultimodalEmbedder()
        examples = [
            MultimodalExample(prompt="Q1", response="A1", image_path=sample_image_path),
            MultimodalExample(prompt="Q2", response="A2", image_path=sample_image_path),
        ]
        fake_emb = np.array([0.1] * 384)
        with patch.object(e, "_load_clip", return_value=MagicMock()) as mock_clip:
            mock_clip.return_value.encode.return_value = fake_emb
            with patch.object(e, "_load_sentence_model", return_value=MagicMock()) as mock_txt:
                mock_txt.return_value.encode.return_value = fake_emb
                results = e.embed_batch(examples, num_workers=2)
                assert len(results) == 2
                assert results[0].image_embedding is not None

    def test_empty_input(self):
        e = MultimodalEmbedder()
        assert e.embed_batch([]) == []


class TestComputeSimilarity:
    def test_normalized_dot_product(self):
        e = MultimodalEmbedder(config={"normalize_embeddings": True})
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = e.compute_similarity(a, b)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        e = MultimodalEmbedder(config={"normalize_embeddings": True})
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = e.compute_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_cosine_similarity(self):
        e = MultimodalEmbedder(config={"normalize_embeddings": False})
        a = [2.0, 0.0]
        b = [1.0, 0.0]
        sim = e.compute_similarity(a, b)
        assert abs(sim - 1.0) < 1e-6


class TestFallbackImageEmbedding:
    def test_returns_embedding(self, sample_image_path):
        e = MultimodalEmbedder()
        emb = e._fallback_image_embedding(sample_image_path)
        assert emb is not None
        assert len(emb) == 512

    def test_normalized(self, sample_image_path):
        e = MultimodalEmbedder(config={"normalize_embeddings": True})
        emb = e._fallback_image_embedding(sample_image_path)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-4

    def test_nonexistent_returns_none(self):
        e = MultimodalEmbedder()
        assert e._fallback_image_embedding("nonexistent.png") is None


class TestFallbackTextEmbedding:
    def test_deterministic(self):
        e = MultimodalEmbedder()
        emb1 = e._fallback_text_embedding("Hello world")
        emb2 = e._fallback_text_embedding("Hello world")
        assert emb1 == emb2

    def test_different_text_different_embeddings(self):
        e = MultimodalEmbedder()
        emb1 = e._fallback_text_embedding("Hello")
        emb2 = e._fallback_text_embedding("World")
        assert emb1 != emb2

    def test_normalized(self):
        e = MultimodalEmbedder(config={"normalize_embeddings": True})
        emb = e._fallback_text_embedding("Test")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-4


class TestStats:
    def test_initial_stats(self):
        e = MultimodalEmbedder()
        assert e.get_stats()["images_embedded"] == 0
        assert e.get_stats()["texts_embedded"] == 0
        assert e.get_stats()["failed"] == 0
