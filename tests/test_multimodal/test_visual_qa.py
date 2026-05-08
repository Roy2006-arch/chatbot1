import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from multimodal.visual_qa import VisualQAGenerator
from multimodal.schema import MultimodalExample, MediaType


class TestVisualQAGeneratorInit:
    def test_default_config(self):
        gen = VisualQAGenerator()
        assert gen.enabled is True
        assert gen.min_caption_length == 5
        assert gen.max_caption_length == 200
        assert gen.use_local_model is False

    def test_custom_config(self):
        gen = VisualQAGenerator(config={
            "enabled": False, "min_caption_length": 10, "use_local_model": True,
        })
        assert gen.enabled is False
        assert gen.min_caption_length == 10
        assert gen.use_local_model is True


class TestModelAvailable:
    def test_false_when_not_using_local_model(self):
        gen = VisualQAGenerator(config={"use_local_model": False})
        assert gen.model_available is False

    def test_false_when_model_fails_to_load(self):
        gen = VisualQAGenerator(config={"use_local_model": True})
        with patch.object(gen, "_load_model", return_value=None):
            assert gen.model_available is False

    def test_true_when_model_loads(self):
        gen = VisualQAGenerator(config={"use_local_model": True})
        with patch.object(gen, "_load_model", return_value=MagicMock()):
            assert gen.model_available is True


class TestGenerateCaption:
    def test_disabled_returns_empty(self, sample_image_path):
        gen = VisualQAGenerator(config={"enabled": False})
        assert gen.generate_caption(sample_image_path) == ""

    def test_uses_model_caption_when_available(self, sample_image_path):
        gen = VisualQAGenerator(config={"use_local_model": True})
        with patch.object(gen, "_load_model", return_value=MagicMock()):
            with patch.object(gen, "_model_caption", return_value="A cat sitting on a mat."):
                with patch.object(gen, "_template_caption", return_value=""):
                    caption = gen.generate_caption(sample_image_path)
                    assert "cat" in caption

    def test_falls_back_to_template(self, sample_image_path):
        gen = VisualQAGenerator()
        with patch.object(gen, "_model_caption", return_value=""):
            caption = gen.generate_caption(sample_image_path, "some text")
            assert "text content" in caption or "Image" in caption

    def test_falls_back_when_caption_too_short(self, sample_image_path):
        gen = VisualQAGenerator(config={"min_caption_length": 20})
        with patch.object(gen, "_model_caption", return_value="Short"):
            caption = gen.generate_caption(sample_image_path)
            assert len(caption) >= 20

    def test_truncates_long_caption(self, sample_image_path):
        gen = VisualQAGenerator(config={"max_caption_length": 5})
        long_caption = "word " * 20
        with patch.object(gen, "_model_caption", return_value=""):
            with patch.object(gen, "_template_caption", return_value=long_caption):
                caption = gen.generate_caption(sample_image_path)
                assert len(caption.split()) <= 5

    def test_tracks_stats(self, sample_image_path):
        gen = VisualQAGenerator()
        with patch.object(gen, "_model_caption", return_value=""):
            with patch.object(gen, "_template_caption", return_value="A valid caption for the image shown."):
                gen.generate_caption(sample_image_path)
                assert gen.stats["captions"] == 1

    def test_uses_ocr_text_in_template(self, sample_image_path):
        gen = VisualQAGenerator()
        with patch.object(gen, "_model_caption", return_value=""):
            caption = gen.generate_caption(sample_image_path, "Hello World OCR text")
            assert "Hello World" in caption or "text content" in caption


class TestModelCaption:
    def test_empty_on_exception(self, sample_image_path):
        gen = VisualQAGenerator()
        gen._processor = MagicMock()
        gen._model = MagicMock()
        gen._processor.side_effect = Exception("model error")
        assert gen._model_caption(sample_image_path) == ""

    def test_returns_caption(self, sample_image_path):
        gen = VisualQAGenerator()
        gen._processor = MagicMock()
        gen._model = MagicMock()
        mock_inputs = MagicMock()
        gen._processor.return_value = mock_inputs
        mock_out = MagicMock()
        gen._model.generate.return_value = mock_out
        gen._processor.decode.return_value = "A valid model caption."
        result = gen._model_caption(sample_image_path)
        assert result == "A valid model caption."


class TestTemplateCaption:
    def test_basic_template(self, sample_image_path):
        gen = VisualQAGenerator()
        caption = gen._template_caption(sample_image_path, "")
        assert "sample_image" in caption or os.path.basename(sample_image_path) in caption

    def test_with_ocr_text(self, sample_image_path):
        gen = VisualQAGenerator()
        caption = gen._template_caption(sample_image_path, "Some OCR text here")
        assert "text content" in caption

    def test_portrait_detection(self, sample_image_path):
        gen = VisualQAGenerator()
        caption = gen._template_caption(sample_image_path, "")
        assert "200x200" in caption


class TestFallbackCaption:
    def test_with_text(self):
        gen = VisualQAGenerator()
        caption = gen._fallback_caption("Hello world this is some text")
        assert "Hello world" in caption

    def test_without_text(self):
        gen = VisualQAGenerator()
        caption = gen._fallback_caption("")
        assert caption == "Image with no visible text content."

    def test_truncates_long_text(self):
        gen = VisualQAGenerator()
        long_text = "word " * 50
        caption = gen._fallback_caption(long_text)
        words = caption.split()
        assert "..." in caption


class TestGenerateQAPairs:
    def test_returns_requested_number(self, sample_image_path):
        gen = VisualQAGenerator()
        pairs = gen.generate_qa_pairs(sample_image_path, "some text", "A caption.", max_pairs=2)
        assert len(pairs) == 2

    def test_without_text(self, sample_image_path):
        gen = VisualQAGenerator()
        pairs = gen.generate_qa_pairs(sample_image_path, "", "A caption.", max_pairs=3)
        assert len(pairs) == 3
        assert "No readable text" in pairs[1]["answer"]

    def test_tracks_stats(self, sample_image_path):
        gen = VisualQAGenerator()
        pairs = gen.generate_qa_pairs(sample_image_path, "text", "Caption.", max_pairs=2)
        assert gen.stats["qa_pairs"] == 2

    def test_qa_structure(self, sample_image_path):
        gen = VisualQAGenerator()
        pairs = gen.generate_qa_pairs(sample_image_path, "Hello", "A nice image.", max_pairs=1)
        assert "question" in pairs[0]
        assert "answer" in pairs[0]


class TestBuildExample:
    def test_returns_none_on_failed_caption(self, sample_image_path):
        gen = VisualQAGenerator()
        with patch.object(gen, "generate_caption", return_value=""):
            result = gen.build_example(sample_image_path)
            assert result is None
            assert gen.stats["failed"] == 1

    def test_returns_example(self, sample_image_path):
        gen = VisualQAGenerator()
        with patch.object(gen, "generate_caption", return_value="A detailed caption for testing purposes here."):
            result = gen.build_example(sample_image_path, "OCR text")
            assert result is not None
            assert isinstance(result, MultimodalExample)
            assert result.caption == "A detailed caption for testing purposes here."
            assert result.image_path == sample_image_path

    def test_includes_qa_pairs(self, sample_image_path):
        gen = VisualQAGenerator()
        with patch.object(gen, "generate_caption", return_value="A valid caption for this image."):
            result = gen.build_example(sample_image_path, "text")
            assert "qa_pairs" in result.metadata
            assert len(result.metadata["qa_pairs"]) > 0

    def test_quality_score(self, sample_image_path):
        gen = VisualQAGenerator()
        with patch.object(gen, "generate_caption", return_value="A detailed caption for quality scoring here."):
            result = gen.build_example(sample_image_path, "some OCR text that is long enough")
            assert result.quality_score > 0.3


class TestBuildBatch:
    def test_disabled_returns_nones(self, sample_image_path):
        gen = VisualQAGenerator(config={"enabled": False})
        assert gen.build_batch([sample_image_path]) == [None]

    def test_returns_results(self, sample_image_path, landscape_image_path):
        gen = VisualQAGenerator()
        with patch.object(gen, "generate_caption", return_value="A caption for testing."):
            results = gen.build_batch([sample_image_path, landscape_image_path], num_workers=2)
            assert len(results) == 2
            assert results[0] is not None
            assert results[1] is not None

    def test_empty_input(self):
        gen = VisualQAGenerator()
        assert gen.build_batch([]) == []


class TestScore:
    def test_minimum_score(self):
        gen = VisualQAGenerator()
        assert gen._score("short", "", []) == 0.3

    def test_long_caption(self):
        gen = VisualQAGenerator()
        assert gen._score("word " * 10, "", [{"q": "a"}]) == 0.5

    def test_with_ocr_text(self):
        gen = VisualQAGenerator()
        assert gen._score("word " * 10, "text " * 10, [{"q": "a"}]) == 0.7

    def test_max_score(self):
        gen = VisualQAGenerator()
        score = gen._score("word " * 10, "text " * 10, [{"q": "a"}, {"q": "b"}, {"q": "c"}])
        assert score == 1.0


class TestStats:
    def test_initial_stats(self):
        gen = VisualQAGenerator()
        assert gen.get_stats()["captions"] == 0
        assert gen.get_stats()["qa_pairs"] == 0
        assert gen.get_stats()["failed"] == 0
