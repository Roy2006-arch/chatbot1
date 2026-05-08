import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image
from multimodal.image_preprocessor import ImagePreprocessor, PreprocessResult
from multimodal.schema import ImageExample, MediaType


class TestImagePreprocessorInit:
    def test_default_config(self):
        p = ImagePreprocessor()
        assert p.enabled is True
        assert p.max_width == 1024
        assert p.max_height == 1024
        assert p.min_width == 32
        assert p.min_height == 32
        assert p.max_size_mb == 20

    def test_custom_config(self):
        p = ImagePreprocessor(config={"enabled": False, "max_width": 512, "min_width": 64})
        assert p.enabled is False
        assert p.max_width == 512
        assert p.min_width == 64


class TestLoadImage:
    def test_loads_valid_image(self, sample_image_path):
        p = ImagePreprocessor()
        img = p.load_image(sample_image_path)
        assert img is not None
        assert img.size == (200, 200)

    def test_nonexistent_path(self):
        p = ImagePreprocessor()
        assert p.load_image("nonexistent.png") is None


class TestValidate:
    def test_none_image(self):
        p = ImagePreprocessor()
        valid, reason = p.validate(None)
        assert valid is False
        assert reason == "failed_to_load"

    def test_too_small(self, small_image_path):
        p = ImagePreprocessor()
        from PIL import Image
        img = Image.open(small_image_path)
        valid, reason = p.validate(img)
        assert valid is False
        assert "too_small" in reason

    def test_too_large(self, sample_image_path):
        p = ImagePreprocessor(config={"max_size_mb": 0.001})
        from PIL import Image
        img = Image.open(sample_image_path)
        valid, reason = p.validate(img, file_size=10000)
        assert valid is False
        assert "too_large" in reason

    def test_valid(self, sample_image_path):
        p = ImagePreprocessor()
        from PIL import Image
        img = Image.open(sample_image_path)
        valid, reason = p.validate(img)
        assert valid is True
        assert reason == ""


class TestResize:
    def test_no_resize_needed(self, sample_image_path):
        p = ImagePreprocessor()
        from PIL import Image
        img = Image.open(sample_image_path)
        result = p.resize(img)
        assert result.size == (200, 200)

    def test_resizes_large_image(self, large_image_path):
        p = ImagePreprocessor(config={"max_width": 800, "max_height": 800})
        from PIL import Image
        img = Image.open(large_image_path)
        assert img.size == (2048, 2048)
        result = p.resize(img)
        assert result.size[0] <= 800
        assert result.size[1] <= 800

    def test_tracks_resize_stat(self, large_image_path):
        p = ImagePreprocessor(config={"max_width": 800, "max_height": 800})
        from PIL import Image
        img = Image.open(large_image_path)
        p.resize(img)
        assert p.stats["resized"] == 1


class TestToRGB:
    def test_rgb_unchanged(self, sample_image_path):
        p = ImagePreprocessor()
        from PIL import Image
        img = Image.open(sample_image_path)
        assert img.mode == "RGB"
        result = p.to_rgb(img)
        assert result.mode == "RGB"

    def test_converts_palette(self):
        p = ImagePreprocessor()
        img = Image.new("P", (100, 100))
        result = p.to_rgb(img)
        assert result.mode == "RGB"


class TestProcess:
    def test_disabled_returns_none(self, sample_image_path):
        p = ImagePreprocessor(config={"enabled": False})
        assert p.process(sample_image_path) is None

    def test_nonexistent_path(self):
        p = ImagePreprocessor()
        assert p.process("nonexistent.png") is None

    def test_oversized_file(self, sample_image_path):
        p = ImagePreprocessor(config={"max_size_mb": 0.000001})
        result = p.process(sample_image_path)
        assert result is None
        assert p.stats["skipped"] == 1

    def test_success(self, sample_image_path):
        p = ImagePreprocessor()
        result = p.process(sample_image_path)
        assert result is not None
        assert isinstance(result, ImageExample)
        assert result.file_path == sample_image_path
        assert result.width == 200
        assert result.height == 200
        assert result.media_type == MediaType.IMAGE

    def test_small_image_skipped(self, small_image_path):
        p = ImagePreprocessor()
        result = p.process(small_image_path)
        assert result is None

    def test_tracks_stats(self, sample_image_path):
        p = ImagePreprocessor()
        p.process(sample_image_path)
        assert p.stats["loaded"] == 1

    def test_resizes_large(self, large_image_path):
        p = ImagePreprocessor()
        result = p.process(large_image_path, media_type=MediaType.IMAGE)
        assert result is not None
        assert result.width <= p.max_width
        assert result.height <= p.max_height


class TestProcessBatch:
    def test_returns_results_in_order(self, sample_image_path, landscape_image_path):
        p = ImagePreprocessor()
        paths = [sample_image_path, landscape_image_path]
        results = p.process_batch(paths, num_workers=2)
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is not None

    def test_empty_input(self):
        p = ImagePreprocessor()
        assert p.process_batch([]) == []


class TestSaveProcessed:
    def test_saves_image(self, sample_image_path):
        p = ImagePreprocessor()
        from PIL import Image
        img = Image.open(sample_image_path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        try:
            result = p.save_processed(img, out_path)
            assert os.path.exists(result)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_saves_jpeg_with_quality(self, sample_image_path):
        p = ImagePreprocessor(config={"jpeg_quality": 85})
        from PIL import Image
        img = Image.open(sample_image_path).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            out_path = f.name
        try:
            result = p.save_processed(img, out_path, fmt="JPEG")
            assert os.path.exists(result)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestDetectMediaType:
    def test_pdf(self):
        p = ImagePreprocessor()
        assert p.detect_media_type("doc.pdf") == MediaType.PDF

    def test_screenshot_keywords(self):
        p = ImagePreprocessor()
        assert p.detect_media_type("screenshot_001.png") == MediaType.SCREENSHOT
        assert p.detect_media_type("capture.jpg") == MediaType.SCREENSHOT

    def test_code_screenshot(self):
        p = ImagePreprocessor()
        assert p.detect_media_type("code_example.png") == MediaType.CODE_SCREENSHOT
        assert p.detect_media_type("terminal_output.png") == MediaType.CODE_SCREENSHOT

    def test_chart(self):
        p = ImagePreprocessor()
        assert p.detect_media_type("chart_2024.png") == MediaType.CHART
        assert p.detect_media_type("graph_result.png") == MediaType.CHART

    def test_technical_diagram(self):
        p = ImagePreprocessor()
        assert p.detect_media_type("schema_design.png") == MediaType.TECHNICAL_DIAGRAM
        assert p.detect_media_type("network_topology.png") == MediaType.TECHNICAL_DIAGRAM

    def test_generic_image(self):
        p = ImagePreprocessor()
        assert p.detect_media_type("photo.png") == MediaType.IMAGE
        assert p.detect_media_type("image_001.jpg") == MediaType.IMAGE

    def test_case_insensitive(self):
        p = ImagePreprocessor()
        assert p.detect_media_type("SCREENSHOT_001.PNG") == MediaType.SCREENSHOT


class TestStats:
    def test_initial_stats(self):
        p = ImagePreprocessor()
        stats = p.get_stats()
        assert stats["loaded"] == 0
        assert stats["resized"] == 0
        assert stats["failed"] == 0
        assert stats["skipped"] == 0
