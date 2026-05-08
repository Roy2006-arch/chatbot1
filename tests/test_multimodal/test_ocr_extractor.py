import os
import sys
from unittest.mock import patch, MagicMock
import pytest
from multimodal.ocr_extractor import OCRExtractor
from multimodal.schema import OCRResult


def _make_mock_pytesseract():
    mock = MagicMock()
    mock.get_tesseract_version.return_value = "5.0"
    mock.image_to_string.return_value = "Hello World"
    mock.image_to_data.return_value = {
        "text": ["Hello", "World", ""],
        "conf": [95, 85, -1],
        "left": [0, 50, 0],
        "top": [0, 20, 0],
        "width": [40, 60, 0],
        "height": [15, 15, 0],
    }
    mock.Output.DICT = "dict"
    return mock


def _make_mock_easyocr():
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0, 0], [50, 0], [50, 15], [0, 15]], "Hello", 0.95),
        ([[50, 20], [100, 20], [100, 35], [50, 35]], "World", 0.85),
    ]
    mock_module = MagicMock()
    mock_module.Reader.return_value = mock_reader
    return mock_module


class TestOCRExtractorInit:
    def test_default_config(self):
        ocr = OCRExtractor()
        assert ocr.enabled is True
        assert ocr.languages == "eng"
        assert ocr.min_confidence == 0.3
        assert ocr.preprocess_image is True
        assert ocr.cache_results is True

    def test_custom_config(self):
        ocr = OCRExtractor(config={"enabled": False, "languages": "fra", "min_confidence": 0.5})
        assert ocr.enabled is False
        assert ocr.languages == "fra"
        assert ocr.min_confidence == 0.5


class TestCheckDeps:
    def test_tesseract_available(self):
        mock_ts = _make_mock_pytesseract()
        with patch.dict("sys.modules", {"pytesseract": mock_ts}):
            ocr = OCRExtractor()
            assert ocr.available is True
            assert ocr._tesseract_available is True

    def test_no_deps(self):
        ocr = OCRExtractor()
        with patch.dict("sys.modules", {"pytesseract": None, "easyocr": None}):
            ocr._check_deps()
            assert ocr._tesseract_available is False
            assert ocr._easyocr_available is False


class TestExtractFromImage:
    def test_disabled_returns_none(self, sample_image_path):
        ocr = OCRExtractor(config={"enabled": False})
        assert ocr.extract_from_image(sample_image_path) is None

    def test_nonexistent_path(self):
        ocr = OCRExtractor()
        assert ocr.extract_from_image("nonexistent.png") is None

    def test_caches_result(self, sample_image_path):
        ocr = OCRExtractor()
        with patch.object(ocr, "_extract_fallback", return_value=OCRResult(raw_text="cached", confidence=0.5)) as mock:
            r1 = ocr.extract_from_image(sample_image_path)
            r2 = ocr.extract_from_image(sample_image_path)
            assert mock.call_count == 1
            assert r1.raw_text == "cached"
            assert r2.raw_text == "cached"

    def test_uses_tesseract_when_available(self, sample_image_path):
        ocr = OCRExtractor()
        ocr._tesseract_available = True
        mock_result = OCRResult(raw_text="tesseract output", confidence=0.9)
        with patch.object(ocr, "_extract_tesseract", return_value=mock_result) as mock_ts:
            with patch.object(ocr, "_extract_easyocr") as mock_eo:
                result = ocr.extract_from_image(sample_image_path)
                mock_ts.assert_called_once()
                mock_eo.assert_not_called()
                assert result.raw_text == "tesseract output"

    def test_uses_easyocr_when_no_tesseract(self, sample_image_path):
        ocr = OCRExtractor()
        ocr._tesseract_available = False
        ocr._easyocr_available = True
        mock_result = OCRResult(raw_text="easyocr output", confidence=0.8)
        with patch.object(ocr, "_extract_easyocr", return_value=mock_result) as mock_eo:
            with patch.object(ocr, "_extract_fallback") as mock_fb:
                result = ocr.extract_from_image(sample_image_path)
                mock_eo.assert_called_once()
                mock_fb.assert_not_called()
                assert result.raw_text == "easyocr output"

    def test_fallback_when_no_ocr(self, sample_image_path):
        ocr = OCRExtractor()
        ocr._tesseract_available = False
        ocr._easyocr_available = False
        mock_result = OCRResult(raw_text="", confidence=0.0)
        with patch.object(ocr, "_extract_fallback", return_value=mock_result) as mock_fb:
            result = ocr.extract_from_image(sample_image_path)
            mock_fb.assert_called_once()
            assert result.raw_text == ""

    def test_sets_processing_time(self, sample_image_path):
        ocr = OCRExtractor()
        ocr._tesseract_available = False
        ocr._easyocr_available = False
        with patch.object(ocr, "_extract_fallback", return_value=OCRResult(raw_text="test", confidence=0.5)):
            result = ocr.extract_from_image(sample_image_path)
            assert result.processing_time >= 0


class TestExtractTesseract:
    def test_success(self, sample_image_path):
        ocr = OCRExtractor()
        mock_ts = _make_mock_pytesseract()
        with patch.dict("sys.modules", {"pytesseract": mock_ts}):
            result = ocr._extract_tesseract(sample_image_path)
            assert result is not None
            assert result.raw_text == "Hello World"
            assert result.confidence > 0
            assert len(result.bounding_boxes) == 2

    def test_failure_returns_none(self, sample_image_path):
        ocr = OCRExtractor()
        mock_ts = MagicMock()
        mock_ts.image_to_string.side_effect = Exception("no tesseract")
        with patch.dict("sys.modules", {"pytesseract": mock_ts}):
            result = ocr._extract_tesseract(sample_image_path)
            assert result is None


class TestExtractEasyOCR:
    def test_success(self, sample_image_path):
        ocr = OCRExtractor()
        mock_eo = _make_mock_easyocr()
        with patch.dict("sys.modules", {"easyocr": mock_eo}):
            result = ocr._extract_easyocr(sample_image_path)
            assert result is not None
            assert "Hello" in result.raw_text
            assert result.confidence > 0
            assert len(result.bounding_boxes) == 2

    def test_failure_returns_none(self, sample_image_path):
        ocr = OCRExtractor()
        mock_eo = MagicMock()
        mock_eo.Reader.side_effect = Exception("no easyocr")
        with patch.dict("sys.modules", {"easyocr": mock_eo}):
            result = ocr._extract_easyocr(sample_image_path)
            assert result is None


class TestExtractFallback:
    def test_returns_empty_result(self, sample_image_path):
        ocr = OCRExtractor()
        result = ocr._extract_fallback(sample_image_path)
        assert result is not None
        assert result.raw_text == ""
        assert result.confidence == 0.0
        assert result.bounding_boxes == []

    def test_nonexistent_path(self):
        ocr = OCRExtractor()
        result = ocr._extract_fallback("nonexistent.png")
        assert result is None


class TestExtractText:
    def test_returns_raw_text(self, sample_image_path):
        ocr = OCRExtractor()
        with patch.object(ocr, "extract_from_image", return_value=OCRResult(raw_text="extracted text", confidence=0.5)):
            assert ocr.extract_text(sample_image_path) == "extracted text"

    def test_returns_empty_on_failure(self, sample_image_path):
        ocr = OCRExtractor()
        with patch.object(ocr, "extract_from_image", return_value=None):
            assert ocr.extract_text(sample_image_path) == ""


class TestExtractBatch:
    def test_returns_all_results(self, sample_image_path, small_image_path):
        ocr = OCRExtractor()
        paths = [sample_image_path, small_image_path]
        with patch.object(ocr, "_extract_fallback", return_value=OCRResult(raw_text="text", confidence=0.5)):
            results = ocr.extract_batch(paths, num_workers=2)
            assert len(results) == 2
            assert all(r is not None for r in results)

    def test_disabled_returns_nones(self):
        ocr = OCRExtractor(config={"enabled": False})
        assert ocr.extract_batch(["a.png", "b.png"]) == [None, None]

    def test_empty_input(self):
        ocr = OCRExtractor()
        assert ocr.extract_batch([]) == []


class TestPreprocess:
    def test_preprocess_dark_image(self, sample_image_path):
        ocr = OCRExtractor()
        from PIL import Image
        img = Image.open(sample_image_path)
        processed = ocr._preprocess(img)
        assert processed is not None
        assert processed.size == img.size


class TestStats:
    def test_initial_stats(self):
        ocr = OCRExtractor()
        stats = ocr.get_stats()
        assert stats["tesseract_available"] is False
        assert stats["easyocr_available"] is False
        assert stats["cache_size"] == 0
