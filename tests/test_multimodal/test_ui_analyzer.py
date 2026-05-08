import os
import pytest
from unittest.mock import patch, MagicMock
from multimodal.ui_analyzer import UIAnalyzer
from multimodal.schema import UIAnalysisResult


class TestUIAnalyzerInit:
    def test_default_config(self):
        ua = UIAnalyzer()
        assert ua.enabled is True
        assert ua.min_element_area == 16
        assert ua.max_elements == 50

    def test_custom_config(self):
        ua = UIAnalyzer(config={"enabled": False, "min_element_area": 100})
        assert ua.enabled is False
        assert ua.min_element_area == 100


class TestAnalyze:
    def test_disabled_returns_empty(self, sample_image_path):
        ua = UIAnalyzer(config={"enabled": False})
        result = ua.analyze(sample_image_path)
        assert result.element_count == 0
        assert result.layout_type == ""

    def test_uses_cv_when_available(self, sample_image_path):
        ua = UIAnalyzer()
        ua._contour_available = True
        mock_result = UIAnalysisResult(
            elements=[{"type": "button"}], layout_type="grid", element_count=1
        )
        with patch.object(ua, "_analyze_cv", return_value=mock_result) as mock_cv:
            with patch.object(ua, "_analyze_text_based") as mock_txt:
                result = ua.analyze(sample_image_path, "button text")
                mock_cv.assert_called_once()
                mock_txt.assert_not_called()
                assert result.layout_type == "grid"

    def test_falls_back_to_text_based(self, sample_image_path):
        ua = UIAnalyzer()
        with patch.object(ua, "_check_deps", return_value=False):
            with patch.object(ua, "_analyze_text_based", return_value=UIAnalysisResult(elements=[], layout_type="wide_layout")) as mock_txt:
                result = ua.analyze(sample_image_path, "")
                mock_txt.assert_called_once()


class TestAnalyzeTextBased:
    def test_detects_buttons_from_ocr(self):
        ua = UIAnalyzer()
        result = ua._analyze_text_based("Click the submit button", "image.png")
        assert result.has_buttons is True
        assert result.element_count > 0

    def test_detects_forms_from_ocr(self):
        ua = UIAnalyzer()
        result = ua._analyze_text_based("Login form with username", "image.png")
        assert result.has_forms is True

    def test_detects_images_from_ocr(self):
        ua = UIAnalyzer()
        result = ua._analyze_text_based("User avatar image", "image.png")
        assert result.has_images is True

    def test_filename_based_detection(self):
        ua = UIAnalyzer()
        result = ua._analyze_text_based("", "login_screenshot.png")
        assert result.has_forms is True

    def test_wide_layout_from_dimensions(self, landscape_image_path):
        ua = UIAnalyzer()
        result = ua._analyze_text_based("", landscape_image_path)
        assert result.layout_type == "wide_layout"

    def test_empty_ocr_and_generic_name(self, sample_image_path):
        ua = UIAnalyzer()
        result = ua._analyze_text_based("", sample_image_path)
        assert result.element_count == 0

    def test_complexity_scoring(self):
        ua = UIAnalyzer()
        ocr = "button submit form login input image heading link list menu card"
        result = ua._analyze_text_based(ocr, "test.png")
        assert result.complexity_score > 0

    def test_metadata_analysis_method(self):
        ua = UIAnalyzer()
        result = ua._analyze_text_based("submit button", "test.png")
        assert result.metadata["analysis_method"] == "text_based"


class TestClassifyLayout:
    def test_wide_by_dimensions(self):
        ua = UIAnalyzer()
        assert ua._classify_layout(800, 400, []) == "wide_layout"

    def test_tall_by_dimensions(self):
        ua = UIAnalyzer()
        assert ua._classify_layout(400, 800, []) == "tall_layout"

    def test_balanced_by_dimensions(self):
        ua = UIAnalyzer()
        assert ua._classify_layout(500, 500, []) == "balanced_layout"

    def test_left_navigation(self):
        ua = UIAnalyzer()
        elements = [
            {"x": 0, "w": 50, "y": 0, "h": 50},
            {"x": 5, "w": 40, "y": 60, "h": 50},
            {"x": 200, "w": 100, "y": 0, "h": 50},
        ]
        result = ua._classify_layout(800, 600, elements)
        assert result == "left_navigation"

    def test_centered_layout(self):
        ua = UIAnalyzer()
        elements = [
            {"x": 350, "w": 100, "y": 0, "h": 50},
            {"x": 360, "w": 80, "y": 60, "h": 50},
        ]
        result = ua._classify_layout(800, 600, elements)
        assert result == "centered_layout"


class TestClassifyLayoutFromKeywords:
    def test_form_layout(self):
        ua = UIAnalyzer()
        result = ua._classify_layout_from_keywords([{"type": "form"}, {"type": "button"}])
        assert result == "form_layout"

    def test_navigation_layout(self):
        ua = UIAnalyzer()
        result = ua._classify_layout_from_keywords([{"type": "list"}, {"type": "nav"}])
        assert result == "navigation_layout"

    def test_card_grid(self):
        ua = UIAnalyzer()
        result = ua._classify_layout_from_keywords([{"type": "card"}])
        assert result == "card_grid"

    def test_content_layout(self):
        ua = UIAnalyzer()
        result = ua._classify_layout_from_keywords([{"type": "heading"}])
        assert result == "content_layout"

    def test_mixed_layout(self):
        ua = UIAnalyzer()
        result = ua._classify_layout_from_keywords([{"type": "image"}, {"type": "input"}])
        assert result == "mixed_layout"


class TestAnalyzeBatch:
    def test_disabled_returns_empty_results(self):
        ua = UIAnalyzer(config={"enabled": False})
        results = ua.analyze_batch(["a.png", "b.png"])
        assert len(results) == 2
        assert all(r.element_count == 0 for r in results)

    def test_empty_input(self):
        ua = UIAnalyzer()
        assert ua.analyze_batch([]) == []


class TestStats:
    def test_initial_stats(self):
        ua = UIAnalyzer()
        assert ua.get_stats()["analyzed"] == 0
        assert ua.get_stats()["elements_found"] == 0
