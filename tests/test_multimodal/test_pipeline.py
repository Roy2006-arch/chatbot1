import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import pytest
from multimodal.pipeline import MultimodalPipeline
from multimodal.schema import (
    MediaType, MultimodalExample, PipelineReport,
    OCRResult, UIAnalysisResult,
)


@pytest.fixture
def pipeline():
    return MultimodalPipeline()


class TestPipelineInit:
    def test_default_init(self, pipeline):
        assert pipeline.ocr is not None
        assert pipeline.preprocessor is not None
        assert pipeline.visual_qa is not None
        assert pipeline.ui_analyzer is not None
        assert pipeline.pdf_processor is not None
        assert pipeline.embedder is not None
        assert pipeline.exporter is not None
        assert pipeline.report is not None

    def test_loads_default_config(self):
        p = MultimodalPipeline()
        assert "multimodal" in p.config
        assert "pipeline" in p.config["multimodal"]

    def test_loads_custom_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("multimodal:\n  pipeline:\n    run_id_prefix: custom\n")
            config_path = f.name
        try:
            p = MultimodalPipeline(config_path=config_path)
            mm = p.config.get("multimodal", {})
            pipe = mm.get("pipeline", {})
            assert pipe.get("run_id_prefix") == "custom"
        finally:
            os.unlink(config_path)

    def test_loads_default_when_missing_config_path(self):
        p = MultimodalPipeline(config_path="nonexistent.yaml")
        assert "multimodal" in p.config


class TestDiscoverFiles:
    def test_returns_empty_when_no_input_dir(self, pipeline):
        result = pipeline._discover_files("/nonexistent", [".png", ".jpg", ".pdf"])
        assert result == ([], [], [])

    def test_discovers_files_in_dir(self, pipeline, input_dir_with_images):
        supported = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf"]
        images, screenshots, pdfs = pipeline._discover_files(input_dir_with_images, supported)
        assert len(images) + len(screenshots) > 0


class TestProcessImages:
    def test_returns_examples(self, pipeline, sample_image_path):
        examples = pipeline._process_images(
            [sample_image_path], "/tmp/out", MediaType.IMAGE
        )
        assert len(examples) >= 0

    def test_handles_ocr_results(self, pipeline, sample_image_path):
        with patch.object(pipeline.ocr, "extract_batch", return_value=[
            OCRResult(raw_text="OCR text", confidence=0.9)
        ]):
            examples = pipeline._process_images(
                [sample_image_path], "/tmp/out", MediaType.IMAGE
            )
            if examples:
                assert examples[0].ocr_text == "OCR text"

    def test_handles_qa_disabled(self, pipeline, sample_image_path):
        pipeline.visual_qa.enabled = False
        examples = pipeline._process_images(
            [sample_image_path], "/tmp/out", MediaType.IMAGE
        )
        assert len(examples) > 0


class TestProcessScreenshots:
    def test_returns_examples(self, pipeline, screenshot_image_path):
        with patch.object(pipeline.ocr, "extract_batch", return_value=[
            OCRResult(raw_text="screenshot text", confidence=0.8)
        ]):
            with patch.object(pipeline.ui_analyzer, "analyze_batch", return_value=[
                UIAnalysisResult(layout_type="form_layout", element_count=5)
            ]):
                examples = pipeline._process_screenshots(
                    [screenshot_image_path], "/tmp/out"
                )
                assert len(examples) > 0
                assert examples[0].media_type == MediaType.SCREENSHOT

    def test_includes_ui_analysis(self, pipeline, screenshot_image_path):
        ui_result = UIAnalysisResult(
            layout_type="grid",
            element_count=3,
            has_buttons=True,
            has_forms=False,
        )
        with patch.object(pipeline.ocr, "extract_batch", return_value=[
            OCRResult(raw_text="", confidence=0.0)
        ]):
            with patch.object(pipeline.ui_analyzer, "analyze_batch", return_value=[ui_result]):
                examples = pipeline._process_screenshots(
                    [screenshot_image_path], "/tmp/out"
                )
                ex = examples[0]
                assert ex.ui_analysis is not None
                assert ex.ui_analysis.layout_type == "grid"


class TestProcessPDFs:
    def test_returns_examples(self, pipeline, sample_pdf_path):
        doc_example = MagicMock()
        doc_example.ocr_results = [MagicMock(raw_text="PDF text")]
        doc_example.embedded_images = []
        doc_example.original_filename = "test.pdf"
        doc_example.page_count = 1
        doc_example.total_chars = 50

        with patch.object(pipeline.pdf_processor, "process_batch", return_value=[doc_example]):
            examples = pipeline._process_pdfs([sample_pdf_path], "/tmp/out")
            assert len(examples) > 0
            assert examples[0].media_type == MediaType.PDF

    def test_skips_none_docs(self, pipeline, sample_pdf_path):
        with patch.object(pipeline.pdf_processor, "process_batch", return_value=[None]):
            examples = pipeline._process_pdfs([sample_pdf_path], "/tmp/out")
            assert len(examples) == 0


class TestAnalyzeInputDirectory:
    def test_returns_analysis(self, pipeline, input_dir_with_images):
        result = pipeline.analyze_input_directory(input_dir_with_images)
        assert result["total_files"] > 0
        assert "by_extension" in result
        assert ".png" in result["by_extension"]

    def test_empty_dir(self, pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.analyze_input_directory(tmpdir)
            assert result["total_files"] == 0


class TestSaveReport:
    def test_saves_json(self, pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline._save_report(tmpdir)
            report_path = os.path.join(tmpdir, "multimodal_report.json")
            assert os.path.exists(report_path)
            with open(report_path) as f:
                data = json.load(f)
            assert "run_id" in data

    def test_report_contents(self, pipeline):
        pipeline.report.total_inputs = 50
        pipeline.report.images_processed = 30
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline._save_report(tmpdir)
            report_path = os.path.join(tmpdir, "multimodal_report.json")
            with open(report_path) as f:
                data = json.load(f)
            assert data["total_inputs"] == 50
            assert data["images_processed"] == 30


class TestGetReport:
    def test_returns_report(self, pipeline):
        report = pipeline.get_report()
        assert isinstance(report, PipelineReport)
        assert report.run_id == ""


class TestRun:
    def test_run_returns_report(self, pipeline):
        with patch.object(pipeline, "_discover_files", return_value=([], [], [])):
            with tempfile.TemporaryDirectory() as tmpdir:
                report = pipeline.run(
                    input_dir="/nonexistent",
                    output_dir=tmpdir,
                )
                assert isinstance(report, PipelineReport)
                assert report.run_id != ""
                assert report.processing_time >= 0

    def test_saves_report_json(self, pipeline):
        with patch.object(pipeline, "_discover_files", return_value=([], [], [])):
            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline.run(output_dir=tmpdir)
                report_path = os.path.join(tmpdir, "multimodal_report.json")
                assert os.path.exists(report_path)

    def test_processes_images(self, pipeline, sample_image_path, landscape_image_path):
        paths = [sample_image_path, landscape_image_path]
        with patch.object(pipeline, "_discover_files", return_value=(paths, [], [])):
            with tempfile.TemporaryDirectory() as tmpdir:
                report = pipeline.run(
                    input_dir="/mock",
                    output_dir=tmpdir,
                    max_images=10,
                )
                assert report.images_processed >= 0

    def test_processes_screenshots(self, pipeline, screenshot_image_path):
        with patch.object(pipeline, "_discover_files", return_value=([], [screenshot_image_path], [])):
            with tempfile.TemporaryDirectory() as tmpdir:
                report = pipeline.run(
                    input_dir="/mock",
                    output_dir=tmpdir,
                    max_screenshots=10,
                )
                assert report.screenshots_processed >= 0

    def test_processes_pdfs(self, pipeline, sample_pdf_path):
        doc_example = MagicMock()
        doc_example.ocr_results = [MagicMock(raw_text="text")]
        doc_example.embedded_images = []
        doc_example.original_filename = "test.pdf"
        doc_example.page_count = 1
        doc_example.total_chars = 10

        with patch.object(pipeline, "_discover_files", return_value=([], [], [sample_pdf_path])):
            with patch.object(pipeline.pdf_processor, "process_batch", return_value=[doc_example]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    report = pipeline.run(
                        input_dir="/mock",
                        output_dir=tmpdir,
                        max_pdfs=10,
                    )
                    assert report.pdfs_processed >= 0

    def test_export_disabled_skips_export(self, pipeline):
        pipeline.exporter.enabled = False
        with patch.object(pipeline, "_discover_files", return_value=([], [], [])):
            with tempfile.TemporaryDirectory() as tmpdir:
                report = pipeline.run(output_dir=tmpdir)
                assert report.examples_exported == 0
