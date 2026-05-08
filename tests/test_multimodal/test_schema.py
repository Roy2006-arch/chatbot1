from multimodal.schema import (
    MediaType, Modality, OCRResult, UIAnalysisResult,
    ImageExample, DocumentExample, ScreenshotExample,
    MultimodalExample, PipelineReport,
)


class TestMediaType:
    def test_values(self):
        assert MediaType.IMAGE.value == "image"
        assert MediaType.SCREENSHOT.value == "screenshot"
        assert MediaType.PDF.value == "pdf"
        assert MediaType.CODE_SCREENSHOT.value == "code_screenshot"
        assert MediaType.CHART.value == "chart"
        assert MediaType.TECHNICAL_DIAGRAM.value == "technical_diagram"
        assert MediaType.DOCUMENT.value == "document"


class TestModality:
    def test_values(self):
        assert Modality.TEXT_ONLY.value == "text_only"
        assert Modality.IMAGE_ONLY.value == "image_only"
        assert Modality.TEXT_IMAGE.value == "text_image"
        assert Modality.TEXT_IMAGE_STRUCTURE.value == "text_image_structure"


class TestOCRResult:
    def test_minimal_creation(self):
        r = OCRResult(raw_text="hello world")
        assert r.raw_text == "hello world"
        assert r.confidence == 0.0
        assert r.language == "eng"
        assert r.bounding_boxes == []
        assert r.processing_time == 0.0
        assert r.num_regions == 0

    def test_full_creation(self):
        r = OCRResult(
            raw_text="Hello World",
            confidence=0.95,
            language="eng",
            bounding_boxes=[{"x": 0, "y": 0, "w": 10, "h": 20}],
            processing_time=1.234,
            num_regions=1,
        )
        assert r.confidence == 0.95
        assert r.processing_time == 1.234
        assert len(r.bounding_boxes) == 1

    def test_to_dict(self):
        r = OCRResult(raw_text="test", confidence=0.8)
        d = r.to_dict()
        assert d["raw_text"] == "test"
        assert d["confidence"] == 0.8
        assert d["language"] == "eng"


class TestUIAnalysisResult:
    def test_minimal_creation(self):
        r = UIAnalysisResult()
        assert r.elements == []
        assert r.layout_type == ""
        assert r.element_count == 0
        assert r.complexity_score == 0.0

    def test_full_creation(self):
        r = UIAnalysisResult(
            elements=[{"type": "button"}],
            layout_type="form_layout",
            element_count=1,
            has_buttons=True,
            has_forms=True,
            complexity_score=0.5,
        )
        assert r.has_buttons is True
        assert r.has_forms is True
        assert r.layout_type == "form_layout"

    def test_to_dict(self):
        r = UIAnalysisResult(elements=[{"type": "button"}], layout_type="grid")
        d = r.to_dict()
        assert d["layout_type"] == "grid"
        assert len(d["elements"]) == 1


class TestImageExample:
    def test_minimal_creation(self):
        ex = ImageExample(file_path="/path/to/img.png")
        assert ex.file_path == "/path/to/img.png"
        assert ex.media_type == MediaType.IMAGE
        assert ex.width == 0

    def test_full_creation(self):
        ex = ImageExample(
            file_path="/path/to/img.png",
            original_filename="img.png",
            width=200,
            height=100,
            file_size_bytes=1024,
            format="PNG",
            media_type=MediaType.CHART,
        )
        assert ex.width == 200
        assert ex.height == 100
        assert ex.media_type == MediaType.CHART

    def test_to_dict(self):
        ex = ImageExample(file_path="/path/to/img.png")
        d = ex.to_dict()
        assert d["media_type"] == "image"
        assert d["file_path"] == "/path/to/img.png"


class TestDocumentExample:
    def test_minimal_creation(self):
        d = DocumentExample(file_path="/path/to/doc.pdf")
        assert d.file_path == "/path/to/doc.pdf"
        assert d.ocr_results == []
        assert d.embedded_images == []

    def test_with_ocr_results(self):
        ocr = OCRResult(raw_text="page text", confidence=0.9)
        d = DocumentExample(
            file_path="/path/to/doc.pdf",
            page_count=3,
            total_chars=500,
            ocr_results=[ocr],
        )
        assert d.page_count == 3
        assert d.total_chars == 500
        assert len(d.ocr_results) == 1
        assert d.ocr_results[0].raw_text == "page text"

    def test_to_dict(self):
        ocr = OCRResult(raw_text="hello")
        d = DocumentExample(file_path="doc.pdf", ocr_results=[ocr])
        output = d.to_dict()
        assert output["media_type"] == "document"
        assert len(output["ocr_results"]) == 1


class TestScreenshotExample:
    def test_minimal_creation(self):
        s = ScreenshotExample(file_path="/path/to/ss.png")
        assert s.media_type == MediaType.SCREENSHOT
        assert s.caption == ""

    def test_with_analysis(self):
        ui = UIAnalysisResult(layout_type="form_layout", element_count=5)
        s = ScreenshotExample(
            file_path="/path/to/ss.png",
            ui_analysis=ui,
            caption="A login screen",
        )
        assert s.ui_analysis.layout_type == "form_layout"
        assert s.caption == "A login screen"

    def test_to_dict(self):
        ui = UIAnalysisResult(layout_type="grid")
        s = ScreenshotExample(file_path="ss.png", ui_analysis=ui)
        d = s.to_dict()
        assert d["media_type"] == "screenshot"
        assert d["ui_analysis"]["layout_type"] == "grid"


class TestMultimodalExample:
    def test_minimal_creation(self):
        ex = MultimodalExample(prompt="Describe", response="An image")
        assert ex.prompt == "Describe"
        assert ex.media_type == MediaType.IMAGE
        assert ex.modality == Modality.TEXT_IMAGE
        assert ex.quality_score == 0.0

    def test_full_creation(self):
        ex = MultimodalExample(
            id="ex-001",
            prompt="What is in this image?",
            response="A cat",
            image_path="/path/to/cat.png",
            ocr_text="cat",
            caption="A cute cat",
            media_type=MediaType.IMAGE,
            modality=Modality.TEXT_IMAGE,
            category="animals",
            quality_score=0.95,
        )
        assert ex.id == "ex-001"
        assert ex.quality_score == 0.95
        assert ex.category == "animals"

    def test_to_dict(self):
        ex = MultimodalExample(prompt="Q", response="A")
        d = ex.to_dict()
        assert d["prompt"] == "Q"
        assert d["media_type"] == "image"
        assert d["modality"] == "text_image"

    def test_from_dict_string_enums(self):
        d = {
            "prompt": "Q",
            "response": "A",
            "media_type": "screenshot",
            "modality": "text_image_structure",
        }
        ex = MultimodalExample.from_dict(d)
        assert ex.media_type == MediaType.SCREENSHOT
        assert ex.modality == Modality.TEXT_IMAGE_STRUCTURE

    def test_from_dict_with_ui_analysis(self):
        d = {
            "prompt": "Q",
            "response": "A",
            "ui_analysis": {"layout_type": "grid", "element_count": 3},
        }
        ex = MultimodalExample.from_dict(d)
        assert ex.ui_analysis is not None
        assert ex.ui_analysis.layout_type == "grid"
        assert ex.ui_analysis.element_count == 3

    def test_from_dict_ignores_extra_fields(self):
        d = {"prompt": "Q", "response": "A", "unknown": "ignored"}
        ex = MultimodalExample.from_dict(d)
        assert ex.prompt == "Q"
        assert not hasattr(ex, "unknown")

    def test_roundtrip(self):
        original = MultimodalExample(
            prompt="Describe",
            response="An image description",
            media_type=MediaType.SCREENSHOT,
            modality=Modality.TEXT_IMAGE,
            quality_score=0.8,
        )
        d = original.to_dict()
        restored = MultimodalExample.from_dict(d)
        assert restored.prompt == original.prompt
        assert restored.media_type == original.media_type
        assert restored.modality == original.modality
        assert restored.quality_score == original.quality_score


class TestPipelineReport:
    def test_minimal_creation(self):
        r = PipelineReport(run_id="test-run")
        assert r.run_id == "test-run"
        assert r.total_inputs == 0
        assert r.processing_time == 0.0

    def test_full_creation(self):
        r = PipelineReport(
            run_id="mm-001",
            timestamp="2025-01-01T00:00:00",
            total_inputs=100,
            images_processed=50,
            screenshots_processed=30,
            pdfs_processed=20,
            captions_generated=80,
            qa_pairs_generated=200,
            examples_exported=100,
            avg_ocr_confidence=0.85,
            processing_time=12.5,
        )
        assert r.images_processed == 50
        assert r.examples_exported == 100

    def test_to_dict(self):
        r = PipelineReport(run_id="mm-test")
        d = r.to_dict()
        assert d["run_id"] == "mm-test"
        assert d["total_inputs"] == 0
        assert d["processing_time"] == 0.0
