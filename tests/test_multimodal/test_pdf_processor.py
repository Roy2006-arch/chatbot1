import os
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock
import pytest
import types
from multimodal.pdf_processor import PDFProcessor, PDFPageResult
from multimodal.schema import DocumentExample


def _make_mock_fitz():
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Hello from page 1"
    mock_page.get_images.return_value = []

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.__enter__.return_value = mock_doc

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    return mock_fitz


class TestPDFProcessorInit:
    def test_default_config(self):
        p = PDFProcessor()
        assert p.enabled is True
        assert p.extract_text is True
        assert p.extract_images is True
        assert p.max_pages == 50
        assert p.min_page_chars == 10

    def test_custom_config(self):
        p = PDFProcessor(config={"enabled": False, "max_pages": 10})
        assert p.enabled is False
        assert p.max_pages == 10


class TestCheckDeps:
    def test_both_unavailable(self):
        p = PDFProcessor()
        with patch.dict("sys.modules", {"pdfminer": None, "fitz": None}):
            p._check_deps()
            assert p._pdfminer_available is False
            assert p._pymupdf_available is False

    def test_pymupdf_available(self):
        p = PDFProcessor()
        mock_fitz = MagicMock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            p._check_deps()
            assert p._pymupdf_available is True

    def test_pdfminer_available(self):
        p = PDFProcessor()
        mock_pdfminer = MagicMock()
        with patch.dict("sys.modules", {"pdfminer": mock_pdfminer}):
            p._check_deps()
            assert p._pdfminer_available is True


class TestProcess:
    def test_disabled_returns_none(self, sample_pdf_path):
        p = PDFProcessor(config={"enabled": False})
        assert p.process(sample_pdf_path) is None

    def test_nonexistent_path(self):
        p = PDFProcessor()
        assert p.process("nonexistent.pdf") is None

    def test_uses_pymupdf_when_available(self, sample_pdf_path):
        p = PDFProcessor()
        p._pymupdf_available = True
        p._pdfminer_available = False
        mock_result = DocumentExample(
            file_path=sample_pdf_path,
            original_filename=os.path.basename(sample_pdf_path),
            page_count=1,
            total_chars=50,
        )
        with patch.object(p, "_check_deps"):
            with patch.object(p, "_process_pymupdf", return_value=mock_result) as mock_pym:
                with patch.object(p, "_process_pdfminer") as mock_pm:
                    result = p.process(sample_pdf_path)
                    mock_pym.assert_called_once()
                    mock_pm.assert_not_called()
                    assert result is not None

    def test_uses_pdfminer_when_no_pymupdf(self, sample_pdf_path):
        p = PDFProcessor()
        p._pymupdf_available = False
        p._pdfminer_available = True
        mock_result = DocumentExample(
            file_path=sample_pdf_path,
            original_filename=os.path.basename(sample_pdf_path),
            page_count=1,
            total_chars=30,
        )
        with patch.object(p, "_check_deps"):
            with patch.object(p, "_process_pdfminer", return_value=mock_result) as mock_pm:
                with patch.object(p, "_process_fallback") as mock_fb:
                    result = p.process(sample_pdf_path)
                    mock_pm.assert_called_once()
                    mock_fb.assert_not_called()

    def test_fallback_when_no_pdf_libs(self, sample_pdf_path):
        p = PDFProcessor()
        p._pymupdf_available = False
        p._pdfminer_available = False
        mock_result = DocumentExample(
            file_path=sample_pdf_path, original_filename="test.pdf", page_count=1
        )
        with patch.object(p, "_process_fallback", return_value=mock_result) as mock_fb:
            result = p.process(sample_pdf_path)
            mock_fb.assert_called_once()


class TestProcessPymupdf:
    def test_success(self, sample_pdf_path):
        p = PDFProcessor()
        mock_fitz = _make_mock_fitz()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = p._process_pymupdf(sample_pdf_path)
            assert result is not None
            assert result.page_count == 1
            assert result.total_chars > 0

    def test_empty_text_page(self, sample_pdf_path):
        p = PDFProcessor(config={"min_page_chars": 10})
        mock_fitz = _make_mock_fitz()
        mock_fitz.open.return_value.__getitem__.return_value.get_text.return_value = "short"
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = p._process_pymupdf(sample_pdf_path)
            assert result is not None
            assert result.total_chars == 0

    def test_extracts_images(self, sample_pdf_path):
        p = PDFProcessor()
        mock_fitz = _make_mock_fitz()
        mock_page = mock_fitz.open.return_value.__getitem__.return_value
        mock_page.get_text.return_value = "Text"
        mock_page.get_images.return_value = [(1, 0, 0, 0, 0, 0, 0)]
        mock_fitz.open.return_value.extract_image.return_value = {"image": b"fake_image_bytes", "ext": "png"}
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = p._process_pymupdf(sample_pdf_path)
            assert result is not None
            assert len(result.embedded_images) == 1
            assert result.embedded_images[0].format == "PNG"

    def test_exception_returns_none(self, sample_pdf_path):
        p = PDFProcessor()
        with patch.dict("sys.modules", {"fitz": None}):
            result = p._process_pymupdf(sample_pdf_path)
            assert result is None
            assert p.stats["failed"] == 1


class TestProcessPdfminer:
    def _make_pdfminer_mock(self, return_value="Page 1 text\fPage 2 text", side_effect=None):
        mock_hl = types.ModuleType("pdfminer.high_level")
        if side_effect:
            mock_hl.extract_text = MagicMock(side_effect=side_effect)
        else:
            mock_hl.extract_text = MagicMock(return_value=return_value)
        mock_hl.extract_pages = MagicMock(return_value=[])
        mock_pm = types.ModuleType("pdfminer")
        mock_pm.high_level = mock_hl
        mock_pm.layout = types.ModuleType("pdfminer.layout")
        mock_pm.layout.LTTextBox = object
        mock_pm.layout.LTFigure = object
        mock_pm.layout.LTImage = object
        return mock_pm

    @pytest.fixture
    def pdfminer_mock_modules(self, request):
        mock_pm = self._make_pdfminer_mock()
        with patch.dict("sys.modules", {
            "pdfminer": mock_pm,
            "pdfminer.high_level": mock_pm.high_level,
            "pdfminer.layout": mock_pm.layout,
        }):
            yield

    def test_success(self, sample_pdf_path, pdfminer_mock_modules):
        p = PDFProcessor()
        result = p._process_pdfminer(sample_pdf_path)
        assert result is not None
        assert result.page_count >= 2
        assert result.total_chars > 0

    def test_empty_text_returns_none(self, sample_pdf_path):
        p = PDFProcessor()
        with patch.dict("sys.modules", {"pdfminer": None}):
            result = p._process_pdfminer(sample_pdf_path)
            assert result is None

    def test_exception_returns_none(self, sample_pdf_path):
        p = PDFProcessor()
        mock_pm = self._make_pdfminer_mock(side_effect=Exception("no pdfminer"))
        with patch.dict("sys.modules", {
            "pdfminer": mock_pm,
            "pdfminer.high_level": mock_pm.high_level,
            "pdfminer.layout": mock_pm.layout,
        }):
            result = p._process_pdfminer(sample_pdf_path)
            assert result is None


class TestProcessFallback:
    def test_valid_pdf(self, sample_pdf_path):
        p = PDFProcessor()
        result = p._process_fallback(sample_pdf_path)
        assert result is not None
        assert result.page_count >= 0
        assert result.total_chars == 0

    def test_invalid_pdf(self):
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        with open(path, "w") as f:
            f.write("not a pdf")
        try:
            p = PDFProcessor()
            result = p._process_fallback(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_exception_returns_none(self, sample_pdf_path):
        p = PDFProcessor()
        with patch("builtins.open", side_effect=Exception("no file")):
            result = p._process_fallback(sample_pdf_path)
            assert result is None


class TestExtractTextFromPDF:
    def test_returns_text(self, sample_pdf_path):
        p = PDFProcessor()
        mock_doc = DocumentExample(
            file_path=sample_pdf_path,
            original_filename="test.pdf",
            ocr_results=[MagicMock(raw_text="Page text")],
        )
        with patch.object(p, "process", return_value=mock_doc):
            text = p.extract_text_from_pdf(sample_pdf_path)
            assert "Page text" in text

    def test_empty_on_failure(self, sample_pdf_path):
        p = PDFProcessor()
        with patch.object(p, "process", return_value=None):
            assert p.extract_text_from_pdf(sample_pdf_path) == ""


class TestProcessBatch:
    def test_returns_results(self, sample_pdf_path):
        p = PDFProcessor()
        mock_fitz = _make_mock_fitz()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            results = p.process_batch([sample_pdf_path], num_workers=1)
            assert len(results) == 1
            assert results[0] is not None

    def test_empty_input(self):
        p = PDFProcessor()
        assert p.process_batch([]) == []


class TestStats:
    def test_initial_stats(self):
        p = PDFProcessor()
        stats = p.get_stats()
        assert stats["pdfs"] == 0
        assert stats["pages"] == 0
        assert stats["images_extracted"] == 0
        assert stats["failed"] == 0
