import os
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .schema import DocumentExample, ImageExample, OCRResult, MediaType


@dataclass
class PDFPageResult:
    page_num: int
    text: str
    images: List[ImageExample] = None
    ocr: Optional[OCRResult] = None

    def __post_init__(self):
        if self.images is None:
            self.images = []


class PDFProcessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.extract_text = self.config.get("extract_text", True)
        self.extract_images = self.config.get("extract_images", True)
        self.min_page_chars = self.config.get("min_page_chars", 10)
        self.max_pages = self.config.get("max_pages", 50)
        self.image_dpi = self.config.get("image_dpi", 200)
        self._pdfminer_available = False
        self._pymupdf_available = False
        self.stats = {"pdfs": 0, "pages": 0, "images_extracted": 0, "failed": 0}

    def _check_deps(self) -> Tuple[bool, bool]:
        try:
            import pdfminer
            self._pdfminer_available = True
        except ImportError:
            self._pdfminer_available = False
        try:
            import fitz
            self._pymupdf_available = True
        except ImportError:
            self._pymupdf_available = False
        return self._pdfminer_available, self._pymupdf_available

    @property
    def available(self) -> bool:
        self._check_deps()
        return self._pdfminer_available or self._pymupdf_available

    def process(self, pdf_path: str) -> Optional[DocumentExample]:
        if not self.enabled or not os.path.exists(pdf_path):
            return None
        self._check_deps()

        if self._pymupdf_available:
            return self._process_pymupdf(pdf_path)
        elif self._pdfminer_available:
            return self._process_pdfminer(pdf_path)
        return self._process_fallback(pdf_path)

    def _process_pymupdf(self, pdf_path: str) -> Optional[DocumentExample]:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            pages_to_read = min(len(doc), self.max_pages)

            ocr_results = []
            all_images = []
            total_chars = 0

            for page_num in range(pages_to_read):
                page = doc[page_num]
                text = page.get_text("text").strip() if self.extract_text else ""

                if len(text) < self.min_page_chars:
                    text = ""

                page_images = []
                if self.extract_images:
                    for img_index, img_info in enumerate(page.get_images(full=True)):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            img_bytes = base_image["image"]
                            img_ext = base_image["ext"]

                            out_dir = os.path.join(os.path.dirname(pdf_path), "_extracted")
                            os.makedirs(out_dir, exist_ok=True)
                            img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num}_{img_index}.{img_ext}"
                            img_path = os.path.join(out_dir, img_filename)

                            with open(img_path, "wb") as f:
                                f.write(img_bytes)

                            ie = ImageExample(
                                file_path=img_path,
                                original_filename=img_filename,
                                media_type=MediaType.IMAGE,
                                file_size_bytes=len(img_bytes),
                                format=img_ext.upper(),
                            )
                            page_images.append(ie)
                        except Exception:
                            continue

                total_chars += len(text)
                all_images.extend(page_images)

                ocr_result = OCRResult(
                    raw_text=text,
                    confidence=1.0,
                    num_regions=max(1, len(text) // 100) if text else 0,
                )
                ocr_results.append(ocr_result)

            doc.close()
            self.stats["pdfs"] += 1
            self.stats["pages"] += pages_to_read
            self.stats["images_extracted"] += len(all_images)

            return DocumentExample(
                file_path=pdf_path,
                original_filename=os.path.basename(pdf_path),
                page_count=pages_to_read,
                total_chars=total_chars,
                ocr_results=ocr_results,
                embedded_images=all_images,
            )
        except Exception:
            self.stats["failed"] += 1
            return None

    def _process_pdfminer(self, pdf_path: str) -> Optional[DocumentExample]:
        try:
            from pdfminer.high_level import extract_text, extract_pages
            from pdfminer.layout import LTTextBox, LTFigure, LTImage

            pages_to_read = self.max_pages
            ocr_results = []
            all_images = []
            total_chars = 0

            full_text = extract_text(pdf_path)
            if not full_text:
                return None

            text_pages = full_text.split("\f")
            pages_to_read = min(len(text_pages), self.max_pages)

            for i in range(pages_to_read):
                text = text_pages[i].strip()
                if len(text) < self.min_page_chars:
                    text = ""
                total_chars += len(text)

                ocr_results.append(OCRResult(
                    raw_text=text,
                    confidence=1.0,
                    num_regions=max(1, len(text) // 100) if text else 0,
                ))

            self.stats["pdfs"] += 1
            self.stats["pages"] += pages_to_read

            return DocumentExample(
                file_path=pdf_path,
                original_filename=os.path.basename(pdf_path),
                page_count=pages_to_read,
                total_chars=total_chars,
                ocr_results=ocr_results,
                embedded_images=all_images,
            )
        except Exception:
            self.stats["failed"] += 1
            return None

    def _process_fallback(self, pdf_path: str) -> Optional[DocumentExample]:
        try:
            with open(pdf_path, "rb") as f:
                header = f.read(1024)
            is_pdf = header[:5] == b"%PDF-"
            if not is_pdf:
                return None

            match = re.search(rb"/Type\s*/Pages[^>]*?/Count\s*(\d+)", header)
            page_count = int(match.group(1)) if match else 1
            page_count = min(page_count, self.max_pages)

            ocr_results = [OCRResult(raw_text="", confidence=0.0) for _ in range(page_count)]

            self.stats["pdfs"] += 1
            self.stats["pages"] += page_count

            return DocumentExample(
                file_path=pdf_path,
                original_filename=os.path.basename(pdf_path),
                page_count=page_count,
                total_chars=0,
                ocr_results=ocr_results,
            )
        except Exception:
            self.stats["failed"] += 1
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        doc = self.process(pdf_path)
        if not doc:
            return ""
        return "\n".join(r.raw_text for r in doc.ocr_results if r)

    def process_batch(
        self, pdf_paths: List[str], num_workers: int = 2
    ) -> List[Optional[DocumentExample]]:
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process, p): i for i, p in enumerate(pdf_paths)}
            for future in as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def get_stats(self) -> Dict:
        return self.stats
