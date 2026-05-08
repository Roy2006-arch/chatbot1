import json
import os
import time
import glob
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime

from .schema import (
    MediaType, MultimodalExample, PipelineReport,
)
from .ocr_extractor import OCRExtractor
from .image_preprocessor import ImagePreprocessor
from .visual_qa import VisualQAGenerator
from .ui_analyzer import UIAnalyzer
from .pdf_processor import PDFProcessor
from .embedder import MultimodalEmbedder
from .exporters import MultimodalExporter


class MultimodalPipeline:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.ocr = OCRExtractor(self.config.get("ocr", {}))
        self.preprocessor = ImagePreprocessor(self.config.get("image_preprocessor", {}))
        self.visual_qa = VisualQAGenerator(self.config.get("visual_qa", {}))
        self.ui_analyzer = UIAnalyzer(self.config.get("ui_analyzer", {}))
        self.pdf_processor = PDFProcessor(self.config.get("pdf_processor", {}))
        self.embedder = MultimodalEmbedder(self.config.get("embedder", {}))
        self.exporter = MultimodalExporter(self.config.get("exporters", {}))
        self.report = PipelineReport(run_id="", timestamp=datetime.utcnow().isoformat())

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        pkg_dir = Path(__file__).parent
        default = pkg_dir / "config.yaml"
        if default.exists():
            import yaml
            with open(default, "r") as f:
                return yaml.safe_load(f)
        return {}

    def run(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_images: int = 500,
        max_screenshots: int = 500,
        max_pdfs: int = 100,
    ) -> PipelineReport:
        mm_config = self.config.get("multimodal", self.config)
        pipeline_cfg = mm_config.get("pipeline", {})

        input_dir = input_dir or pipeline_cfg.get("input_dir", "multimodal_data/input")
        output_dir = output_dir or pipeline_cfg.get("output_dir", "multimodal_data/output")
        max_images = max_images or pipeline_cfg.get("max_images", 500)
        max_screenshots = max_screenshots or pipeline_cfg.get("max_screenshots", 500)
        max_pdfs = max_pdfs or pipeline_cfg.get("max_pdfs", 100)

        supported = pipeline_cfg.get("supported_formats", [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf"])
        run_id = f"{pipeline_cfg.get('run_id_prefix', 'mm')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start = time.time()

        os.makedirs(output_dir, exist_ok=True)

        self.report.run_id = run_id
        print(f"\n{'='*60}")
        print(f"Multimodal Pipeline (run: {run_id})")
        print(f"{'='*60}")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")

        image_files, screenshot_files, pdf_files = self._discover_files(input_dir, supported)

        image_files = image_files[:max_images]
        screenshot_files = screenshot_files[:max_screenshots]
        pdf_files = pdf_files[:max_pdfs]

        self.report.total_inputs = len(image_files) + len(screenshot_files) + len(pdf_files)
        print(f"Found: {len(image_files)} images, {len(screenshot_files)} screenshots, {len(pdf_files)} PDFs")

        all_examples: List[MultimodalExample] = []

        if image_files:
            img_examples = self._process_images(image_files, output_dir, MediaType.IMAGE)
            all_examples.extend(img_examples)
            self.report.images_processed = len(img_examples)
            print(f"  Images processed: {len(img_examples)}")

        if screenshot_files:
            ss_examples = self._process_screenshots(screenshot_files, output_dir)
            all_examples.extend(ss_examples)
            self.report.screenshots_processed = len(ss_examples)
            print(f"  Screenshots processed: {len(ss_examples)}")

        if pdf_files:
            pdf_examples = self._process_pdfs(pdf_files, output_dir)
            all_examples.extend(pdf_examples)
            self.report.pdfs_processed = len(pdf_examples)
            print(f"  PDFs processed: {len(pdf_examples)}")

        if all_examples and self.embedder.enabled:
            print("Computing embeddings...")
            all_examples = self.embedder.embed_batch(all_examples)

        self.report.captions_generated = sum(1 for ex in all_examples if ex.caption)
        self.report.qa_pairs_generated = sum(
            len(ex.metadata.get("qa_pairs", [])) for ex in all_examples
        )

        if all_examples and self.exporter.enabled:
            print("Exporting...")
            exported = self.exporter.export_all(all_examples, output_dir)
            self.report.examples_exported = len(all_examples)
            print(f"  Exported {len(all_examples)} examples to {len(exported)} files")
            self.report.metadata["exported_files"] = exported

        ocr_confidences = [
            ex.metadata.get("ocr_confidence", 0)
            for ex in all_examples if ex.metadata.get("ocr_confidence", 0) > 0
        ]
        self.report.avg_ocr_confidence = round(
            sum(ocr_confidences) / max(len(ocr_confidences), 1), 4
        )
        self.report.processing_time = round(time.time() - start, 2)
        self.report.metadata["input_dir"] = input_dir
        self.report.metadata["output_dir"] = output_dir
        self._save_report(output_dir)

        print(f"\nSummary:")
        print(f"  Total inputs:          {self.report.total_inputs}")
        print(f"  Images processed:      {self.report.images_processed}")
        print(f"  Screenshots processed: {self.report.screenshots_processed}")
        print(f"  PDFs processed:        {self.report.pdfs_processed}")
        print(f"  Captions generated:    {self.report.captions_generated}")
        print(f"  QA pairs generated:    {self.report.qa_pairs_generated}")
        print(f"  Examples exported:     {self.report.examples_exported}")
        print(f"  Processing time:       {self.report.processing_time}s")
        print(f"{'='*60}")

        return self.report

    def _discover_files(
        self, input_dir: str, supported_extensions: List[str]
    ) -> tuple:
        image_files = []
        screenshot_files = []
        pdf_files = []

        if not os.path.exists(input_dir):
            return image_files, screenshot_files, pdf_files

        img_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

        for ext in supported_extensions:
            if ext == ".pdf":
                continue
            pattern = os.path.join(input_dir, f"**/*{ext}")
            for fp in glob.glob(pattern, recursive=True):
                if self.preprocessor.enabled:
                    mt = self.preprocessor.detect_media_type(fp)
                    if mt == MediaType.SCREENSHOT:
                        screenshot_files.append(fp)
                    else:
                        image_files.append(fp)
                else:
                    image_files.append(fp)

        if ".pdf" in supported_extensions:
            for fp in glob.glob(os.path.join(input_dir, "**/*.pdf"), recursive=True):
                pdf_files.append(fp)

        image_files.sort()
        screenshot_files.sort()
        pdf_files.sort()
        return image_files, screenshot_files, pdf_files

    def _process_images(
        self, image_paths: List[str], output_dir: str, media_type: MediaType
    ) -> List[MultimodalExample]:
        examples = []

        if self.ocr.enabled:
            ocr_results = self.ocr.extract_batch(image_paths)
        else:
            ocr_results = [None] * len(image_paths)

        for img_path, ocr_r in zip(image_paths, ocr_results):
            ocr_text = ocr_r.raw_text if ocr_r else ""

            if self.visual_qa.enabled:
                ex = self.visual_qa.build_example(img_path, ocr_text, media_type.value)
            else:
                ex = MultimodalExample(
                    prompt=f"Describe this image: {os.path.basename(img_path)}",
                    response=ocr_text or "",
                    image_path=img_path,
                    ocr_text=ocr_text,
                    media_type=media_type,
                    category=media_type.value,
                )

            if ex:
                if ocr_r:
                    ex.metadata["ocr_confidence"] = ocr_r.confidence
                examples.append(ex)

        return examples

    def _process_screenshots(
        self, screenshot_paths: List[str], output_dir: str
    ) -> List[MultimodalExample]:
        examples = []

        if self.ocr.enabled:
            ocr_results = self.ocr.extract_batch(screenshot_paths)
        else:
            ocr_results = [None] * len(screenshot_paths)

        ui_results = self.ui_analyzer.analyze_batch(
            screenshot_paths,
            [r.raw_text if r else "" for r in ocr_results] if ocr_results else None,
        )

        for path, ocr_r, ui_r in zip(screenshot_paths, ocr_results, ui_results):
            ocr_text = ocr_r.raw_text if ocr_r else ""

            prompt_parts = [f"Analyze this screenshot: {os.path.basename(path)}"]
            if ui_r.layout_type:
                prompt_parts.append(f"Layout: {ui_r.layout_type}")
            if ui_r.element_count:
                prompt_parts.append(f"Elements: {ui_r.element_count}")
            if ui_r.has_buttons:
                prompt_parts.append("Has interactive elements")
            if ocr_text:
                prompt_parts.append(f"Text: {ocr_text[:100]}")

            response = (
                f"This screenshot shows a {ui_r.layout_type.replace('_', ' ')} layout "
                f"with {ui_r.element_count} detected elements."
            )
            if ocr_text:
                response += f" Extracted text ({ocr_r.confidence:.0%} confidence): {ocr_text[:200]}"

            if self.visual_qa.enabled:
                caption = self.visual_qa.generate_caption(path, ocr_text)
            else:
                caption = response

            ex = MultimodalExample(
                prompt=" ".join(prompt_parts),
                response=caption or response,
                image_path=path,
                ocr_text=ocr_text,
                caption=caption,
                ui_analysis=ui_r,
                media_type=MediaType.SCREENSHOT,
                category="screenshot",
            )
            if ocr_r:
                ex.metadata["ocr_confidence"] = ocr_r.confidence
            examples.append(ex)

        return examples

    def _process_pdfs(
        self, pdf_paths: List[str], output_dir: str
    ) -> List[MultimodalExample]:
        examples = []
        docs = self.pdf_processor.process_batch(pdf_paths)

        for pdf_path, doc in zip(pdf_paths, docs):
            if doc is None:
                continue

            text_content = "\n".join(r.raw_text for r in doc.ocr_results if r and r.raw_text)
            image_examples = doc.embedded_images

            ex = MultimodalExample(
                prompt=f"Process this PDF: {doc.original_filename} ({doc.page_count} pages)",
                response=text_content[:1000] if text_content else f"PDF with {doc.page_count} pages.",
                ocr_text=text_content[:2000],
                media_type=MediaType.PDF,
                category="document",
                metadata={
                    "page_count": doc.page_count,
                    "total_chars": doc.total_chars,
                    "embedded_images": [ie.to_dict() for ie in image_examples],
                    "num_embedded_images": len(image_examples),
                },
            )

            if self.visual_qa.enabled and text_content:
                ex.caption = self.visual_qa.generate_caption(pdf_path, text_content)

            examples.append(ex)

        return examples

    def process_file(self, file_path: str) -> Optional[MultimodalExample]:
        """Process a single file (image or PDF) through the pipeline.

        Returns a MultimodalExample with extracted OCR text, caption,
        QA pairs, UI analysis (for screenshots), and embedding.
        """
        if not os.path.exists(file_path):
            return None

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            doc = self.pdf_processor.process(file_path)
            if doc is None:
                return None

            text_content = "\n".join(r.raw_text for r in doc.ocr_results if r and r.raw_text)
            ex = MultimodalExample(
                prompt=f"Process this PDF: {doc.original_filename} ({doc.page_count} pages)",
                response=text_content[:1000] if text_content else f"PDF with {doc.page_count} pages.",
                ocr_text=text_content[:2000],
                media_type=MediaType.PDF,
                category="document",
                metadata={
                    "page_count": doc.page_count,
                    "total_chars": doc.total_chars,
                    "num_embedded_images": len(doc.embedded_images),
                },
            )
            if self.visual_qa.enabled and text_content:
                ex.caption = self.visual_qa.generate_caption(file_path, text_content)
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            mt = MediaType.IMAGE
            if self.preprocessor.enabled:
                mt = self.preprocessor.detect_media_type(file_path)

            ocr_result = self.ocr.extract(file_path) if self.ocr.enabled else None
            ocr_text = ocr_result.raw_text if ocr_result else ""

            if self.visual_qa.enabled:
                ex = self.visual_qa.build_example(file_path, ocr_text, mt.value)
            else:
                ex = MultimodalExample(
                    prompt=f"Describe this file: {os.path.basename(file_path)}",
                    response=ocr_text or "",
                    image_path=file_path,
                    ocr_text=ocr_text,
                    media_type=mt,
                    category=mt.value,
                )

            if ex and ocr_result:
                ex.metadata["ocr_confidence"] = ocr_result.confidence
        else:
            return None

        if ex and self.embedder.enabled:
            ex = self.embedder.embed_batch([ex])[0]

        return ex

    def analyze_input_directory(self, input_dir: str) -> Dict:
        supported = self.config.get("multimodal", self.config).get("pipeline", {}).get(
            "supported_formats", [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf"]
        )
        img_files, ss_files, pdf_files = self._discover_files(input_dir, supported)
        total = len(img_files) + len(ss_files) + len(pdf_files)

        from collections import Counter
        ext_counts: Counter = Counter()
        for fp in img_files + ss_files:
            ext_counts[os.path.splitext(fp)[1]] += 1
        ext_counts[".pdf"] = len(pdf_files)

        return {
            "total_files": total,
            "images": len(img_files),
            "screenshots": len(ss_files),
            "pdfs": len(pdf_files),
            "by_extension": dict(ext_counts.most_common()),
            "input_dir": input_dir,
        }

    def _save_report(self, output_dir: str):
        path = os.path.join(output_dir, "multimodal_report.json")
        with open(path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2)

    def get_report(self) -> PipelineReport:
        return self.report
