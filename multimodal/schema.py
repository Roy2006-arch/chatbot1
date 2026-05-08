from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class MediaType(Enum):
    IMAGE = "image"
    SCREENSHOT = "screenshot"
    PDF = "pdf"
    CODE_SCREENSHOT = "code_screenshot"
    CHART = "chart"
    TECHNICAL_DIAGRAM = "technical_diagram"
    DOCUMENT = "document"


class Modality(Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    TEXT_IMAGE = "text_image"
    TEXT_IMAGE_STRUCTURE = "text_image_structure"


@dataclass
class OCRResult:
    raw_text: str
    confidence: float = 0.0
    language: str = "eng"
    bounding_boxes: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0
    num_regions: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class UIAnalysisResult:
    elements: List[Dict] = field(default_factory=list)
    layout_type: str = ""
    element_count: int = 0
    has_buttons: bool = False
    has_forms: bool = False
    has_images: bool = False
    has_text_blocks: bool = False
    complexity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ImageExample:
    file_path: str
    original_filename: str = ""
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    format: str = ""
    media_type: MediaType = MediaType.IMAGE

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["media_type"] = self.media_type.value
        return d


@dataclass
class DocumentExample:
    file_path: str
    original_filename: str = ""
    page_count: int = 0
    total_chars: int = 0
    ocr_results: List[OCRResult] = field(default_factory=list)
    embedded_images: List[ImageExample] = field(default_factory=list)
    media_type: MediaType = MediaType.DOCUMENT

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["media_type"] = self.media_type.value
        d["ocr_results"] = [r.to_dict() for r in self.ocr_results]
        return d


@dataclass
class ScreenshotExample:
    file_path: str
    original_filename: str = ""
    width: int = 0
    height: int = 0
    ui_analysis: Optional[UIAnalysisResult] = None
    ocr_result: Optional[OCRResult] = None
    caption: str = ""
    media_type: MediaType = MediaType.SCREENSHOT

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["media_type"] = self.media_type.value
        if self.ui_analysis:
            d["ui_analysis"] = self.ui_analysis.to_dict()
        if self.ocr_result:
            d["ocr_result"] = self.ocr_result.to_dict()
        return d


@dataclass
class MultimodalExample:
    id: str = ""
    prompt: str = ""
    response: str = ""
    image_path: Optional[str] = None
    image_embedding: Optional[List[float]] = None
    ocr_text: str = ""
    caption: str = ""
    ui_analysis: Optional[UIAnalysisResult] = None
    media_type: MediaType = MediaType.IMAGE
    modality: Modality = Modality.TEXT_IMAGE
    category: str = ""
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["media_type"] = self.media_type.value
        d["modality"] = self.modality.value
        if self.ui_analysis:
            d["ui_analysis"] = self.ui_analysis.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "MultimodalExample":
        if isinstance(d.get("media_type"), str):
            d["media_type"] = MediaType(d["media_type"])
        if isinstance(d.get("modality"), str):
            d["modality"] = Modality(d["modality"])
        if isinstance(d.get("ui_analysis"), dict) and d["ui_analysis"]:
            d["ui_analysis"] = UIAnalysisResult(**d["ui_analysis"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineReport:
    run_id: str
    timestamp: str = ""
    total_inputs: int = 0
    images_processed: int = 0
    screenshots_processed: int = 0
    pdfs_processed: int = 0
    captions_generated: int = 0
    qa_pairs_generated: int = 0
    examples_exported: int = 0
    avg_ocr_confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)
