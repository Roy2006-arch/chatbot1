import os
import io
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .schema import ImageExample, MediaType


@dataclass
class PreprocessResult:
    success: bool
    image_path: str
    image: Optional[Any] = None
    width: int = 0
    height: int = 0
    format: str = ""
    error: str = ""


class ImagePreprocessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.max_width = self.config.get("max_width", 1024)
        self.max_height = self.config.get("max_height", 1024)
        self.min_width = self.config.get("min_width", 32)
        self.min_height = self.config.get("min_height", 32)
        self.max_size_mb = self.config.get("max_size_mb", 20)
        self.normalize = self.config.get("normalize", True)
        self.default_format = self.config.get("default_format", "PNG")
        self.jpeg_quality = self.config.get("jpeg_quality", 90)
        self.stats = {"loaded": 0, "resized": 0, "failed": 0, "skipped": 0}

    def load_image(self, path: str) -> Optional[Any]:
        try:
            from PIL import Image
            img = Image.open(path)
            img.verify()
            img = Image.open(path)
            img.load()
            return img
        except Exception:
            return None

    def validate(self, img: Any, file_size: int = 0) -> Tuple[bool, str]:
        if img is None:
            return False, "failed_to_load"
        w, h = img.size
        if w < self.min_width or h < self.min_height:
            return False, f"too_small:{w}x{h}"
        if file_size > 0 and file_size > self.max_size_mb * 1024 * 1024:
            return False, f"too_large:{file_size}bytes"
        return True, ""

    def resize(self, img: Any) -> Any:
        w, h = img.size
        if w <= self.max_width and h <= self.max_height:
            return img

        ratio = min(self.max_width / w, self.max_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        self.stats["resized"] += 1
        return img.resize((new_w, new_h), self._get_resample())

    def to_rgb(self, img: Any) -> Any:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _get_resample(self) -> int:
        try:
            from PIL import Image
            return Image.LANCZOS
        except AttributeError:
            try:
                from PIL import Image
                return Image.ANTIALIAS
            except AttributeError:
                return 1

    def process(self, file_path: str, media_type: MediaType = MediaType.IMAGE) -> Optional[ImageExample]:
        if not self.enabled:
            return None
        if not os.path.exists(file_path):
            self.stats["failed"] += 1
            return None

        file_size = os.path.getsize(file_path)
        if file_size > self.max_size_mb * 1024 * 1024:
            self.stats["skipped"] += 1
            return None

        img = self.load_image(file_path)
        valid, reason = self.validate(img, file_size)
        if not valid:
            self.stats["failed" if "failed" in reason else "skipped"] += 1
            return None

        img = self.to_rgb(img)
        img = self.resize(img)
        self.stats["loaded"] += 1

        return ImageExample(
            file_path=file_path,
            original_filename=os.path.basename(file_path),
            width=img.width,
            height=img.height,
            file_size_bytes=file_size,
            format=img.format or self.default_format,
            media_type=media_type,
        )

    def process_batch(
        self, file_paths: List[str], media_type: MediaType = MediaType.IMAGE, num_workers: int = 4
    ) -> List[Optional[ImageExample]]:
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process, p, media_type): i for i, p in enumerate(file_paths)}
            for future in as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def save_processed(
        self, image: Any, output_path: str, fmt: Optional[str] = None
    ) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fmt = (fmt or self.default_format).upper()
        save_kwargs = {}
        if fmt == "JPEG":
            save_kwargs["quality"] = self.jpeg_quality
        image.save(output_path, format=fmt, **save_kwargs)
        return output_path

    def detect_media_type(self, file_path: str) -> MediaType:
        name = os.path.basename(file_path).lower()
        ext = os.path.splitext(name)[1].lower()
        if ext == ".pdf":
            return MediaType.PDF
        screen_keywords = ["screenshot", "screen", "capture", "snip", "ss_"]
        code_keywords = ["code", "snippet", "editor", "ide_", "terminal", "vscode"]
        chart_keywords = ["chart", "graph", "plot", "figure", "diagram"]
        diagram_keywords = ["architecture", "flowchart", "uml", "network", "schema"]

        for kw in screen_keywords:
            if kw in name:
                return MediaType.SCREENSHOT
        for kw in code_keywords:
            if kw in name:
                return MediaType.CODE_SCREENSHOT
        for kw in chart_keywords:
            if kw in name:
                return MediaType.CHART
        for kw in diagram_keywords:
            if kw in name:
                return MediaType.TECHNICAL_DIAGRAM
        return MediaType.IMAGE

    def get_stats(self) -> Dict:
        return self.stats
