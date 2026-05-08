import os
import time
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schema import OCRResult


class OCRExtractor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.languages = self.config.get("languages", "eng")
        self.min_confidence = self.config.get("min_confidence", 0.3)
        self.preprocess_image = self.config.get("preprocess_image", True)
        self.cache_results = self.config.get("cache_results", True)
        self.max_regions = self.config.get("max_regions", 100)
        self._cache: Dict[str, OCRResult] = {}
        self._tesseract_available = False
        self._easyocr_available = False

    @property
    def available(self) -> bool:
        return self._check_deps()

    def _check_deps(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            return True
        except Exception:
            pass
        try:
            import easyocr
            self._easyocr_available = True
            return True
        except ImportError:
            pass
        return False

    def extract_from_image(self, image_path: str) -> Optional[OCRResult]:
        if not self.enabled:
            return None
        if not os.path.exists(image_path):
            return None

        if self.cache_results and image_path in self._cache:
            return self._cache[image_path]

        start = time.time()

        if self._tesseract_available:
            result = self._extract_tesseract(image_path)
        elif self._easyocr_available:
            result = self._extract_easyocr(image_path)
        else:
            result = self._extract_fallback(image_path)

        if result:
            result.processing_time = round(time.time() - start, 3)
            if self.cache_results:
                self._cache[image_path] = result
        return result

    def _extract_tesseract(self, image_path: str) -> Optional[OCRResult]:
        try:
            from PIL import Image
            import pytesseract

            img = Image.open(image_path)
            if self.preprocess_image:
                img = self._preprocess(img)

            data = pytesseract.image_to_data(img, lang=self.languages, output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(img, lang=self.languages).strip()

            confidences = [int(c) for c in data.get("conf", []) if isinstance(c, (int, float)) and c > 0]
            avg_conf = sum(confidences) / max(len(confidences), 1) / 100.0 if confidences else 0.0

            boxes = []
            for i in range(len(data.get("text", []))):
                if int(data["conf"][i]) > 0 and data["text"][i].strip():
                    boxes.append({
                        "text": data["text"][i],
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                        "confidence": int(data["conf"][i]) / 100.0,
                    })

            return OCRResult(
                raw_text=text,
                confidence=round(avg_conf, 4),
                language=self.languages,
                bounding_boxes=boxes[:self.max_regions],
                num_regions=len(boxes),
            )
        except Exception:
            return None

    def _extract_easyocr(self, image_path: str) -> Optional[OCRResult]:
        try:
            import easyocr
            reader = easyocr.Reader([self.languages], gpu=False, verbose=False)
            results = reader.readtext(image_path)

            text_parts = []
            boxes = []
            confidences = []
            for r in results:
                bbox, text, conf = r
                text_parts.append(text)
                confidences.append(conf)
                boxes.append({
                    "text": text,
                    "x": int(min(p[0] for p in bbox)),
                    "y": int(min(p[1] for p in bbox)),
                    "w": int(max(p[0] for p in bbox) - min(p[0] for p in bbox)),
                    "h": int(max(p[1] for p in bbox) - min(p[1] for p in bbox)),
                    "confidence": round(conf, 4),
                })

            full_text = " ".join(text_parts)
            avg_conf = sum(confidences) / max(len(confidences), 1) if confidences else 0.0

            return OCRResult(
                raw_text=full_text,
                confidence=round(avg_conf, 4),
                language=self.languages,
                bounding_boxes=boxes[:self.max_regions],
                num_regions=len(boxes),
            )
        except Exception:
            return None

    def _extract_fallback(self, image_path: str) -> Optional[OCRResult]:
        try:
            from PIL import Image
            img = Image.open(image_path)
            w, h = img.size
            return OCRResult(
                raw_text="",
                confidence=0.0,
                language=self.languages,
                bounding_boxes=[],
                num_regions=0,
            )
        except Exception:
            return None

    def extract_batch(
        self, image_paths: List[str], num_workers: int = 4
    ) -> List[Optional[OCRResult]]:
        if not self.enabled:
            return [None] * len(image_paths)
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.extract_from_image, p): i for i, p in enumerate(image_paths)}
            for future in as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _preprocess(self, img: "Image.Image") -> "Image.Image":
        try:
            import numpy as np
            arr = np.array(img.convert("L"))
            from PIL import ImageFilter, ImageEnhance
            if arr.mean() < 128:
                import numpy as np
                arr = 255 - arr
                img = Image.fromarray(arr)
            img = img.filter(ImageFilter.SHARPEN)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            return img
        except Exception:
            return img

    def extract_text(self, image_path: str) -> str:
        result = self.extract_from_image(image_path)
        return result.raw_text if result else ""

    def get_stats(self) -> Dict:
        return {
            "tesseract_available": self._tesseract_available,
            "easyocr_available": self._easyocr_available,
            "cache_size": len(self._cache),
        }
