import os
import re
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schema import UIAnalysisResult


UI_ELEMENT_KEYWORDS = {
    "button": ["button", "btn", "submit", "cancel", "ok", "save", "delete", "edit", "click"],
    "input": ["input", "textfield", "text field", "search", "type", "enter"],
    "form": ["form", "login", "register", "signup", "sign up", "sign-in"],
    "image": ["img", "image", "photo", "picture", "avatar", "icon"],
    "heading": ["h1", "h2", "h3", "title", "header", "heading"],
    "link": ["link", "url", "href", "hyper", "anchor"],
    "list": ["list", "menu", "nav", "navigation", "sidebar"],
    "card": ["card", "panel", "widget", "box", "container"],
}


class UIAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.min_element_area = self.config.get("min_element_area", 16)
        self.max_elements = self.config.get("max_elements", 50)
        self.detect_buttons = self.config.get("detect_buttons", True)
        self.detect_forms = self.config.get("detect_forms", True)
        self.detect_text_blocks = self.config.get("detect_text_blocks", True)
        self.detect_images = self.config.get("detect_images", True)
        self.layout_threshold = self.config.get("layout_threshold", 0.5)
        self._contour_available = False
        self.stats = {"analyzed": 0, "elements_found": 0}

    def _check_deps(self) -> bool:
        try:
            import cv2
            self._contour_available = True
            return True
        except ImportError:
            return False

    def analyze(self, image_path: str, ocr_text: str = "") -> UIAnalysisResult:
        if not self.enabled:
            return UIAnalysisResult()

        if self._check_deps():
            return self._analyze_cv(image_path, ocr_text)
        return self._analyze_text_based(ocr_text, image_path)

    def _analyze_cv(self, image_path: str, ocr_text: str) -> UIAnalysisResult:
        try:
            import cv2
            import numpy as np

            img = cv2.imread(image_path)
            if img is None:
                return self._analyze_text_based(ocr_text, image_path)

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            elements = []
            has_buttons = False
            has_forms = False
            has_imgs = False
            has_text = False

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                if area < self.min_element_area:
                    continue

                element = {
                    "x": int(x), "y": int(y), "w": int(cw), "h": int(ch),
                    "area": int(area), "aspect_ratio": round(cw / max(ch, 1), 2),
                    "type": "region",
                }

                if ch > h * 0.02 and cw > w * 0.02:
                    if cw / max(ch, 1) < 3 and ch / max(cw, 1) < 3:
                        element["type"] = "button_like"
                        has_buttons = True
                    if cw > w * 0.3:
                        element["type"] = "text_block"
                        has_text = True
                    if cw > w * 0.5 and ch > h * 0.3:
                        element["type"] = "form_region"
                        has_forms = True
                    if cw < w * 0.1 and ch < h * 0.1:
                        element["type"] = "icon_image"
                        has_imgs = True

                elements.append(element)

            elements = elements[:self.max_elements]

            layout = self._classify_layout(w, h, elements)
            complexity = min(1.0, len(elements) / 30)

            return UIAnalysisResult(
                elements=elements,
                layout_type=layout,
                element_count=len(elements),
                has_buttons=has_buttons,
                has_forms=has_forms,
                has_images=has_imgs,
                has_text_blocks=has_text,
                complexity_score=round(complexity, 4),
                metadata={"image_size": f"{w}x{h}", "analysis_method": "cv_contour"},
            )
        except Exception:
            return self._analyze_text_based(ocr_text, image_path)

    def _analyze_text_based(self, ocr_text: str, image_path: str) -> UIAnalysisResult:
        elements = []
        has_buttons = False
        has_forms = False
        has_imgs = False
        has_text = False
        text_lower = ocr_text.lower()

        for element_type, keywords in UI_ELEMENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    elements.append({
                        "type": element_type,
                        "keyword": kw,
                        "source": "ocr_text",
                    })
                    if element_type == "button":
                        has_buttons = True
                    if element_type == "form":
                        has_forms = True
                    if element_type == "image":
                        has_imgs = True
                    if element_type in ("heading", "text_block"):
                        has_text = True
                    break

        if not elements:
            from PIL import Image
            try:
                img = Image.open(image_path)
                w, h_ = img.size
            except Exception:
                w, h_ = 0, 0

            name_lower = os.path.basename(image_path).lower()
            if any(k in name_lower for k in ["login", "signup", "form", "settings"]):
                has_forms = True
                elements.append({"type": "form", "source": "filename"})
            if any(k in name_lower for k in ["button", "btn", "click"]):
                has_buttons = True
                elements.append({"type": "button", "source": "filename"})

            if w > h_ * 1.2:
                layout = "wide_layout"
            elif h_ > w * 1.2:
                layout = "tall_layout"
            else:
                layout = "balanced_layout"
        else:
            layout = self._classify_layout_from_keywords(elements)

        complexity = min(1.0, len(elements) / 10)

        return UIAnalysisResult(
            elements=elements,
            layout_type=layout,
            element_count=len(elements),
            has_buttons=has_buttons,
            has_forms=has_forms,
            has_images=has_imgs,
            has_text_blocks=has_text,
            complexity_score=round(complexity, 4),
            metadata={"analysis_method": "text_based"},
        )

    def _classify_layout(self, w: int, h: int, elements: List[Dict]) -> str:
        if not elements:
            if w > h * 1.5:
                return "wide_layout"
            elif h > w * 1.5:
                return "tall_layout"
            return "balanced_layout"

        mid_x = w / 2
        left_elems = sum(1 for e in elements if e["x"] + e["w"] / 2 < mid_x - w * 0.1)
        right_elems = sum(1 for e in elements if e["x"] + e["w"] / 2 > mid_x + w * 0.1)
        center_elems = len(elements) - left_elems - right_elems

        if left_elems > right_elems + center_elems:
            return "left_navigation"
        elif right_elems > left_elems + center_elems:
            return "right_panel"
        elif center_elems > left_elems + right_elems:
            return "centered_layout"
        elif w > h * 1.5:
            return "wide_layout"
        elif h > w * 1.5:
            return "tall_layout"
        return "grid_layout"

    def _classify_layout_from_keywords(self, elements: List[Dict]) -> str:
        types = [e["type"] for e in elements]
        if "form" in types:
            return "form_layout"
        if "list" in types or "nav" in types:
            return "navigation_layout"
        if "card" in types:
            return "card_grid"
        if "heading" in types:
            return "content_layout"
        return "mixed_layout"

    def analyze_batch(
        self, image_paths: List[str], ocr_texts: Optional[List[str]] = None,
        num_workers: int = 4,
    ) -> List[UIAnalysisResult]:
        if not self.enabled:
            return [UIAnalysisResult() for _ in image_paths]
        ocr_texts = ocr_texts or [""] * len(image_paths)
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.analyze, p, t): i
                for i, (p, t) in enumerate(zip(image_paths, ocr_texts))
            }
            for future in as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def get_stats(self) -> Dict:
        return self.stats
