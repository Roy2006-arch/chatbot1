import os
import json
import re
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schema import MultimodalExample, MediaType


CAPTION_TEMPLATES = [
    "Describe this image in detail.",
    "What is shown in this image?",
    "List the key elements visible in this image.",
    "What can you infer from this image?",
    "Generate a detailed description of this image.",
]

QA_TEMPLATES = [
    ("What is the main subject of this image?", "The main subject is {subject}."),
    ("Describe the colors and composition.", "The composition features {composition}."),
    ("What text is visible in this image?", "The visible text is: {text}"),
    ("What can you infer from the visual elements?", "{inference}"),
    ("Describe the layout and structure.", "The layout shows {layout}."),
]


class VisualQAGenerator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.min_caption_length = self.config.get("min_caption_length", 5)
        self.max_caption_length = self.config.get("max_caption_length", 200)
        self.caption_templates = self.config.get("caption_templates", CAPTION_TEMPLATES)
        self.use_local_model = self.config.get("use_local_model", False)
        self.batch_size = self.config.get("batch_size", 8)
        self._model = None
        self._processor = None
        self.stats = {"captions": 0, "qa_pairs": 0, "failed": 0}

    @property
    def model_available(self) -> bool:
        if self.use_local_model:
            return self._load_model() is not None
        return False

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            self._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self._model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            return self._model
        except Exception:
            return None

    def generate_caption(self, image_path: str, ocr_text: str = "") -> str:
        if not self.enabled:
            return ""

        model_caption = self._model_caption(image_path) if self.model_available else ""
        template_caption = self._template_caption(image_path, ocr_text)

        caption = model_caption or template_caption
        if not caption:
            caption = self._fallback_caption(ocr_text)

        if len(caption) < self.min_caption_length:
            caption = self._fallback_caption(ocr_text)

        words = caption.split()
        if len(words) > self.max_caption_length:
            caption = " ".join(words[:self.max_caption_length])

        if caption:
            self.stats["captions"] += 1
        return caption

    def _model_caption(self, image_path: str) -> str:
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            inputs = self._processor(img, return_tensors="pt")
            out = self._model.generate(**inputs, max_new_tokens=50)
            return self._processor.decode(out[0], skip_special_tokens=True).strip()
        except Exception:
            return ""

    def _template_caption(self, image_path: str, ocr_text: str) -> str:
        from PIL import Image
        try:
            img = Image.open(image_path)
            w, h = img.size
            fmt = img.format or "unknown"
            elements = []
            if ocr_text:
                elements.append(f"text content detected: {ocr_text[:100]}")
            if w > 0 and h > 0:
                orientation = "landscape" if w > h else "portrait" if h > w else "square"
                elements.append(f"{orientation} image ({w}x{h})")
            elements.append(f"format: {fmt}")
            detail = "; ".join(elements)
            caption = f"This is a {os.path.basename(image_path)}. {detail}."
            return caption
        except Exception:
            return ""

    def _fallback_caption(self, ocr_text: str) -> str:
        if ocr_text:
            words = ocr_text.split()
            snippet = " ".join(words[:20])
            return f"Image containing text: \"{snippet}\"" + ("..." if len(words) > 20 else "")
        return "Image with no visible text content."

    def generate_qa_pairs(
        self, image_path: str, ocr_text: str, caption: str, max_pairs: int = 3
    ) -> List[Dict[str, str]]:
        pairs = []
        subject = caption.split(".")[0] if caption else "unknown"
        composition = f"a mix of visual elements with {len(ocr_text)} characters of detected text" if ocr_text else "primarily visual elements"

        qa_configs = [
            ("What is shown in this image?", f"{caption}" if caption else "An image."),
            ("What text can be read in this image?", f"The image contains the following text: {ocr_text[:200]}" if ocr_text else "No readable text detected."),
            ("Describe the visual composition.", composition),
            ("What is the main subject?", subject),
            ("What can you infer from this image?", f"Based on the visual analysis, this image contains {composition}."),
        ]

        for q, a in qa_configs[:max_pairs]:
            pairs.append({"question": q, "answer": a})

        self.stats["qa_pairs"] += len(pairs)
        return pairs

    def build_example(
        self, image_path: str, ocr_text: str = "", category: str = ""
    ) -> Optional[MultimodalExample]:
        caption = self.generate_caption(image_path, ocr_text)
        if not caption:
            self.stats["failed"] += 1
            return None

        from .image_preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        img_info = preprocessor.process(image_path)

        media_type = preprocessor.detect_media_type(image_path) if img_info else MediaType.IMAGE

        qa_pairs = self.generate_qa_pairs(image_path, ocr_text, caption)
        qa_text = "\n".join(f"Q: {p['question']}\nA: {p['answer']}" for p in qa_pairs)

        example = MultimodalExample(
            prompt=f"Describe this image: {os.path.basename(image_path)}",
            response=caption,
            image_path=image_path,
            ocr_text=ocr_text,
            caption=caption,
            media_type=media_type,
            category=category or media_type.value,
            quality_score=self._score(caption, ocr_text, qa_pairs),
            metadata={
                "num_qa_pairs": len(qa_pairs),
                "qa_pairs": qa_pairs,
                "caption": caption,
                "qa_text": qa_text,
                "width": img_info.width if img_info else 0,
                "height": img_info.height if img_info else 0,
                "format": img_info.format if img_info else "",
            },
        )
        return example

    def build_batch(
        self, image_paths: List[str], ocr_texts: Optional[List[str]] = None,
        num_workers: int = 4,
    ) -> List[Optional[MultimodalExample]]:
        if not self.enabled:
            return [None] * len(image_paths)
        ocr_texts = ocr_texts or [""] * len(image_paths)
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.build_example, p, t): i
                for i, (p, t) in enumerate(zip(image_paths, ocr_texts))
            }
            for future in as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _score(self, caption: str, ocr_text: str, qa_pairs: List[Dict]) -> float:
        score = 0.3
        if len(caption.split()) >= 10:
            score += 0.2
        if ocr_text and len(ocr_text) >= 20:
            score += 0.2
        if len(qa_pairs) >= 2:
            score += 0.15
        if len(qa_pairs) >= 3:
            score += 0.15
        return min(1.0, score)

    def get_stats(self) -> Dict:
        return self.stats
