import os
import numpy as np
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schema import MultimodalExample, MediaType


class MultimodalEmbedder:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.model_name = self.config.get("model_name", "clip-ViT-B-32")
        self.text_model_name = self.config.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = self.config.get("batch_size", 16)
        self.normalize_embeddings = self.config.get("normalize_embeddings", True)
        self.cache_dir = self.config.get("cache_dir", "")
        self._clip_model = None
        self._text_model = None
        self._sentence_model = None
        self.stats = {"images_embedded": 0, "texts_embedded": 0, "failed": 0}

    @property
    def clip_available(self) -> bool:
        return self._load_clip() is not None

    @property
    def text_model_available(self) -> bool:
        return self._load_text_model() is not None or self._load_sentence_model() is not None

    def _load_clip(self):
        if self._clip_model is not None:
            return self._clip_model
        try:
            import sentence_transformers
            self._clip_model = sentence_transformers.SentenceTransformer(
                self.model_name, cache_folder=self.cache_dir or None
            )
            return self._clip_model
        except Exception:
            return None

    def _load_text_model(self):
        if self._text_model is not None:
            return self._text_model
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            self._text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self._text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            return self._text_model
        except Exception:
            return None

    def _load_sentence_model(self):
        if self._sentence_model is not None:
            return self._sentence_model
        try:
            import sentence_transformers
            self._sentence_model = sentence_transformers.SentenceTransformer(
                self.text_model_name, cache_folder=self.cache_dir or None
            )
            return self._sentence_model
        except Exception:
            return None

    def embed_image(self, image_path: str) -> Optional[List[float]]:
        if not self.enabled or not os.path.exists(image_path):
            return None

        model = self._load_clip()
        if model is None:
            return self._fallback_image_embedding(image_path)

        try:
            embedding = model.encode(image_path, normalize_embeddings=self.normalize_embeddings)
            self.stats["images_embedded"] += 1
            return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        except Exception:
            self.stats["failed"] += 1
            return None

    def embed_text(self, text: str) -> Optional[List[float]]:
        if not self.enabled or not text:
            return None

        model = self._load_sentence_model()
        if model is None:
            return self._fallback_text_embedding(text)

        try:
            embedding = model.encode(text, normalize_embeddings=self.normalize_embeddings)
            self.stats["texts_embedded"] += 1
            return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        except Exception:
            self.stats["failed"] += 1
            return None

    def embed_example(self, example: MultimodalExample) -> Optional[MultimodalExample]:
        if not self.enabled:
            return example

        if example.image_path and example.image_embedding is None:
            img_emb = self.embed_image(example.image_path)
            if img_emb:
                example.image_embedding = img_emb

        if example.response and not example.metadata.get("text_embedding"):
            text_emb = self.embed_text(example.response)
            if text_emb:
                example.metadata["text_embedding"] = text_emb

        return example

    def embed_batch(
        self, examples: List[MultimodalExample], num_workers: int = 4
    ) -> List[MultimodalExample]:
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.embed_example, ex): i for i, ex in enumerate(examples)}
            for future in as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def compute_similarity(
        self, embedding_a: List[float], embedding_b: List[float]
    ) -> float:
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        if self.normalize_embeddings:
            return float(np.dot(a, b))
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / max(norm, 1e-8))

    def _fallback_image_embedding(self, image_path: str) -> Optional[List[float]]:
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(image_path).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32).flatten() / 255.0
            emb = arr[:512].tolist()
            if self.normalize_embeddings:
                norm = np.linalg.norm(emb)
                emb = (np.array(emb) / max(norm, 1e-8)).tolist()
            return emb
        except Exception:
            return None

    def _fallback_text_embedding(self, text: str) -> Optional[List[float]]:
        try:
            import numpy as np
            import hashlib
            h = hashlib.md5(text.encode()).hexdigest()
            seed = int(h[:8], 16)
            rng = np.random.RandomState(seed)
            emb = rng.randn(384).astype(np.float32)
            if self.normalize_embeddings:
                emb = emb / np.linalg.norm(emb)
            return emb.tolist()
        except Exception:
            return None

    def get_stats(self) -> Dict:
        return self.stats
