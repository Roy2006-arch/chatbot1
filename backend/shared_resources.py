import contextvars
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

logger = logging.getLogger("chatbot.shared_resources")


_request_embedding_cache: contextvars.ContextVar = contextvars.ContextVar(
    "request_embedding_cache", default=None
)


class RequestEmbeddingCache:
    """Per-request cache to eliminate redundant embedding calls across components."""

    def __init__(self):
        self._cache: dict = {}

    def encode(self, embedder, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        texts_key = tuple(texts) if isinstance(texts, list) else texts

        filtered_kwargs = {
            k: v
            for k, v in sorted(kwargs.items())
            if k not in ("batch_size", "show_progress_bar")
        }
        key = (texts_key, tuple(filtered_kwargs.items()))

        if key not in self._cache:
            result = embedder.encode(texts, **kwargs)
            self._cache[key] = np.array(result, dtype="float32")
        return self._cache[key]

    def clear(self):
        self._cache.clear()


def get_request_cache() -> Optional[RequestEmbeddingCache]:
    return _request_embedding_cache.get()


def set_request_cache(cache: Optional[RequestEmbeddingCache]):
    _request_embedding_cache.set(cache)


def reset_request_cache():
    """Clear and reset the per-request embedding cache."""
    _request_embedding_cache.set(None)


class ModelRegistry:
    """
    Singleton registry for shared heavy models to prevent redundant memory usage.
    """
    _embedder: Optional[SentenceTransformer] = None
    _embed_model_name = "all-MiniLM-L6-v2"

    @classmethod
    def get_embedder(cls) -> SentenceTransformer:
        if cls._embedder is None:
            logger.info(f"Loading shared embedder model: {cls._embed_model_name}")
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._embedder = SentenceTransformer(cls._embed_model_name, device=device)
        return cls._embedder

    @classmethod
    def clear_cache(cls):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
