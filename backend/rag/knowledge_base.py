"""
rag/knowledge_base.py
=====================
Persistent global FAISS knowledge base.

Key design decisions vs. the existing document_processor.py:
  - GLOBAL (not per-session) — shared across all users/sessions.
  - PERSISTENT — index + metadata saved to disk; survives server restarts.
  - METADATA-RICH — stores source, chunk_index, char offsets per vector.
  - THREAD-SAFE — read/write protected by threading.RLock.
  - INCREMENTAL — new documents can be added without rebuilding the full index.

Storage layout (under `index_dir`):
    knowledge_base/
        index.faiss       ← FAISS flat index (IVFFlat when large, Flat when small)
        metadata.json     ← parallel list of chunk metadata dicts
        manifest.json     ← ingested source registry (deduplicate re-ingestion)

Usage:
    from rag.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    kb.add_chunks(chunks)          # list[Chunk]
    results = kb.search("query", top_k=5)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk

logger = logging.getLogger("rag.knowledge_base")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge_base"
_EMBED_MODEL       = "all-MiniLM-L6-v2"
_EMBED_DIM         = 384

# Switch from Flat to IVF when we exceed this many vectors (better search speed)
_IVF_THRESHOLD = 10_000
_IVF_NLIST     = 100    # number of Voronoi cells
_IVF_NPROBE    = 10     # cells searched per query (accuracy vs. speed trade-off)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class SearchResult:
    """A single retrieval result with text, score, and provenance."""
    __slots__ = ("text", "source", "chunk_index", "score", "metadata")

    def __init__(self, text: str, source: str, chunk_index: int, score: float, metadata: dict):
        self.text        = text
        self.source      = source
        self.chunk_index = chunk_index
        self.score       = score          # cosine similarity [0, 1]
        self.metadata    = metadata

    def to_dict(self) -> dict:
        return {
            "text":        self.text,
            "source":      self.source,
            "chunk_index": self.chunk_index,
            "score":       round(self.score, 4),
            "metadata":    self.metadata,
        }

    def __repr__(self) -> str:
        return f"<SearchResult source={self.source!r} score={self.score:.3f} text={self.text[:60]!r}>"


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Thread-safe, disk-persistent FAISS knowledge base.

    Args:
        index_dir  : directory where index.faiss, metadata.json, manifest.json live
        embed_model: SentenceTransformer model name
        embed_dim  : embedding dimension (must match embed_model output)
    """

    def __init__(
        self,
        index_dir:   Path | str = _DEFAULT_INDEX_DIR,
        embed_model: str        = _EMBED_MODEL,
        embed_dim:   int        = _EMBED_DIM,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._dim      = embed_dim
        self._lock     = threading.RLock()
        self._embedder = SentenceTransformer(embed_model)

        # Parallel data structures (index position ↔ metadata row)
        self._metadata: list[dict]  = []   # one dict per vector
        self._manifest: dict        = {}   # source → ingestion timestamp
        self._index: faiss.Index    = faiss.IndexFlatIP(embed_dim)  # Inner Product ≈ cosine after L2-norm

        self._load()

    # ------------------------------------------------------------------ #
    #  Public: Write                                                        #
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Embed and index a list of Chunk objects.
        Returns the number of vectors actually added.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self._embed(texts)

        with self._lock:
            self._index.add(vectors)
            for chunk, vec in zip(chunks, vectors):
                self._metadata.append({
                    "text":        chunk.text,
                    "source":      chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "char_start":  chunk.char_start,
                    "char_end":    chunk.char_end,
                    "metadata":    chunk.metadata,
                    "added_at":    time.time(),
                })
            self._maybe_upgrade_index()
            self._save()

        logger.info("[KnowledgeBase] Added %d chunks | Total vectors: %d", len(chunks), self._index.ntotal)
        return len(chunks)

    def register_source(self, source: str) -> None:
        """Mark a source as ingested in the manifest (prevents re-ingestion)."""
        with self._lock:
            self._manifest[source] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._save_manifest()

    def is_ingested(self, source: str) -> bool:
        """Return True if this source is already in the manifest."""
        return source in self._manifest

    def clear(self) -> None:
        """Wipe the entire knowledge base (index + metadata + manifest)."""
        with self._lock:
            self._index    = faiss.IndexFlatIP(self._dim)
            self._metadata = []
            self._manifest = {}
            self._save()
        logger.warning("[KnowledgeBase] Knowledge base cleared.")

    # ------------------------------------------------------------------ #
    #  Public: Read                                                         #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query:       str,
        top_k:       int   = 5,
        score_floor: float = 0.20,
    ) -> list[SearchResult]:
        """
        Embed `query` and return the top_k most similar chunks.

        Args:
            query      : natural language query string
            top_k      : maximum results to return
            score_floor: minimum cosine similarity; results below are dropped

        Returns:
            List of SearchResult, sorted descending by score.
        """
        if self._index.ntotal == 0:
            return []

        query_vec = self._embed([query])

        with self._lock:
            k = min(top_k, self._index.ntotal)
            scores, idxs = self._index.search(query_vec, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1 or score < score_floor:
                continue
            meta = self._metadata[idx]
            results.append(SearchResult(
                text=meta["text"],
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                score=float(score),
                metadata=meta.get("metadata", {}),
            ))

        return sorted(results, key=lambda r: r.score, reverse=True)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    @property
    def ingested_sources(self) -> list[str]:
        return list(self._manifest.keys())

    def stats(self) -> dict:
        return {
            "total_vectors":    self.total_vectors,
            "total_sources":    len(self._manifest),
            "ingested_sources": self.ingested_sources,
            "index_type":       type(self._index).__name__,
        }

    # ------------------------------------------------------------------ #
    #  Internal: Embedding                                                  #
    # ------------------------------------------------------------------ #

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Produce L2-normalised embeddings (required for IndexFlatIP = cosine similarity).
        """
        vecs = self._embedder.encode(texts, batch_size=64, show_progress_bar=False)
        vecs = np.array(vecs, dtype="float32")
        faiss.normalize_L2(vecs)
        return vecs

    # ------------------------------------------------------------------ #
    #  Internal: Index Upgrade                                             #
    # ------------------------------------------------------------------ #

    def _maybe_upgrade_index(self) -> None:
        """
        Automatically upgrade from IndexFlatIP → IndexIVFFlat once the
        vector count crosses _IVF_THRESHOLD for O(log n) search speed.
        """
        if (
            self._index.ntotal >= _IVF_THRESHOLD
            and isinstance(self._index, faiss.IndexFlatIP)
        ):
            logger.info("[KnowledgeBase] Upgrading to IVFFlat index (n=%d)", self._index.ntotal)
            quantizer = faiss.IndexFlatIP(self._dim)
            new_index = faiss.IndexIVFFlat(quantizer, self._dim, _IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
            # Reconstruct all existing vectors to train the new index
            existing = np.zeros((self._index.ntotal, self._dim), dtype="float32")
            for i in range(self._index.ntotal):
                self._index.reconstruct(i, existing[i])
            new_index.train(existing)
            new_index.add(existing)
            new_index.nprobe = _IVF_NPROBE
            self._index = new_index

    # ------------------------------------------------------------------ #
    #  Internal: Persistence                                               #
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        try:
            faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
            with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False)
            self._save_manifest()
        except OSError as exc:
            logger.error("[KnowledgeBase] Save failed: %s", exc)

    def _save_manifest(self) -> None:
        with open(self.index_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        index_path    = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.json"
        manifest_path = self.index_dir / "manifest.json"

        if index_path.exists() and metadata_path.exists():
            try:
                self._index    = faiss.read_index(str(index_path))
                with open(metadata_path, encoding="utf-8") as f:
                    self._metadata = json.load(f)
                logger.info(
                    "[KnowledgeBase] Loaded from disk: %d vectors, %d metadata rows",
                    self._index.ntotal, len(self._metadata)
                )
            except Exception as exc:
                logger.error("[KnowledgeBase] Failed to load from disk: %s — starting fresh.", exc)
                self._index    = faiss.IndexFlatIP(self._dim)
                self._metadata = []

        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    self._manifest = json.load(f)
            except Exception:
                self._manifest = {}
