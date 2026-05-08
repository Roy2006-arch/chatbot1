import json
import logging
import threading
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .chunker import Chunk

logger = logging.getLogger("rag.knowledge_base")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge_base"
_EMBED_MODEL       = "all-MiniLM-L6-v2"
_EMBED_DIM         = 384

# Switch from Flat to IVF when we exceed this many vectors
_IVF_THRESHOLD = 10_000
_IVF_NLIST     = 100
_IVF_NPROBE    = 10

# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class SearchResult:
    """A single retrieval result with text, score, and provenance."""
    __slots__ = ("text", "source", "chunk_index", "score", "metadata", "hash")

    def __init__(self, text: str, source: str, chunk_index: int, score: float, metadata: dict, content_hash: str = ""):
        self.text        = text
        self.source      = source
        self.chunk_index = chunk_index
        self.score       = score
        self.metadata    = metadata
        self.hash        = content_hash

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
    Thread-safe, disk-persistent Hybrid Knowledge Base (FAISS + BM25).
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

        # Semantic Index (FAISS)
        self._index: faiss.Index = faiss.IndexFlatIP(embed_dim)
        
        # Keyword Index (BM25)
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

        # Metadata & Manifest
        self._metadata: list[dict]  = []
        self._manifest: dict        = {}
        self._hashes: set[str]      = set() # For deduplication
        
        # Embedding Cache
        from functools import lru_cache
        self._cached_encode = lru_cache(maxsize=1024)(self._embedder.encode)

        self._load()

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Embed and index a list of Chunk objects with deduplication.
        """
        if not chunks:
            return 0

        # Deduplicate against current session and persistent hashes
        unique_chunks = []
        for c in chunks:
            h = self._get_hash(c.text)
            if h not in self._hashes:
                unique_chunks.append(c)
                self._hashes.add(h)

        if not unique_chunks:
            return 0

        texts = [c.text for c in unique_chunks]
        vectors = self._embed(texts)

        with self._lock:
            self._index.add(vectors)
            for chunk in unique_chunks:
                meta = {
                    "text":        chunk.text,
                    "source":      chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "char_start":  chunk.char_start,
                    "char_end":    chunk.char_end,
                    "metadata":    chunk.metadata,
                    "hash":        self._get_hash(chunk.text),
                    "added_at":    time.time(),
                }
                self._metadata.append(meta)
                # Update BM25 corpus
                self._tokenized_corpus.append(chunk.text.lower().split())
            
            # Re-initialize BM25 with new corpus
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            
            self._maybe_upgrade_index()
            self._save()

        logger.info("[KnowledgeBase] Added %d unique chunks | Total: %d", len(unique_chunks), self._index.ntotal)
        return len(unique_chunks)

    def search(
        self,
        query:       str,
        top_k:       int   = 20,
        score_floor: float = 0.20,
        hybrid:      bool  = True
    ) -> list[SearchResult]:
        """
        Hybrid search combining FAISS (Semantic) and BM25 (Keyword).
        Uses Reciprocal Rank Fusion (RRF) if hybrid is enabled.
        """
        if self._index.ntotal == 0:
            return []

        # 1. Semantic Search
        query_vec = self._embed([query])
        with self._lock:
            k = min(top_k * 2, self._index.ntotal)
            sem_scores, sem_idxs = self._index.search(query_vec, k)

        sem_results = []
        sem_scores_filtered = []
        for score, idx in zip(sem_scores[0], sem_idxs[0]):
            if idx == -1 or score < score_floor: continue
            sem_results.append(idx)
            sem_scores_filtered.append(score)

        if not hybrid or not self._bm25:
            return self._hydrate(sem_results[:top_k], sem_scores_filtered[:top_k])

        # 2. Keyword Search
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        bm25_idxs = np.argsort(bm25_scores)[::-1][:k]

        # 3. Reciprocal Rank Fusion (RRF)
        # rrf_score = 1 / (rank + k_constant)
        K = 60 
        rrf_map: Dict[int, float] = {}
        
        for rank, idx in enumerate(sem_results):
            rrf_map[idx] = rrf_map.get(idx, 0) + 1.0 / (rank + K)
        
        for rank, idx in enumerate(bm25_idxs):
            rrf_map[idx] = rrf_map.get(idx, 0) + 1.0 / (rank + K)

        # Sort by RRF score
        fused_idxs = sorted(rrf_map.keys(), key=lambda i: rrf_map[i], reverse=True)[:top_k]
        
        return self._hydrate(fused_idxs, [rrf_map[i] for i in fused_idxs])

    def _hydrate(self, idxs: List[int], scores: List[float]) -> List[SearchResult]:
        results = []
        for idx, score in zip(idxs, scores):
            meta = self._metadata[idx]
            results.append(SearchResult(
                text=meta["text"],
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                score=float(score),
                metadata=meta.get("metadata", {}),
                content_hash=meta.get("hash", "")
            ))
        return results

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Produce L2-normalised embeddings.
        Uses LRU cache for single queries.
        """
        if len(texts) == 1:
            # Use cached version for single queries
            vecs = self._cached_encode(texts, batch_size=1, show_progress_bar=False)
        else:
            vecs = self._embedder.encode(texts, batch_size=64, show_progress_bar=False)
            
        vecs = np.array(vecs, dtype="float32")
        faiss.normalize_L2(vecs)
        return vecs

    def _maybe_upgrade_index(self) -> None:
        if self._index.ntotal >= _IVF_THRESHOLD and isinstance(self._index, faiss.IndexFlatIP):
            logger.info("[KnowledgeBase] Upgrading index...")
            quantizer = faiss.IndexFlatIP(self._dim)
            new_index = faiss.IndexIVFFlat(quantizer, self._dim, _IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
            existing = np.zeros((self._index.ntotal, self._dim), dtype="float32")
            for i in range(self._index.ntotal):
                self._index.reconstruct(i, existing[i])
            new_index.train(existing)
            new_index.add(existing)
            new_index.nprobe = _IVF_NPROBE
            self._index = new_index

    def stats(self) -> dict:
        return {
            "total_vectors": self._index.ntotal,
            "total_sources": len(self._manifest),
            "index_type": type(self._index).__name__,
            "bm25_enabled": self._bm25 is not None
        }

    def register_source(self, source: str) -> None:
        with self._lock:
            self._manifest[source] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._save_manifest()

    def is_ingested(self, source: str) -> bool:
        return source in self._manifest

    def _save(self) -> None:
        faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)
        self._save_manifest()

    def _save_manifest(self) -> None:
        with open(self.index_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.json"
        manifest_path = self.index_dir / "manifest.json"

        if index_path.exists() and metadata_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                with open(metadata_path, encoding="utf-8") as f:
                    self._metadata = json.load(f)
                # Rebuild hashes and BM25 corpus
                self._hashes = {m.get("hash") for m in self._metadata if m.get("hash")}
                self._tokenized_corpus = [m["text"].lower().split() for m in self._metadata]
                if self._tokenized_corpus:
                    self._bm25 = BM25Okapi(self._tokenized_corpus)
                logger.info("[KnowledgeBase] Loaded %d vectors", self._index.ntotal)
            except Exception as exc:
                logger.error("[KnowledgeBase] Load failed: %s", exc)
                self._index = faiss.IndexFlatIP(self._dim)
                self._metadata = []

        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                self._manifest = json.load(f)
