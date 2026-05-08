import logging
import time
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from .knowledge_base import KnowledgeBase, SearchResult

logger = logging.getLogger("rag.retriever")

# ---------------------------------------------------------------------------
# Constants & Models
# ---------------------------------------------------------------------------
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_HAS_CROSS_ENCODER = True # Assume available since we're in a high-perf environment

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Structured result from the production retriever."""
    chunks:        list[SearchResult]
    context_block: str
    original_query: str
    expanded_queries: List[str]
    total_candidates: int
    used_reranker: bool   = False
    used_mmr:      bool   = False
    latency_ms:    float  = 0.0

try:
    from feedback.mistake_memory import MistakeMemory
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from feedback.mistake_memory import MistakeMemory

# ---------------------------------------------------------------------------
# RAGRetriever (Production Grade)
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    Production-grade multi-stage retrieval orchestrator.
    Stages: Rewrite -> Hybrid Search -> RRF Fusion -> Rerank -> MMR -> Compress.
    """

    def __init__(
        self,
        knowledge_base:    KnowledgeBase,
        top_k_candidates:  int   = 40,
        top_n_results:     int   = 6,
        score_floor:       float = 0.25,
        use_reranker:      bool  = True,
        use_mmr:           bool  = True,
        mmr_lambda:        float = 0.5,
        compress_context:  bool  = True
    ):
        self.kb               = knowledge_base
        self.top_k            = top_k_candidates
        self.top_n            = top_n_results
        self.score_floor      = score_floor
        self.use_reranker     = use_reranker
        self.use_mmr          = use_mmr
        self.mmr_lambda       = mmr_lambda
        self.compress_context = compress_context
        
        self._embedder        = knowledge_base._embedder
        self._reranker        = None # Lazy load
        
        # Long-term Learning: Mistake Memory
        self.mistake_memory   = MistakeMemory()
        
    def _get_reranker(self):
        if self._reranker is None:
            logger.info("[RAGRetriever] Loading cross-encoder...")
            self._reranker = CrossEncoder(_CROSS_ENCODER_MODEL)
        return self._reranker

    def _rewrite_query(self, query: str) -> List[str]:
        """
        Simple query expansion for now. 
        In production, this would call the LLM to generate HyDE or variations.
        """
        # For now, we'll just return the original query and some variations
        variations = [query]
        # Basic heuristic expansion (e.g. removing stop words or adding context)
        # Placeholder for LLM-based rewriting
        return variations

    def _mmr(
        self,
        query_vec:    np.ndarray,
        candidates:   list[SearchResult],
        top_n:        int,
        lambda_param: float = 0.5,
    ) -> list[SearchResult]:
        if not candidates: return []
        if len(candidates) <= top_n: return candidates

        # Use the bi-encoder to get embeddings for MMR comparison
        cand_texts = [c.text for c in candidates]
        cand_vecs = self._embedder.encode(cand_texts, show_progress_bar=False)
        cand_vecs = np.array(cand_vecs, dtype="float32")
        # L2-normalize
        norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9
        cand_vecs = cand_vecs / norms
        
        query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(top_n):
            best_idx, best_score = -1, float("-inf")
            for i in remaining:
                relevance = float(query_vec_norm @ cand_vecs[i])
                redundancy = max(
                    (float(cand_vecs[i] @ cand_vecs[j]) for j in selected_indices),
                    default=0.0,
                )
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if mmr_score > best_score:
                    best_score, best_idx = mmr_score, i

            if best_idx == -1: break
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def _compress_chunk(self, query: str, text: str) -> str:
        """
        Sentence-level compression: Keep only the most relevant sentences.
        """
        if not self.compress_context: return text
        
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) <= 2: return text
        
        # Simple keyword-based relevance scoring for compression
        query_words = set(query.lower().split())
        scored_sentences = []
        for s in sentences:
            s_words = set(s.lower().split())
            overlap = len(query_words.intersection(s_words))
            scored_sentences.append((overlap, s))
        
        # Sort and keep top N sentences (preserving order)
        top_indices = sorted(
            range(len(scored_sentences)), 
            key=lambda i: scored_sentences[i][0], 
            reverse=True
        )[:3] # Keep top 3 relevant sentences
        
        compressed = " ".join([sentences[i] for i in sorted(top_indices)])
        return compressed

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        
        # Stage 1: Query Expansion
        queries = self._rewrite_query(query)
        
        # Stage 2: Hybrid Retrieval
        all_candidates: Dict[str, SearchResult] = {}
        for q in queries:
            results = self.kb.search(q, top_k=self.top_k, score_floor=self.score_floor, hybrid=True)
            for r in results:
                # Deduplicate by content hash
                all_candidates[r.hash] = r
        
        candidates = list(all_candidates.values())
        total_candidates = len(candidates)

        if not candidates:
            return RetrievalResult([], "", query, queries, 0, latency_ms=0.0)

        # Stage 3: Cross-Encoder Reranking
        used_reranker = False
        if self.use_reranker and len(candidates) > 1:
            try:
                ce = self._get_reranker()
                pairs = [(query, c.text) for c in candidates]
                scores = ce.predict(pairs)
                for i, c in enumerate(candidates):
                    c.score = float(scores[i])
                candidates.sort(key=lambda x: x.score, reverse=True)
                used_reranker = True
                # Keep top K after reranking
                candidates = candidates[:self.top_k // 2]
            except Exception as e:
                logger.error(f"[RAGRetriever] Reranking error: {e}")

        # Stage 4: MMR Diversification
        used_mmr = False
        if self.use_mmr and len(candidates) > self.top_n:
            query_vec = self._embedder.encode([query], show_progress_bar=False)[0]
            candidates = self._mmr(query_vec, candidates, self.top_n, self.mmr_lambda)
            used_mmr = True
        else:
            candidates = candidates[:self.top_n]

        # Stage 5: Context Compression
        for c in candidates:
            c.text = self._compress_chunk(query, c.text)

        # Stage 6: Long-term Learning (Mistake Memory)
        corrections_block = self.mistake_memory.format_corrections_for_prompt(query)

        # Stage 7: Formatting
        context_block = self._format(candidates, corrections_block)

        latency = round((time.perf_counter() - t0) * 1000, 2)
        return RetrievalResult(
            chunks=candidates,
            context_block=context_block,
            original_query=query,
            expanded_queries=queries,
            total_candidates=total_candidates,
            used_reranker=used_reranker,
            used_mmr=used_mmr,
            latency_ms=latency
        )

    def _format(self, results: list[SearchResult], corrections_block: str = "") -> str:
        if not results and not corrections_block: return ""
        
        lines = ["### Verified Knowledge Base Context"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] Source: {r.source}")
            lines.append(r.text.strip())
            lines.append("")
        
        if corrections_block:
            lines.append(corrections_block)
            lines.append("")

        lines.append("Instructions: Use the provided context to answer the query. If the information is not present, say so. Do not hallucinate.")
        return "\n".join(lines)

    def get_context(self, query: str) -> str:
        return self.retrieve(query).context_block
