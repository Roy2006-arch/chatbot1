"""
rag/retriever.py
================
RAG Retriever — the public interface used by the chat endpoint.

Responsibilities:
  1. Embed a user query
  2. Search the global KnowledgeBase
  3. Optionally re-rank results with a cross-encoder (if available)
  4. Apply MMR (Maximal Marginal Relevance) to diversify results
  5. Format retrieved chunks into a clean context block for injection into the LLM prompt

Architecture:
    User Query
        │
        ▼
    Bi-encoder (SentenceTransformer)   ← fast ANN search
        │
        ▼
    top_k_candidates (e.g. 20)
        │
        ├─► [Optional] Cross-encoder reranking    ← slow but accurate
        │
        ▼
    MMR Diversification
        │
        ▼
    top_n final results (e.g. 5)
        │
        ▼
    Format as context block → injected into LLM prompt

Usage:
    from rag.retriever import RAGRetriever
    from rag.knowledge_base import KnowledgeBase

    kb        = KnowledgeBase()
    retriever = RAGRetriever(kb)

    context_block = retriever.get_context("What is the refund policy?")
    # → "### Retrieved Knowledge\n[1] Source: faq.pdf ..."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .knowledge_base import KnowledgeBase, SearchResult

logger = logging.getLogger("rag.retriever")

# ---------------------------------------------------------------------------
# Optional: Cross-Encoder reranking
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _cross_encoder: Optional[CrossEncoder] = None

    def _get_cross_encoder() -> CrossEncoder:
        global _cross_encoder
        if _cross_encoder is None:
            logger.info("[RAGRetriever] Loading cross-encoder: %s", _CROSS_ENCODER_MODEL)
            _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
        return _cross_encoder

    _HAS_CROSS_ENCODER = True
except Exception:
    _HAS_CROSS_ENCODER = False
    logger.info("[RAGRetriever] Cross-encoder not available — using bi-encoder scores only.")


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Structured result from the retriever, ready for prompt injection."""
    chunks:        list[SearchResult]
    context_block: str                     # formatted string for LLM prompt
    query:         str
    total_found:   int
    used_reranker: bool   = False
    used_mmr:      bool   = False
    latency_ms:    float  = 0.0


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance)
# ---------------------------------------------------------------------------

def _mmr(
    query_vec:    np.ndarray,
    candidates:   list[SearchResult],
    embedder:     SentenceTransformer,
    top_n:        int,
    lambda_param: float = 0.6,
) -> list[SearchResult]:
    """
    Maximal Marginal Relevance selection to maximise relevance while
    minimising redundancy between selected chunks.

    Args:
        lambda_param : trade-off between relevance (1.0) and diversity (0.0)
    """
    if len(candidates) <= top_n:
        return candidates

    cand_vecs = embedder.encode([c.text for c in candidates], show_progress_bar=False)
    cand_vecs = np.array(cand_vecs, dtype="float32")
    # L2-normalise for cosine sim
    norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9
    cand_vecs = cand_vecs / norms
    query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(top_n):
        best_idx, best_score = -1, float("-inf")
        for i in remaining:
            relevance  = float(query_vec_norm @ cand_vecs[i])
            redundancy = max(
                (float(cand_vecs[i] @ cand_vecs[j]) for j in selected_indices),
                default=0.0,
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i

        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    Retrieves and formats relevant knowledge-base chunks for LLM context injection.

    Args:
        knowledge_base   : the global KnowledgeBase instance
        top_k_candidates : how many candidates to fetch from FAISS before reranking/MMR
        top_n_results    : how many final chunks to inject into the prompt
        score_floor      : minimum cosine similarity to include a result
        use_reranker     : whether to apply cross-encoder reranking (if available)
        use_mmr          : whether to diversify with MMR
        mmr_lambda       : MMR relevance-diversity trade-off (0=diverse, 1=relevant)
    """

    def __init__(
        self,
        knowledge_base:    KnowledgeBase,
        top_k_candidates:  int   = 20,
        top_n_results:     int   = 5,
        score_floor:       float = 0.45,   # raised from 0.20: avoid weak/off-topic context
        use_reranker:      bool  = True,
        use_mmr:           bool  = True,
        mmr_lambda:        float = 0.6,
    ):
        self.kb               = knowledge_base
        self.top_k_candidates = top_k_candidates
        self.top_n            = top_n_results
        self.score_floor      = score_floor
        self.use_reranker     = use_reranker and _HAS_CROSS_ENCODER
        self.use_mmr          = use_mmr
        self.mmr_lambda       = mmr_lambda
        # Reuse the same embedder as the knowledge base for the MMR step
        self._embedder        = knowledge_base._embedder

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Full retrieval pipeline: search → rerank → MMR → format.

        Returns a RetrievalResult with context_block ready for prompt injection.
        """
        import time
        t0 = time.perf_counter()

        # ── Step 1: Bi-encoder ANN search ─────────────────────────────
        candidates = self.kb.search(query, top_k=self.top_k_candidates, score_floor=self.score_floor)
        total_found = len(candidates)

        if not candidates:
            return RetrievalResult(
                chunks=[], context_block="", query=query, total_found=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2)
            )

        # ── Step 2: Optional cross-encoder reranking ──────────────────
        used_reranker = False
        if self.use_reranker and len(candidates) > self.top_n:
            try:
                ce = _get_cross_encoder()
                pairs  = [(query, c.text) for c in candidates]
                scores = ce.predict(pairs)
                for i, result in enumerate(candidates):
                    result.score = float(scores[i])
                candidates.sort(key=lambda r: r.score, reverse=True)
                used_reranker = True
                logger.debug("[RAGRetriever] Cross-encoder reranked %d candidates.", len(candidates))
            except Exception as exc:
                logger.warning("[RAGRetriever] Cross-encoder failed: %s", exc)

        # ── Step 3: MMR diversification ───────────────────────────
        used_mmr = False
        if self.use_mmr and len(candidates) > self.top_n:
            query_vec = self._embedder.encode([query], show_progress_bar=False)[0]
            candidates = _mmr(query_vec, candidates, self._embedder, self.top_n, self.mmr_lambda)
            used_mmr = True
        else:
            candidates = candidates[:self.top_n]

        # ── Step 4: Format into context block ───────────────────────
        context_block = self._format(candidates, query)

        latency = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "[RAGRetriever] query=%r | candidates=%d | returned=%d | reranker=%s | mmr=%s | %.1fms",
            query[:60], total_found, len(candidates), used_reranker, used_mmr, latency
        )

        return RetrievalResult(
            chunks=candidates,
            context_block=context_block,
            query=query,
            total_found=total_found,
            used_reranker=used_reranker,
            used_mmr=used_mmr,
            latency_ms=latency,
        )

    def get_context(self, query: str) -> str:
        """
        Convenience wrapper — returns just the formatted context string.
        Returns empty string if the knowledge base is empty or no results found.
        """
        result = self.retrieve(query)
        return result.context_block

    # ------------------------------------------------------------------ #
    #  Formatting                                                          #
    # ------------------------------------------------------------------ #

    def _format(self, results: list[SearchResult], query: str) -> str:
        """
        Produce a clean, structured context block for injection into the LLM prompt.
        """
        if not results:
            return ""

        lines = ["### Retrieved Knowledge\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] Source: {r.source} (relevance: {r.score:.2f})")
            lines.append(r.text.strip())
            lines.append("")

        lines.append("---")
        lines.append(
            "Base your answer on the retrieved knowledge above. Quote or paraphrase it "
            "and, where useful, reference the source number (e.g. [1]). If the context "
            "does not contain the answer, say you don't have that information rather than "
            "guessing. Never fabricate details that contradict the context."
        )
        return "\n".join(lines)
