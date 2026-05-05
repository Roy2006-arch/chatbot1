"""
rag/
====
Persistent Retrieval-Augmented Generation subsystem.

Two-tier retrieval architecture:
  Tier 1 — Global Knowledge Base  (this package)
            Pre-loaded domain documents; persisted to disk; shared across sessions.
  Tier 2 — Session Document Store  (../document_processor.py)
            Per-user PDF uploads; ephemeral; kept as-is.

Public API:
    from rag import KnowledgeBase, DocumentIngestionPipeline, RAGRetriever
"""

from .knowledge_base import KnowledgeBase
from .ingestion import DocumentIngestionPipeline
from .retriever import RAGRetriever

__all__ = ["KnowledgeBase", "DocumentIngestionPipeline", "RAGRetriever"]
