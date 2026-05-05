"""
rag/chunker.py
==============
Smart text chunker with sentence-boundary awareness and sliding-window overlap.

Why not simple fixed-size splits?
  Fixed 500-char splits cut mid-sentence, destroying semantic coherence.
  This chunker respects sentence boundaries and adds configurable overlap so
  that retrieval context is always complete and coherent.

Usage:
    from rag.chunker import TextChunker
    chunker = TextChunker(chunk_size=400, overlap=80)
    chunks = chunker.chunk("Long document text...", source="my_doc.pdf")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""
    text:        str
    source:      str                    # filename / URL / identifier
    chunk_index: int                    # position within the source document
    char_start:  int                    # character offset in original text
    char_end:    int
    metadata:    dict = field(default_factory=dict)  # arbitrary extra fields

    def __len__(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "text":        self.text,
            "source":      self.source,
            "chunk_index": self.chunk_index,
            "char_start":  self.char_start,
            "char_end":    self.char_end,
            "metadata":    self.metadata,
        }


# ---------------------------------------------------------------------------
# Sentence Splitter
# ---------------------------------------------------------------------------

# Regex that splits on sentence-ending punctuation followed by whitespace,
# while not breaking on abbreviations like "Dr.", "U.S.A.", numbers "3.14".
_SENT_BOUNDARY = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
)


def _split_sentences(text: str) -> list[str]:
    """Split text into a list of sentences (best-effort)."""
    sentences = _SENT_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Sentence-aware sliding-window chunker.

    Args:
        chunk_size : target chunk size in characters (default 400)
        overlap    : character overlap between consecutive chunks (default 80)
        min_size   : discard chunks shorter than this (default 30)
    """

    def __init__(
        self,
        chunk_size: int = 400,
        overlap:    int = 80,
        min_size:   int = 30,
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.min_size   = min_size

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def chunk(
        self,
        text:     str,
        source:   str = "unknown",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Chunk a document into overlapping, sentence-aligned segments.

        Returns a list of Chunk dataclass instances.
        """
        text = self._normalize(text)
        if not text:
            return []

        sentences = _split_sentences(text)
        return self._build_chunks(sentences, text, source, metadata or {})

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip excess whitespace and normalize newlines."""
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def _build_chunks(
        self,
        sentences: list[str],
        original:  str,
        source:    str,
        metadata:  dict,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        buf:    list[str]   = []
        buf_len = 0
        chunk_idx = 0
        char_cursor = 0

        for sent in sentences:
            sent_len = len(sent) + 1  # +1 for space separator

            # --- Buffer would overflow: emit current chunk, start new with overlap ---
            if buf and buf_len + sent_len > self.chunk_size:
                chunk_text = " ".join(buf)
                char_start = original.find(chunk_text, max(0, char_cursor - len(chunk_text) - 20))
                char_end   = char_start + len(chunk_text)

                if len(chunk_text) >= self.min_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        source=source,
                        chunk_index=chunk_idx,
                        char_start=max(0, char_start),
                        char_end=char_end,
                        metadata=metadata,
                    ))
                    chunk_idx += 1

                # Slide window: carry over overlap sentences
                overlap_buf: list[str] = []
                overlap_len = 0
                for s in reversed(buf):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                buf     = overlap_buf
                buf_len = overlap_len

            buf.append(sent)
            buf_len += sent_len
            char_cursor += sent_len

        # --- Flush remaining sentences ---
        if buf:
            chunk_text = " ".join(buf)
            if len(chunk_text) >= self.min_size:
                char_start = original.rfind(chunk_text[:50]) if len(chunk_text) > 50 else original.rfind(chunk_text)
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_idx,
                    char_start=max(0, char_start),
                    char_end=max(0, char_start) + len(chunk_text),
                    metadata=metadata,
                ))

        return chunks
