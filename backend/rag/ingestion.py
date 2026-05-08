"""
rag/ingestion.py
================
Multi-format document ingestion pipeline.

Supported formats:
    .pdf   — PyPDF2 page-by-page extraction
    .txt   — plain text
    .md    — Markdown (stripped to plain text)
    .json  — JSON with configurable text field extraction
    .jsonl — JSONL (line-by-line JSON objects)
    URL    — basic web page fetch + HTML→text conversion

Pipeline per document:
    1. Load raw text from source
    2. Clean (normalise whitespace, strip boilerplate)
    3. Chunk (sentence-aware, overlapping)
    4. Return list[Chunk] ready for KnowledgeBase.add_chunks()

Usage:
    from rag.ingestion import DocumentIngestionPipeline
    pipeline = DocumentIngestionPipeline()

    chunks = pipeline.ingest_file("docs/manual.pdf")
    chunks = pipeline.ingest_text("raw text...", source="inline_faq")
    chunks = pipeline.ingest_url("https://example.org/docs")  # replace with real URL
    chunks = pipeline.ingest_directory("docs/")        # batch
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

from .chunker import Chunk, TextChunker

logger = logging.getLogger("rag.ingestion")

# ---------------------------------------------------------------------------
# Optional heavy dependencies (graceful degradation)
# ---------------------------------------------------------------------------

try:
    import PyPDF2
    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False
    logger.warning("[Ingestion] PyPDF2 not installed — PDF ingestion unavailable.")

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False
    logger.warning("[Ingestion] beautifulsoup4 not installed — URL ingestion will use raw HTML stripping.")


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Remove HTML tags. Uses BeautifulSoup if available, else regex fallback."""
    if _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ")
    # Regex fallback
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&[a-z]+;', ' ', text)
    return text


def _clean_text(text: str) -> str:
    """Normalize whitespace, strip control chars."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


def _strip_markdown(text: str) -> str:
    """Convert Markdown to plain text (keep content, remove formatting symbols)."""
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # headers
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)          # bold/italic
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)               # code
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)                   # images
    text = re.sub(r'\[(.+?)\]\(.*?\)', r'\1', text)              # links
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)    # bullets
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)    # numbered lists
    return text


# ---------------------------------------------------------------------------
# Loaders (one per format)
# ---------------------------------------------------------------------------

class _Loader:
    def load(self, *args, **kwargs) -> str:
        raise NotImplementedError


class PDFLoader(_Loader):
    def load(self, path: Path) -> str:
        if not _HAS_PYPDF2:
            raise RuntimeError("PyPDF2 is required for PDF ingestion: pip install PyPDF2")
        reader = PyPDF2.PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)

    def load_bytes(self, data: bytes) -> str:
        if not _HAS_PYPDF2:
            raise RuntimeError("PyPDF2 is required for PDF ingestion.")
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)


class TextLoader(_Loader):
    def load(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


class MarkdownLoader(_Loader):
    def load(self, path: Path) -> str:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return _strip_markdown(raw)


class JSONLoader(_Loader):
    """Extract text from JSON. Specify the key(s) to extract."""
    def __init__(self, text_keys: list[str] | None = None):
        self.text_keys = text_keys or ["text", "content", "body", "description", "answer"]

    def load(self, path: Path) -> str:
        data = json.loads(path.read_text(encoding="utf-8"))
        return self._extract(data)

    def _extract(self, data) -> str:
        parts: list[str] = []
        if isinstance(data, dict):
            for key in self.text_keys:
                if key in data and isinstance(data[key], str):
                    parts.append(data[key])
            # Recurse into nested dicts
            for v in data.values():
                if isinstance(v, (dict, list)):
                    parts.append(self._extract(v))
        elif isinstance(data, list):
            for item in data:
                parts.append(self._extract(item))
        elif isinstance(data, str):
            parts.append(data)
        return "\n".join(filter(None, parts))


class JSONLLoader(_Loader):
    def __init__(self, text_keys: list[str] | None = None):
        self._json_loader = JSONLoader(text_keys)

    def load(self, path: Path) -> str:
        parts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    parts.append(self._json_loader._extract(obj))
                except json.JSONDecodeError:
                    pass
        return "\n".join(parts)


class URLLoader(_Loader):
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def load(self, url: str) -> str:
        try:
            with urlopen(url, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except URLError as exc:
            raise RuntimeError(f"Failed to fetch URL {url!r}: {exc}") from exc
        return _strip_html(raw)


# ---------------------------------------------------------------------------
# DocumentIngestionPipeline
# ---------------------------------------------------------------------------

_LOADER_MAP: dict[str, _Loader] = {
    ".pdf":   PDFLoader(),
    ".txt":   TextLoader(),
    ".md":    MarkdownLoader(),
    ".json":  JSONLoader(),
    ".jsonl": JSONLLoader(),
}


class DocumentIngestionPipeline:
    """
    High-level pipeline: load → clean → chunk → return list[Chunk].

    Args:
        chunk_size : target chunk size in characters
        overlap    : sliding-window overlap in characters
        min_size   : minimum chunk length (shorter chunks discarded)
    """

    def __init__(
        self,
        chunk_size: int = 400,
        overlap:    int = 80,
        min_size:   int = 30,
    ):
        self.chunker    = TextChunker(chunk_size=chunk_size, overlap=overlap, min_size=min_size)
        self._url_loader = URLLoader()

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def ingest_file(self, path: Path | str, extra_metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Ingest a single file. Format detected from extension.

        Args:
            path           : absolute or relative path to the document
            extra_metadata : optional dict merged into each Chunk.metadata

        Returns:
            list[Chunk] ready to be added to KnowledgeBase
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext    = path.suffix.lower()
        loader = _LOADER_MAP.get(ext)
        if loader is None:
            raise ValueError(f"Unsupported file extension: {ext!r}. Supported: {list(_LOADER_MAP)}")

        t0      = time.perf_counter()
        raw     = loader.load(path)
        cleaned = _clean_text(raw)
        chunks  = self.chunker.chunk(
            cleaned,
            source=path.name,
            metadata={**(extra_metadata or {}), "file_path": str(path), "format": ext},
        )
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "[Ingestion] %s → %d chunks (%.1fms)",
            path.name, len(chunks), elapsed
        )
        return chunks

    def ingest_bytes(self, data: bytes, source: str, fmt: str = ".pdf", extra_metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Ingest from raw bytes (e.g., uploaded file data).
        `fmt` should be the file extension: '.pdf', '.txt', etc.
        """
        if fmt == ".pdf":
            raw = PDFLoader().load_bytes(data)
        elif fmt in (".txt", ".md"):
            raw = data.decode("utf-8", errors="replace")
            if fmt == ".md":
                raw = _strip_markdown(raw)
        else:
            raise ValueError(f"ingest_bytes does not support format {fmt!r}")

        cleaned = _clean_text(raw)
        return self.chunker.chunk(
            cleaned,
            source=source,
            metadata={**(extra_metadata or {}), "format": fmt},
        )

    def ingest_text(self, text: str, source: str = "inline", extra_metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Ingest a raw text string directly (e.g., inline FAQ, API response).
        """
        cleaned = _clean_text(text)
        chunks  = self.chunker.chunk(
            cleaned,
            source=source,
            metadata={**(extra_metadata or {}), "format": "text"},
        )
        logger.info("[Ingestion] inline:%s → %d chunks", source, len(chunks))
        return chunks

    def ingest_url(self, url: str, extra_metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Fetch a web page and ingest its text content.
        """
        t0  = time.perf_counter()
        raw = self._url_loader.load(url)
        cleaned = _clean_text(raw)
        chunks  = self.chunker.chunk(
            cleaned,
            source=url,
            metadata={**(extra_metadata or {}), "format": "url", "url": url},
        )
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("[Ingestion] %s → %d chunks (%.1fms)", url, len(chunks), elapsed)
        return chunks

    def ingest_directory(
        self,
        directory:      Path | str,
        recursive:      bool          = True,
        extensions:     list[str]     = None,
        extra_metadata: Optional[dict] = None,
    ) -> dict[str, list[Chunk]]:
        """
        Batch-ingest all supported files in a directory.

        Returns:
            dict mapping filename → list[Chunk]
        """
        directory  = Path(directory)
        extensions = extensions or list(_LOADER_MAP.keys())
        pattern    = "**/*" if recursive else "*"

        results: dict[str, list[Chunk]] = {}
        for file in directory.glob(pattern):
            if file.suffix.lower() in extensions and file.is_file():
                try:
                    chunks = self.ingest_file(file, extra_metadata=extra_metadata)
                    results[file.name] = chunks
                except Exception as exc:
                    logger.error("[Ingestion] Skipping %s: %s", file.name, exc)

        total = sum(len(v) for v in results.values())
        logger.info(
            "[Ingestion] Directory %s → %d files → %d total chunks",
            directory, len(results), total
        )
        return results
