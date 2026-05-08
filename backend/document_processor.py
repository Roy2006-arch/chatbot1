import faiss
import numpy as np
import PyPDF2
import io
import time
import logging
import threading
from backend.shared_resources import ModelRegistry, get_request_cache
from backend.rag.chunker import TextChunker

logger = logging.getLogger("chatbot.document_processor")


class DocumentProcessor:
    def __init__(self):
        self.embedder = ModelRegistry.get_embedder()
        self.embedding_dim = 384

        self.indices: dict[str, faiss.Index] = {}
        self.chunks: dict[str, list[str]] = {}
        self.last_accessed: dict[str, float] = {}
        self._lock = threading.RLock()

        self.chunker = TextChunker(chunk_size=500, overlap=100)

    def _ensure_session(self, session_id):
        with self._lock:
            if session_id not in self.indices:
                self.indices[session_id] = faiss.IndexFlatIP(self.embedding_dim)
                self.chunks[session_id] = []
            self.last_accessed[session_id] = time.time()

    def _cached_encode(self, texts, **kwargs):
        cache = get_request_cache()
        if cache is not None:
            return cache.encode(self.embedder, texts, **kwargs)
        return self.embedder.encode(texts, **kwargs)

    def process_pdf(self, session_id, file_bytes):
        self._ensure_session(session_id)

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            full_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + "\n"
        except Exception as e:
            logger.error("Error reading PDF for session %s: %s", session_id, e)
            return 0

        smart_chunks = self.chunker.chunk(full_text, source="user_upload")

        if not smart_chunks:
            return 0

        texts = [c.text for c in smart_chunks]
        vectors = self._cached_encode(texts, batch_size=32, show_progress_bar=False)
        vectors = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors)

        with self._lock:
            self.indices[session_id].add(vectors)
            self.chunks[session_id].extend(texts)

        logger.info("Indexed %d smart chunks for session %s", len(smart_chunks), session_id)
        return len(smart_chunks)

    def query_documents(self, session_id, query, top_k=3):
        with self._lock:
            if session_id not in self.indices or self.indices[session_id].ntotal == 0:
                return ""
            self.last_accessed[session_id] = time.time()

        query_vector = self._cached_encode([query]).astype('float32')
        faiss.normalize_L2(query_vector)

        with self._lock:
            distances, idxs = self.indices[session_id].search(query_vector, k=top_k)

        results = []
        for i in idxs[0]:
            if i != -1 and i < len(self.chunks.get(session_id, [])):
                results.append(self.chunks[session_id][i])

        return "\n\n---\n\n".join(results)

    def cleanup_idle_sessions(self, idle_time_seconds=3600):
        current_time = time.time()
        with self._lock:
            to_delete = [
                sid for sid, last_acc in self.last_accessed.items()
                if current_time - last_acc > idle_time_seconds
            ]
            for sid in to_delete:
                del self.indices[sid]
                del self.chunks[sid]
                del self.last_accessed[sid]
                logger.info("Cleaned up document session: %s", sid)
        return len(to_delete)
