import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import io
import time

class DocumentProcessor:
    """
    Handles PDF ingestion and vectorization (RAG) per session.
    """
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Scoped by session_id to prevent data leakage
        self.indices = {}   # session_id -> faiss.IndexFlatL2
        self.chunks = {}    # session_id -> list of chunks
        self.last_accessed = {} # session_id -> timestamp for cleanup
        
    def _ensure_session(self, session_id):
        if session_id not in self.indices:
            self.indices[session_id] = faiss.IndexFlatL2(self.embedding_dim)
            self.chunks[session_id] = []
        self.last_accessed[session_id] = time.time()

    def process_pdf(self, session_id, file_bytes):
        self._ensure_session(session_id)
        
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        full_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
            
        raw_chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
        
        added = 0
        for chunk in raw_chunks:
            if chunk.strip():
                vector = self.embedder.encode([chunk])
                self.indices[session_id].add(np.array(vector).astype('float32'))
                self.chunks[session_id].append(chunk)
                added += 1
                
        return added
        
    def query_documents(self, session_id, query, top_k=3):
        if session_id not in self.indices or self.indices[session_id].ntotal == 0:
            return ""
            
        self.last_accessed[session_id] = time.time()
        query_vector = self.embedder.encode([query]).astype('float32')
        distances, idxs = self.indices[session_id].search(query_vector, k=top_k)
        
        results = []
        for i in idxs[0]:
            if i != -1 and i < len(self.chunks[session_id]):
                results.append(self.chunks[session_id][i])
                
        return "\n\n---\n\n".join(results)
        
    def cleanup_idle_sessions(self, idle_time_seconds=3600):
        current_time = time.time()
        to_delete = [sid for sid, last_acc in self.last_accessed.items() 
                     if current_time - last_acc > idle_time_seconds]
        for sid in to_delete:
            del self.indices[sid]
            del self.chunks[sid]
            del self.last_accessed[sid]
