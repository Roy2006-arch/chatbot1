"""
feedback/mistake_memory.py
---------------------------
Logic for long-term "mistake memory". This module tracks past failed responses,
detects repeated mistakes via similarity search, and provides "past corrections"
to the generator to prevent repeating known errors.
"""

import json
import logging
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timezone
from .db_schema import get_conn, _now_utc
from sentence_transformers import SentenceTransformer

log = logging.getLogger("chatbot.mistake_memory")

# Threshold for similarity (0.0 to 1.0)
# If a new failed query is > 0.9 similar to an old one, we increment occurrence_count
SIMILARITY_THRESHOLD = 0.90

class MistakeMemory:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def record_failure(
        self,
        *,
        conv_id: str,
        session_id: str,
        prompt: str,
        response: str,
        source: str = "auto",
        composite_score: Optional[float] = None,
        grade: str = "?",
        failure_reasons: List[str] = None
    ):
        """
        Records a failed response. If a similar failure exists, increments its count.
        """
        embedding = self._embed(prompt)
        embedding_blob = embedding.tobytes()
        reasons_json = json.dumps(failure_reasons or [])
        
        conn = get_conn()
        
        # 1. Search for similar PREVIOUS UNRESOLVED failures
        # Note: In a large system, we'd use FAISS. For mistake memory (usually small), 
        # a simple scan of unresolved failures is fine.
        rows = conn.execute(
            "SELECT id, prompt, embedding, occurrence_count FROM failed_queries WHERE resolved = 0"
        ).fetchall()
        
        best_match_id = None
        for row in rows:
            if row["embedding"]:
                past_emb = np.frombuffer(row["embedding"], dtype=np.float32)
                # Cosine similarity (SentenceTransformers usually produce normalised vectors)
                sim = np.dot(embedding, past_emb) / (np.linalg.norm(embedding) * np.linalg.norm(past_emb) + 1e-9)
                
                if sim > SIMILARITY_THRESHOLD:
                    best_match_id = row["id"]
                    break
        
        if best_match_id:
            # Increment count
            conn.execute(
                "UPDATE failed_queries SET occurrence_count = occurrence_count + 1, timestamp_utc = ? WHERE id = ?",
                (_now_utc(), best_match_id)
            )
            log.info("[MistakeMemory] Repeated mistake detected (id=%d). Count incremented.", best_match_id)
        else:
            # Insert new failure
            conn.execute(
                """
                INSERT INTO failed_queries 
                (conv_id, session_id, prompt, response, composite_score, grade, 
                 failure_reasons, source, occurrence_count, embedding, timestamp_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (conv_id, session_id, prompt, response, composite_score, grade, 
                 reasons_json, source, embedding_blob, _now_utc())
            )
            log.info("[MistakeMemory] New failure recorded.")
            
        conn.commit()

    def get_relevant_corrections(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Searches for resolved or annotated mistakes that are similar to the current query.
        Returns the 'preferred_response' as a correction.
        """
        embedding = self._embed(query)
        conn = get_conn()
        
        # We look for failures that have a preferred_response (the "Correction")
        rows = conn.execute(
            "SELECT prompt, response as rejected, preferred_response as chosen, embedding "
            "FROM failed_queries WHERE preferred_response != ''"
        ).fetchall()
        
        matches = []
        for row in rows:
            if row["embedding"]:
                past_emb = np.frombuffer(row["embedding"], dtype=np.float32)
                sim = np.dot(embedding, past_emb) / (np.linalg.norm(embedding) * np.linalg.norm(past_emb) + 1e-9)
                
                if sim > 0.7: # Lower threshold for retrieval to provide "related" corrections
                    matches.append({
                        "prompt": row["prompt"],
                        "rejected": row["rejected"],
                        "correction": row["chosen"],
                        "similarity": float(sim)
                    })
        
        # Sort by similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_k]

    def format_corrections_for_prompt(self, query: str) -> str:
        """
        Formats relevant corrections into a string for LLM context.
        """
        corrections = self.get_relevant_corrections(query)
        if not corrections:
            return ""
            
        lines = ["### Past Learning (Anti-Mistake Memory)"]
        lines.append("The following are corrections to past mistakes I made on similar topics. "
                     "Use these to ensure the current response is accurate.")
        
        for i, c in enumerate(corrections, 1):
            lines.append(f"\n[{i}] Similar Past Query: {c['prompt']}")
            lines.append(f"    Previous Error: {c['rejected'][:200]}...")
            lines.append(f"    Correction: {c['correction']}")
            
        lines.append("\n---")
        return "\n".join(lines)
