import json
import logging
import numpy as np
from typing import List, Optional, Dict

from .db_schema import get_conn, _now_utc
from backend.shared_resources import ModelRegistry, get_request_cache

log = logging.getLogger("chatbot.mistake_memory")

SIMILARITY_THRESHOLD = 0.90


class MistakeMemory:
    def __init__(self):
        self.model = ModelRegistry.get_embedder()

    def _embed(self, text: str) -> np.ndarray:
        cache = get_request_cache()
        if cache is not None:
            return cache.encode(self.model, [text])[0]
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
        failure_reasons: List[str] = None,
    ):
        embedding = self._embed(prompt)
        embedding_blob = embedding.tobytes()
        reasons_json = json.dumps(failure_reasons or [])

        conn = get_conn()

        rows = conn.execute(
            "SELECT id, prompt, embedding, occurrence_count FROM failed_queries WHERE resolved = 0"
        ).fetchall()

        best_match_id = None
        for row in rows:
            if row["embedding"]:
                past_emb = np.frombuffer(row["embedding"], dtype=np.float32)
                sim = np.dot(embedding, past_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(past_emb) + 1e-9
                )
                if sim > SIMILARITY_THRESHOLD:
                    best_match_id = row["id"]
                    break

        if best_match_id:
            conn.execute(
                "UPDATE failed_queries SET occurrence_count = occurrence_count + 1, timestamp_utc = ? WHERE id = ?",
                (_now_utc(), best_match_id),
            )
            log.info("[MistakeMemory] Repeated mistake (id=%d). Count incremented.", best_match_id)
        else:
            conn.execute(
                """
                INSERT INTO failed_queries
                (conv_id, session_id, prompt, response, composite_score, grade,
                 failure_reasons, source, occurrence_count, embedding, timestamp_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (conv_id, session_id, prompt, response, composite_score, grade,
                 reasons_json, source, embedding_blob, _now_utc()),
            )
            log.info("[MistakeMemory] New failure recorded.")

        conn.commit()

    def get_relevant_corrections(self, query: str, top_k: int = 2) -> List[Dict]:
        embedding = self._embed(query)
        conn = get_conn()

        rows = conn.execute(
            "SELECT prompt, response as rejected, preferred_response as chosen, embedding "
            "FROM failed_queries WHERE preferred_response != ''"
        ).fetchall()

        matches = []
        for row in rows:
            if row["embedding"]:
                past_emb = np.frombuffer(row["embedding"], dtype=np.float32)
                sim = np.dot(embedding, past_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(past_emb) + 1e-9
                )
                if sim > 0.7:
                    matches.append({
                        "prompt": row["prompt"],
                        "rejected": row["rejected"],
                        "correction": row["chosen"],
                        "similarity": float(sim),
                    })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_k]

    def format_corrections_for_prompt(self, query: str) -> str:
        corrections = self.get_relevant_corrections(query)
        if not corrections:
            return ""

        lines = ["### Past Learning (Anti-Mistake Memory)"]
        lines.append(
            "The following are corrections to past mistakes I made on similar topics. "
            "Use these to ensure the current response is accurate."
        )
        for i, c in enumerate(corrections, 1):
            lines.append(f"\n[{i}] Similar Past Query: {c['prompt']}")
            lines.append(f"    Previous Error: {c['rejected'][:200]}...")
            lines.append(f"    Correction: {c['correction']}")

        lines.append("\n---")
        return "\n".join(lines)
