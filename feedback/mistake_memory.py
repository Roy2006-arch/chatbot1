import json
import logging
import numpy as np
from typing import List, Optional, Dict

from .db_schema import get_conn, close_conn, _now_utc
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

    def _batch_similarity(self, query_emb: np.ndarray, emb_list: List[np.ndarray]) -> np.ndarray:
        if not emb_list:
            return np.array([], dtype="float32")
        stack = np.stack(emb_list, axis=0)
        dot = np.dot(stack, query_emb)
        norms = np.linalg.norm(stack, axis=1) * np.linalg.norm(query_emb) + 1e-9
        return dot / norms

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
        try:
            rows = conn.execute(
                "SELECT id, embedding, occurrence_count FROM failed_queries WHERE resolved = 0 AND embedding IS NOT NULL"
            ).fetchall()

            if rows:
                emb_list = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
                sims = self._batch_similarity(embedding, emb_list)
                max_idx = int(np.argmax(sims)) if len(sims) > 0 else -1
                best_match_id = rows[max_idx]["id"] if max_idx >= 0 and sims[max_idx] > SIMILARITY_THRESHOLD else None
            else:
                best_match_id = None

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
        finally:
            close_conn(conn)

    def get_relevant_corrections(self, query: str, top_k: int = 2) -> List[Dict]:
        embedding = self._embed(query)
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT prompt, response as rejected, preferred_response as chosen, embedding "
                "FROM failed_queries WHERE preferred_response != '' AND embedding IS NOT NULL"
            ).fetchall()
        finally:
            close_conn(conn)

        if not rows:
            return []

        emb_list = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        sims = self._batch_similarity(embedding, emb_list)

        results = []
        for i in np.argsort(sims)[::-1]:
            if sims[i] > 0.7:
                results.append({
                    "prompt": rows[i]["prompt"],
                    "rejected": rows[i]["rejected"],
                    "correction": rows[i]["chosen"],
                    "similarity": float(sims[i]),
                })
                if len(results) >= top_k:
                    break

        return results

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
