"""
feedback/conversation_logger.py
--------------------------------
Logs every conversation turn (user + assistant) to the SQLite database,
runs auto-evaluation on each assistant turn, and flags bad responses.

Integrates with:
  • backend/main.py  (call log_turn() after each stream completes)
  • evaluation/scorer.py  (reused scoring engine — zero duplication)
  • feedback/db_schema.py (persistence)
"""

import json
import time
import uuid
import logging
from typing import Optional

from .db_schema import get_conn, _now_utc
from .mistake_memory import MistakeMemory

# Lazy import to avoid loading sentence-transformers unless scoring is needed
_scorer = None

log = logging.getLogger("chatbot.conv_logger")


def _get_scorer():
    global _scorer
    if _scorer is None:
        import sys, os
        eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
        if eval_path not in sys.path:
            sys.path.insert(0, eval_path)
        import scorer as s
        _scorer = s
    return _scorer

# Global singleton for mistake memory (expensive to reload model)
_MISTAKE_MEM: MistakeMemory | None = None

def _get_mistake_memory() -> MistakeMemory:
    global _MISTAKE_MEM
    if _MISTAKE_MEM is None:
        _MISTAKE_MEM = MistakeMemory()
    return _MISTAKE_MEM


# ── Core logger ───────────────────────────────────────────────────────────────

def log_turn(
    *,
    conv_id: str,
    session_id: str,
    turn_index: int,
    role: str,                         # 'user' | 'assistant'
    content: str,
    model_name: str = "gpt2",
    prompt: Optional[str] = None,      # required for scoring assistant turns
    expected_keywords: Optional[list] = None,
    ttft_seconds: float = 0.0,
    total_time_seconds: float = 0.0,
) -> dict:
    """
    Persist a single conversation turn.
    For assistant turns, auto-evaluates the response and flags bad ones.
    Returns the inserted record as a dict.
    """
    now = _now_utc()

    score_rel = score_coh = score_acc = composite = None
    grade = None
    is_bad = 0
    failure_reasons: list = []

    if role == "assistant" and prompt:
        s = _get_scorer()
        result = s.evaluate_response(content, prompt, expected_keywords)
        scores = result.get("scores", {})
        score_rel = scores.get("relevance")
        score_coh = scores.get("coherence")
        score_acc = scores.get("accuracy")
        composite = result.get("composite_score")
        grade     = result.get("grade", "F")

        BAD_THRESHOLD = 0.55
        if composite is not None and composite < BAD_THRESHOLD:
            is_bad = 1
            # Failure reason detection (reuse evaluation logic)
            eval_mod = _get_scorer()
            from evaluation.logger import detect_failure_reasons
            failure_reasons = detect_failure_reasons(prompt, content, scores)

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO conversations
            (conv_id, session_id, turn_index, role, content, model_name,
             timestamp_utc, score_relevance, score_coherence, score_accuracy,
             composite_score, grade, ttft_seconds, total_time_seconds,
             is_bad_response, failure_reasons)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            conv_id, session_id, turn_index, role, content, model_name,
            now, score_rel, score_coh, score_acc,
            composite, grade, ttft_seconds, total_time_seconds,
            is_bad, json.dumps(failure_reasons),
        ),
    )
    conn.commit()

    record = {
        "conv_id": conv_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "role": role,
        "content": content,
        "model_name": model_name,
        "timestamp_utc": now,
        "score_relevance": score_rel,
        "score_coherence": score_coh,
        "score_accuracy": score_acc,
        "composite_score": composite,
        "grade": grade,
        "is_bad_response": bool(is_bad),
        "failure_reasons": failure_reasons,
    }

    # Auto-enqueue bad responses into failed_queries
    if is_bad and prompt:
        _store_failed_query(
            conv_id=conv_id,
            session_id=session_id,
            prompt=prompt,
            response=content,
            composite_score=composite,
            grade=grade,
            failure_reasons=failure_reasons,
            source="auto",
        )

    return record


def _store_failed_query(
    *,
    conv_id: str,
    session_id: str,
    prompt: str,
    response: str,
    composite_score: Optional[float],
    grade: Optional[str],
    failure_reasons: list,
    source: str = "auto",   # 'auto' | 'user'
) -> None:
    """Records a failed query into failed_queries via MistakeMemory."""
    mm = _get_mistake_memory()
    mm.record_failure(
        conv_id=conv_id,
        session_id=session_id,
        prompt=prompt,
        response=response,
        source=source,
        composite_score=composite_score,
        grade=grade,
        failure_reasons=failure_reasons
    )
    log.warning(
        "[ConvLogger] Bad response recorded → MistakeMemory  source=%s  score=%.3f",
        source, composite_score or 0,
    )


# ── Session helpers ────────────────────────────────────────────────────────────

def new_conv_id() -> str:
    """Generate a fresh conversation UUID."""
    return str(uuid.uuid4())


def load_conversation(conv_id: str) -> list[dict]:
    """Return all turns for a conversation, ordered by turn_index."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM conversations WHERE conv_id=? ORDER BY turn_index",
        (conv_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def load_session_conversations(session_id: str, limit: int = 50) -> list[dict]:
    """Return recent conversations for a session (newest first)."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT DISTINCT conv_id, MAX(timestamp_utc) as last_ts
        FROM conversations
        WHERE session_id = ?
        GROUP BY conv_id
        ORDER BY last_ts DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]
