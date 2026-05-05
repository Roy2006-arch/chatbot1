"""
feedback/feedback_store.py
--------------------------
All logic for recording and querying user 👍 / 👎 votes.

Scoring philosophy
──────────────────
• Each 👍 vote raises the response's "trust score" by +1
• Each 👎 vote lowers it by -1 AND enqueues the prompt into failed_queries
• A "net score" per (conv_id, turn_index) is computed as:
      net_score = thumbs_up_count - thumbs_down_count
• Responses with net_score ≤ -2 are automatically promoted to retraining
  priority, even if the auto-eval gave them a passing grade.

Feedback score formula (0-1 range for dashboards)
──────────────────────────────────────────────────
  feedback_score = sigmoid(net_score)
  where sigmoid(x) = 1 / (1 + e^(-x))

This maps:
  net = -3  →  0.047  (very negative)
  net = -1  →  0.269  (thumbs down)
  net =  0  →  0.500  (neutral)
  net = +1  →  0.731  (one thumbs up)
  net = +3  →  0.953  (very positive)
"""

import json
import math
import logging
from typing import Optional

from .db_schema import get_conn, _now_utc

log = logging.getLogger("chatbot.feedback_store")

# Responses with net_score at or below this are promoted for retraining
RETRAIN_NET_SCORE_THRESHOLD = -2


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── Record a vote ─────────────────────────────────────────────────────────────

def record_feedback(
    *,
    conv_id: str,
    turn_index: int,
    session_id: str,
    vote: int,          # +1 (👍) or -1 (👎)
    comment: str = "",
    prompt: str = "",
    response: str = "",
) -> dict:
    """
    Persist a user vote and trigger downstream logic.

    Args:
        vote: +1 for 👍, -1 for 👎
        comment: optional free-text user comment
    Returns:
        Inserted feedback record as dict.
    """
    if vote not in (1, -1):
        raise ValueError(f"vote must be +1 or -1, got {vote}")

    now = _now_utc()
    conn = get_conn()

    conn.execute(
        """
        INSERT INTO feedback
            (conv_id, turn_index, session_id, vote, comment,
             timestamp_utc, prompt, response)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (conv_id, turn_index, session_id, vote, comment, now, prompt, response),
    )
    conn.commit()

    vote_label = "👍 thumbs_up" if vote == 1 else "👎 thumbs_down"
    log.info("[Feedback] %s  conv=%s  turn=%d", vote_label, conv_id, turn_index)

    # After recording, check aggregate net score and enqueue if necessary
    if vote == -1:
        _handle_thumbs_down(
            conv_id=conv_id,
            turn_index=turn_index,
            session_id=session_id,
            prompt=prompt,
            response=response,
        )
    else:
        _check_net_score_for_recovery(conv_id, turn_index)

    return {
        "conv_id": conv_id,
        "turn_index": turn_index,
        "session_id": session_id,
        "vote": vote,
        "comment": comment,
        "timestamp_utc": now,
    }


def _handle_thumbs_down(
    *,
    conv_id: str,
    turn_index: int,
    session_id: str,
    prompt: str,
    response: str,
) -> None:
    """
    On thumbs-down: store as a failed query (source='user') unless already
    present, and promote to retraining if net score is severely negative.
    """
    conn = get_conn()

    # Only insert if not already in failed_queries for this turn
    existing = conn.execute(
        "SELECT id FROM failed_queries WHERE conv_id=? AND prompt=?",
        (conv_id, prompt),
    ).fetchone()

    if not existing:
        # Fetch auto-eval scores from conversations table if available
        conv_row = conn.execute(
            """
            SELECT composite_score, grade, failure_reasons
            FROM conversations
            WHERE conv_id=? AND turn_index=? AND role='assistant'
            """,
            (conv_id, turn_index),
        ).fetchone()

        composite = conv_row["composite_score"] if conv_row else None
        grade     = conv_row["grade"] if conv_row else "?"
        reasons   = json.loads(conv_row["failure_reasons"]) if conv_row else []
        reasons.append("User reported 👎 thumbs-down")

        conn.execute(
            """
            INSERT INTO failed_queries
                (conv_id, session_id, prompt, response, composite_score,
                 grade, failure_reasons, source, timestamp_utc)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                conv_id, session_id, prompt, response, composite,
                grade, json.dumps(reasons), "user", _now_utc(),
            ),
        )
        conn.commit()
        log.warning(
            "[Feedback] Thumbs-down → failed_queries  conv=%s  turn=%d",
            conv_id, turn_index,
        )


def _check_net_score_for_recovery(conv_id: str, turn_index: int) -> None:
    """
    If a previously down-voted response now has net_score > 0 after upvotes,
    mark the failed_query as resolved (user changed their mind).
    """
    net = get_net_score(conv_id, turn_index)
    if net > 0:
        conn = get_conn()
        conn.execute(
            """
            UPDATE failed_queries SET resolved=1
            WHERE conv_id=? AND source='user' AND resolved=0
            """,
            (conv_id,),
        )
        conn.commit()
        log.info(
            "[Feedback] Net score recovered → marking user-flagged record resolved  conv=%s",
            conv_id,
        )


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_net_score(conv_id: str, turn_index: int) -> int:
    """Sum of all votes for a specific turn (up=+1, down=-1)."""
    conn = get_conn()
    row = conn.execute(
        "SELECT COALESCE(SUM(vote), 0) FROM feedback WHERE conv_id=? AND turn_index=?",
        (conv_id, turn_index),
    ).fetchone()
    return int(row[0])


def get_feedback_score(conv_id: str, turn_index: int) -> float:
    """Sigmoid-normalised feedback score (0–1) for dashboard display."""
    return round(sigmoid(get_net_score(conv_id, turn_index)), 4)


def load_all_feedback(limit: int = 1000, vote_filter: Optional[int] = None) -> list[dict]:
    """
    Return recent feedback records.
    vote_filter: None=all, 1=only 👍, -1=only 👎
    """
    conn = get_conn()
    if vote_filter is not None:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE vote=? ORDER BY id DESC LIMIT ?",
            (vote_filter, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_failed_queries(
    limit: int = 500,
    source_filter: Optional[str] = None,   # 'auto' | 'user' | None
    unresolved_only: bool = True,
) -> list[dict]:
    """Return failed queries for retraining pipeline consumption."""
    conn = get_conn()
    clauses = []
    params: list = []

    if unresolved_only:
        clauses.append("resolved = 0")
    if source_filter:
        clauses.append("source = ?")
        params.append(source_filter)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    rows = conn.execute(
        f"SELECT * FROM failed_queries {where} ORDER BY id DESC LIMIT ?",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


# ── Dashboard summary ─────────────────────────────────────────────────────────

def feedback_summary() -> dict:
    """Aggregate statistics for the feedback dashboard."""
    conn = get_conn()

    totals = conn.execute(
        """
        SELECT
            COUNT(*)                             AS total_votes,
            SUM(CASE WHEN vote=1 THEN 1 ELSE 0 END) AS thumbs_up,
            SUM(CASE WHEN vote=-1 THEN 1 ELSE 0 END) AS thumbs_down
        FROM feedback
        """
    ).fetchone()

    failed = conn.execute(
        """
        SELECT
            COUNT(*)                                   AS total,
            SUM(CASE WHEN source='auto' THEN 1 ELSE 0 END) AS auto_flagged,
            SUM(CASE WHEN source='user' THEN 1 ELSE 0 END) AS user_flagged,
            SUM(CASE WHEN resolved=1   THEN 1 ELSE 0 END)  AS resolved
        FROM failed_queries
        """
    ).fetchone()

    avg_score = conn.execute(
        "SELECT AVG(composite_score) FROM conversations WHERE role='assistant'"
    ).fetchone()[0]

    return {
        "total_votes":   totals["total_votes"] or 0,
        "thumbs_up":     totals["thumbs_up"] or 0,
        "thumbs_down":   totals["thumbs_down"] or 0,
        "satisfaction_rate": round(
            (totals["thumbs_up"] or 0) /
            max((totals["total_votes"] or 1), 1),
            4,
        ),
        "failed_queries": {
            "total":       failed["total"] or 0,
            "auto_flagged": failed["auto_flagged"] or 0,
            "user_flagged": failed["user_flagged"] or 0,
            "resolved":    failed["resolved"] or 0,
        },
        "avg_composite_score": round(avg_score, 4) if avg_score else None,
    }
