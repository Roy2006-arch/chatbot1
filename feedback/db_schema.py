"""
feedback/db_schema.py
---------------------
SQLite database schema for the chatbot feedback loop system.

Tables
──────
  conversations   – full turn-level log of every chat session
  feedback        – user 👍 / 👎 votes with optional free-text comment
  failed_queries  – queries flagged as bad (low score OR user thumbs-down)
  retrain_runs    – audit log of every retraining pipeline invocation

Design notes
────────────
• SQLite is used (zero-infra, single-file database at data/feedback.db).
  Switch to PostgreSQL trivially by changing DB_URL below.
• The schema is created via raw DDL rather than an ORM to keep
  dependencies minimal — this file imports only stdlib sqlite3.
"""

import os
import sqlite3
from datetime import datetime, timezone

# ── Path ──────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(__file__)
DB_DIR   = os.path.join(_HERE, "..", "data", "feedback")
DB_PATH  = os.path.join(DB_DIR, "feedback.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent writes
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ── DDL ───────────────────────────────────────────────────────────────────────
_SCHEMA = """
-- ── conversations ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id            TEXT    NOT NULL,   -- UUID per conversation
    session_id         TEXT    NOT NULL,   -- matches backend session
    turn_index         INTEGER NOT NULL DEFAULT 0,
    role               TEXT    NOT NULL CHECK(role IN ('user','assistant')),
    content            TEXT    NOT NULL,
    model_name         TEXT    NOT NULL DEFAULT 'gpt2',
    timestamp_utc      TEXT    NOT NULL,

    -- Auto-eval scores (filled by scorer after every assistant turn)
    score_relevance    REAL,
    score_coherence    REAL,
    score_accuracy     REAL,
    composite_score    REAL,
    grade              TEXT,

    -- Latency
    ttft_seconds       REAL,
    total_time_seconds REAL,

    -- Failure flags
    is_bad_response    INTEGER NOT NULL DEFAULT 0,   -- 0/1
    failure_reasons    TEXT    NOT NULL DEFAULT '[]' -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_conv_session
    ON conversations(session_id, conv_id);
CREATE INDEX IF NOT EXISTS idx_conv_bad
    ON conversations(is_bad_response);


-- ── feedback ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id         TEXT    NOT NULL,
    turn_index      INTEGER NOT NULL,   -- references conversations.turn_index
    session_id      TEXT    NOT NULL,
    vote            INTEGER NOT NULL CHECK(vote IN (-1, 1)),  -- 1=👍  -1=👎
    comment         TEXT    NOT NULL DEFAULT '',
    timestamp_utc   TEXT    NOT NULL,

    -- Denormalised for quick dashboard queries
    prompt          TEXT    NOT NULL DEFAULT '',
    response        TEXT    NOT NULL DEFAULT '',

    FOREIGN KEY (conv_id, turn_index)
        REFERENCES conversations(conv_id, turn_index)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_conv
    ON feedback(conv_id);
CREATE INDEX IF NOT EXISTS idx_feedback_vote
    ON feedback(vote);


-- ── failed_queries ──────────────────────────────────────────────────────────
-- Superset of bad_responses: anything that was auto-flagged OR user-downvoted.
CREATE TABLE IF NOT EXISTS failed_queries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id         TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    prompt          TEXT    NOT NULL,
    response        TEXT    NOT NULL,
    composite_score REAL,
    grade           TEXT,
    failure_reasons TEXT    NOT NULL DEFAULT '[]',   -- JSON array
    source          TEXT    NOT NULL DEFAULT 'auto', -- 'auto' | 'user'
    resolved        INTEGER NOT NULL DEFAULT 0,      -- 1 once retrained
    preferred_response TEXT NOT NULL DEFAULT '',     -- annotator fills this
    occurrence_count   INTEGER NOT NULL DEFAULT 1,   -- track repeated mistakes
    embedding          BLOB,                         -- prompt embedding for similarity
    timestamp_utc      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_failed_resolved
    ON failed_queries(resolved);
CREATE INDEX IF NOT EXISTS idx_failed_source
    ON failed_queries(source);
CREATE INDEX IF NOT EXISTS idx_failed_occurrence
    ON failed_queries(occurrence_count);


-- ── retrain_runs ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retrain_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,
    started_utc     TEXT    NOT NULL,
    finished_utc    TEXT,
    status          TEXT    NOT NULL DEFAULT 'running',  -- running|done|failed
    candidates_used INTEGER NOT NULL DEFAULT 0,
    model_before    TEXT    NOT NULL DEFAULT '',
    model_after     TEXT    NOT NULL DEFAULT '',
    avg_score_before REAL,
    avg_score_after  REAL,
    notes           TEXT    NOT NULL DEFAULT ''
);
"""


def init_db() -> None:
    """Create all tables (idempotent – safe to call every startup)."""
    conn = _connect()
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    print(f"[FeedbackDB] Database ready at {os.path.abspath(DB_PATH)}")


# ── Singleton connection pool (thread-safe via WAL) ───────────────────────────
_CONN: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        init_db()
        _CONN = _connect()
    return _CONN


# ── Convenience helpers ───────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)
