import os
import sqlite3
import threading
import queue
from contextlib import contextmanager
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_DIR = os.path.join(_PROJECT_ROOT, "data", "feedback")
DB_PATH = os.path.join(DB_DIR, "feedback.db")

_db_lock = threading.RLock()
_db_initialized = False
_connection_pool: queue.Queue[sqlite3.Connection] = queue.Queue()
_POOL_SIZE = 3


def _create_conn() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_pool():
    global _db_initialized
    if _db_initialized:
        return
    with _db_lock:
        if _db_initialized:
            return
        for _ in range(_POOL_SIZE):
            _connection_pool.put(_create_conn())
        conn = borrow_conn()
        conn.executescript(_SCHEMA)
        conn.commit()
        return_conn(conn)
        _db_initialized = True


def borrow_conn() -> sqlite3.Connection:
    try:
        return _connection_pool.get_nowait()
    except queue.Empty:
        return _create_conn()


def return_conn(conn: sqlite3.Connection):
    try:
        _connection_pool.put_nowait(conn)
    except queue.Full:
        conn.close()


@contextmanager
def get_conn_ctx():
    """Provide a transactional scope around a series of operations."""
    conn = borrow_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id            TEXT    NOT NULL,
    session_id         TEXT    NOT NULL,
    turn_index         INTEGER NOT NULL DEFAULT 0,
    role               TEXT    NOT NULL CHECK(role IN ('user','assistant')),
    content            TEXT    NOT NULL,
    model_name         TEXT    NOT NULL DEFAULT 'gpt2',
    timestamp_utc      TEXT    NOT NULL,
    score_relevance    REAL,
    score_coherence    REAL,
    score_accuracy     REAL,
    composite_score    REAL,
    grade              TEXT,
    ttft_seconds       REAL,
    total_time_seconds REAL,
    is_bad_response    INTEGER NOT NULL DEFAULT 0,
    failure_reasons    TEXT    NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, conv_id);
CREATE INDEX IF NOT EXISTS idx_conv_bad ON conversations(is_bad_response);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id         TEXT    NOT NULL,
    turn_index      INTEGER NOT NULL,
    session_id      TEXT    NOT NULL,
    vote            INTEGER NOT NULL CHECK(vote IN (-1, 1)),
    comment         TEXT    NOT NULL DEFAULT '',
    timestamp_utc   TEXT    NOT NULL,
    prompt          TEXT    NOT NULL DEFAULT '',
    response        TEXT    NOT NULL DEFAULT '',
    FOREIGN KEY (conv_id, turn_index)
        REFERENCES conversations(conv_id, turn_index)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_conv ON feedback(conv_id);
CREATE INDEX IF NOT EXISTS idx_feedback_vote ON feedback(vote);

CREATE TABLE IF NOT EXISTS failed_queries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id         TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    prompt          TEXT    NOT NULL,
    response        TEXT    NOT NULL,
    composite_score REAL,
    grade           TEXT,
    failure_reasons TEXT    NOT NULL DEFAULT '[]',
    source          TEXT    NOT NULL DEFAULT 'auto',
    resolved        INTEGER NOT NULL DEFAULT 0,
    preferred_response TEXT NOT NULL DEFAULT '',
    occurrence_count   INTEGER NOT NULL DEFAULT 1,
    embedding          BLOB,
    timestamp_utc      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_failed_resolved ON failed_queries(resolved);
CREATE INDEX IF NOT EXISTS idx_failed_source ON failed_queries(source);
CREATE INDEX IF NOT EXISTS idx_failed_occurrence ON failed_queries(occurrence_count);

CREATE TABLE IF NOT EXISTS retrain_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,
    started_utc     TEXT    NOT NULL,
    finished_utc    TEXT,
    status          TEXT    NOT NULL DEFAULT 'running',
    candidates_used INTEGER NOT NULL DEFAULT 0,
    model_before    TEXT    NOT NULL DEFAULT '',
    model_after     TEXT    NOT NULL DEFAULT '',
    avg_score_before REAL,
    avg_score_after  REAL,
    notes           TEXT    NOT NULL DEFAULT ''
);
"""


def init_db() -> None:
    init_pool()
    print(f"[FeedbackDB] Database ready at {os.path.abspath(DB_PATH)}")


def get_conn() -> sqlite3.Connection:
    """Get a connection from the pool. Caller must commit and call close_conn() when done."""
    init_pool()
    return borrow_conn()


def close_conn(conn: sqlite3.Connection):
    """Return a connection to the pool (or close if pool is full)."""
    return_conn(conn)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)
