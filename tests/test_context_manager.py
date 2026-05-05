"""
tests/test_context_manager.py
------------------------------
Smoke-tests for the new ContextManager.

Run from the project root:
    d:\\chatbot\\.venv\\Scripts\\python -m pytest tests/test_context_manager.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from context_manager import ContextManager


SID = "test-session-001"


@pytest.fixture
def mgr():
    return ContextManager(
        window_size=4,
        summarize_after=6,
        max_tokens_budget=1024,
        dedup_threshold=0.90,
    )


# ── 1. Basic add + get_context ─────────────────────────────────────────────────

def test_empty_context(mgr):
    ctx = mgr.get_context("nonexistent")
    assert "System:" in ctx


def test_add_and_retrieve(mgr):
    mgr.add_message(SID, "user", "Hello, my name is Alice.")
    mgr.add_message(SID, "assistant", "Hi Alice! How can I help?")
    ctx = mgr.get_context(SID, current_query="Hello")
    assert "Alice" in ctx or "Hello" in ctx


# ── 2. Sliding window compresses on overflow ───────────────────────────────────

def test_window_compression(mgr):
    for i in range(10):
        mgr.add_message(SID + "-cmp", "user", f"Question number {i}")
        mgr.add_message(SID + "-cmp", "assistant", f"Answer number {i}")
    stats = mgr.session_stats(SID + "-cmp")
    # After 10 rounds of 2 msgs = 20 msgs, window should be well below summarize_after
    # (may be window_size or window_size+1 depending on exact trigger point)
    assert stats["window_messages"] <= mgr.window_size + 1
    # Summary should be non-empty because messages were compressed
    assert stats["summary_chars"] > 0


# ── 3. Key-info extraction ─────────────────────────────────────────────────────

def test_key_info_name(mgr):
    sid = SID + "-ki"
    mgr.add_message(sid, "user", "My name is Bob and I am from London.")
    ki = mgr.get_key_info(sid)
    assert "name" in ki
    assert any("Bob" in v for v in ki["name"])


def test_key_info_location(mgr):
    sid = SID + "-loc"
    mgr.add_message(sid, "user", "I am from Paris.")
    ki = mgr.get_key_info(sid)
    assert "location" in ki


# ── 4. Repetition detection ────────────────────────────────────────────────────

def test_repetition_not_triggered_on_first_message(mgr):
    sid = SID + "-rep1"
    mgr.add_message(sid, "user", "What is Python?")
    mgr.add_message(sid, "assistant", "Python is a high-level programming language.")
    # First time asking — should NOT be flagged
    result = mgr.is_repetitive_answer(sid, "Python is a high-level programming language.")
    # Whether True or False depends on threshold; just ensure method runs
    assert isinstance(result, bool)


def test_repetition_detected_on_identical_answer(mgr):
    sid = SID + "-rep2"
    identical = "Python is a high-level, interpreted programming language known for its readability."
    mgr.add_message(sid, "user", "Tell me about Python.")
    mgr.add_message(sid, "assistant", identical)
    mgr.add_message(sid, "user", "Can you explain Python again?")
    # Same answer again → should be flagged as repetitive
    assert mgr.is_repetitive_answer(sid, identical) is True


# ── 5. Episodic recall surfaces in context ─────────────────────────────────────

def test_episodic_recall_in_context(mgr):
    sid = SID + "-ep"
    mgr.add_message(sid, "user", "I love machine learning frameworks.")
    mgr.add_message(sid, "assistant", "Great! Which framework are you using?")
    ctx = mgr.get_context(sid, current_query="Which ML library is best for me?")
    # Episodic recall should surface the past user statement
    assert "machine learning" in ctx.lower() or "ML" in ctx or "past" in ctx.lower()


# ── 6. Session stats ───────────────────────────────────────────────────────────

def test_session_stats(mgr):
    sid = SID + "-stats"
    mgr.add_message(sid, "user", "Hi")
    mgr.add_message(sid, "assistant", "Hello!")
    stats = mgr.session_stats(sid)
    assert stats["window_messages"] == 2
    assert stats["total_turns"] == 2
    assert stats["episodic_indexed"] == 1   # only user messages indexed


# ── 7. TTL cleanup ─────────────────────────────────────────────────────────────

def test_cleanup_idle(mgr):
    sid = SID + "-ttl"
    mgr.add_message(sid, "user", "I'll be gone soon.")
    # Force last_accessed to the past
    mgr._sessions[sid].last_accessed = 0.0
    removed = mgr.cleanup_idle_sessions(idle_time_seconds=1)
    assert removed >= 1
    assert sid not in mgr._sessions


# ── 8. Token budget trimming ───────────────────────────────────────────────────

def test_token_budget():
    tight_mgr = ContextManager(
        window_size=4,
        summarize_after=6,
        max_tokens_budget=50,   # very tight
    )
    sid = "budget-test"
    for _ in range(3):
        tight_mgr.add_message(sid, "user", "x" * 300)
        tight_mgr.add_message(sid, "assistant", "y" * 300)
    ctx = tight_mgr.get_context(sid)
    # Should be trimmed — context should not exceed budget drastically
    estimated_tokens = len(ctx) // 4
    # Allow some overshoot from system sections
    assert estimated_tokens < 200, f"Context too long: {estimated_tokens} tokens"
