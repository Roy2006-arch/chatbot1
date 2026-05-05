"""
logger.py
---------
Structured JSON evaluation logger.

Two log files:
  1. eval_log.jsonl         – every evaluation result (newline-delimited JSON)
  2. bad_responses.jsonl    – only responses that fall below the quality threshold
     (these are the candidates flagged for human review and retraining)

Log format
----------
Each line is a self-contained JSON object (see LOG_SCHEMA below for the spec).
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LOG_DIR          = os.path.join(os.path.dirname(__file__), "logs")
EVAL_LOG_PATH    = os.path.join(LOG_DIR, "eval_log.jsonl")
BAD_LOG_PATH     = os.path.join(LOG_DIR, "bad_responses.jsonl")
SUMMARY_LOG_PATH = os.path.join(LOG_DIR, "run_summaries.json")

os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Quality threshold — responses below this composite score are "bad"
# ---------------------------------------------------------------------------
BAD_RESPONSE_THRESHOLD = 0.55   # Grade C or below


# ---------------------------------------------------------------------------
# Log schema (for documentation purposes)
# ---------------------------------------------------------------------------
LOG_SCHEMA = {
    "eval_id":            "str  — unique UUID for this evaluation event",
    "run_id":             "str  — groups all entries from a single evaluation run",
    "timestamp_utc":      "str  — ISO-8601 UTC timestamp",
    "session_id":         "str  — chatbot session ID (for multi-turn tracing)",
    "prompt":             "str  — user input",
    "response":           "str  — chatbot output",
    "expected_keywords":  "list — keywords the response should contain (optional)",
    "scores": {
        "accuracy":       "float|null — keyword coverage (null if no keywords given)",
        "relevance":      "float      — semantic prompt↔response similarity (0-1)",
        "coherence":      "float      — structural quality score (0-1)",
    },
    "composite_score":    "float — weighted overall score (0-1)",
    "grade":              "str   — A/B/C/D/F",
    "grade_label":        "str   — e.g. 'Excellent', 'Poor'",
    "is_bad_response":    "bool  — true if composite_score < BAD_RESPONSE_THRESHOLD",
    "failure_reasons":    "list  — human-readable list of detected failure modes",
    "ttft_seconds":       "float — time-to-first-token (0 if unavailable)",
    "total_time_seconds": "float — full end-to-end latency",
    "model_name":         "str   — model identifier for the run",
    "retrain_candidate":  "bool  — same as is_bad_response; convenient alias for pipelines",
    "human_review_notes": "str   — reserved for manual annotator notes (blank by default)",
}


# ---------------------------------------------------------------------------
# Failure Reason Detection
# ---------------------------------------------------------------------------

def detect_failure_reasons(prompt: str, response: str, scores: dict) -> list[str]:
    """Return a list of human-readable failure reasons for a bad response."""
    reasons = []
    if not response.strip():
        reasons.append("Empty response")
        return reasons
    if scores.get("accuracy") is not None and scores["accuracy"] < 0.30:
        reasons.append("Low factual accuracy — key facts missing from response")
    if scores.get("relevance", 1.0) < 0.30:
        reasons.append("Off-topic — response has low semantic similarity to the prompt")
    if scores.get("coherence", 1.0) < 0.40:
        reasons.append("Incoherent — response is repetitive, too short, or structurally broken")
    if "user:" in response.lower() or "assistant:" in response.lower():
        reasons.append("Role leakage — model hallucinated conversational role markers")
    words = response.split()
    if len(words) > 500:
        reasons.append(f"Runaway generation — response is {len(words)} words (> 500)")
    if len(words) < 5:
        reasons.append("Truncated — response is fewer than 5 words")
    if not reasons:
        reasons.append("Composite score below threshold")
    return reasons


# ---------------------------------------------------------------------------
# Core Logging Functions
# ---------------------------------------------------------------------------

def _append_jsonl(path: str, record: dict) -> None:
    """Append a single JSON object as a line to a .jsonl file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_evaluation(
    prompt: str,
    response: str,
    eval_result: dict,
    run_id: str,
    session_id: str = "unknown",
    expected_keywords: Optional[list] = None,
    ttft_seconds: float = 0.0,
    total_time_seconds: float = 0.0,
    model_name: str = "gpt2",
) -> dict:
    """
    Build and persist a structured evaluation log entry.

    Args:
        prompt            : The user's input.
        response          : The chatbot's output.
        eval_result       : Output from scorer.evaluate_response().
        run_id            : ID linking all entries in one evaluation run.
        session_id        : The chatbot session_id for conversation tracing.
        expected_keywords : Keywords used for accuracy scoring.
        ttft_seconds      : Time-to-first-token in seconds.
        total_time_seconds: Full response latency in seconds.
        model_name        : The model identifier.

    Returns:
        The complete log record dict.
    """
    scores     = eval_result.get("scores", {})
    composite  = eval_result.get("composite_score", 0.0)
    is_bad     = composite < BAD_RESPONSE_THRESHOLD

    failure_reasons = detect_failure_reasons(prompt, response, scores) if is_bad else []

    record = {
        "eval_id":            str(uuid.uuid4()),
        "run_id":             run_id,
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
        "session_id":         session_id,
        "model_name":         model_name,
        "prompt":             prompt,
        "response":           response,
        "expected_keywords":  expected_keywords or [],
        "scores":             scores,
        "composite_score":    composite,
        "grade":              eval_result.get("grade", "F"),
        "grade_label":        eval_result.get("grade_label", "Failing"),
        "is_bad_response":    is_bad,
        "retrain_candidate":  is_bad,
        "failure_reasons":    failure_reasons,
        "ttft_seconds":       round(ttft_seconds, 4),
        "total_time_seconds": round(total_time_seconds, 4),
        "human_review_notes": "",
    }

    # Write to main eval log
    _append_jsonl(EVAL_LOG_PATH, record)

    # Mirror to bad-response log if flagged
    if is_bad:
        _append_jsonl(BAD_LOG_PATH, record)

    return record


# ---------------------------------------------------------------------------
# Run Summary
# ---------------------------------------------------------------------------

def save_run_summary(run_id: str, records: list[dict], model_name: str = "gpt2") -> dict:
    """
    Compute and persist aggregate statistics for a completed evaluation run.
    Appends to run_summaries.json as a list of summary objects.
    """
    if not records:
        return {}

    total = len(records)
    bad   = sum(1 for r in records if r.get("is_bad_response", False))

    scores_by_dim = {dim: [] for dim in ["accuracy", "relevance", "coherence"]}
    composites    = []

    for r in records:
        composites.append(r.get("composite_score", 0.0))
        for dim in scores_by_dim:
            val = r.get("scores", {}).get(dim)
            if val is not None:
                scores_by_dim[dim].append(val)

    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    grade_dist = {}
    for r in records:
        g = r.get("grade", "F")
        grade_dist[g] = grade_dist.get(g, 0) + 1

    summary = {
        "run_id":              run_id,
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
        "model_name":          model_name,
        "total_evaluated":     total,
        "bad_responses":       bad,
        "bad_response_rate":   round(bad / total, 4),
        "avg_scores": {
            "accuracy":        _avg(scores_by_dim["accuracy"]),
            "relevance":       _avg(scores_by_dim["relevance"]),
            "coherence":       _avg(scores_by_dim["coherence"]),
            "composite":       _avg(composites),
        },
        "grade_distribution":  grade_dist,
        "retrain_candidates":  bad,
    }

    # Load existing summaries, append, save
    summaries = []
    if os.path.exists(SUMMARY_LOG_PATH):
        try:
            with open(SUMMARY_LOG_PATH, "r", encoding="utf-8") as f:
                summaries = json.load(f)
        except (json.JSONDecodeError, IOError):
            summaries = []

    summaries.append(summary)
    with open(SUMMARY_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------------
# Retrieval Helpers (for feedback loop / retraining pipeline)
# ---------------------------------------------------------------------------

def load_bad_responses(limit: int = 500) -> list[dict]:
    """Load the most recent `limit` bad-response records from the log."""
    if not os.path.exists(BAD_LOG_PATH):
        return []
    records = []
    with open(BAD_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records[-limit:]


def load_all_evaluations(run_id: Optional[str] = None) -> list[dict]:
    """Load all eval log entries, optionally filtered by run_id."""
    if not os.path.exists(EVAL_LOG_PATH):
        return []
    records = []
    with open(EVAL_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if run_id is None or r.get("run_id") == run_id:
                    records.append(r)
            except json.JSONDecodeError:
                continue
    return records
