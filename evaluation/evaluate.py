"""
evaluate.py
-----------
Main evaluation script — the entry point for a full automated evaluation run.

Features:
  ✓ Loads test cases from a JSON dataset (data/eval_dataset.json)
  ✓ Queries the live FastAPI chatbot (/chat/stream)
  ✓ Scores each response for Accuracy, Relevance, Coherence
  ✓ Assigns letter grades (A–F)
  ✓ Logs every result + flags bad responses to dedicated logs
  ✓ Saves a run-level summary (pass rates, avg scores, grade distribution)
  ✓ Prints a live progress table to the terminal

Usage:
    python evaluate.py                          # run all test cases
    python evaluate.py --limit 10              # run first 10 cases
    python evaluate.py --base-url http://...   # custom backend URL
    python evaluate.py --model my-mistral-7b   # tag results with model name
"""

import argparse
import json
import os
import sys
import time
import uuid
from typing import Optional

import requests

# Ensure evaluation package is importable regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from scorer import evaluate_response
from logger import log_evaluation, save_run_summary, BAD_RESPONSE_THRESHOLD

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL  = "http://localhost:8000"
DEFAULT_MODEL     = "gpt2"
DATASET_PATH      = os.path.join(os.path.dirname(__file__), "data", "eval_dataset.json")
STREAM_TIMEOUT    = 60   # seconds


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """Load test cases from a JSON file."""
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found at: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both a plain list and {"test_cases": [...]}
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "test_cases" in data:
        return data["test_cases"]
    raise ValueError("Dataset must be a JSON list or a dict with key 'test_cases'.")


# ---------------------------------------------------------------------------
# Chatbot Query
# ---------------------------------------------------------------------------

def query_chatbot(
    prompt: str,
    session_id: str,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[str, float, float]:
    """
    Send a prompt to the chatbot's streaming endpoint.
    Returns (response_text, time_to_first_token, total_time).
    """
    payload    = {"message": prompt, "session_id": session_id}
    full_resp  = ""
    ttft       = 0.0
    start      = time.time()
    first_seen = False

    try:
        resp = requests.post(
            f"{base_url}/chat/stream",
            json=payload,
            stream=True,
            timeout=STREAM_TIMEOUT,
        )
        if resp.status_code != 200:
            return f"[HTTP {resp.status_code}]", 0.0, time.time() - start

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                token = chunk.get("content", "")
                if token and not first_seen:
                    ttft = time.time() - start
                    first_seen = True
                full_resp += token
            except json.JSONDecodeError:
                continue

        total_time = time.time() - start
        return full_resp.strip(), ttft, total_time

    except requests.exceptions.ConnectionError:
        return "[CONNECTION ERROR — Is the backend running?]", 0.0, 0.0
    except requests.exceptions.Timeout:
        return "[TIMEOUT — Backend did not respond in time]", 0.0, STREAM_TIMEOUT
    except Exception as exc:
        return f"[ERROR — {exc}]", 0.0, 0.0


# ---------------------------------------------------------------------------
# Progress Table Printer
# ---------------------------------------------------------------------------

_GRADE_COLORS = {
    "A": "\033[92m",  # green
    "B": "\033[96m",  # cyan
    "C": "\033[93m",  # yellow
    "D": "\033[33m",  # orange-ish
    "F": "\033[91m",  # red
}
_RESET = "\033[0m"


def _grade_str(grade: str) -> str:
    return f"{_GRADE_COLORS.get(grade, '')}{grade}{_RESET}"


def print_table_header() -> None:
    print()
    print(f"{'#':>4}  {'Grade':<6}  {'Score':>6}  {'Acc':>5}  {'Rel':>5}  {'Coh':>5}  {'TTFT':>5}  Prompt")
    print("─" * 100)


def print_row(idx: int, record: dict) -> None:
    s       = record.get("scores", {})
    acc     = f"{s.get('accuracy', 0) or 0:.2f}" if s.get("accuracy") is not None else "  N/A"
    rel     = f"{s.get('relevance', 0):.2f}"
    coh     = f"{s.get('coherence', 0):.2f}"
    comp    = record.get("composite_score", 0.0)
    grade   = record.get("grade", "F")
    ttft    = record.get("ttft_seconds", 0.0)
    prompt  = record.get("prompt", "")[:55]
    bad_tag = " ⚑" if record.get("is_bad_response") else ""
    print(
        f"{idx:>4}  {_grade_str(grade):<6}  {comp:>6.3f}  {acc:>5}  {rel:>5}  {coh:>5}  {ttft:>4.1f}s  {prompt}{bad_tag}"
    )


# ---------------------------------------------------------------------------
# Main Evaluation Runner
# ---------------------------------------------------------------------------

def run_evaluation(
    base_url: str  = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL,
    dataset_path: str = DATASET_PATH,
    limit: Optional[int] = None,
    use_fresh_sessions: bool = True,
) -> dict:
    """
    Execute a full evaluation run.
    Returns the run summary dict.
    """
    run_id   = str(uuid.uuid4())
    cases    = load_dataset(dataset_path)
    if limit:
        cases = cases[:limit]

    print(f"\n{'═'*60}")
    print(f"  🤖  CHATBOT EVALUATION FRAMEWORK")
    print(f"  Model     : {model_name}")
    print(f"  Backend   : {base_url}")
    print(f"  Test cases: {len(cases)}")
    print(f"  Run ID    : {run_id}")
    print(f"{'═'*60}")

    # Connectivity check
    try:
        requests.get(base_url, timeout=5)
    except Exception:
        print(f"\n[WARN] Cannot reach {base_url} — responses will be connection errors.\n")

    print_table_header()

    records      = []
    bad_count    = 0
    session_pool = {}  # category → session_id (to test multi-turn context)

    for idx, case in enumerate(cases, start=1):
        prompt     = case.get("prompt", "")
        keywords   = case.get("expected_keywords", [])
        category   = case.get("category", "general")

        # Re-use session per category to test memory, or use fresh per case
        if use_fresh_sessions:
            session_id = str(uuid.uuid4())
        else:
            if category not in session_pool:
                session_pool[category] = str(uuid.uuid4())
            session_id = session_pool[category]

        response, ttft, total_t = query_chatbot(prompt, session_id, base_url)

        # Abort on connection issues after the first case
        if "CONNECTION ERROR" in response and idx == 1:
            print(f"\n[FATAL] Cannot reach the backend. Aborting.")
            break

        eval_result = evaluate_response(prompt, response, expected_keywords=keywords)
        record = log_evaluation(
            prompt=prompt,
            response=response,
            eval_result=eval_result,
            run_id=run_id,
            session_id=session_id,
            expected_keywords=keywords,
            ttft_seconds=ttft,
            total_time_seconds=total_t,
            model_name=model_name,
        )
        records.append(record)

        if record.get("is_bad_response"):
            bad_count += 1

        print_row(idx, record)

        # Pace requests slightly to avoid hammering a local server
        time.sleep(0.5)

    # Final summary
    summary = save_run_summary(run_id, records, model_name)

    print("\n" + "─" * 100)
    print(f"\n  Run complete!")
    print(f"  Evaluated      : {len(records)} cases")
    print(f"  Bad responses  : {bad_count} ({summary.get('bad_response_rate', 0)*100:.1f}%)")
    print(f"  Avg composite  : {summary.get('avg_scores', {}).get('composite', 0):.3f}")
    print(f"  Grade dist.    : {summary.get('grade_distribution', {})}")
    print(f"\n  📁 Logs saved to: evaluation/logs/")
    if bad_count > 0:
        print(f"  ⚑  Run feedback_loop.py to export retraining data.")
    print()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated chatbot evaluation")
    parser.add_argument("--base-url",  default=DEFAULT_BASE_URL, help="FastAPI backend URL")
    parser.add_argument("--model",     default=DEFAULT_MODEL,    help="Model name tag for logs")
    parser.add_argument("--dataset",   default=DATASET_PATH,     help="Path to eval_dataset.json")
    parser.add_argument("--limit",     type=int, default=None,   help="Max test cases to run")
    parser.add_argument(
        "--shared-sessions", action="store_true",
        help="Reuse session_id per category (tests multi-turn memory)"
    )
    args = parser.parse_args()

    run_evaluation(
        base_url=args.base_url,
        model_name=args.model,
        dataset_path=args.dataset,
        limit=args.limit,
        use_fresh_sessions=not args.shared_sessions,
    )
