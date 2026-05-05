"""
feedback_loop.py
----------------
Retraining Feedback Loop

Reads the bad_responses.jsonl log produced by logger.py and:
  1. Exports a DPO-format JSONL training file (prompt + rejected response).
  2. Generates a human-review CSV for manual annotation of preferred responses.
  3. Prints a summary report of the most common failure modes.

Usage:
    python feedback_loop.py                        # use default log path
    python feedback_loop.py --limit 200            # cap at 200 bad records
    python feedback_loop.py --output my_dpo.jsonl  # custom output path
"""

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone

from logger import load_bad_responses, SUMMARY_LOG_PATH, LOG_DIR

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------
RETRAIN_JSONL_PATH  = os.path.join(LOG_DIR, "retrain_dpo_candidates.jsonl")
HUMAN_REVIEW_CSV    = os.path.join(LOG_DIR, "human_review_queue.csv")
FAILURE_REPORT_PATH = os.path.join(LOG_DIR, "failure_mode_report.json")


# ---------------------------------------------------------------------------
# DPO Export
# ---------------------------------------------------------------------------

def export_dpo_candidates(records: list[dict], output_path: str) -> int:
    """
    Write DPO-format JSONL for fine-tuning.

    Each line:
    {
        "prompt":   "...",
        "rejected": "...",   ← the bad chatbot response
        "chosen":   ""       ← blank; to be filled by a human annotator
    }
    """
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            entry = {
                "prompt":             r.get("prompt", ""),
                "rejected":           r.get("response", ""),
                "chosen":             "",          # Human fills this in
                "failure_reasons":    r.get("failure_reasons", []),
                "composite_score":    r.get("composite_score", 0.0),
                "grade":              r.get("grade", "F"),
                "eval_id":            r.get("eval_id", ""),
                "session_id":         r.get("session_id", ""),
                "timestamp_utc":      r.get("timestamp_utc", ""),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Human Review CSV
# ---------------------------------------------------------------------------

def export_human_review_csv(records: list[dict], output_path: str) -> int:
    """
    Write a CSV file for human annotators to add 'preferred' responses.
    Columns: eval_id, prompt, bad_response, preferred_response (blank), notes
    """
    fieldnames = [
        "eval_id",
        "prompt",
        "bad_response",
        "composite_score",
        "grade",
        "failure_reasons",
        "preferred_response",   # to be filled by human
        "annotator_notes",
    ]
    written = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "eval_id":            r.get("eval_id", ""),
                "prompt":             r.get("prompt", ""),
                "bad_response":       r.get("response", ""),
                "composite_score":    r.get("composite_score", 0.0),
                "grade":              r.get("grade", "F"),
                "failure_reasons":    "; ".join(r.get("failure_reasons", [])),
                "preferred_response": "",
                "annotator_notes":    "",
            })
            written += 1
    return written


# ---------------------------------------------------------------------------
# Failure Mode Analysis
# ---------------------------------------------------------------------------

def analyse_failure_modes(records: list[dict]) -> dict:
    """Count and rank the most common failure reasons across bad responses."""
    counter: Counter = Counter()
    for r in records:
        for reason in r.get("failure_reasons", []):
            counter[reason] += 1

    total = len(records)
    ranked = [
        {
            "reason":     reason,
            "count":      count,
            "percentage": round(count / total * 100, 1) if total else 0,
        }
        for reason, count in counter.most_common()
    ]

    report = {
        "generated_utc":  datetime.now(timezone.utc).isoformat(),
        "total_bad_responses": total,
        "top_failure_modes": ranked,
    }

    with open(FAILURE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


# ---------------------------------------------------------------------------
# Pretty Console Report
# ---------------------------------------------------------------------------

def print_feedback_report(records: list[dict], failure_report: dict) -> None:
    total = len(records)
    if total == 0:
        print("✅  No bad responses in the log. Nothing to retrain on.")
        return

    avg_score = sum(r.get("composite_score", 0) for r in records) / total

    print("\n" + "═" * 60)
    print("   🔁  RETRAINING FEEDBACK LOOP REPORT")
    print("═" * 60)
    print(f"  Bad responses loaded   : {total}")
    print(f"  Avg composite score    : {avg_score:.3f}")
    print(f"  DPO candidates exported: {RETRAIN_JSONL_PATH}")
    print(f"  Human review CSV       : {HUMAN_REVIEW_CSV}")
    print(f"  Failure mode report    : {FAILURE_REPORT_PATH}")
    print()
    print("  Top Failure Modes:")
    for entry in failure_report.get("top_failure_modes", [])[:5]:
        bar = "█" * int(entry["percentage"] / 5)
        print(f"    [{bar:<20}] {entry['percentage']:5.1f}%  {entry['reason']}")
    print("═" * 60)
    print()
    print("  Next Steps:")
    print("  1. Open  human_review_queue.csv  and fill in 'preferred_response'")
    print("  2. Feed  retrain_dpo_candidates.jsonl  into your DPO training script")
    print("  3. Re-run the evaluation suite to track improvement")
    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def run_feedback_loop(limit: int = 500, dpo_output: str = RETRAIN_JSONL_PATH) -> None:
    print(f"Loading up to {limit} bad responses from log...")
    records = load_bad_responses(limit=limit)

    if not records:
        print("No bad responses found. Log may be empty or path incorrect.")
        print(f"Expected log at: {os.path.abspath(dpo_output)}")
        return

    # Export DPO training candidates
    n_dpo = export_dpo_candidates(records, dpo_output)
    print(f"Exported {n_dpo} DPO candidates → {dpo_output}")

    # Export human review CSV
    n_csv = export_human_review_csv(records, HUMAN_REVIEW_CSV)
    print(f"Exported {n_csv} records to human review CSV → {HUMAN_REVIEW_CSV}")

    # Failure mode analysis
    failure_report = analyse_failure_modes(records)

    # Print summary
    print_feedback_report(records, failure_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot Retraining Feedback Loop")
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Maximum number of bad responses to process (default: 500)"
    )
    parser.add_argument(
        "--output", type=str, default=RETRAIN_JSONL_PATH,
        help="Output path for DPO candidates JSONL"
    )
    args = parser.parse_args()
    run_feedback_loop(limit=args.limit, dpo_output=args.output)
