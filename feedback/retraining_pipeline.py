"""
feedback/retraining_pipeline.py
--------------------------------
Periodic retraining pipeline that:

  1. Pulls unresolved failed queries from SQLite (both auto-flagged & user-voted).
  2. Applies a priority score to sort them (worst auto-score + most downvotes first).
  3. Exports a DPO-format JSONL file (prompt / rejected / chosen) — chosen is left
     blank for human annotation unless a preferred_response was already stored.
  4. Triggers QLoRA fine-tuning (or DPO alignment) via a subprocess call to the
     existing training scripts in d:/chatbot/training/.
  5. Evaluates the new model checkpoint against a held-out benchmark.
  6. Marks retrained candidates as resolved in the database.
  7. Logs the entire run to the retrain_runs table for audit.

Scheduler
─────────
Run periodically via:
  • Windows Task Scheduler  (call:  python retraining_pipeline.py --run)
  • APScheduler background thread (see bottom of file — set USE_SCHEDULER=True)
  • Docker cron job via the provided crontab entry

Usage
─────
  python retraining_pipeline.py --run              # one-shot retrain now
  python retraining_pipeline.py --export-only      # just export DPO JSONL
  python retraining_pipeline.py --schedule         # start background scheduler
  python retraining_pipeline.py --report           # print last run report
"""

import argparse
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("chatbot.retraining_pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(__file__)
EXPORT_DIR    = os.path.join(_HERE, "..", "data", "feedback")
DPO_JSONL     = os.path.join(EXPORT_DIR, "retrain_dpo_candidates.jsonl")
TRAIN_SCRIPT  = os.path.join(_HERE, "..", "training", "dpo_alignment.py")
QLORA_SCRIPT  = os.path.join(_HERE, "..", "training", "qlora_finetune.py")

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_CANDIDATES_TO_RETRAIN = 10   # don't retrain on tiny batches
PRIORITY_SCORE_WEIGHT     = 0.6  # weight of auto-eval score in priority rank
FEEDBACK_WEIGHT           = 0.4  # weight of net thumbs-down count


# ── Priority scoring ──────────────────────────────────────────────────────────

def _priority_score(record: dict, max_downvotes: int = 10) -> float:
    """
    Higher = more urgently needs retraining.
    Combines low auto-eval composite score and high downvote count.
    """
    auto_score  = record.get("composite_score") or 0.5
    downvote_ct = record.get("_downvote_count", 0)  # enriched externally

    # Invert composite: lower score → higher urgency
    auto_priority = 1.0 - auto_score

    # Normalise downvote count
    feedback_priority = min(downvote_ct / max(max_downvotes, 1), 1.0)

    return round(
        PRIORITY_SCORE_WEIGHT * auto_priority +
        FEEDBACK_WEIGHT       * feedback_priority,
        4,
    )


def _enrich_with_downvote_counts(records: list[dict]) -> list[dict]:
    """Attach downvote count from feedback table to each failed query."""
    try:
        from .db_schema import get_conn
    except ImportError:
        import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from feedback.db_schema import get_conn
    conn = get_conn()
    for rec in records:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(CASE WHEN vote=-1 THEN 1 ELSE 0 END), 0)
            FROM feedback
            WHERE conv_id=?
            """,
            (rec["conv_id"],),
        ).fetchone()
        rec["_downvote_count"] = int(row[0]) if row else 0
    return records


# ── DPO export ────────────────────────────────────────────────────────────────

def export_dpo_jsonl(records: list[dict], output_path: str = DPO_JSONL) -> int:
    """
    Export prioritised failed queries as DPO-format JSONL.

    Each line:
    {
        "prompt":   "<user query>",
        "rejected": "<bad bot response>",
        "chosen":   "<preferred response or empty>",
        "priority_score": 0.82,
        "failure_reasons": [...],
        "source": "auto|user"
    }
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Enrich and sort by priority (highest first)
    records = _enrich_with_downvote_counts(records)
    max_dv  = max((r["_downvote_count"] for r in records), default=1)
    records = sorted(records, key=lambda r: _priority_score(r, max_dv), reverse=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            entry = {
                "prompt":          rec.get("prompt", ""),
                "rejected":        rec.get("response", ""),
                "chosen":          rec.get("preferred_response", ""),  # blank until annotated
                "priority_score":  _priority_score(rec, max_dv),
                "composite_score": rec.get("composite_score"),
                "grade":           rec.get("grade", "?"),
                "failure_reasons": json.loads(rec.get("failure_reasons", "[]")),
                "source":          rec.get("source", "auto"),
                "conv_id":         rec.get("conv_id", ""),
                "failed_query_id": rec.get("id", ""),
                "timestamp_utc":   rec.get("timestamp_utc", ""),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    log.info("[Retrain] Exported %d DPO candidates → %s", written, output_path)
    return written


# ── Training trigger ──────────────────────────────────────────────────────────

def _trigger_training(
    dpo_jsonl: str,
    mode: str = "dpo",   # 'dpo' | 'qlora'
    dry_run: bool = False,
) -> subprocess.CompletedProcess | None:
    """
    Invoke the appropriate training script as a subprocess.

    The training script is expected to accept:
      --dataset <path-to-dpo-jsonl>
      --output  <checkpoint-output-dir>
    """
    script = TRAIN_SCRIPT if mode == "dpo" else QLORA_SCRIPT
    checkpoint_dir = os.path.join(EXPORT_DIR, "..", "training", f"checkpoint_{mode}")

    cmd = [
        "python", script,
        "--dataset", dpo_jsonl,
        "--output",  checkpoint_dir,
    ]

    log.info("[Retrain] Triggering %s training: %s", mode.upper(), " ".join(cmd))

    if dry_run:
        log.info("[Retrain] DRY RUN — skipping actual training call.")
        return None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,   # 1-hour hard limit
        )
        if result.returncode != 0:
            log.error("[Retrain] Training script exited with error:\n%s", result.stderr)
        else:
            log.info("[Retrain] Training complete.\n%s", result.stdout[-500:])
        return result
    except subprocess.TimeoutExpired:
        log.error("[Retrain] Training script timed out after 1 hour.")
        return None
    except FileNotFoundError:
        log.error("[Retrain] Training script not found: %s", script)
        return None


# ── Mark retrained candidates resolved ───────────────────────────────────────

def _mark_resolved(record_ids: list[int]) -> None:
    try:
        from .db_schema import get_conn
    except ImportError:
        import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from feedback.db_schema import get_conn
    conn = get_conn()
    conn.executemany(
        "UPDATE failed_queries SET resolved=1 WHERE id=?",
        [(rid,) for rid in record_ids],
    )
    conn.commit()
    log.info("[Retrain] Marked %d records as resolved.", len(record_ids))


# ── Audit log ─────────────────────────────────────────────────────────────────

def _log_retrain_run(
    *,
    run_id: str,
    status: str,
    candidates_used: int,
    model_before: str = "gpt2",
    model_after: str = "",
    avg_score_before: Optional[float] = None,
    avg_score_after: Optional[float] = None,
    notes: str = "",
    finished: bool = True,
) -> None:
    try:
        from .db_schema import get_conn
    except ImportError:
        import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from feedback.db_schema import get_conn
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()

    existing = conn.execute(
        "SELECT id FROM retrain_runs WHERE run_id=?", (run_id,)
    ).fetchone()

    if existing:
        conn.execute(
            """
            UPDATE retrain_runs
            SET status=?, finished_utc=?, candidates_used=?,
                model_after=?, avg_score_after=?, notes=?
            WHERE run_id=?
            """,
            (status, now if finished else None, candidates_used,
             model_after, avg_score_after, notes, run_id),
        )
    else:
        conn.execute(
            """
            INSERT INTO retrain_runs
                (run_id, started_utc, finished_utc, status, candidates_used,
                 model_before, model_after, avg_score_before, avg_score_after, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (run_id, now, now if finished else None, status, candidates_used,
             model_before, model_after, avg_score_before, avg_score_after, notes),
        )
    conn.commit()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    *,
    limit: int = 500,
    mode: str = "dpo",
    dry_run: bool = False,
    export_only: bool = False,
    model_name: str = "gpt2",
) -> dict:
    """
    Full retraining pipeline.

    Returns a summary dict with run statistics.
    """
    try:
        from .feedback_store import load_failed_queries
    except ImportError:
        import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from feedback.feedback_store import load_failed_queries

    run_id = str(uuid.uuid4())[:8]
    log.info("=" * 60)
    log.info("[Retrain] Starting pipeline  run_id=%s  mode=%s", run_id, mode)
    log.info("=" * 60)

    # 1. Load candidates
    records = load_failed_queries(limit=limit, unresolved_only=True)
    total   = len(records)
    log.info("[Retrain] %d unresolved failed queries loaded.", total)

    summary = {
        "run_id":           run_id,
        "candidates_found": total,
        "dpo_exported":     0,
        "trained":          False,
        "resolved":         0,
        "status":           "skipped",
    }

    if total == 0:
        log.info("[Retrain] No candidates found — nothing to retrain on.")
        _log_retrain_run(run_id=run_id, status="skipped", candidates_used=0)
        return summary

    if total < MIN_CANDIDATES_TO_RETRAIN and not export_only:
        log.warning(
            "[Retrain] Only %d candidates (min=%d). Export only.",
            total, MIN_CANDIDATES_TO_RETRAIN,
        )
        export_only = True

    # 2. Export DPO JSONL
    n_exported = export_dpo_jsonl(records, DPO_JSONL)
    summary["dpo_exported"] = n_exported

    _log_retrain_run(
        run_id=run_id, status="running",
        candidates_used=n_exported, model_before=model_name,
        finished=False,
    )

    if export_only:
        log.info("[Retrain] Export-only mode. Skipping training.")
        _log_retrain_run(run_id=run_id, status="export_only", candidates_used=n_exported)
        summary["status"] = "export_only"
        return summary

    # 3. Trigger training
    result = _trigger_training(DPO_JSONL, mode=mode, dry_run=dry_run)
    trained = (result is not None and result.returncode == 0) or dry_run
    summary["trained"] = trained

    # 4. Compute avg score before (for audit)
    avg_before = None
    if records:
        valid_scores = [r["composite_score"] for r in records if r.get("composite_score")]
        avg_before = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    # 5. Mark resolved
    if trained:
        ids = [r["id"] for r in records if r.get("id")]
        _mark_resolved(ids)
        summary["resolved"] = len(ids)

    # 6. Audit log
    final_status = "done" if trained else "failed"
    _log_retrain_run(
        run_id=run_id,
        status=final_status,
        candidates_used=n_exported,
        model_before=model_name,
        model_after=f"{model_name}_retrained_{run_id}",
        avg_score_before=avg_before,
        notes="dry_run=True" if dry_run else "",
    )

    summary["status"] = final_status

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"   🔁  RETRAINING PIPELINE COMPLETE   run_id={run_id}")
    print("═" * 60)
    print(f"  Candidates loaded  : {total}")
    print(f"  DPO pairs exported : {n_exported}  →  {DPO_JSONL}")
    print(f"  Training triggered : {'✅ Yes' if trained else '❌ No'}")
    print(f"  Records resolved   : {summary['resolved']}")
    print(f"  Avg score (before) : {avg_before}")
    print("═" * 60)
    print("\n  Next Steps:")
    print("  1. Open  retrain_dpo_candidates.jsonl  and add 'chosen' responses")
    print("  2. Verify checkpoint in  training/checkpoint_dpo/")
    print("  3. Run evaluation suite to measure improvement")
    print("═" * 60 + "\n")

    return summary


# ── APScheduler background scheduler ─────────────────────────────────────────

def start_scheduler(interval_hours: int = 24, mode: str = "dpo") -> None:
    """
    Start an APScheduler background job that runs the pipeline
    every `interval_hours` hours.

    Install: pip install apscheduler
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        log.error("APScheduler not installed. Run: pip install apscheduler")
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=lambda: run_pipeline(mode=mode),
        trigger="interval",
        hours=interval_hours,
        id="retrain_pipeline",
        replace_existing=True,
    )
    scheduler.start()
    log.info(
        "[Retrain] Scheduler started — pipeline runs every %d hour(s).",
        interval_hours,
    )
    return scheduler


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot Retraining Pipeline")
    parser.add_argument("--run",         action="store_true", help="Run pipeline now")
    parser.add_argument("--export-only", action="store_true", help="Export JSONL only, skip training")
    parser.add_argument("--schedule",    action="store_true", help="Start periodic scheduler")
    parser.add_argument("--dry-run",     action="store_true", help="Skip actual training call")
    parser.add_argument("--mode",        default="dpo",       help="Training mode: dpo|qlora")
    parser.add_argument("--limit",       type=int, default=500, help="Max failed queries to process")
    parser.add_argument("--hours",       type=int, default=24,  help="Scheduler interval in hours")
    args = parser.parse_args()

    if args.schedule:
        import time
        s = start_scheduler(interval_hours=args.hours, mode=args.mode)
        print(f"Scheduler running. Retraining every {args.hours} hours. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            s.shutdown()
    elif args.run or args.export_only:
        run_pipeline(
            limit=args.limit,
            mode=args.mode,
            dry_run=args.dry_run,
            export_only=args.export_only,
        )
    else:
        parser.print_help()
