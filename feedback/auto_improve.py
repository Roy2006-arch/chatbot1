"""
feedback/auto_improve.py
------------------------
Scheduled auto-improvement job that:
  1. Polls feedback.db for new entries since last check
  2. Runs self-improvement pipeline
  3. Quality-filters results
  4. Computes composite score gain
  5. If gain > threshold, triggers QLoRA retraining

Usage:
    from feedback.auto_improve import start_auto_improve_scheduler
    scheduler = start_auto_improve_scheduler(min_new_entries=10)
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List
from threading import Lock

from .db_schema import get_conn_ctx, _now_utc
from self_improvement.pipeline import SelfImprovementPipeline
from self_improvement.schema import ImprovementReport
from evaluation.scorer import evaluate_response

log = logging.getLogger("chatbot.auto_improve")

_DEFAULT_MIN_NEW_ENTRIES = 10
_DEFAULT_MIN_COMPOSITE_GAIN = 0.05
_DEFAULT_POLL_INTERVAL_SECONDS = 3600
_LAST_CHECK_KEY = "auto_improve_last_check"


class AutoImprover:
    """Checks for new feedback entries, runs self-improvement, evaluates gain, and triggers retraining."""

    def __init__(
        self,
        min_new_entries: int = _DEFAULT_MIN_NEW_ENTRIES,
        min_composite_gain: float = _DEFAULT_MIN_COMPOSITE_GAIN,
    ):
        self.min_new_entries = min_new_entries
        self.min_composite_gain = min_composite_gain
        self._lock = Lock()
        self._last_check_time = self._load_last_check()

    def _load_last_check(self) -> str:
        """Load last check timestamp from DB."""
        with get_conn_ctx() as conn:
            row = conn.execute(
                "SELECT notes FROM retrain_runs WHERE run_id=? ORDER BY id DESC LIMIT 1",
                (_LAST_CHECK_KEY,),
            ).fetchone()
        if row:
            return row["notes"]
        return ""

    def _save_last_check(self, timestamp: str):
        """Save last check timestamp to DB."""
        with get_conn_ctx() as conn:
            existing = conn.execute(
                "SELECT id FROM retrain_runs WHERE run_id=?", (_LAST_CHECK_KEY,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE retrain_runs SET notes=?, started_utc=? WHERE run_id=?",
                    (timestamp, timestamp, _LAST_CHECK_KEY),
                )
            else:
                conn.execute(
                    "INSERT INTO retrain_runs (run_id, started_utc, notes) VALUES (?, ?, ?)",
                    (_LAST_CHECK_KEY, timestamp, timestamp),
                )

    def count_new_entries(self) -> int:
        """Count feedback entries since last check."""
        with get_conn_ctx() as conn:
            if self._last_check_time:
                row = conn.execute(
                    "SELECT COUNT(*) FROM feedback WHERE timestamp_utc > ?",
                    (self._last_check_time,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()
        return int(row[0]) if row else 0

    def get_last_check_time(self) -> str:
        return self._last_check_time

    def run(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run one auto-improvement cycle. Returns summary dict with results."""
        with self._lock:
            now = _now_utc()
            new_count = self.count_new_entries()

            summary = {
                "checked_at": now,
                "new_entries_since_last_check": new_count,
                "threshold_met": False,
                "self_improvement_run": None,
                "composite_gain": 0.0,
                "retraining_triggered": False,
                "dry_run": dry_run,
            }

            if new_count < self.min_new_entries:
                log.info(
                    "[AutoImprove] Skipped: %d new entries < min %d",
                    new_count, self.min_new_entries,
                )
                self._save_last_check(now)
                return summary

            summary["threshold_met"] = True

            log.info("[AutoImprove] Running self-improvement pipeline...")
            pipeline = SelfImprovementPipeline()
            report = pipeline.run(
                max_failed=min(new_count, 500),
                max_corrections=min(new_count, 200),
            )
            summary["self_improvement_run"] = {
                "run_id": report.run_id,
                "corrections": report.corrections_generated,
                "total_processed": report.total_failed_queries,
            }

            if report.total_failed_queries > 0:
                failed = pipeline.correction_gen.load_failed_queries(
                    limit=min(new_count, 500), unresolved_only=True
                )
                if failed:
                    scores_before = [
                        float(r.get("composite_score", 0) or 0) for r in failed
                    ]
                    generated = pipeline.correction_gen.generate_batch(failed[:100])
                    if generated:
                        scores_after = [g.score_after for g in generated]
                        score_before = sum(scores_before) / len(scores_before)
                        score_after = sum(scores_after) / len(scores_after)
                        gain = score_after - score_before
                        summary["composite_gain"] = round(gain, 4)
                        summary["score_before"] = round(score_before, 4)
                        summary["score_after"] = round(score_after, 4)

                        log.info(
                            "[AutoImprove] Score: before=%.4f after=%.4f gain=%.4f",
                            score_before, score_after, gain,
                        )

            if summary["composite_gain"] >= self.min_composite_gain and not dry_run:
                log.info("[AutoImprove] Gain %.4f >= %.4f -- triggering retraining",
                         summary["composite_gain"], self.min_composite_gain)

                dpo_path = os.path.join(
                    "data", "feedback",
                    f"auto_improve_dpo_{report.run_id}.jsonl",
                )
                pipeline.export_dpo_for_retraining(dpo_path)

                try:
                    from .retraining_pipeline import run_pipeline
                    retrain_result = run_pipeline(
                        dry_run=False,
                        export_only=False,
                    )
                    summary["retraining_triggered"] = retrain_result.get("status") == "done"
                    summary["retrain_run_id"] = retrain_result.get("run_id", "")
                except Exception as e:
                    log.error("[AutoImprove] Retraining error: %s", e, exc_info=True)
                    summary["retraining_error"] = str(e)

            elif summary["composite_gain"] < self.min_composite_gain:
                log.info(
                    "[AutoImprove] Gain %.4f below threshold %.4f -- skipping retraining",
                    summary["composite_gain"], self.min_composite_gain,
                )

            self._save_last_check(now)
            return summary


_SCHEDULER = None


def start_auto_improve_scheduler(
    interval_seconds: int = _DEFAULT_POLL_INTERVAL_SECONDS,
    min_new_entries: int = _DEFAULT_MIN_NEW_ENTRIES,
    min_composite_gain: float = _DEFAULT_MIN_COMPOSITE_GAIN,
    dry_run: bool = False,
):
    """
    Start APScheduler background job for auto-improvement.

    Returns the scheduler instance (call .shutdown() to stop).
    """
    global _SCHEDULER

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        log.error("APScheduler not installed. Run: pip install apscheduler")
        return None

    improver = AutoImprover(
        min_new_entries=min_new_entries,
        min_composite_gain=min_composite_gain,
    )

    def _job():
        log.info("[AutoImprove] Scheduled job triggered")
        try:
            result = improver.run(dry_run=dry_run)
            log.info("[AutoImprove] Job complete: %s", json.dumps(result, default=str))
        except Exception as e:
            log.error("[AutoImprove] Job failed: %s", e, exc_info=True)

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=_job,
        trigger="interval",
        seconds=interval_seconds,
        id="auto_improve",
        replace_existing=True,
    )
    scheduler.start()

    _SCHEDULER = scheduler
    log.info(
        "[AutoImprove] Scheduler started -- polling every %d seconds",
        interval_seconds,
    )
    return scheduler


def stop_auto_improve_scheduler():
    """Stop the auto-improve scheduler."""
    global _SCHEDULER
    if _SCHEDULER:
        _SCHEDULER.shutdown(wait=False)
        _SCHEDULER = None
        log.info("[AutoImprove] Scheduler stopped")


def run_once(
    min_new_entries: int = _DEFAULT_MIN_NEW_ENTRIES,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single auto-improvement cycle (one-shot)."""
    improver = AutoImprover(min_new_entries=min_new_entries)
    return improver.run(dry_run=dry_run)
