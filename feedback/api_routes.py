"""
feedback/api_routes.py
-----------------------
FastAPI router that exposes all feedback loop endpoints.

Mount in backend/main.py:
    from feedback.api_routes import router as feedback_router
    app.include_router(feedback_router, prefix="/feedback", tags=["Feedback"])

Endpoints
─────────
  POST /feedback/vote           — submit 👍 / 👎
  GET  /feedback/summary        — dashboard aggregates
  GET  /feedback/failed-queries — list failed queries for review
  GET  /feedback/conversations  — list logged conversations
  GET  /feedback/conversation/{conv_id} — full turn log
  POST /feedback/annotate       — save preferred_response for a failed query
  POST /feedback/retrain        — trigger retraining pipeline (admin)
  GET  /feedback/retrain/status — last retrain run status
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, validator

from .db_schema import get_conn, init_db
from .feedback_store import (
    record_feedback,
    load_all_feedback,
    load_failed_queries,
    feedback_summary,
    get_feedback_score,
)
from .conversation_logger import load_conversation, load_session_conversations

router = APIRouter()

# Ensure DB is initialised when this module loads
init_db()


# ── Pydantic models ───────────────────────────────────────────────────────────

class VoteRequest(BaseModel):
    conv_id:    str
    turn_index: int
    session_id: str
    vote:       int        # +1 (👍) or -1 (👎)
    comment:    str = ""
    prompt:     str = ""   # denormalised for quick lookup
    response:   str = ""

    @validator("vote")
    def validate_vote(cls, v):
        if v not in (1, -1):
            raise ValueError("vote must be +1 (👍) or -1 (👎)")
        return v


class AnnotateRequest(BaseModel):
    failed_query_id:    int
    preferred_response: str
    annotator_notes:    str = ""


class RetrainRequest(BaseModel):
    mode:     str = "dpo"    # 'dpo' | 'qlora'
    limit:    int = 500
    dry_run:  bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/vote", summary="Submit 👍 or 👎 feedback for a response")
def submit_vote(req: VoteRequest):
    """
    Record user feedback on a specific assistant turn.

    - vote = +1  →  👍  (thumbs up)
    - vote = -1  →  👎  (thumbs down — auto-queues for retraining)
    """
    record = record_feedback(
        conv_id=req.conv_id,
        turn_index=req.turn_index,
        session_id=req.session_id,
        vote=req.vote,
        comment=req.comment,
        prompt=req.prompt,
        response=req.response,
    )
    feedback_score = get_feedback_score(req.conv_id, req.turn_index)
    return {
        "status":         "recorded",
        "vote_label":     "👍 thumbs_up" if req.vote == 1 else "👎 thumbs_down",
        "feedback_score": feedback_score,
        **record,
    }


@router.get("/summary", summary="Feedback dashboard aggregates")
def get_summary():
    """
    Returns aggregate statistics:
    - total votes, thumbs up/down counts, satisfaction rate
    - failed query counts (auto-flagged vs user-flagged)
    - average composite score across all assistant turns
    """
    return feedback_summary()


@router.get("/failed-queries", summary="List failed / low-quality queries")
def get_failed_queries(
    limit: int = Query(100, le=1000),
    source: Optional[str] = Query(None, description="auto | user | None=all"),
    unresolved_only: bool = Query(True),
):
    """Return failed queries sorted by recency, ready for human annotation."""
    results = load_failed_queries(
        limit=limit,
        source_filter=source,
        unresolved_only=unresolved_only,
    )
    return {
        "count": len(results),
        "results": results,
    }


@router.get("/conversations", summary="List logged conversations for a session")
def get_conversations(
    session_id: str = Query(...),
    limit: int = Query(50, le=200),
):
    convs = load_session_conversations(session_id, limit=limit)
    return {"session_id": session_id, "conversations": convs}


@router.get("/conversation/{conv_id}", summary="Full turn-level log for a conversation")
def get_conversation(conv_id: str):
    turns = load_conversation(conv_id)
    if not turns:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conv_id": conv_id, "turns": turns}


@router.post("/annotate", summary="Save preferred response for a failed query")
def annotate_failed_query(req: AnnotateRequest):
    """
    Allows human annotators to provide the 'chosen' (ideal) response for a
    failed query. This will be included in the next DPO training export.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT id FROM failed_queries WHERE id=?", (req.failed_query_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Failed query #{req.failed_query_id} not found")

    conn.execute(
        """
        UPDATE failed_queries
        SET preferred_response=?, resolved=0
        WHERE id=?
        """,
        (req.preferred_response, req.failed_query_id),
    )
    conn.commit()
    return {
        "status": "annotated",
        "failed_query_id": req.failed_query_id,
        "message": "Preferred response saved. Will be included in next retraining export.",
    }


@router.post("/retrain", summary="[Admin] Trigger retraining pipeline")
def trigger_retrain(req: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Triggers the full retraining pipeline in the background.
    Returns immediately with a run_id; check /retrain/status for progress.
    """
    import uuid
    run_id = str(uuid.uuid4())[:8]

    def _run():
        from .retraining_pipeline import run_pipeline
        run_pipeline(
            limit=req.limit,
            mode=req.mode,
            dry_run=req.dry_run,
        )

    background_tasks.add_task(_run)
    return {
        "status":  "queued",
        "run_id":  run_id,
        "message": "Retraining pipeline started in background. Check /feedback/retrain/status.",
    }


@router.get("/retrain/status", summary="Status of the last retraining run")
def retrain_status():
    """Returns the most recent entry from retrain_runs."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM retrain_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return {"status": "no_runs", "message": "No retraining runs recorded yet."}
    return dict(row)


@router.get("/feedback-scores", summary="Feedback scores for recent responses")
def get_feedback_scores(
    limit: int = Query(50, le=500),
    vote_filter: Optional[int] = Query(None, description="+1 or -1"),
):
    """Returns raw feedback records with sigmoid-normalised scores."""
    records = load_all_feedback(limit=limit, vote_filter=vote_filter)
    # Enrich with feedback_score
    for rec in records:
        rec["feedback_score"] = get_feedback_score(rec["conv_id"], rec["turn_index"])
    return {"count": len(records), "records": records}
