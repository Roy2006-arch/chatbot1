"""
response_middleware.py
======================
FastAPI middleware that logs reasoning traces for every chat request.

Registers as a Starlette BaseHTTPMiddleware so every request passing
through /chat/* gets timing headers + a server-side reasoning audit log.

Install in main.py:
    from response_middleware import ReasoningAuditMiddleware
    app.add_middleware(ReasoningAuditMiddleware, pipeline=reasoning_pipeline)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from backend.reasoning_pipeline import ReasoningPipeline

AUDIT_LOG_PATH = Path(__file__).parent.parent / "evaluation" / "logs" / "reasoning_audit.jsonl"
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("chatbot.middleware")


class ReasoningAuditMiddleware(BaseHTTPMiddleware):
    """
    Attaches timing and reasoning-trace metadata to every /chat/* response.

    Response Headers added (non-sensitive — debug only; strip in prod):
        X-Pipeline-Intent     : classified intent category
        X-Pipeline-Latency-Ms : total pre/post processing time
        X-Pipeline-Refined    : 'true' | 'false'
    """

    def __init__(self, app: ASGIApp, pipeline: ReasoningPipeline):
        super().__init__(app)
        self.pipeline = pipeline

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Only annotate chat endpoints
        if request.url.path.startswith("/chat"):
            trace = self.pipeline.get_last_trace()
            if trace:
                response.headers["X-Pipeline-Intent"]     = trace.intent_category
                response.headers["X-Pipeline-Latency-Ms"] = str(trace.latency_ms)
                response.headers["X-Pipeline-Refined"]    = str(trace.refinement_applied).lower()

                # Append to JSONL audit log
                audit_entry = {
                    "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "path":               str(request.url.path),
                    "intent":             trace.intent,
                    "intent_category":    trace.intent_category,
                    "reasoning_steps":    trace.steps,
                    "draft_issues":       trace.draft_issues,
                    "refinement_applied": trace.refinement_applied,
                    "pipeline_ms":        trace.latency_ms,
                    "total_request_ms":   elapsed_ms,
                }
                try:
                    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(audit_entry) + "\n")
                except OSError as exc:
                    logger.warning("Could not write audit log: %s", exc)

        return response
