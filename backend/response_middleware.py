from __future__ import annotations

import json
import logging
import time
import contextvars
from pathlib import Path

import aiofiles

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from backend.reasoning_pipeline import ReasoningPipeline

AUDIT_LOG_PATH = Path(__file__).parent.parent / "evaluation" / "logs" / "reasoning_audit.jsonl"
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("chatbot.middleware")

# Global request-scoped context variable for the reasoning trace
request_trace_var: contextvars.ContextVar = contextvars.ContextVar("request_trace", default=None)


class ReasoningAuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, pipeline: ReasoningPipeline):
        super().__init__(app)
        self.pipeline = pipeline

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        t0 = time.perf_counter()
        # Initialize context variable for this request execution context
        token = request_trace_var.set(None)
        
        response = await call_next(request)
        
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        if request.url.path.startswith("/chat"):
            trace = request_trace_var.get()
            if trace:
                response.headers["X-Pipeline-Intent"] = trace.intent_category
                response.headers["X-Pipeline-Latency-Ms"] = str(trace.latency_ms)
                response.headers["X-Pipeline-Refined"] = str(trace.refinement_applied).lower()

                audit_entry = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "path": str(request.url.path),
                    "intent": trace.intent,
                    "intent_category": trace.intent_category,
                    "reasoning_steps": trace.steps,
                    "draft_issues": trace.draft_issues,
                    "refinement_applied": trace.refinement_applied,
                    "pipeline_ms": trace.latency_ms,
                    "total_request_ms": elapsed_ms,
                }
                try:
                    async with aiofiles.open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                        await f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
                except OSError as exc:
                    logger.warning("Could not write audit log: %s", exc)
        
        request_trace_var.reset(token)
        return response
