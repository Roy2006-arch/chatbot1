import asyncio
import logging
import time
import gc
import torch
from typing import Optional
from backend.shared_resources import ModelRegistry

logger = logging.getLogger("chatbot.lifecycle")


class MemoryLifecycleManager:
    def __init__(
        self,
        context_manager,
        orchestrator,
        doc_processor=None,
        cleanup_interval_seconds: int = 300,
        session_ttl_seconds: int = 3600,
        stream_ttl_seconds: int = 600,
    ):
        self.context_manager = context_manager
        self.orchestrator = orchestrator
        self.doc_processor = doc_processor
        self.cleanup_interval = cleanup_interval_seconds
        self.session_ttl = session_ttl_seconds
        self.stream_ttl = stream_ttl_seconds

        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._run_cleanup_loop())
        logger.info("[LifecycleManager] Background cleanup started.")

    async def stop(self):
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._run_final_cleanup()
        logger.info("[LifecycleManager] Stopped.")

    async def _run_cleanup_loop(self):
        while self._is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.perform_cleanup()
            except Exception as e:
                logger.error("[LifecycleManager] Cleanup loop error: %s", e)

    async def perform_cleanup(self):
        t0 = time.time()
        logger.info("[LifecycleManager] Starting scheduled cleanup...")

        sessions_cleaned = self.context_manager.cleanup_idle_sessions(self.session_ttl)
        orchestration_cleaned = self.orchestrator.cleanup_stale_sessions(self.stream_ttl)

        docs_cleaned = 0
        if self.doc_processor:
            docs_cleaned = self.doc_processor.cleanup_idle_sessions(self.session_ttl)

        gc.collect()

        if torch.cuda.is_available():
            ModelRegistry.clear_cache()
            logger.debug("[LifecycleManager] CUDA cache cleared.")

        duration = (time.time() - t0) * 1000
        logger.info(
            "[LifecycleManager] Cleanup: Context=-%d, Orchestrator=-%d, Docs=-%d. Took %.1fms",
            sessions_cleaned, orchestration_cleaned, docs_cleaned, duration,
        )

    def cleanup_session(self, session_id: str):
        self.orchestrator.reset_session(session_id)
        self.context_manager.cleanup_idle_sessions(0)
        if self.doc_processor:
            self.doc_processor.cleanup_idle_sessions(0)
        logger.debug("[LifecycleManager] Cleaned up session: %s", session_id)

    def force_cleanup_request(self, session_id: str):
        self.cleanup_session(session_id)

    def _run_final_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            ModelRegistry.clear_cache()
