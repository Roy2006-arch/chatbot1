import asyncio
import logging
import time
import gc
import torch
from typing import Optional

logger = logging.getLogger("chatbot.lifecycle")

class MemoryLifecycleManager:
    """
    Manages the lifecycle of sessions, streams, and system resources.
    Ensures that stale resources are purged to maintain performance and stability.
    """
    
    def __init__(
        self, 
        context_manager, 
        orchestrator,
        cleanup_interval_seconds: int = 300, # Clean up every 5 minutes
        session_ttl_seconds: int = 3600,     # Sessions expire after 1 hour
        stream_ttl_seconds: int = 600        # Active generation state expires after 10 mins
    ):
        self.context_manager = context_manager
        self.orchestrator = orchestrator
        self.cleanup_interval = cleanup_interval_seconds
        self.session_ttl = session_ttl_seconds
        self.stream_ttl = stream_ttl_seconds
        
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self):
        """Starts the background cleanup task."""
        if self._is_running:
            return
            
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._run_cleanup_loop())
        logger.info("[LifecycleManager] Background cleanup task started.")

    async def stop(self):
        """Stops the background cleanup task."""
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("[LifecycleManager] Background cleanup task stopped.")

    async def _run_cleanup_loop(self):
        """Continuous loop for resource management."""
        while self._is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.perform_cleanup()
            except Exception as e:
                logger.error(f"[LifecycleManager] Cleanup loop error: {e}")

    async def perform_cleanup(self):
        """Executes all cleanup protocols."""
        t0 = time.time()
        logger.info("[LifecycleManager] Starting scheduled cleanup...")
        
        # 1. Cleanup Idle Context Sessions
        sessions_cleaned = self.context_manager.cleanup_idle_sessions(self.session_ttl)
        
        # 2. Cleanup Stale Orchestration States (Abandoned Requests)
        orchestration_cleaned = self.orchestrator.cleanup_stale_sessions(self.stream_ttl)
        
        # 3. Explicit Garbage Collection
        # This helps clear large string buffers and unreferenced tensors
        gc.collect()
        
        # 4. GPU Memory Maintenance (if not using vLLM's exclusive mode)
        # Note: vLLM usually handles its own pool, but this helps for other components (Embedders, etc.)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("[LifecycleManager] CUDA cache cleared.")

        duration = (time.time() - t0) * 1000
        logger.info(
            f"[LifecycleManager] Cleanup complete. "
            f"Context: -{sessions_cleaned}, Orchestrator: -{orchestration_cleaned}. "
            f"Took {duration:.1f}ms"
        )

    def force_cleanup_request(self, session_id: str):
        """Immediately purges ephemeral state for a specific request/session."""
        self.orchestrator.reset_session(session_id)
        logger.debug(f"[LifecycleManager] Forced cleanup for session: {session_id}")
