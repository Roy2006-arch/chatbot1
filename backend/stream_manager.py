import asyncio
import json
import logging
import re
from typing import AsyncGenerator, Optional

logger = logging.getLogger("chatbot.stream_manager")

class StreamManager:
    """
    Orchestrates the SSE streaming pipeline.
    Handles queue management, hidden reasoning, and SSE formatting.
    Integrates with ContinuityManager for markdown safety.
    """
    def __init__(self, conv_id: str, session_id: str, continuity_manager=None):
        self.conv_id = conv_id
        self.session_id = session_id
        self.queue = asyncio.Queue()
        self.continuity_manager = continuity_manager
        self.is_thinking = False
        self.thinking_buffer = ""
        self.stop_requested = False
        self.tokens_yielded = 0

    async def put(self, token: str):
        """Put a token into the stream queue."""
        await self.queue.put(token)

    async def finalize(self):
        """Signal the end of the stream."""
        await self.queue.put(None)

    async def event_generator(self, timeout: float = 60.0) -> AsyncGenerator[str, None]:
        """
        Consumes the queue and yields SSE formatted strings.
        Does NOT yield final 'done' or '[DONE]' as main.py handles refinement.
        """
        try:
            while True:
                try:
                    token = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning("Stream timeout reached.")
                    yield f"data: {json.dumps({'error': 'stream_timeout'})}\n\n"
                    break

                if token is None or self.stop_requested:
                    break

                # ── Hidden Reasoning Suppression ──────────────────────────────
                combined = self.thinking_buffer + token
                if "<thought>" in combined.lower() and not self.is_thinking:
                    self.is_thinking = True
                
                if self.is_thinking:
                    self.thinking_buffer += token
                    if "</thought>" in self.thinking_buffer.lower():
                        self.is_thinking = False
                        parts = re.split(r"</thought>", self.thinking_buffer, flags=re.IGNORECASE)
                        token = parts[1] if len(parts) > 1 else ""
                        self.thinking_buffer = ""
                        if not token: continue
                    else:
                        continue

                # ── Hallucination / Stop Token Guard ──────────────────────────
                stop_tokens = ["<|user|>", "<|assistant|>", "<|system|>", "\nuser:", "\nassistant:"]
                if any(t in token.lower() for t in stop_tokens):
                    break

                # ── Yield Safe Content via ContinuityManager ──────────────────
                if self.continuity_manager:
                    safe_text = self.continuity_manager.process_chunk(self.session_id, token)
                else:
                    safe_text = token

                if safe_text:
                    yield f"data: {json.dumps({'content': safe_text, 'conv_id': self.conv_id})}\n\n"
                    self.tokens_yielded += len(safe_text)
                
                await asyncio.sleep(0.005)

        except Exception as e:
            logger.error(f"Error in event_generator: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        # No finally-yield [DONE] here, main.py controls completion.

    def stop(self):
        """Request the stream to stop."""
        self.stop_requested = True

    def reset(self):
        """Resets the state for a new generation pass."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.is_thinking = False
        self.thinking_buffer = ""
        self.stop_requested = False
        # Do not reset tokens_yielded as we want to track total across passes if needed
        # Or if main.py expects it, we can. For now, just reset flags.
