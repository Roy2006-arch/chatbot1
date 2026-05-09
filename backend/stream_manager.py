import asyncio
import json
import logging
import re
from typing import AsyncGenerator, Optional

logger = logging.getLogger("chatbot.stream_manager")


class StreamManager:
    def __init__(self, conv_id: str, session_id: str, continuity_manager=None):
        self.conv_id = conv_id
        self.session_id = session_id
        self.queue = asyncio.Queue()
        self.continuity_manager = continuity_manager

        self.stop_requested = False
        self.tokens_yielded = 0
        self._finalized = False
        self._buffer_for_thought = ""
        self._in_thought = False
        self._thought_content = ""

    async def put(self, token: str):
        if token and not self.stop_requested:
            await self.queue.put(token)

    async def finalize(self):
        if not self._finalized:
            self._finalized = True
            await self.queue.put(None)

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    async def event_generator(self, timeout: float = 60.0) -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    token = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning("Stream timeout reached for session=%s", self.session_id)
                    yield f"data: {json.dumps({'error': 'stream_timeout'})}\n\n"
                    break

                if token is None or self.stop_requested:
                    break

                clean_tokens = self._suppress_thought_tags(token)
                for clean in clean_tokens:
                    if clean:
                        async for event in self._yield_safe(clean):
                            yield event

        except Exception as e:
            logger.error("Error in event_generator: %s", str(e), exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def _suppress_thought_tags(self, token: str) -> list[str]:
        MAX_THOUGHT_BUFFER = 8192
        self._buffer_for_thought += token
        if len(self._buffer_for_thought) > MAX_THOUGHT_BUFFER:
            overflow = self._buffer_for_thought[:MAX_THOUGHT_BUFFER // 2]
            self._buffer_for_thought = self._buffer_for_thought[MAX_THOUGHT_BUFFER // 2:]
            if not self._in_thought:
                return [overflow] if overflow else []
            self._thought_content += overflow
            return []
        if not self._in_thought:
            parts = self._buffer_for_thought.split("<thought>")
            if len(parts) > 1:
                output = [parts[0]] if parts[0] else []
                self._buffer_for_thought = "<thought>".join(parts[1:])
                self._in_thought = True
                self._thought_content = ""
                return output
            else:
                cutoff = max(0, len(self._buffer_for_thought) - 8)
                safe, self._buffer_for_thought = self._buffer_for_thought[:cutoff], self._buffer_for_thought[cutoff:]
                return [safe] if safe else []
        else:
            end_idx = self._buffer_for_thought.find("</thought>")
            if end_idx != -1:
                self._thought_content += self._buffer_for_thought[:end_idx]
                self._buffer_for_thought = self._buffer_for_thought[end_idx + 10:]
                self._in_thought = False
                self._thought_content = ""
                return self._suppress_thought_tags("")
            else:
                self._thought_content += self._buffer_for_thought
                self._buffer_for_thought = ""
                return []

    async def _yield_safe(self, text: str) -> AsyncGenerator[str, None]:
        stop_tokens = [
            "<|user|>", "<|assistant|>", "<|system|>",
            "\nuser:", "\nassistant:", "\nsystem:",
            "\n\nuser:", "\n\nassistant:", "\n\nsystem:",
            "\nUser:", "\nAssistant:", "\nSystem:",
            "\n\nUser:", "\n\nAssistant:", "\n\nSystem:",
            "\nhuman:", "\nHuman:",
            "\n\nhuman:", "\n\nHuman:",
        ]
        for t in stop_tokens:
            if t in text:
                text = text.split(t)[0]
                self.stop_requested = True
                logger.warning(
                    "Stop token '%s' detected in stream for session=%s. Truncating.",
                    t, self.session_id,
                )
                break

        if not text:
            return

        if self.continuity_manager:
            safe_text = self.continuity_manager.process_chunk(self.session_id, text)
        else:
            safe_text = text

        if safe_text:
            yield f"data: {json.dumps({'content': safe_text, 'conv_id': self.conv_id})}\n\n"
            self.tokens_yielded += len(safe_text)

    def stop(self):
        self.stop_requested = True

    def reset(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer_for_thought = ""
        self._in_thought = False
        self._thought_content = ""
        self.stop_requested = False
        self._finalized = False
        self.tokens_yielded = 0
