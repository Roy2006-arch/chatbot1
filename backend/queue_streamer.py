import asyncio
from transformers import TextStreamer
from typing import Optional

class QueueStreamer(TextStreamer):
    """
    HuggingFace Streamer that pushes tokens into an asyncio.Queue.
    Supports asynchronous consumption.
    """
    def __init__(self, tokenizer, queue: asyncio.Queue, skip_prompt: bool = True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.queue = queue
        self.loop = asyncio.get_event_loop()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        Called when a new chunk of text is ready.
        We use call_soon_threadsafe because this might be called from a background thread.
        """
        if text:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, text)
        
        if stream_end:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, None)
