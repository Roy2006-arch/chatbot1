import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, AsyncGenerator
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

logger = logging.getLogger("chatbot.inference_manager")

class InferenceManager:
    """
    Manages concurrent inference requests using vLLM AsyncLLMEngine.
    Provides high throughput via continuous batching and PagedAttention.
    """
    def __init__(
        self, 
        model_name: str,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        trust_remote_code: bool = True
    ):
        engine_args = AsyncEngineArgs(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            # Continuous batching is enabled by default in vLLM
            enforce_eager=True, # Often faster for smaller models like TinyLlama
            disable_log_requests=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"[vLLM] Engine initialized for model: {model_name}")

    async def generate_stream(
        self, 
        prompt: str,
        request_id: str,
        sampling_kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Generates a stream of tokens using vLLM.
        Native cancellation is supported: if the generator is closed, 
        vLLM stops processing the request.
        """
        # Prepare sampling parameters
        sampling_params = SamplingParams(
            n=sampling_kwargs.get("n", 1),
            temperature=sampling_kwargs.get("temperature", 0.7),
            top_p=sampling_kwargs.get("top_p", 0.9),
            max_tokens=sampling_kwargs.get("max_new_tokens", 512),
            repetition_penalty=sampling_kwargs.get("repetition_penalty", 1.1),
            stop=sampling_kwargs.get("stop", None),
        )

        logger.info(f"[vLLM] Starting stream for {request_id}")
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        last_text = ""
        async for request_output in results_generator:
            # request_output contains the full text generated so far
            current_text = request_output.outputs[0].text
            new_text = current_text[len(last_text):]
            last_text = current_text
            
            if new_text:
                yield new_text

    async def generate_full(
        self,
        prompt: str,
        request_id: str,
        sampling_kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Generates full responses without streaming.
        Useful for secondary candidate generation or validation.
        """
        sampling_params = SamplingParams(
            n=sampling_kwargs.get("n", 1),
            temperature=sampling_kwargs.get("temperature", 0.7),
            top_p=sampling_kwargs.get("top_p", 0.9),
            max_tokens=sampling_kwargs.get("max_new_tokens", 512),
            repetition_penalty=sampling_kwargs.get("repetition_penalty", 1.1),
        )

        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output:
            return [out.text for out in final_output.outputs]
        return []

    def cleanup_gpu(self):
        """
        Note: vLLM manages GPU memory automatically. 
        Manual cleanup is usually not required unless shutting down.
        """
        pass
