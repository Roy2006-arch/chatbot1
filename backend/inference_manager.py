import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

import torch

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

logger = logging.getLogger("chatbot.inference_manager")


class InferenceManager:
    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.use_vllm = HAS_VLLM and torch.cuda.is_available()
        self.engine = None
        self.fallback_pipeline = None

        if self.use_vllm:
            try:
                engine_args = AsyncEngineArgs(
                    model=model_name,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    trust_remote_code=trust_remote_code,
                    enforce_eager=True,
                    disable_log_requests=True,
                )
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                logger.info("[InferenceManager] vLLM Engine initialized for %s", model_name)
            except Exception as e:
                logger.error("[InferenceManager] vLLM init failed, falling back: %s", e)
                self.use_vllm = False

        if not self.use_vllm:
            logger.info("[InferenceManager] Using Transformers fallback for %s", model_name)
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.fallback_pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=device,
                trust_remote_code=trust_remote_code,
            )

    async def generate_stream(
        self,
        prompt: str,
        request_id: str,
        sampling_kwargs: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        if self.use_vllm:
            sampling_params = SamplingParams(
                n=sampling_kwargs.get("n", 1),
                temperature=sampling_kwargs.get("temperature", 0.7),
                top_p=sampling_kwargs.get("top_p", 0.9),
                max_tokens=sampling_kwargs.get("max_new_tokens", 1024),
                repetition_penalty=sampling_kwargs.get("repetition_penalty", 1.1),
                stop=sampling_kwargs.get("stop", None),
            )

            results_generator = self.engine.generate(prompt, sampling_params, request_id)
            last_text = ""
            async for request_output in results_generator:
                current_text = request_output.outputs[0].text
                new_text = current_text[len(last_text):]
                last_text = current_text
                if new_text:
                    yield new_text
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.fallback_pipeline(
                    prompt,
                    max_new_tokens=sampling_kwargs.get("max_new_tokens", 1024),
                    temperature=sampling_kwargs.get("temperature", 0.7),
                    top_p=sampling_kwargs.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=50256,
                ),
            )
            full_text = result[0]["generated_text"]
            if full_text.startswith(prompt):
                yield full_text[len(prompt):]
            else:
                yield full_text

    async def generate_full(
        self,
        prompt: str,
        request_id: str,
        sampling_kwargs: Dict[str, Any],
    ) -> List[str]:
        if self.use_vllm:
            sampling_params = SamplingParams(
                n=sampling_kwargs.get("n", 1),
                temperature=sampling_kwargs.get("temperature", 0.7),
                top_p=sampling_kwargs.get("top_p", 0.9),
                max_tokens=sampling_kwargs.get("max_new_tokens", 1024),
                repetition_penalty=sampling_kwargs.get("repetition_penalty", 1.1),
            )
            results_generator = self.engine.generate(prompt, sampling_params, request_id)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            if final_output:
                return [out.text for out in final_output.outputs]
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.fallback_pipeline(
                    prompt,
                    max_new_tokens=sampling_kwargs.get("max_new_tokens", 1024),
                    num_return_sequences=sampling_kwargs.get("n", 1),
                    do_sample=True,
                ),
            )
            return [r["generated_text"][len(prompt):] for r in result]
        return []

    def cleanup_gpu(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("[InferenceManager] CUDA cache cleared.")
