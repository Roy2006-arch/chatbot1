import asyncio
import gc
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.document_processor import DocumentProcessor
from backend.reasoning_pipeline import ReasoningPipeline
from backend.unified_orchestrator import UnifiedOrchestrator, ResponseLifecycle
from backend.rag import KnowledgeBase, DocumentIngestionPipeline, RAGRetriever
from backend.stream_manager import StreamManager
from backend.inference_manager import InferenceManager
from backend.context_manager import ContextManager
from backend.lifecycle_manager import MemoryLifecycleManager
from backend.response_middleware import ReasoningAuditMiddleware
from backend.refinement_middleware import ResponseRefinementASGIMiddleware
from backend.shared_resources import set_request_cache, get_request_cache, RequestEmbeddingCache
from backend.realtime_utils import realtime_handler
from backend.url_verifier import URLVerifier
from feedback import init_db, log_turn, new_conv_id, router as feedback_router
from feedback.mistake_memory import MistakeMemory
from feedback.auto_improve import start_auto_improve_scheduler, stop_auto_improve_scheduler
from multimodal.chat_integration import router as multimodal_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot.main")

_CPU_POOL: ProcessPoolExecutor | None = None

def get_cpu_pool() -> ProcessPoolExecutor:
    global _CPU_POOL
    if _CPU_POOL is None:
        _CPU_POOL = ProcessPoolExecutor(max_workers=max(4, os.cpu_count() or 4))
    return _CPU_POOL

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    init_db()
    await lifecycle_manager.start()

    # Start auto-improve scheduler (checks feedback.db hourly)
    _auto_scheduler = start_auto_improve_scheduler(
        interval_seconds=3600,
        min_new_entries=10,
        min_composite_gain=0.05,
    )
    if _auto_scheduler:
        logger.info("[Main] Auto-improve scheduler started.")

    yield

    logger.info("Shutting down...")
    stop_auto_improve_scheduler()
    await lifecycle_manager.stop()
    await url_verifier.close()
    global _CPU_POOL
    if _CPU_POOL:
        _CPU_POOL.shutdown(wait=False)
        _CPU_POOL = None

app = FastAPI(title="Custom Chatbot API - Streaming Enabled", lifespan=lifespan)

app.include_router(feedback_router, prefix="/feedback", tags=["Feedback"])
app.include_router(multimodal_router, prefix="/api/chat", tags=["Multimodal Chat"])

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' http://127.0.0.1:8000 https://kaustav2006-chatbot-api.hf.space; "
        "img-src 'self' data:;"
    )
    return response

ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://kaustav2006-chatbot-api.hf.space",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

MODE = "best"
MODELS = {
    "fast": "gpt2",
    "best": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}
MODEL_NAME = MODELS.get(MODE, "gpt2")

logger.info("Initializing vLLM with %s...", MODEL_NAME)
inference_manager = InferenceManager(
    model_name=MODEL_NAME,
    tokenizer=tokenizer,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
logger.info("Tokenizer loaded for %s", MODEL_NAME)

memory_manager = ContextManager(
    window_size=10,
    summarize_after=16,
    max_tokens_budget=3072,
    dedup_threshold=0.92,
)
doc_processor = DocumentProcessor()

knowledge_base = KnowledgeBase()
rag_pipeline = DocumentIngestionPipeline()
retriever = RAGRetriever(
    knowledge_base,
    top_k_candidates=30,
    top_n_results=3,
    score_floor=0.30,
    use_reranker=True,
    use_mmr=True,
)

reasoning_pipeline = ReasoningPipeline(refinement_threshold=0.55)
url_verifier = URLVerifier()
orchestrator = UnifiedOrchestrator()

lifecycle_manager = MemoryLifecycleManager(
    context_manager=memory_manager,
    orchestrator=orchestrator,
    doc_processor=doc_processor,
    cleanup_interval_seconds=300,
    session_ttl_seconds=3600,
)

app.add_middleware(ReasoningAuditMiddleware, pipeline=reasoning_pipeline)
app.add_middleware(ResponseRefinementASGIMiddleware)

INTENTS_REQUIRING_RAG = [
    "coding_problem", "debugging", "explanation",
    "architecture", "optimization", "document_query", "general",
]
MAX_STREAM_SECONDS = 60
MAX_AUTORETRY = 3
MAX_CONTINUATION_NEW_TOKENS = 1024

GREETING_RESPONSES = OrderedDict([
    ("hello", "Hey!"),
    ("hi", "Hey!"),
    ("hey", "Hey!"),
    ("good morning", "Morning!"),
    ("good evening", "Evening!"),
    ("how are you", "Doing great! What's up?"),
    ("how is the day", "All good here! What can I do for you?"),
    ("how's the day", "All good here! What can I do for you?"),
    ("how's your day", "Doing great! What's up?"),
    ("how was your day", "Doing great! What's up?"),
    ("how is your day going", "Doing great! What's up?"),
    ("what's your name", "I'm a chatbot."),
    ("who are you", "I'm a chatbot."),
    ("who made you", "I was built to help with questions and coding."),
    ("thanks", "No problem!"),
    ("thank you", "No problem!"),
])

class SimpleResponseCache:
    def __init__(self, maxsize=512, ttl=600):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, message: str, session_id: str):
        key = self._make_key(message, session_id)
        entry = self._cache.get(key)
        if entry and (time.time() - entry["time"] < self._ttl):
            self._cache.move_to_end(key)
            return entry["response"]
        if entry:
            del self._cache[key]
        return None

    def set(self, message: str, session_id: str, response: str):
        key = self._make_key(message, session_id)
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = {"response": response, "time": time.time()}

    def _make_key(self, message: str, session_id: str) -> str:
        normalized = message.lower().strip()
        return f"{session_id}:{hashlib.md5(normalized.encode()).hexdigest()}"

response_cache = SimpleResponseCache()

def _get_greeting_response(message: str):
    normalized = message.lower().strip().rstrip("?!.,;:")
    # Exact match first (handles "hi", "hello!", etc.)
    for greeting, response in GREETING_RESPONSES.items():
        if normalized == greeting:
            return response
    # Extract first alphabetic word (handles "hi there", "hello world", "hi, how are you")
    match = re.match(r'[^a-z]*([a-z]+)', normalized)
    first_word = match.group(1) if match else ""
    single_word = {k: v for k, v in GREETING_RESPONSES.items() if " " not in k}
    if first_word in single_word:
        return single_word[first_word]
    # Multi-word greetings (check longest first to avoid partial matches)
    multi_word = {k: v for k, v in GREETING_RESPONSES.items() if " " in k}
    for greeting, response in sorted(multi_word.items(), key=lambda x: -len(x[0])):
        if normalized.startswith(greeting):
            return response
    return None

def _is_simple_query(category: str, message: str) -> bool:
    if category in ("casual_chat", "general"):
        return True
    if category in ("explanation", "document_query") and len(message.split()) <= 10:
        return True
    return False

def _get_token_budget(category: str, message: str) -> int:
    budgets = {
        "casual_chat": 100,
        "general": 200,
        "document_query": 512,
        "explanation": 512,
        "architecture": 1024,
        "optimization": 1024,
        "debugging": 1024,
        "coding_problem": 2048,
    }
    return budgets.get(category, 1024)


class ChatRequest(BaseModel):
    message: str
    session_id: str
    conv_id: str = ""
    is_continuation: bool = False
    timezone: str = ""

    @validator("session_id")
    def validate_session_id(cls, v):
        if not v or len(v) > 64:
            raise ValueError("session_id must be 1-64 characters")
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("session_id must be alphanumeric, dash, or underscore only")
        return v

    @validator("message")
    def validate_message(cls, v):
        if len(v) > 32000:
            raise ValueError("message exceeds 32000 character limit")
        return v

    def get_or_create_conv_id(self) -> str:
        return self.conv_id if self.conv_id else new_conv_id()


def _has_chat_template(tokenizer) -> bool:
    return hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None


def _format_prompt(tokenizer, messages, mode: str) -> str:
    if _has_chat_template(tokenizer) and mode == "best":
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = []
    for m in messages:
        parts.append(f"{m['role']}: {m['content']}\n\n")
    parts.append("assistant:")
    return "".join(parts)


async def _run_cpu_bound(fn, *args, **kwargs):
    from functools import partial
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(get_cpu_pool(), partial(fn, *args, **kwargs))
    return await loop.run_in_executor(get_cpu_pool(), fn, *args)


@app.get("/context/stats/{session_id}")
def context_stats(session_id: str):
    stats = memory_manager.session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Session not found")
    stats["summary_preview"] = memory_manager.get_summary(session_id)[:300]
    stats["key_info"] = memory_manager.get_key_info(session_id)
    return stats


@app.get("/rag/stats")
async def rag_stats():
    return knowledge_base.stats()


@app.post("/realtime/timezone")
async def set_user_timezone(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "")
    tz_name = body.get("timezone", "UTC")
    if session_id:
        realtime_handler.set_user_timezone(session_id, tz_name)
    return {"status": "ok", "timezone": tz_name}


@app.post("/rag/ingest")
async def rag_ingest(file: UploadFile = File(...)):
    contents = await file.read()
    ext = os.path.splitext(file.filename)[1].lower() or ".pdf"
    chunks = await asyncio.to_thread(rag_pipeline.ingest_bytes, contents, file.filename, ext)
    added = await asyncio.to_thread(knowledge_base.add_chunks, chunks)
    knowledge_base.register_source(file.filename)
    return {"status": "success", "filename": file.filename, "chunks_indexed": added}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(...)):
    contents = await file.read()
    num_chunks = await asyncio.to_thread(doc_processor.process_pdf, session_id, contents)
    return {"status": "success", "filename": file.filename, "chunks_indexed": num_chunks}


@app.post("/admin/corrections/reload")
async def reload_correction_patterns():
    """
    Hot-reload correction patterns from the self-improvement config.
    Useful when correction_generator config.yaml is updated at runtime.
    """
    try:
        from self_improvement.correction_generator import CorrectionGenerator
        from self_improvement.pipeline import SelfImprovementPipeline
        pipeline = SelfImprovementPipeline()
        pipeline.correction_gen = CorrectionGenerator(
            pipeline.config.get("correction_generator", {})
        )
        return {
            "status": "ok",
            "message": "Correction patterns reloaded from config",
            "patterns_available": pipeline.correction_gen.enabled,
        }
    except Exception as e:
        logger.error("[Admin] Correction reload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/status")
async def admin_status():
    """Return system status including scheduler health and dataset bridge info."""
    status = {
        "model": MODEL_NAME,
        "mode": MODE,
        "scheduler_active": True,
        "mistake_memory_available": True,
        "multimodal_pipeline_available": True,
    }
    try:
        from training.dataset_bridge import build_dataset
        ds = build_dataset(max_examples=100)
        status["dataset_bridge"] = ds.to_dict()
    except Exception as e:
        status["dataset_bridge_error"] = str(e)
    return status


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conv_id = request.get_or_create_conv_id()
    turn_start = time.time()

    # Register timezone from request if provided (supports per-request timezone override)
    if request.timezone:
        realtime_handler.set_user_timezone(request.session_id, request.timezone)

    # Fast-path realtime check - highest priority, no LLM needed
    if not request.is_continuation:
        realtime_response = realtime_handler.handle(request.message, request.session_id)
        if realtime_response:
            logger.info("[FastPath] Realtime response for session=%s", request.session_id)
            await asyncio.to_thread(
                log_turn, conv_id=conv_id, session_id=request.session_id, turn_index=0,
                role="user", content=request.message, model_name=MODEL_NAME,
            )
            await asyncio.to_thread(
                memory_manager.add_message, request.session_id, "user", request.message
            )
            await asyncio.to_thread(
                memory_manager.add_message, request.session_id, "assistant", realtime_response
            )
            await asyncio.to_thread(
                log_turn, conv_id=conv_id, session_id=request.session_id, turn_index=1,
                role="assistant", content=realtime_response, model_name=MODEL_NAME,
                prompt=request.message, ttft_seconds=0.0,
                total_time_seconds=time.time() - turn_start,
            )
            orchestrator.reset_session(request.session_id)

            _cleanup_request_cache()
            async def realtime_stream():
                yield f"data: {json.dumps({'content': realtime_response, 'conv_id': conv_id})}\n\n"
                yield f"data: {json.dumps({'done': True, 'conv_id': conv_id})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(realtime_stream(), media_type="text/event-stream")

    # Fast-path greeting check - run before any expensive processing
    if not request.is_continuation:
        greeting_response = _get_greeting_response(request.message)
        if greeting_response:
            logger.info("[FastPath] Greeting for session=%s", request.session_id)
            await asyncio.to_thread(
                log_turn, conv_id=conv_id, session_id=request.session_id, turn_index=0,
                role="user", content=request.message, model_name=MODEL_NAME,
            )
            await asyncio.to_thread(
                memory_manager.add_message, request.session_id, "user", request.message
            )
            await asyncio.to_thread(
                memory_manager.add_message, request.session_id, "assistant", greeting_response
            )
            await asyncio.to_thread(
                log_turn, conv_id=conv_id, session_id=request.session_id, turn_index=1,
                role="assistant", content=greeting_response, model_name=MODEL_NAME,
                prompt=request.message, ttft_seconds=0.0,
                total_time_seconds=time.time() - turn_start,
            )
            orchestrator.reset_session(request.session_id)

            _cleanup_request_cache()
            async def greeting_stream():
                yield f"data: {json.dumps({'content': greeting_response, 'conv_id': conv_id})}\n\n"
                yield f"data: {json.dumps({'done': True, 'conv_id': conv_id})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(greeting_stream(), media_type="text/event-stream")

    # Early cache check (before expensive processing)
    if not request.is_continuation:
        cached_response = response_cache.get(request.message, request.session_id)
        if cached_response:
            logger.info("[FastPath] Cache hit for session=%s", request.session_id)
            await asyncio.to_thread(
                memory_manager.add_message, request.session_id, "user", request.message
            )
            await asyncio.to_thread(
                memory_manager.add_message, request.session_id, "assistant", cached_response
            )
            await asyncio.to_thread(
                log_turn, conv_id=conv_id, session_id=request.session_id, turn_index=0,
                role="user", content=request.message, model_name=MODEL_NAME,
            )
            await asyncio.to_thread(
                log_turn, conv_id=conv_id, session_id=request.session_id, turn_index=1,
                role="assistant", content=cached_response, model_name=MODEL_NAME,
                prompt=request.message, ttft_seconds=0.0,
                total_time_seconds=time.time() - turn_start,
            )
            orchestrator.reset_session(request.session_id)

            _cleanup_request_cache()
            async def cached_stream():
                yield f"data: {json.dumps({'content': cached_response, 'conv_id': conv_id})}\n\n"
                yield f"data: {json.dumps({'done': True, 'conv_id': conv_id})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cached_stream(), media_type="text/event-stream")

    request_id = f"req_{uuid.uuid4().hex[:8]}"
    request_cache = RequestEmbeddingCache()
    set_request_cache(request_cache)
    first_token_time: list[float] = []
    orchestrator.transition(request.session_id, ResponseLifecycle.INIT)

    def _cleanup_request_cache():
        cache = get_request_cache()
        if cache:
            cache.clear()
        set_request_cache(None)

    if not request.is_continuation:
        await asyncio.to_thread(
            log_turn,
            conv_id=conv_id,
            session_id=request.session_id,
            turn_index=0,
            role="user",
            content=request.message,
            model_name=MODEL_NAME,
        )
        await asyncio.to_thread(
            memory_manager.add_message, request.session_id, "user", request.message
        )
        orchestrator.transition(request.session_id, ResponseLifecycle.RETRIEVAL)

        messages = await asyncio.to_thread(
            memory_manager.get_messages, request.session_id, current_query=request.message
        )

        augmented_messages, reasoning_trace = await _run_cpu_bound(
            reasoning_pipeline.prepare_messages, request.message, messages
        )
    else:
        messages = await asyncio.to_thread(
            memory_manager.get_messages, request.session_id, current_query=""
        )
        orchestrator.transition(request.session_id, ResponseLifecycle.RETRIEVAL)
        recovery_prompt = orchestrator.get_recovery_prompt(request.session_id)
        content_msg = (
            recovery_prompt
            if recovery_prompt
            else "Please continue exactly where you left off. Output ONLY the continued text, no conversational intro."
        )
        augmented_messages = messages + [{"role": "user", "content": content_msg}]

        class FakeTrace:
            intent_category = "coding"
            steps = []
        reasoning_trace = FakeTrace()
        orchestrator.transition(request.session_id, ResponseLifecycle.GENERATION)

    category = getattr(reasoning_trace, "intent_category", "general")
    is_simple = _is_simple_query(category, request.message)
    rag_result = None

    # Inject realtime context into ALL system messages (prevents hallucination of time/date)
    if augmented_messages and augmented_messages[0]["role"] == "system":
        system_content = augmented_messages[0]["content"]
        realtime_block = realtime_handler.get_realtime_context_block(request.session_id)
        system_content = system_content.replace(
            "[INTERNAL PLANNING]",
            f"[REALTIME DATA]\n{realtime_block}\n\n[INTERNAL PLANNING]",
        )
        augmented_messages[0]["content"] = system_content

    # MistakeMemory corrections (single call, shared with RAG context)
    corrections_block = None
    if not request.is_continuation and not is_simple and augmented_messages and augmented_messages[0]["role"] == "system":
        try:
            mm = MistakeMemory()
            corrections_block = await asyncio.to_thread(mm.format_corrections_for_prompt, request.message)
        except Exception as e:
            logger.warning("[MistakeMemory] Injection error: %s", e)

    if not request.is_continuation and not is_simple and category in INTENTS_REQUIRING_RAG:
        rag_task = asyncio.to_thread(retriever.retrieve, request.message)
        doc_task = asyncio.to_thread(
            doc_processor.query_documents, request.session_id, request.message
        )
        rag_result, doc_context = await asyncio.gather(rag_task, doc_task)

        if augmented_messages and augmented_messages[0]["role"] == "system":
            system_content = augmented_messages[0]["content"]
            if corrections_block:
                system_content += f"\n\n{corrections_block}"
            if rag_result and rag_result.context_block:
                system_content += f"\n\n[USE THE FOLLOWING VERIFIED KNOWLEDGE]\n{rag_result.context_block}"
                logger.info("[RAG] Knowledge injected for category: %s", category)
            if doc_context:
                system_content += f"\n\n[USER-UPLOADED DOCUMENT CONTEXT]\n{doc_context}"
            augmented_messages[0]["content"] = system_content
    else:
        if not request.is_continuation:
            logger.info("[RAG] Bypassed RAG for category: %s", category)

    logger.info(
        "[ReasoningPipeline] session=%s intent=%s steps=%d",
        request.session_id, getattr(reasoning_trace, "intent_category", "unknown"),
        len(getattr(reasoning_trace, "steps", [])),
    )
    orchestrator.transition(request.session_id, ResponseLifecycle.GENERATION)

    formatted_prompt = _format_prompt(tokenizer, augmented_messages, MODE)

    orchestrator.transition(request.session_id, ResponseLifecycle.STREAMING)
    stream_manager = StreamManager(
        conv_id=conv_id,
        session_id=request.session_id,
        continuity_manager=orchestrator,
    )

    if request.is_continuation:
        max_new_tokens = MAX_CONTINUATION_NEW_TOKENS
    elif is_simple:
        max_new_tokens = _get_token_budget(category, request.message)
    else:
        max_new_tokens = 2048
    num_candidates = 1

    stop_tokens = [tokenizer.eos_token] if tokenizer.eos_token is not None else []
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        repetition_penalty=1.1,
        top_p=0.9,
        stop=stop_tokens,
    )

    async def generate_primary():
        try:
            sampling_kwargs = generation_kwargs.copy()
            async for token in inference_manager.generate_stream(
                prompt=formatted_prompt,
                request_id=f"{request_id}_p",
                sampling_kwargs=sampling_kwargs,
            ):
                await stream_manager.put(token)
        except asyncio.CancelledError:
            logger.info("Primary generation cancelled for %s", request_id)
        except Exception as e:
            logger.error("Primary generation error for %s: %s", request_id, e)
        finally:
            await stream_manager.finalize()

    async def generate_secondary():
        if num_candidates <= 1:
            return []
        try:
            sec_kwargs = generation_kwargs.copy()
            sec_kwargs.update({"temperature": 0.85, "n": num_candidates - 1})
            outputs = await inference_manager.generate_full(
                prompt=formatted_prompt,
                request_id=f"{request_id}_s",
                sampling_kwargs=sec_kwargs,
            )
            return outputs
        except asyncio.CancelledError:
            return []
        except Exception as e:
            logger.error("Secondary generation error for %s: %s", request_id, e)
            return []

    primary_task = asyncio.create_task(generate_primary())
    secondary_task = None
    if num_candidates > 1:
        secondary_task = asyncio.create_task(generate_secondary())

    async def cleanup_session_resources():
        orchestrator.reset_session(request.session_id)
        await asyncio.to_thread(memory_manager.add_message, request.session_id, "assistant", "[Response interrupted]")

    async def event_generator():
        nonlocal rag_result
        draft_text = ""
        stream_finished = False

        async def check_disconnect():
            nonlocal stream_finished
            while not stream_finished:
                try:
                    if await http_request.is_disconnected():
                        logger.warning("Client disconnected for %s — stopping.", request_id)
                        primary_task.cancel()
                        if secondary_task:
                            secondary_task.cancel()
                        stream_manager.stop()
                        orchestrator.set_interrupted(request.session_id)
                        return
                except Exception:
                    return
                await asyncio.sleep(0.5)

        disconnect_monitor = asyncio.create_task(check_disconnect())

        try:
            async def stream_with_timeout():
                async for event in stream_manager.event_generator():
                    if "content" in event:
                        try:
                            data = json.loads(event.replace("data: ", "", 1))
                            content = data.get("content", "")
                            if content:
                                draft_text += content
                        except (json.JSONDecodeError, IndexError):
                            pass
                    if not first_token_time and stream_manager.tokens_yielded > 0:
                        first_token_time.append(time.time() - turn_start)
                    yield event

            async for event in stream_with_timeout():
                yield event
        except asyncio.TimeoutError:
            logger.error("Stream timed out for %s", request_id)
            yield f"data: {json.dumps({'error': 'stream_timeout', 'conv_id': conv_id})}\n\n"
        finally:
            stream_finished = True
            disconnect_monitor.cancel()
            primary_task.cancel()
            if secondary_task:
                secondary_task.cancel()
            try:
                await primary_task
            except asyncio.CancelledError:
                pass
            _cleanup_request_cache()

        secondary_candidates = []
        if secondary_task:
            secondary_candidates = await secondary_task

        all_candidates = [draft_text.strip()] + (secondary_candidates or [])
        rag_ctx = rag_result.context_block if rag_result and rag_result.context_block else ""

        needs_refine = draft_text.strip() and not is_simple and category in ("coding_problem", "debugging", "architecture", "optimization")
        if needs_refine:
            final_response, reasoning_trace_updated = await _run_cpu_bound(
                reasoning_pipeline.refine,
                request.message,
                draft_text.strip(),
                reasoning_trace,
                candidates=all_candidates if len(all_candidates) > 1 else None,
                context=rag_ctx,
            )
        else:
            final_response = draft_text.strip()
            reasoning_trace_updated = reasoning_trace

        if not is_simple and url_verifier.has_any_urls(final_response):
            url_report = await url_verifier.verify_response(final_response)
            if not url_report.all_verified or url_report.has_fake_urls:
                logger.warning(
                    "[URLVerifier] %d unverifiable URLs in response for session=%s",
                    len([u for u in url_report.urls if u.confidence < url_verifier.confidence_threshold]),
                    request.session_id,
                )

        if getattr(reasoning_trace_updated, "refinement_applied", False):
            if final_response.strip() != draft_text.strip():
                yield f"data: {json.dumps({'content': '', 'refined': True, 'full': final_response, 'conv_id': conv_id})}\n\n"

        if final_response.strip():
            response_cache.set(request.message, request.session_id, final_response)

        await asyncio.to_thread(
            memory_manager.add_message, request.session_id, "assistant", final_response
        )

        total_elapsed = time.time() - turn_start
        ttft = first_token_time[0] if first_token_time else 0.0
        await asyncio.to_thread(
            log_turn,
            conv_id=conv_id,
            session_id=request.session_id,
            turn_index=1,
            role="assistant",
            content=final_response,
            model_name=MODEL_NAME,
            prompt=request.message,
            ttft_seconds=ttft,
            total_time_seconds=total_elapsed,
        )

        retry_count = 0
        if is_simple:
            orchestrator.transition(request.session_id, ResponseLifecycle.FINALIZED)
        else:
            validation = orchestrator.validate_and_finalize(request.session_id)
            if not validation["is_valid"] and validation["repair_suffix"]:
                yield f"data: {json.dumps({'content': validation['repair_suffix'], 'repaired': True, 'conv_id': conv_id})}\n\n"

            while (
                orchestrator.get_current_state(request.session_id) == ResponseLifecycle.RECOVERY
                and retry_count < MAX_AUTORETRY
            ):
                retry_count += 1
                logger.info(
                    "[AutoContinuation] Retry %d/%d for %s",
                    retry_count, MAX_AUTORETRY, request_id,
                )

                recovery_prompt = orchestrator.get_recovery_prompt(request.session_id)
                continuation_messages = augmented_messages + [
                    {"role": "user", "content": recovery_prompt}
                ]
                formatted_prompt_cont = _format_prompt(tokenizer, continuation_messages, MODE)

                stream_manager.reset()
                cont_kwargs = generation_kwargs.copy()
                cont_kwargs["max_new_tokens"] = MAX_CONTINUATION_NEW_TOKENS

                async def generate_continuation():
                    try:
                        async for token in inference_manager.generate_stream(
                            prompt=formatted_prompt_cont,
                            request_id=f"{request_id}_cont_{retry_count}",
                            sampling_kwargs=cont_kwargs,
                        ):
                            await stream_manager.put(token)
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error("Continuation error: %s", e)
                    finally:
                        await stream_manager.finalize()

                cont_task = asyncio.create_task(generate_continuation())
                try:
                    async for event in stream_manager.event_generator():
                        if "content" in event:
                            try:
                                data = json.loads(event.replace("data: ", "", 1))
                                content = data.get("content", "")
                                if content:
                                    draft_text += content
                            except (json.JSONDecodeError, IndexError):
                                pass
                        yield event
                except asyncio.TimeoutError:
                    logger.error("Continuation stream timed out for %s", request_id)
                await cont_task

                validation = orchestrator.validate_and_finalize(request.session_id)
                if not validation["is_valid"] and validation["repair_suffix"]:
                    yield f"data: {json.dumps({'content': validation['repair_suffix'], 'repaired': True, 'conv_id': conv_id})}\n\n"

        if orchestrator.get_current_state(request.session_id) == ResponseLifecycle.FINALIZED:
            orchestrator.reset_session(request.session_id)
            yield f"data: {json.dumps({'done': True, 'conv_id': conv_id})}\n\n"
        else:
            logger.warning("Session %s not finalized after %d retries.", request.session_id, retry_count)
            yield f"data: {json.dumps({'done': True, 'interrupted': True, 'conv_id': conv_id})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
