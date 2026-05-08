from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading
from transformers import AutoTokenizer
import json
import re
import asyncio
import logging
import time
import uuid
from fastapi import UploadFile, File
from backend.document_processor import DocumentProcessor
from backend.reasoning_pipeline import ReasoningPipeline
from backend.unified_orchestrator import UnifiedOrchestrator, ResponseLifecycle
from backend.rag import KnowledgeBase, DocumentIngestionPipeline, RAGRetriever
from backend.stream_manager import StreamManager
from backend.inference_manager import InferenceManager

# inference_manager is now initialized after MODEL_NAME is defined

# ── Feedback Loop System ──────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feedback import init_db, log_turn, new_conv_id, router as feedback_router
from feedback.db_schema import get_conn   # ensure DB is ready at import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot.main")

app = FastAPI(title="Custom Chatbot API - Streaming Enabled")

# ── Feedback router ────────────────────────────────────────────────────────────
app.include_router(feedback_router, prefix="/feedback", tags=["Feedback"])

# Initialise feedback DB on startup
init_db()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES & FRONTEND ---
import os
from fastapi.responses import FileResponse

# Get absolute path to the frontend directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Static files will be mounted at the end of the file to avoid route shadowing

# --- 1. MODEL LOADING & OPTIMIZATION (vLLM Managed) ---
MODE = "best" 
MODELS = {
    "fast": "gpt2",
    "best": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}
MODEL_NAME = MODELS.get(MODE, "gpt2")

print(f"Initializing vLLM with {MODEL_NAME}...")
# vLLM manages its own model loading and memory
inference_manager = InferenceManager(
    model_name=MODEL_NAME,
    gpu_memory_utilization=0.85,
    max_model_len=4096
)

# We still need a tokenizer for prompt formatting (apply_chat_template)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded for {MODEL_NAME}")

# --- 2. CONTEXT MEMORY ---
from backend.context_manager import ContextManager
memory_manager = ContextManager(
    window_size=10,        # Slightly larger window
    summarize_after=16,
    max_tokens_budget=3072, # Increased budget
    dedup_threshold=0.92,
)
doc_processor = DocumentProcessor()

# --- 3. GLOBAL RAG KNOWLEDGE BASE ---
# Persistent across restarts. Pre-load documents via:
#   cd backend && python -m rag.ingest_cli --dir ../data/knowledge/
knowledge_base = KnowledgeBase()
rag_pipeline   = DocumentIngestionPipeline()
retriever      = RAGRetriever(
    knowledge_base,
    top_k_candidates=30,     # Increased candidate pool
    top_n_results=3,        # Reduced N to focus on most relevant
    score_floor=0.30,       # Lowered floor to catch more relevant context
    use_reranker=True,
    use_mmr=True,
)

# --- Reasoning & Refinement Middleware ---
reasoning_pipeline = ReasoningPipeline(refinement_threshold=0.55)
orchestrator = UnifiedOrchestrator()

from backend.lifecycle_manager import MemoryLifecycleManager
lifecycle_manager = MemoryLifecycleManager(
    context_manager=memory_manager,
    orchestrator=orchestrator,
    cleanup_interval_seconds=300,
    session_ttl_seconds=3600
)

@app.on_event("startup")
async def startup_event():
    await lifecycle_manager.start()

@app.on_event("shutdown")
async def shutdown_event():
    await lifecycle_manager.stop()

from backend.response_middleware import ReasoningAuditMiddleware
app.add_middleware(ReasoningAuditMiddleware, pipeline=reasoning_pipeline)

class ChatRequest(BaseModel):
    message:    str
    session_id: str
    conv_id:    str = ""   # optional; generated server-side if blank
    is_continuation: bool = False

    def get_or_create_conv_id(self) -> str:
        return self.conv_id if self.conv_id else new_conv_id()

@app.get("/context/stats/{session_id}")
def context_stats(session_id: str):
    """Inspect context-manager state for a session (debug endpoint)."""
    stats = memory_manager.session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Session not found")
    stats["summary_preview"] = memory_manager.get_summary(session_id)[:300]
    stats["key_info"] = memory_manager.get_key_info(session_id)
    return stats

@app.get("/rag/stats")
async def rag_stats():
    """Inspect the global knowledge base."""
    return knowledge_base.stats()

@app.post("/rag/ingest")
async def rag_ingest(file: UploadFile = File(...)):
    """
    Ingest a document into the GLOBAL knowledge base (persisted across sessions).
    Supports PDF, TXT, MD.
    """
    import os
    contents = await file.read()
    ext = os.path.splitext(file.filename)[1].lower() or ".pdf"
    chunks = await asyncio.to_thread(
        rag_pipeline.ingest_bytes, contents, file.filename, ext
    )
    added = await asyncio.to_thread(knowledge_base.add_chunks, chunks)
    knowledge_base.register_source(file.filename)
    return {"status": "success", "filename": file.filename, "chunks_indexed": added}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(...)):
    """Upload a document to the SESSION knowledge base (ephemeral, per-user)."""
    contents = await file.read()
    num_chunks = await asyncio.to_thread(doc_processor.process_pdf, session_id, contents)
    return {"status": "success", "filename": file.filename, "chunks_indexed": num_chunks}

# --- 3. STREAMING ENDPOINT (ChatGPT Style) ---
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # ── Feedback: conversation tracking ────────────────────────────────────
    conv_id    = request.get_or_create_conv_id()
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    turn_start = time.time()
    first_token_time: list[float] = []   # mutable container for closure capture
    orchestrator.transition(request.session_id, ResponseLifecycle.INIT)

    # ── Feedback & Memory Tracking ─────────────────────────────────────────
    if not request.is_continuation:
        # Log the user turn immediately
        await asyncio.to_thread(
            log_turn,
            conv_id=conv_id,
            session_id=request.session_id,
            turn_index=0,
            role="user",
            content=request.message,
            model_name=MODEL_NAME,
        )

        # ── Memory: record user turn ──────────────────────────────────────────
        await asyncio.to_thread(memory_manager.add_message, request.session_id, "user", request.message)
        orchestrator.transition(request.session_id, ResponseLifecycle.RETRIEVAL)

        # ── Memory: retrieve context ──────────────────────────────────────────
        messages = await asyncio.to_thread(
            memory_manager.get_messages, request.session_id, current_query=request.message
        )

        # ── REASONING PIPELINE: Phase A — augment prompt ─────────────────────
        augmented_messages, reasoning_trace = await asyncio.to_thread(
            reasoning_pipeline.prepare_messages, request.message, messages
        )
        
        if getattr(reasoning_trace, 'is_coding_challenge', False) or reasoning_trace.intent_category == "coding_problem":
            # continuity_manager.set_state(...) - currently handled via status in state
            pass
    else:
        # Continuation flow: skip logging user turn, just get context
        messages = await asyncio.to_thread(
            memory_manager.get_messages, request.session_id, current_query=""
        )
        orchestrator.transition(request.session_id, ResponseLifecycle.RETRIEVAL)
        recovery_prompt = orchestrator.get_recovery_prompt(request.session_id)
        content_msg = recovery_prompt if recovery_prompt else "Please continue exactly where you left off. Output ONLY the continued text, no conversational intro."
        augmented_messages = messages + [{
            "role": "user", 
            "content": content_msg
        }]
        class FakeTrace:
            intent_category = "coding"  # Give a high token budget
            steps = []
        reasoning_trace = FakeTrace()
        orchestrator.transition(request.session_id, ResponseLifecycle.GENERATION)

    # ── RAG ROUTING: Only use RAG if intent requires factual knowledge ────
    INTENTS_REQUIRING_RAG = ["coding_problem", "debugging", "explanation", "architecture", "optimization", "document_query", "general"]
    category = reasoning_trace.intent_category
    
    if not request.is_continuation and category in INTENTS_REQUIRING_RAG:
        # ── RAG Tier 1: Global Knowledge Base ───────────────────────────────
        rag_result = await asyncio.to_thread(retriever.retrieve, request.message)
        
        # ── RAG Tier 2: Session-scoped uploaded documents ────────────────────
        doc_context = await asyncio.to_thread(
            doc_processor.query_documents, request.session_id, request.message
        )

        # Update system message with RAG context
        if augmented_messages and augmented_messages[0]["role"] == "system":
            system_content = augmented_messages[0]["content"]
            
            if rag_result.context_block:
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
        request.session_id, reasoning_trace.intent_category, len(reasoning_trace.steps)
    )
    orchestrator.transition(request.session_id, ResponseLifecycle.GENERATION)

    # ── Prompt Formatting (vLLM takes raw string or tokens) ───────────────
    if hasattr(tokenizer, "apply_chat_template") and MODE == "best":
        formatted_prompt = tokenizer.apply_chat_template(augmented_messages, tokenize=False, add_generation_prompt=True)
    else:
        full_text = ""
        for m in augmented_messages:
            full_text += f"{m['role']}: {m['content']}\n\n"
        full_text += "assistant:"
        formatted_prompt = full_text

    # ── Stream Orchestration ─────────────────────────────────────────────
    orchestrator.transition(request.session_id, ResponseLifecycle.STREAMING)
    stream_manager = StreamManager(
        conv_id=conv_id, 
        session_id=request.session_id,
        continuity_manager=orchestrator
    )
    # queue_streamer is no longer needed with vLLM

    max_new_tokens = 8192
    num_candidates = 2 if category == "coding_problem" and not request.is_continuation else 1
    
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        repetition_penalty=1.1,
        top_p=0.9,
        stop=[tokenizer.eos_token] if hasattr(tokenizer, "eos_token") else None
    )

    # ── Background Generation Tasks (vLLM Optimized) ──────────────────────
    async def generate_primary():
        """Generates the first candidate with streaming."""
        try:
            sampling_kwargs = generation_kwargs.copy()
            # vLLM doesn't use 'streamer', it uses an async generator
            
            async for token in inference_manager.generate_stream(
                prompt=formatted_prompt,
                request_id=f"{request_id}_p",
                sampling_kwargs=sampling_kwargs
            ):
                # Feed tokens to the existing stream_manager logic
                await stream_manager.put(token)
                
        except Exception as e:
            logger.error(f"Primary generation error for {request_id}: {e}")
        finally:
            await stream_manager.finalize()

    async def generate_secondary():
        """Generates additional candidates in the background."""
        if num_candidates <= 1:
            return []
        try:
            sec_kwargs = generation_kwargs.copy()
            sec_kwargs.update({
                "temperature": 0.85,
                "n": num_candidates - 1
            })
            
            outputs = await inference_manager.generate_full(
                prompt=formatted_prompt,
                request_id=f"{request_id}_s",
                sampling_kwargs=sec_kwargs
            )
            return outputs
        except Exception as e:
            logger.error(f"Secondary generation error for {request_id}: {e}")
            return []

    # Start primary generation immediately
    primary_task = asyncio.create_task(generate_primary())
    # Start secondary generation if needed
    secondary_task = asyncio.create_task(generate_secondary())

    # ── SSE streaming generator ───────────────────────────────────────────
    async def event_generator():
        draft_text = ""
        
        # Monitor client disconnection
        async def check_disconnect():
            while not primary_task.done():
                if await http_request.is_disconnected():
                    logger.warning(f"Client disconnected for {request_id} — stopping generation.")
                    # vLLM handles cancellation when the generator is dropped/cancelled
                    primary_task.cancel()
                    secondary_task.cancel()
                    stream_manager.stop()
                    
                    orchestrator.set_interrupted(request.session_id)
                    break
                await asyncio.sleep(0.5)

        disconnect_monitor = asyncio.create_task(check_disconnect())

        # Consume the stream
        async for event in stream_manager.event_generator():
            # Extract content from event for local tracking
            if "content" in event:
                try:
                    data = json.loads(event.replace("data: ", ""))
                    if "content" in data:
                        draft_text += data["content"]
                except:
                    pass
            
            # Record time-to-first-token
            if not first_token_time and stream_manager.tokens_yielded > 0:
                first_token_time.append(time.time() - turn_start)
                
            yield event

        # Wait for both tasks to complete
        await primary_task
        secondary_candidates = await secondary_task
        disconnect_monitor.cancel()

        all_candidates = [draft_text.strip()] + secondary_candidates
        
        # ── REASONING PIPELINE: Phase B — validate & refine ──────────────
        final_response, reasoning_trace_updated = await asyncio.to_thread(
            reasoning_pipeline.refine, 
            request.message, 
            draft_text.strip(), 
            reasoning_trace,
            candidates=all_candidates if len(all_candidates) > 1 else None,
            context=rag_result.context_block if 'rag_result' in locals() else ""
        )

        if reasoning_trace_updated.refinement_applied:
            logger.info("[ReasoningPipeline] Refinement applied.")
            if final_response.strip() != draft_text.strip():
                yield f"data: {json.dumps({'content': '', 'refined': True, 'full': final_response, 'conv_id': conv_id})}\n\n"

        # Persist response and log
        await asyncio.to_thread(memory_manager.add_message, request.session_id, "assistant", final_response)
        
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

        # --- AUTOMATIC CONTINUATION LOOP ---
        MAX_AUTORETRY = 1
        retry_count = 0
        
        # Initial Validation
        validation = orchestrator.validate_and_finalize(request.session_id)
        if not validation["is_valid"] and validation["repair_suffix"]:
            yield f"data: {json.dumps({'content': validation['repair_suffix'], 'repaired': True, 'conv_id': conv_id})}\n\n"
        
        while orchestrator.get_current_state(request.session_id) == ResponseLifecycle.RECOVERY and retry_count < MAX_AUTORETRY:
            retry_count += 1
            logger.info(f"[AutoContinuation] Triggering automatic continuation (Retry {retry_count}/{MAX_AUTORETRY})")
            
            # Prepare recovery context
            recovery_prompt = orchestrator.get_recovery_prompt(request.session_id)
            # Use current messages + recovery prompt
            continuation_messages = augmented_messages + [{"role": "user", "content": recovery_prompt}]
            
            # Re-tokenize
            if hasattr(tokenizer, "apply_chat_template") and MODE == "best":
                formatted_prompt_cont = tokenizer.apply_chat_template(continuation_messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt_cont = formatted_prompt + "\n\n" + recovery_prompt + "\nassistant:"
            
            # Reset stream manager for new pass
            stream_manager.reset() # Need to implement reset or create new one
            # queue_streamer is no longer needed
            
            # Define new generation task
            async def generate_continuation():
                try:
                    cont_kwargs = generation_kwargs.copy()
                    
                    async for token in inference_manager.generate_stream(
                        prompt=formatted_prompt_cont,
                        request_id=f"{request_id}_cont_{retry_count}",
                        sampling_kwargs=cont_kwargs
                    ):
                        await stream_manager.put(token)
                finally:
                    await stream_manager.finalize()
            
            cont_task = asyncio.create_task(generate_continuation())
            
            # Yield from continuation stream
            async for event in stream_manager.event_generator():
                if "content" in event:
                    try:
                        data = json.loads(event.replace("data: ", ""))
                        if "content" in data:
                            draft_text += data["content"]
                    except: pass
                yield event
            
            await cont_task
            
            # Re-validate
            validation = orchestrator.validate_and_finalize(request.session_id)
            if not validation["is_valid"] and validation["repair_suffix"]:
                yield f"data: {json.dumps({'content': validation['repair_suffix'], 'repaired': True, 'conv_id': conv_id})}\n\n"

        # Final check
        if orchestrator.get_current_state(request.session_id) == ResponseLifecycle.FINALIZED:
            orchestrator.reset_session(request.session_id)
            yield f"data: {json.dumps({'done': True, 'conv_id': conv_id})}\n\n"
        else:
            logger.warning(f"Session {request.session_id} still not finalized after {retry_count} retries.")
            yield f"data: {json.dumps({'done': True, 'interrupted': True, 'conv_id': conv_id})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Mount the static files at the root. We do this at the end so it doesn't shadow API routes.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
