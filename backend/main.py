from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import json
import asyncio
import logging
import time
import uuid
from fastapi import UploadFile, File
from document_processor import DocumentProcessor
from reasoning_pipeline import ReasoningPipeline
from rag import KnowledgeBase, DocumentIngestionPipeline, RAGRetriever

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

# --- 1. MODEL LOADING & OPTIMIZATION ---
# Toggle between "fast" (GPT-2) and "best" (TinyLlama / Phi-3)
# GPT-2 is very fast but poor at reasoning. TinyLlama is superior for custom tasks.
MODE = "best" 

MODELS = {
    "fast": "gpt2",
    "best": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

MODEL_NAME = MODELS.get(MODE, "gpt2")

print(f"Loading {MODEL_NAME} model in {MODE} mode...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# OPTIMIZATION: Use 4-bit/8-bit quantization if 'bitsandbytes' is available
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
print(f"Model '{MODEL_NAME}' loaded successfully!")

# --- 2. CONTEXT MEMORY ---
from context_manager import ContextManager
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

from response_middleware import ReasoningAuditMiddleware
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
    turn_start = time.time()
    first_token_time: list[float] = []   # mutable container for closure capture

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

        # ── Memory: retrieve context ──────────────────────────────────────────
        messages = await asyncio.to_thread(
            memory_manager.get_messages, request.session_id, current_query=request.message
        )

        # ── REASONING PIPELINE: Phase A — augment prompt ─────────────────────
        augmented_messages, reasoning_trace = await asyncio.to_thread(
            reasoning_pipeline.prepare_messages, request.message, messages
        )
    else:
        # Continuation flow: skip logging user turn, just get context
        messages = await asyncio.to_thread(
            memory_manager.get_messages, request.session_id, current_query=""
        )
        augmented_messages = messages + [{
            "role": "user", 
            "content": "Please continue exactly where you left off. Output ONLY the continued text, no conversational intro."
        }]
        class FakeTrace:
            intent_category = "coding"  # Give a high token budget
            steps = []
        reasoning_trace = FakeTrace()

    # ── RAG ROUTING: Only use RAG if intent requires factual knowledge ────
    INTENTS_REQUIRING_RAG = ["factual", "coding", "math", "instruction", "general", "clarification", "debugging", "brainstorming"]
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

    # ── Tokenize ──────────────────────────────────────────────────────────
    # ── Tokenize ──────────────────────────────────────────────────────────
    if hasattr(tokenizer, "apply_chat_template") and MODE == "best":
        formatted_prompt = tokenizer.apply_chat_template(augmented_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
    else:
        # Fallback for models without templates
        full_text = ""
        for m in augmented_messages:
            full_text += f"{m['role']}: {m['content']}\n\n"
        full_text += "assistant:"
        inputs = tokenizer(full_text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # ── Rule 3: Unrestricted length control ──────────────────────────────────
    # Removed arbitrary intent-based token constraints to prevent the model 
    # from artificially cutting off code blocks or long explanations.
    max_new_tokens = 8192

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        repetition_penalty=1.1,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Start generation in background thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # ── SSE streaming generator ───────────────────────────────────────────
    async def event_generator():
        draft_text = ""
        for new_text in streamer:
            if await http_request.is_disconnected():
                logger.warning("Client disconnected — stopping stream.")
                break

            # Record time-to-first-token
            if not first_token_time:
                first_token_time.append(time.time() - turn_start)

            temp_text = draft_text + new_text
            
            # Halt on role-leakage hallucination or special tokens
            # Optimized: Only check the tail to prevent O(N^2) lag
            stop_tokens = ["<|user|>", "<|assistant|>", "<|system|>", "</s>", "\nuser:", "\nassistant:", "\nsystem:"]
            tail = temp_text[-50:].lower()
            if any(token in tail for token in stop_tokens):
                break

            draft_text += new_text
            yield f"data: {json.dumps({'content': new_text, 'conv_id': conv_id})}\n\n"
            await asyncio.sleep(0.01)

        # ── REASONING PIPELINE: Phase B — validate & refine ──────────────
        final_response, reasoning_trace_updated = await asyncio.to_thread(
            reasoning_pipeline.refine, request.message, draft_text.strip(), reasoning_trace
        )

        if reasoning_trace_updated.refinement_applied:
            logger.info(
                "[ReasoningPipeline] Refinement applied | issues=%s",
                reasoning_trace_updated.draft_issues
            )
            # Push correction delta to frontend so it shows the refined version
            # (only if the text actually changed)
            if final_response.strip() != draft_text.strip():
                yield f"data: {json.dumps({'content': '', 'refined': True, 'full': final_response, 'conv_id': conv_id})}\n\n"

        # ── Repetition guard ──────────────────────────────────────────────
        is_repeat = await asyncio.to_thread(
            memory_manager.is_repetitive_answer,
            request.session_id, final_response
        )
        if is_repeat:
            logger.info(
                "[ContextManager] Suppressing repetitive answer for session=%s",
                request.session_id
            )
            # Append a gentle note so the user knows the model noticed
            final_response += (
                "\n\n*(I noticed I may have answered this before — "
                "let me know if you'd like a different perspective!)*"
            )

        # Persist the refined (higher-quality) response to memory
        await asyncio.to_thread(
            memory_manager.add_message, request.session_id, "assistant", final_response
        )

        # ── Feedback: log assistant turn with auto-eval ────────────────────
        total_elapsed = time.time() - turn_start
        ttft           = first_token_time[0] if first_token_time else 0.0
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

        yield f"data: {json.dumps({'done': True, 'conv_id': conv_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Mount the static files at the root. We do this at the end so it doesn't shadow API routes.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
