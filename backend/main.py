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
from pathlib import Path
from fastapi import UploadFile, File
from backend.document_processor import DocumentProcessor
from backend.reasoning_pipeline import ReasoningPipeline
from backend.response_state_manager import ResponseStateManager, ResponseState
from backend.rag import KnowledgeBase, DocumentIngestionPipeline, RAGRetriever

# ── Feedback Loop System ─────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feedback import init_db, log_turn, new_conv_id, router as feedback_router
from feedback.db_schema import get_conn   # ensure DB is ready at import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot.main")

app = FastAPI(title="Custom Chatbot API - Streaming Enabled")

# ── Feedback router ────────────────────────────────────────
app.include_router(feedback_router, prefix="/feedback", tags=["Feedback"])

# Initialise feedback DB on startup
init_db()

# --- CORS ---
# SECURITY: "*" origins with credentials is invalid per the CORS spec and unsafe.
# Configure allowed origins via CORS_ALLOW_ORIGINS (comma-separated). Default "*"
# is only valid when credentials are disabled.
_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
if _origins_env == "*":
    _allow_origins = ["*"]
    _allow_credentials = False
else:
    _allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
    _allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES & FRONTEND ---
from fastapi.responses import FileResponse

# Get absolute path to the frontend directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Static files will be mounted at the end of the file to avoid route shadowing

# --- 1. MODEL LOADING & OPTIMIZATION ---
# Small-but-strong instruct models that run on CPU. Qwen2.5-1.5B-Instruct is the
# default sweet spot for a ~2 vCPU / 16GB deployment: proper chat template (fixes
# role leakage), strong coding/reasoning for its size, and a large context window.
# Override with MODEL_MODE=fast|best|quality or MODEL_NAME=<hf-id>.
MODELS = {
    "fast":    "Qwen/Qwen2.5-0.5B-Instruct",
    "best":    "Qwen/Qwen2.5-1.5B-Instruct",
    "quality": "Qwen/Qwen2.5-3B-Instruct",
}

MODE = os.getenv("MODEL_MODE", "best")
MODEL_NAME = os.getenv("MODEL_NAME") or MODELS.get(MODE, MODELS["best"])

print(f"Loading {MODEL_NAME} (mode={MODE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Keep dtype selection simple and robust across torch versions:
# fp16 on GPU, fp32 on CPU (bf16 CPU kernels are inconsistent across torch builds).
_use_cuda = torch.cuda.is_available()
_dtype = torch.float16 if _use_cuda else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if _use_cuda else None,
    torch_dtype=_dtype,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"Model '{MODEL_NAME}' loaded successfully!")

# Model context window (for clamping max_new_tokens). Fall back to a safe default.
MODEL_MAX_CONTEXT = int(
    getattr(model.config, "max_position_embeddings", 0) or 4096
)

# A single process holds one model; generation is not thread-safe when interleaved.
# Serialize generate() calls with a lock. Scale out with more replicas, not threads.
_GEN_LOCK = threading.Lock()

# --- Coding developer prompt (only attached for coding/debugging turns) ---
_CODING_PROMPT_PATH = Path(__file__).parent / "coding_developer_prompt.txt"
try:
    CODING_DEVELOPER_PROMPT = _CODING_PROMPT_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    CODING_DEVELOPER_PROMPT = ""

# --- 2. CONTEXT MEMORY ---
from backend.context_manager import ContextManager
memory_manager = ContextManager(
    window_size=10,
    summarize_after=16,
    max_tokens_budget=3072,
    dedup_threshold=0.92,
)
doc_processor = DocumentProcessor()

# --- 3. GLOBAL RAG KNOWLEDGE BASE ---
knowledge_base = KnowledgeBase()
rag_pipeline   = DocumentIngestionPipeline()
retriever      = RAGRetriever(
    knowledge_base,
    top_k_candidates=30,
    top_n_results=3,
    score_floor=0.45,       # raised from 0.30 to avoid injecting weak/off-topic context
    use_reranker=True,
    use_mmr=True,
)

# --- Reasoning & Refinement Middleware ---
reasoning_pipeline = ReasoningPipeline(refinement_threshold=0.55)
state_manager = ResponseStateManager()

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

# --- Decoding presets per intent -------------------------------------------------
def _generation_params(intent_category: str) -> dict:
    """Per-intent decoding presets. Deterministic for code/math, sampled otherwise."""
    if intent_category in ("coding", "debugging", "math"):
        return dict(do_sample=False, repetition_penalty=1.1)
    return dict(do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.1)


# --- 4. STREAMING ENDPOINT (ChatGPT Style) ---
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # ── Feedback: conversation tracking ───────────────────────────
    conv_id    = request.get_or_create_conv_id()
    turn_start = time.time()
    first_token_time: list[float] = []   # mutable container for closure capture

    # ── Feedback & Memory Tracking ──────────────────────────────
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

        await asyncio.to_thread(memory_manager.add_message, request.session_id, "user", request.message)

        messages = await asyncio.to_thread(
            memory_manager.get_messages, request.session_id, current_query=request.message
        )

        # ── REASONING PIPELINE: Phase A — classify intent ────────────────────
        augmented_messages, reasoning_trace = await asyncio.to_thread(
            reasoning_pipeline.prepare_messages, request.message, messages
        )

        is_coding = getattr(reasoning_trace, 'is_coding_challenge', False) or \
            reasoning_trace.intent_category in ("coding", "debugging")

        # Attach the coding developer prompt ONLY for coding turns.
        if is_coding and CODING_DEVELOPER_PROMPT and augmented_messages \
                and augmented_messages[0]["role"] == "system":
            augmented_messages[0]["content"] += "\n\n" + CODING_DEVELOPER_PROMPT

        if is_coding:
            state_manager.set_state(request.session_id, ResponseState.CODING_SOLVER)
        else:
            state_manager.set_state(request.session_id, ResponseState.NORMAL_CHAT)
    else:
        # Continuation flow: skip logging user turn, just get context
        messages = await asyncio.to_thread(
            memory_manager.get_messages, request.session_id, current_query=""
        )
        recovery_prompt = state_manager.get_recovery_prompt(request.session_id)
        content_msg = recovery_prompt if recovery_prompt else "Please continue exactly where you left off. Output ONLY the continued text, no conversational intro."
        augmented_messages = messages + [{
            "role": "user",
            "content": content_msg
        }]
        class FakeTrace:
            intent_category = "coding"  # Give a high token budget
            steps = []
        reasoning_trace = FakeTrace()

    # ── RAG ROUTING: Only use RAG if intent requires factual knowledge ────
    INTENTS_REQUIRING_RAG = ["factual", "coding", "math", "instruction", "general", "clarification", "debugging", "brainstorming"]
    category = reasoning_trace.intent_category

    if not request.is_continuation and category in INTENTS_REQUIRING_RAG:
        rag_result = await asyncio.to_thread(retriever.retrieve, request.message)
        doc_context = await asyncio.to_thread(
            doc_processor.query_documents, request.session_id, request.message
        )

        if augmented_messages and augmented_messages[0]["role"] == "system":
            system_content = augmented_messages[0]["content"]

            if rag_result.context_block:
                system_content += (
                    "\n\n[USE THE FOLLOWING VERIFIED KNOWLEDGE]\n"
                    "Answer using only the facts in this block. Quote or paraphrase it; "
                    "if it does not contain the answer, say you don't have that information "
                    "rather than guessing.\n"
                    f"{rag_result.context_block}"
                )
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

    # ── Tokenize (chat template) ────────────────────────────────
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(
            augmented_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {"input_ids": inputs}
    else:
        # Fallback for models without a chat template
        full_text = ""
        for m in augmented_messages:
            full_text += f"{m['role']}: {m['content']}\n\n"
        full_text += "assistant:"
        inputs = tokenizer(full_text, return_tensors="pt")

    if _use_cuda:
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[-1]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # ── Bounded length control ───────────────────────────────────
    # Cap new tokens AND ensure prompt + new tokens stay within the model's
    # context window (the old code requested 8192 on a 2048-ctx model → garbage).
    default_new = 1536 if category in ("coding", "debugging") else 1024
    room = max(64, MODEL_MAX_CONTEXT - prompt_len - 8)
    max_new_tokens = min(int(os.getenv("MAX_NEW_TOKENS", default_new)), room)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **_generation_params(category),
    )

    # Start generation in background thread (serialized by the global lock).
    def _run_generation():
        with _GEN_LOCK, torch.no_grad():
            model.generate(**generation_kwargs)

    thread = threading.Thread(target=_run_generation)
    thread.start()

    # ── SSE streaming generator ─────────────────────────────────
    async def event_generator():
        draft_text = ""
        for new_text in streamer:
            if await http_request.is_disconnected():
                logger.warning("Client disconnected — stopping stream.")
                state_manager.flag_interrupted(request.session_id)
                break

            if not first_token_time:
                first_token_time.append(time.time() - turn_start)

            temp_text = draft_text + new_text

            # Safety net: halt on role-leakage / special tokens (the chat template +
            # eos should already prevent this for Qwen, but keep a guard).
            stop_tokens = ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>",
                           "<|system|>", "</s>", "\nuser:", "\nassistant:", "\nsystem:"]
            tail = temp_text[-50:].lower()
            if any(token.lower() in tail for token in stop_tokens):
                break

            draft_text += new_text
            state_manager.update_generation(request.session_id, new_text)
            yield f"data: {json.dumps({'content': new_text, 'conv_id': conv_id})}\n\n"
            await asyncio.sleep(0.01)

        # ── REASONING PIPELINE: Phase B — validate & light cleanup ──────────
        final_response, reasoning_trace_updated = await asyncio.to_thread(
            reasoning_pipeline.refine, request.message, draft_text.strip(), reasoning_trace
        )

        if reasoning_trace_updated.refinement_applied:
            logger.info(
                "[ReasoningPipeline] Cleanup applied | issues=%s",
                reasoning_trace_updated.draft_issues
            )
            if final_response.strip() != draft_text.strip():
                yield f"data: {json.dumps({'content': '', 'refined': True, 'full': final_response, 'conv_id': conv_id})}\n\n"

        # ── Repetition guard ───────────────────────────────────
        is_repeat = await asyncio.to_thread(
            memory_manager.is_repetitive_answer,
            request.session_id, final_response
        )
        if is_repeat:
            logger.info(
                "[ContextManager] Suppressing repetitive answer for session=%s",
                request.session_id
            )
            final_response += (
                "\n\n*(I noticed I may have answered this before — "
                "let me know if you'd like a different perspective!)*"
            )

        await asyncio.to_thread(
            memory_manager.add_message, request.session_id, "assistant", final_response
        )

        # ── Feedback: log assistant turn ───────────────────────────
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

        state_manager.update_generation(request.session_id, "")
        state_manager.finalize_response(request.session_id)
        yield f"data: {json.dumps({'done': True, 'conv_id': conv_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Mount the static files at the root. We do this at the end so it doesn't shadow API routes.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
