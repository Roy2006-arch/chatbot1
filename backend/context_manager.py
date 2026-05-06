"""
backend/context_manager.py
--------------------------
Advanced Conversational Context Manager.

Features:
  - Short-term sliding window (recent messages, configurable)
  - Long-term compression: summarize old turns into rolling summaries
  - Key-information extraction: names, dates, entities, preferences
  - Semantic deduplication: detect + suppress near-duplicate answers
  - Token-budget enforcement: keep prompt under model's context limit
  - FAISS-backed episodic recall: retrieve past user utterances by similarity
  - Thread-safe per-session state with TTL cleanup
  - Drop-in replacement for the old MemoryManager

Usage
-----
    from context_manager import ContextManager

    mgr = ContextManager()
    mgr.add_message(session_id, "user", text)
    prompt_prefix = mgr.get_context(session_id, current_query=text)
    is_repeat = mgr.is_repetitive_answer(session_id, candidate_answer)
    mgr.cleanup_idle_sessions(idle_time_seconds=3600)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger("chatbot.context_manager")

# ── Constants ──────────────────────────────────────────────────────────────────
SYSTEM_IDENTITY = (
    "You are an advanced AI assistant designed to provide accurate, intelligent, concise, and context-aware responses.\n\n"

    "CORE BEHAVIOR RULES:\n\n"

    "1. Understand the user's intent before answering.\n"
    "   - Identify whether the user wants: a short answer, detailed explanation, coding help, casual conversation, "
    "step-by-step guidance, brainstorming, debugging, or factual information.\n\n"

    "2. Never generate multiple unrelated responses.\n"
    "   - Give ONE best answer only.\n"
    "   - Do not provide lists of alternative greetings or unnecessary options unless explicitly asked.\n\n"

    "3. Control response length intelligently.\n"
    "   - For greetings like 'hello', 'hi': reply naturally in 1 short sentence.\n"
    "   - For simple factual questions: answer in 2-5 lines.\n"
    "   - For technical or educational questions: provide structured detailed answers.\n"
    "   - Avoid overexplaining unless requested.\n\n"

    "4. Be conversational and natural.\n"
    "   - Sound human-like.\n"
    "   - Avoid robotic phrases like: 'As an AI assistant...', 'I would be delighted...', "
    "'How may I assist your greeting request?'\n"
    "   - Use modern natural language.\n\n"

    "5. Stay focused on the user query.\n"
    "   - Do not drift into unrelated topics.\n"
    "   - Do not hallucinate information.\n"
    "   - If uncertain, admit uncertainty clearly.\n\n"

    "6. Before generating the final answer, internally check:\n"
    "   - Is the answer relevant? Is it concise enough? Is it accurate? Is it understandable?\n"
    "   Then generate the final response.\n\n"

    "7. Formatting rules:\n"
    "   - Use paragraphs for normal answers.\n"
    "   - Use bullet points only when useful.\n"
    "   - Use code blocks for programming.\n"
    "   - Avoid giant walls of text.\n\n"

    "8. Coding response rules:\n"
    "   - Give clean and correct code.\n"
    "   - Explain only important parts.\n"
    "   - Avoid unnecessary theory unless asked.\n\n"

    "9. Memory and context:\n"
    "   - Remember recent conversation context.\n"
    "   - Do not repeat previous answers unnecessarily.\n"
    "   - Maintain continuity naturally.\n\n"

    "10. Tone adaptation:\n"
    "    - Beginner user: simpler explanations.\n"
    "    - Technical user: more detailed technical depth.\n"
    "    - Casual chat: friendly and short.\n\n"

    "11. Efficiency rule:\n"
    "    - Prioritize quality over quantity.\n"
    "    - The best answer is: correct, direct, useful, readable.\n\n"

    "12. Never do these:\n"
    "    - Never generate 5 alternative answers automatically.\n"
    "    - Never repeat the same sentence.\n"
    "    - Never write filler content.\n"
    "    - Never make the response unnecessarily dramatic.\n\n"

    "Main objective: Generate responses that feel similar to modern high-quality AI assistants — "
    "intelligent, concise, helpful, context-aware, human-like, and efficient. Always think before responding."
)

# How many characters count as approximately 1 token (rough heuristic for GPT-family)
CHARS_PER_TOKEN: int = 4

# ── Key-info extraction patterns ───────────────────────────────────────────────
_KI_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("name",       re.compile(r"\bmy name is ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", re.I)),
    ("name",       re.compile(r"\bcall me ([A-Z][a-z]+)", re.I)),
    ("location",   re.compile(r"\bI(?:'m| am) (?:from|in|at|based in) ([\w\s,]+?)(?:\.|,|$)", re.I)),
    ("language",   re.compile(r"\bI(?:'m| am) (?:learning|using|coding in) ([\w+#]+)", re.I)),
    ("preference", re.compile(r"\bI (?:prefer|like|love|hate|dislike|want) ([\w\s]+?)(?:\.|,|$)", re.I)),
    ("goal",       re.compile(r"\bI(?:'m| am) (?:trying|working|building|creating|making) ([\w\s]+?)(?:\.|,|$)", re.I)),
    ("context",    re.compile(r"\bmy (?:project|app|system|bot|model) (?:is|uses) ([\w\s]+?)(?:\.|,|$)", re.I)),
]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """All memory structures for a single session."""
    # Short-term ring buffer (recent messages)
    window: List[Dict[str, str]] = field(default_factory=list)
    # Compressed rolling summary of older turns
    rolling_summary: str = ""
    # Extracted key info (name, prefs, goals …)
    key_info: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    # FAISS index for episodic user utterances
    faiss_index: Optional[faiss.Index] = None
    # Parallel store: texts mapped to FAISS vectors
    episodic_texts: List[str] = field(default_factory=list)
    # Hash ring for deduplication of assistant answers
    answer_hashes: List[str] = field(default_factory=list)
    # Last access timestamp for TTL cleanup
    last_accessed: float = field(default_factory=time.time)
    # Total turn count (not limited to window)
    total_turns: int = 0


# ── ContextManager ─────────────────────────────────────────────────────────────

class ContextManager:
    """
    Drop-in replacement for MemoryManager with significantly smarter
    context handling for long conversations.

    Parameters
    ----------
    window_size : int
        Number of *recent* message objects to keep in the sliding window.
    summarize_after : int
        Evict + summarize oldest messages once window grows beyond this many.
    max_tokens_budget : int
        Approximate max tokens the final context string may consume.
        Context is truncated (oldest parts removed) if over budget.
    dedup_threshold : float
        Cosine-similarity threshold above which an answer is considered
        a repeat (0–1, higher = stricter).
    embedding_model : str
        SentenceTransformer model name for semantic operations.
    """

    def __init__(
        self,
        window_size: int = 8,
        summarize_after: int = 12,
        max_tokens_budget: int = 2048,
        dedup_threshold: float = 0.92,
        embedding_model: str = "all-MiniLM-L6-v2",
        # Kept for backwards-compat with callers that pass max_context_messages=
        max_context_messages: int = 0,
    ) -> None:
        if max_context_messages:           # honour legacy kwarg
            window_size = max_context_messages

        self.window_size = window_size
        self.summarize_after = summarize_after
        self.max_token_budget = max_tokens_budget
        self.dedup_threshold = dedup_threshold

        log.info("[ContextManager] Loading embedder: %s", embedding_model)
        self._embedder = SentenceTransformer(embedding_model)
        try:
            self._dim = self._embedder.get_embedding_dimension()
        except AttributeError:   # older sentence-transformers
            self._dim = self._embedder.get_sentence_embedding_dimension()

        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Record a new message; triggers summarization when window overflows."""
        state = self._get_or_create(session_id)
        with self._lock:
            state.window.append({"role": role, "content": content})
            state.total_turns += 1
            state.last_accessed = time.time()

            if role == "user":
                self._index_episodic(state, content)
                self._extract_key_info(state, content)

            if len(state.window) > self.summarize_after:
                self._compress_window(state)

    def get_messages(
        self,
        session_id: str,
        current_query: Optional[str] = None,
        include_key_info: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Build and return the list of message objects for this session.
        This follows the standard {"role": "...", "content": "..."} format.
        """
        if session_id not in self._sessions:
            return [{"role": "system", "content": SYSTEM_IDENTITY}]

        state = self._sessions[session_id]
        state.last_accessed = time.time()
        
        messages: List[Dict[str, str]] = []

        # 1. System Prompt (Identity + Key Info + Episodic Context)
        system_content = [SYSTEM_IDENTITY]

        # 1.2 Key info
        if include_key_info and state.key_info:
            ki_lines = []
            for category, values in state.key_info.items():
                unique_vals = list(dict.fromkeys(values))[-3:]  # last 3 unique
                ki_lines.append(f"  {category}: {', '.join(unique_vals)}")
            if ki_lines:
                system_content.append(
                    "[Persistent user facts]\n" + "\n".join(ki_lines)
                )

        # 1.3 Rolling summary
        if state.rolling_summary:
            system_content.append(
                f"[Conversation summary so far]\n{state.rolling_summary}"
            )

        # 1.4 Episodic recall (similar past queries)
        if current_query and state.faiss_index and state.faiss_index.ntotal > 0:
            recalled = self._recall_similar(state, current_query, k=3)
            if recalled:
                system_content.append(
                    "[Relevant past user statements]\n"
                    + "\n".join(f"  - {r}" for r in recalled)
                )

        messages.append({"role": "system", "content": "\n\n".join(system_content)})

        # 2. Sliding window (verbatim)
        # We append the history turns
        messages.extend(state.window)

        # Enforce token budget on the HISTORY (window), not on the system prompt
        return self._trim_messages_to_budget(messages)

    def get_context(self, *args, **kwargs) -> str:
        """Legacy shim: returns concatenated string (deprecated)."""
        msgs = self.get_messages(*args, **kwargs)
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    def is_repetitive_answer(
        self, session_id: str, candidate: str
    ) -> bool:
        """
        Return True if candidate answer is semantically very similar to
        a recent assistant answer in this session (repetition guard).
        """
        if session_id not in self._sessions:
            return False
        state = self._sessions[session_id]
        # Collect recent assistant messages
        recent_assistant = [
            m["content"]
            for m in state.window[-6:]
            if m["role"] == "assistant"
        ]
        if not recent_assistant:
            return False

        cand_vec = self._embed([candidate])
        for past in recent_assistant:
            past_vec = self._embed([past])
            sim = self._cosine(cand_vec[0], past_vec[0])
            if sim >= self.dedup_threshold:
                log.info(
                    "[ContextManager] Repetition detected (sim=%.3f) for session=%s",
                    sim, session_id,
                )
                return True
        return False

    def get_key_info(self, session_id: str) -> Dict[str, List[str]]:
        """Return extracted key-info dict for the session."""
        if session_id not in self._sessions:
            return {}
        return dict(self._sessions[session_id].key_info)

    def get_summary(self, session_id: str) -> str:
        """Return the current rolling summary for the session."""
        if session_id not in self._sessions:
            return ""
        return self._sessions[session_id].rolling_summary

    def cleanup_idle_sessions(self, idle_time_seconds: int = 3600) -> int:
        """Remove sessions idle for longer than idle_time_seconds. Returns count deleted."""
        now = time.time()
        with self._lock:
            to_delete = [
                sid for sid, state in self._sessions.items()
                if now - state.last_accessed > idle_time_seconds
            ]
            for sid in to_delete:
                del self._sessions[sid]
        if to_delete:
            log.info("[ContextManager] Cleaned up %d idle sessions.", len(to_delete))
        return len(to_delete)

    def session_stats(self, session_id: str) -> dict:
        """Return diagnostic stats for a session."""
        if session_id not in self._sessions:
            return {}
        s = self._sessions[session_id]
        return {
            "window_messages": len(s.window),
            "total_turns": s.total_turns,
            "episodic_indexed": s.faiss_index.ntotal if s.faiss_index else 0,
            "summary_chars": len(s.rolling_summary),
            "key_info_categories": len(s.key_info),
            "last_accessed": s.last_accessed,
        }

    # ── Backwards-compat shim for old MemoryManager callers ───────────────────
    # (old code called memory_manager.history, memory_manager.summaries, etc.)

    @property
    def history(self) -> Dict[str, List[Dict]]:
        return {sid: s.window for sid, s in self._sessions.items()}

    @property
    def summaries(self) -> Dict[str, str]:
        return {sid: s.rolling_summary for sid, s in self._sessions.items()}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_or_create(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            state = SessionState()
            state.faiss_index = faiss.IndexFlatIP(self._dim)  # inner-product → cosine
            self._sessions[session_id] = state
        return self._sessions[session_id]

    # ── Summarization ──────────────────────────────────────────────────────────

    def _compress_window(self, state: SessionState) -> None:
        """
        Evict the oldest half of the window, generate a plain-text summary,
        and append it to the rolling summary.
        """
        evict_count = len(state.window) - self.window_size
        if evict_count <= 0:
            return

        evicted = state.window[:evict_count]
        state.window = state.window[evict_count:]  # exactly window_size messages remain

        new_summary = self._summarize_turns(evicted)
        if state.rolling_summary:
            # Re-compress if rolling summary is already long
            if len(state.rolling_summary) > 1500:
                state.rolling_summary = self._merge_summaries(
                    state.rolling_summary, new_summary
                )
            else:
                state.rolling_summary += "\n" + new_summary
        else:
            state.rolling_summary = new_summary

        log.debug(
            "[ContextManager] Compressed %d messages. Summary length=%d chars.",
            evict_count, len(state.rolling_summary),
        )

    def _summarize_turns(self, turns: List[Dict[str, str]]) -> str:
        """
        Produce a concise summary of a list of turns using a rule-based
        extractive approach (no external LLM call required).

        The summary:
          - Captures each user question (trimmed)
          - Captures a 1-sentence snippet of the assistant's reply
        """
        lines = []
        for msg in turns:
            role = msg["role"]
            text = msg["content"].strip()
            if role == "user":
                trimmed = text[:200] + ("…" if len(text) > 200 else "")
                lines.append(f"User asked: {trimmed}")
            elif role == "assistant":
                # Take first sentence as the gist
                first_sent = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
                first_sent = first_sent[:180] + ("…" if len(first_sent) > 180 else "")
                lines.append(f"Assistant said: {first_sent}")
        return " | ".join(lines)

    def _merge_summaries(self, old: str, new: str) -> str:
        """
        Merge an old rolling summary with a new summary segment.
        Keeps the most recent content by truncating old from the front.
        """
        combined = old + "\n" + new
        # Keep last 1800 chars of summary to avoid unbounded growth
        if len(combined) > 1800:
            combined = "…" + combined[-1800:]
        return combined

    # ── Key-info extraction ────────────────────────────────────────────────────

    def _extract_key_info(self, state: SessionState, text: str) -> None:
        for category, pattern in _KI_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if value and len(value) < 100:
                    existing = state.key_info[category]
                    if value not in existing:
                        existing.append(value)
                        # Cap list at 5 per category
                        if len(existing) > 5:
                            state.key_info[category] = existing[-5:]

    # ── Episodic recall ────────────────────────────────────────────────────────

    def _index_episodic(self, state: SessionState, text: str) -> None:
        """Embed and add a user utterance to the FAISS episodic index."""
        vec = self._embed([text])
        faiss.normalize_L2(vec)
        state.faiss_index.add(vec)
        state.episodic_texts.append(text)

    def _recall_similar(
        self, state: SessionState, query: str, k: int = 3
    ) -> List[str]:
        """Retrieve top-k episodic user utterances most similar to query."""
        q_vec = self._embed([query])
        faiss.normalize_L2(q_vec)
        n_total = state.faiss_index.ntotal
        actual_k = min(k, n_total)
        if actual_k == 0:
            return []
        distances, indices = state.faiss_index.search(q_vec, actual_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(state.episodic_texts):
                continue
            # Require at least moderate similarity
            if score > 0.55:
                recalled_text = state.episodic_texts[idx]
                # Don't recall the current query itself
                if recalled_text.strip() != query.strip():
                    trimmed = recalled_text[:180] + (
                        "…" if len(recalled_text) > 180 else ""
                    )
                    results.append(trimmed)
        return results

    # ── Deduplication helpers ──────────────────────────────────────────────────

    @staticmethod
    def _answer_hash(text: str) -> str:
        return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:12]

    # ── Embedding & similarity ─────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._embedder.encode(texts, normalize_embeddings=False)
        return np.array(vecs, dtype="float32")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ── Token budget ───────────────────────────────────────────────────────────

    def _trim_messages_to_budget(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Ensures the total message list stays within token budget.
        Trims from the START of the window (index 1 and onwards), 
        preserving the system prompt (index 0).
        """
        if len(messages) <= 1:
            return messages

        # Estimate tokens
        total_chars = sum(len(m["content"]) for m in messages)
        estimated_tokens = total_chars // CHARS_PER_TOKEN

        if estimated_tokens <= self.max_token_budget:
            return messages

        # We need to remove history turns while keeping the system prompt (messages[0])
        system_msg = messages[0]
        history = messages[1:]

        while history and (sum(len(m["content"]) for m in [system_msg] + history) // CHARS_PER_TOKEN) > self.max_token_budget:
            history.pop(0)  # Remove oldest history turn

        log.debug(
            "[ContextManager] Trimmed %d history messages to fit budget.",
            len(messages) - 1 - len(history)
        )
        return [system_msg] + history
