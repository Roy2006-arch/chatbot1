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
from backend.shared_resources import ModelRegistry, get_request_cache

log = logging.getLogger("chatbot.context_manager")

import os
from pathlib import Path

_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"


def get_system_identity() -> str:
    try:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as _f:
            return _f.read()
    except FileNotFoundError:
        return "You are a helpful AI assistant."


CHARS_PER_TOKEN: int = 4

_KI_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("name",       re.compile(r"\bmy name is ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", re.I)),
    ("name",       re.compile(r"\bcall me ([A-Z][a-z]+)", re.I)),
    ("location",   re.compile(r"\bI(?:'m| am) (?:from|in|at|based in) ([\w\s,]+?)(?:\.|,|$)", re.I)),
    ("language",   re.compile(r"\bI(?:'m| am) (?:learning|using|coding in) ([\w+#]+)", re.I)),
    ("preference", re.compile(r"\bI (?:prefer|like|love|hate|dislike|want) ([\w\s]+?)(?:\.|,|$)", re.I)),
    ("goal",       re.compile(r"\bI(?:'m| am) (?:trying|working|building|creating|making) ([\w\s]+?)(?:\.|,|$)", re.I)),
    ("context",    re.compile(r"\bmy (?:project|app|system|bot|model) (?:is|uses) ([\w\s]+?)(?:\.|,|$)", re.I)),
]


@dataclass
class SessionState:
    window: List[Dict[str, str]] = field(default_factory=list)
    rolling_summary: str = ""
    key_info: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    faiss_index: Optional[faiss.Index] = None
    episodic_texts: List[str] = field(default_factory=list)
    answer_hashes: List[str] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)
    total_turns: int = 0


class ContextManager:
    def __init__(
        self,
        window_size: int = 8,
        summarize_after: int = 12,
        max_tokens_budget: int = 2048,
        dedup_threshold: float = 0.92,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_context_messages: int = 0,
    ) -> None:
        if max_context_messages:
            window_size = max_context_messages

        self.window_size = window_size
        self.summarize_after = summarize_after
        self.max_token_budget = max_tokens_budget
        self.dedup_threshold = dedup_threshold

        log.info("[ContextManager] Using shared embedder.")
        self._embedder = ModelRegistry.get_embedder()
        try:
            self._dim = self._embedder.get_embedding_dimension()
        except AttributeError:
            self._dim = self._embedder.get_sentence_embedding_dimension()

        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        state = self._get_or_create(session_id)
        with self._lock:
            state.window.append({"role": role, "content": content})
            state.total_turns += 1
            state.last_accessed = time.time()

            if role == "user":
                self._extract_key_info(state, content)

            if len(state.window) > self.summarize_after:
                self._compress_window(state)

        if role == "user":
            self._index_episodic(state, content)

    def get_messages(
        self,
        session_id: str,
        current_query: Optional[str] = None,
        include_key_info: bool = True,
    ) -> List[Dict[str, str]]:
        if session_id not in self._sessions:
            return [{"role": "system", "content": get_system_identity()}]

        state = self._sessions[session_id]
        state.last_accessed = time.time()

        messages: List[Dict[str, str]] = []
        system_content = [get_system_identity()]

        if include_key_info and state.key_info:
            ki_lines = []
            for category, values in state.key_info.items():
                unique_vals = list(dict.fromkeys(values))[-3:]
                ki_lines.append(f"  {category}: {', '.join(unique_vals)}")
            if ki_lines:
                system_content.append("[Persistent user facts]\n" + "\n".join(ki_lines))

        if state.rolling_summary:
            system_content.append(f"[Conversation summary so far]\n{state.rolling_summary}")

        if current_query and state.faiss_index and state.faiss_index.ntotal > 0:
            recalled = self._recall_similar(state, current_query, k=3)
            if recalled:
                system_content.append(
                    "[Relevant past user statements]\n"
                    + "\n".join(f"  - {r}" for r in recalled)
                )

        messages.append({"role": "system", "content": "\n\n".join(system_content)})
        messages.extend(state.window)

        return self._trim_messages_to_budget(messages)

    def get_context(self, *args, **kwargs) -> str:
        msgs = self.get_messages(*args, **kwargs)
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    def is_repetitive_answer(self, session_id: str, candidate: str) -> bool:
        if session_id not in self._sessions:
            return False
        state = self._sessions[session_id]

        recent_assistant = [
            m["content"]
            for m in state.window[-6:]
            if m["role"] == "assistant"
        ]
        if not recent_assistant:
            return False

        candidate_tokens = set(candidate.lower().split())
        for past in recent_assistant:
            past_tokens = set(past.lower().split())
            if len(candidate_tokens) < 5 or len(past_tokens) < 5:
                continue
            overlap = len(candidate_tokens & past_tokens)
            jaccard = overlap / max(len(candidate_tokens | past_tokens), 1)
            if jaccard > 0.7:
                log.info("[ContextManager] Repetition detected (jaccard=%.3f)", jaccard)
                return True
        return False

    def get_key_info(self, session_id: str) -> Dict[str, List[str]]:
        if session_id not in self._sessions:
            return {}
        return dict(self._sessions[session_id].key_info)

    def get_summary(self, session_id: str) -> str:
        if session_id not in self._sessions:
            return ""
        return self._sessions[session_id].rolling_summary

    def cleanup_idle_sessions(self, idle_time_seconds: int = 3600) -> int:
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

    @property
    def history(self) -> Dict[str, List[Dict]]:
        return {sid: s.window for sid, s in self._sessions.items()}

    @property
    def summaries(self) -> Dict[str, str]:
        return {sid: s.rolling_summary for sid, s in self._sessions.items()}

    def _get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                state = SessionState()
                state.faiss_index = faiss.IndexFlatIP(self._dim)
                self._sessions[session_id] = state
            return self._sessions[session_id]

    def _compress_window(self, state: SessionState) -> None:
        evict_count = len(state.window) - self.window_size
        if evict_count <= 0:
            return

        evicted = state.window[:evict_count]
        state.window = state.window[evict_count:]

        new_summary = self._summarize_turns(evicted)
        if state.rolling_summary:
            if len(state.rolling_summary) > 1500:
                state.rolling_summary = self._merge_summaries(state.rolling_summary, new_summary)
            else:
                state.rolling_summary += "\n" + new_summary
        else:
            state.rolling_summary = new_summary

        log.debug("[ContextManager] Compressed %d messages.", evict_count)

    def _summarize_turns(self, turns: List[Dict[str, str]]) -> str:
        lines = []
        for msg in turns:
            role = msg["role"]
            text = msg["content"].strip()
            if role == "user":
                trimmed = text[:200] + ("..." if len(text) > 200 else "")
                lines.append(f"User asked: {trimmed}")
            elif role == "assistant":
                first_sent = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
                first_sent = first_sent[:180] + ("..." if len(first_sent) > 180 else "")
                lines.append(f"Assistant said: {first_sent}")
        return " | ".join(lines)

    def _merge_summaries(self, old: str, new: str) -> str:
        combined = old + "\n" + new
        if len(combined) > 1800:
            combined = "..." + combined[-1800:]
        return combined

    def _extract_key_info(self, state: SessionState, text: str) -> None:
        for category, pattern in _KI_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if value and len(value) < 100:
                    existing = state.key_info[category]
                    if value not in existing:
                        existing.append(value)
                        if len(existing) > 5:
                            state.key_info[category] = existing[-5:]

    def _index_episodic(self, state: SessionState, text: str) -> None:
        vec = self._embed([text])
        faiss.normalize_L2(vec)
        state.faiss_index.add(vec)
        state.episodic_texts.append(text)

    def _recall_similar(self, state: SessionState, query: str, k: int = 3) -> List[str]:
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
            if score > 0.55:
                recalled_text = state.episodic_texts[idx]
                if recalled_text.strip() != query.strip():
                    trimmed = recalled_text[:180] + ("..." if len(recalled_text) > 180 else "")
                    results.append(trimmed)
        return results

    @staticmethod
    def _answer_hash(text: str) -> str:
        return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:12]

    def _embed(self, texts: List[str]) -> np.ndarray:
        cache = get_request_cache()
        if cache is not None:
            vecs = cache.encode(self._embedder, texts, normalize_embeddings=False, batch_size=len(texts))
        else:
            vecs = self._embedder.encode(texts, normalize_embeddings=False, batch_size=len(texts))
        return np.array(vecs, dtype="float32")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _trim_messages_to_budget(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(messages) <= 1:
            return messages

        total_chars = sum(len(m["content"]) for m in messages)
        estimated_tokens = total_chars // CHARS_PER_TOKEN

        if estimated_tokens <= self.max_token_budget:
            return messages

        system_msg = messages[0]
        history = messages[1:]

        while history and (
            sum(len(m["content"]) for m in [system_msg] + history) // CHARS_PER_TOKEN
        ) > self.max_token_budget:
            history.pop(0)

        log.debug("[ContextManager] Trimmed history to fit budget.")
        return [system_msg] + history
