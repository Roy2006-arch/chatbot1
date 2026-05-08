"""
backend/unified_orchestrator.py
-------------------------------
Unified Orchestration System for Chatbot Response Lifecycle.
Merges ContinuityManager, ResponseStateManager, and CompetitiveProgrammingValidator.
"""

import logging
import re
import json
import time
import asyncio
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("unified_orchestrator")

class ResponseLifecycle(Enum):
    INIT = auto()
    RETRIEVAL = auto()
    GENERATION = auto()
    STREAMING = auto()
    VALIDATION = auto()
    RECOVERY = auto()
    FINALIZED = auto()

@dataclass
class OrchestrationState:
    session_id: str
    state: ResponseLifecycle = ResponseLifecycle.INIT
    buffer: str = ""
    emitted_length: int = 0
    token_count: int = 0
    markdown_open: bool = False
    open_brackets: Dict[str, int] = field(default_factory=lambda: {"{": 0, "[": 0, "(": 0})
    detected_language: Optional[str] = None
    is_interrupted: bool = False
    last_complete_line: str = ""
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "state": self.state.name,
            "token_count": self.token_count,
            "is_interrupted": self.is_interrupted,
            "markdown_open": self.markdown_open,
            "detected_language": self.detected_language,
            "issues": self.issues
        }

class UnifiedOrchestrator:
    """
    Centralized orchestration system for managing the full lifecycle of an AI response.
    Handles state transitions, streaming safety, validation, and interruption recovery.
    """

    def __init__(self):
        from backend.validation_engine import ResponseValidationEngine
        self._sessions: Dict[str, OrchestrationState] = {}
        self.validator = ResponseValidationEngine()
        
        # Patterns that are unsafe to emit if they appear at the very end of a chunk
        self.UNSAFE_PATTERNS = [
            r"```\w*$",   # Incomplete code block start
            r"`{1,2}$",    # Incomplete backticks
            r"\[[^\]]*$",  # Incomplete markdown link/image
            r"\$[^$]*$",   # Incomplete LaTeX
            r"\*\*$|__$",  # Incomplete bold/italic
            r"^\s*-\s*$",  # Incomplete list item
        ]

    def _get_state(self, session_id: str) -> OrchestrationState:
        if session_id not in self._sessions:
            self._sessions[session_id] = OrchestrationState(session_id=session_id)
        return self._sessions[session_id]

    def transition(self, session_id: str, new_state: ResponseLifecycle):
        """Moves the session to a new state in the lifecycle."""
        state = self._get_state(session_id)
        logger.info(f"Session {session_id} transitioning: {state.state.name} -> {new_state.name}")
        state.state = new_state

    def get_current_state(self, session_id: str) -> ResponseLifecycle:
        return self._get_state(session_id).state

    # --- 1. STREAMING & CONTINUITY ---

    def process_chunk(self, session_id: str, chunk: str) -> str:
        """
        Updates the internal buffer with a new chunk and returns the safe-to-emit portion.
        This merges the continuity logic and code completion checks.
        """
        state = self._get_state(session_id)
        state.buffer += chunk
        state.token_count += 1
        
        # 1. Update Markdown State
        backticks = re.findall(r"```", state.buffer)
        state.markdown_open = len(backticks) % 2 != 0
        
        if state.markdown_open:
            match = re.search(r"```(\w+)\n", state.buffer)
            if match:
                state.detected_language = match.group(1).lower()

        # 2. Update Bracket/Brace State
        for char in chunk:
            if char in state.open_brackets:
                state.open_brackets[char] += 1
            elif char == "}":
                state.open_brackets["{"] = max(0, state.open_brackets["{"] - 1)
            elif char == "]":
                state.open_brackets["["] = max(0, state.open_brackets["["] - 1)
            elif char == ")":
                state.open_brackets["("] = max(0, state.open_brackets["("] - 1)

        # 3. Track last complete line for recovery
        lines = state.buffer.split("\n")
        if len(lines) > 1:
            # -2 is the last full line before the currently forming one
            line = lines[-2].strip()
            if line:
                state.last_complete_line = line

        # 4. Tail Buffering for Safety
        full_buffer = state.buffer
        emit_until = len(full_buffer)
        
        for pattern in self.UNSAFE_PATTERNS:
            match = re.search(pattern, full_buffer)
            if match:
                emit_until = min(emit_until, match.start())

        safe_to_emit = full_buffer[:emit_until]
        new_content = safe_to_emit[state.emitted_length:]
        state.emitted_length += len(new_content)
        
        return new_content

    # --- 2. VALIDATION & RECOVERY ---

    def validate_and_finalize(self, session_id: str) -> Dict[str, Any]:
        """
        Runs comprehensive validation on the final response.
        Uses the advanced ResponseValidationEngine for structural and logical checks.
        """
        state = self._get_state(session_id)
        self.transition(session_id, ResponseLifecycle.VALIDATION)
        
        # Run advanced validation
        report = self.validator.validate(state.buffer)
        state.issues = [i.message for i in report.issues]
        
        is_valid = report.is_valid
        repair_suffix = ""
        
        if not is_valid:
            # Check if it's a critical issue that requires full recovery/regeneration
            if report.needs_regeneration:
                state.is_interrupted = True
                self.transition(session_id, ResponseLifecycle.RECOVERY)
            else:
                # Attempt auto-repair for non-critical issues
                repaired_text = self.validator.repair(state.buffer, report)
                repair_suffix = repaired_text[len(state.buffer):]
                state.buffer = repaired_text
                is_valid = True # Mark as valid after repair
        
        if is_valid and not state.is_interrupted:
            self.transition(session_id, ResponseLifecycle.FINALIZED)

        return {
            "is_valid": is_valid,
            "issues": state.issues,
            "repair_suffix": repair_suffix,
            "is_interrupted": state.is_interrupted
        }

    def _is_repairable(self, issues: List[str]) -> bool:
        # Simple heuristic: can repair if just a few brackets or markdown unclosed
        return len(issues) <= 3 and "python_incomplete_block" not in issues

    def _perform_repair(self, state: OrchestrationState) -> str:
        """Appends closing tokens to fix syntax."""
        repair = ""
        if state.markdown_open:
            repair += "\n```"
            state.markdown_open = False
            
        for bracket, count in reversed(list(state.open_brackets.items())):
            closing = {"{": "}", "[": "]", "(": ")"}[bracket]
            repair += closing * count
            state.open_brackets[bracket] = 0
            
        state.buffer += repair
        return repair

    def get_recovery_prompt(self, session_id: str) -> str:
        """
        Generates a contextual recovery prompt for the next turn.
        Merges recovery logic from both previous managers.
        """
        state = self._get_state(session_id)
        
        last_lines = state.buffer.split("\n")[-3:]
        context_snippet = "\n".join(last_lines)
        
        prompt_parts = ["[SYSTEM RECOVERY]: The previous response was interrupted."]
        
        if state.last_complete_line:
            prompt_parts.append(f"Resume exactly from the last complete line: `{state.last_complete_line}`.")
        else:
            prompt_parts.append(f"Resume after: \"...{context_snippet}\"")

        prompt_parts.append("Do NOT repeat previous text. Do NOT use conversational intros.")

        if state.markdown_open:
            lang = state.detected_language or ""
            prompt_parts.append(f"You are inside a {lang} code block. Continue the code and ensure the block is eventually closed with ```.")
            
        for issue in state.issues:
            if "unclosed" in issue.lower():
                prompt_parts.append("Ensure all pending open braces/brackets/parentheses are properly closed.")
            if "truncated" in issue.lower():
                prompt_parts.append("The previous response was cut off mid-expression. Please resume exactly from where it stopped and complete the logic.")
            if "syntax" in issue.lower():
                prompt_parts.append("The previous code block had a syntax error. Please provide a corrected continuation.")
            if "reasoning" in issue.lower():
                prompt_parts.append("The previous thought process was interrupted. Complete the reasoning before giving the final answer.")

        return " ".join(prompt_parts)

    def reset_session(self, session_id: str):
        """Clears state for a fresh turn."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def set_interrupted(self, session_id: str):
        state = self._get_state(session_id)
        state.is_interrupted = True
        self.transition(session_id, ResponseLifecycle.RECOVERY)

    def cleanup_stale_sessions(self, ttl_seconds: int = 3600) -> int:
        """Removes orchestration states older than ttl_seconds."""
        now = time.time()
        to_delete = [
            sid for sid, state in self._sessions.items()
            if now - state.start_time > ttl_seconds
        ]
        for sid in to_delete:
            del self._sessions[sid]
        return len(to_delete)
