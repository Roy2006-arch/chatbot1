"""
backend/response_state_manager.py
---------------------------------
Manages response generation state, stream interruptions, and strictly enforces 
behavioral modes (like coding_solver) to prevent topic switching.

Features:
- State Machine (Normal, Coding, Debugging, Explanation)
- Lock-in mechanism for CODING_SOLVER mode
- Incomplete code block detection (markdown parsing)
- Unclosed brace counting
- Interruption recovery logic and prompt generation
"""

import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger("response_state_manager")

class ResponseState(Enum):
    NORMAL_CHAT = auto()
    CODING_SOLVER = auto()
    DEBUGGING = auto()
    EXPLANATION_MODE = auto()


@dataclass
class GenerationContext:
    session_id: str
    current_state: ResponseState
    buffer: str = ""
    is_interrupted: bool = False
    incomplete_code_block: bool = False
    unclosed_braces: int = 0
    last_complete_line: str = ""


class ResponseStateManager:
    """
    Tracks the active state of an ongoing generation, monitors completion criteria,
    and constructs recovery prompts if a stream is interrupted or incomplete.
    """

    def __init__(self):
        self._sessions: Dict[str, GenerationContext] = {}

    def _get_context(self, session_id: str) -> GenerationContext:
        if session_id not in self._sessions:
            self._sessions[session_id] = GenerationContext(
                session_id=session_id,
                current_state=ResponseState.NORMAL_CHAT
            )
        return self._sessions[session_id]

    def set_state(self, session_id: str, new_state: ResponseState) -> bool:
        """
        Attempts to update the current mode.
        Returns True if successful, False if the state change was rejected.
        """
        ctx = self._get_context(session_id)
        
        # Lock rule: Once CODING_SOLVER is active, do not switch until reset/completed
        if ctx.current_state == ResponseState.CODING_SOLVER and new_state != ResponseState.CODING_SOLVER:
            logger.warning(f"Session {session_id} is locked in CODING_SOLVER mode. State change to {new_state.name} rejected.")
            return False
            
        ctx.current_state = new_state
        return True

    def get_state(self, session_id: str) -> ResponseState:
        return self._get_context(session_id).current_state

    def update_generation(self, session_id: str, chunk: str) -> bool:
        """
        Feeds a newly generated chunk into the buffer and recalculates
        completion validity (markdown blocks, braces).
        Returns True if the current buffer appears completely formed.
        """
        ctx = self._get_context(session_id)
        ctx.buffer += chunk
        
        # Markdown closure validation
        code_block_count = ctx.buffer.count("```")
        ctx.incomplete_code_block = (code_block_count % 2 != 0)
        
        # Brace closure validation (simple heuristic)
        open_braces = ctx.buffer.count("{")
        close_braces = ctx.buffer.count("}")
        ctx.unclosed_braces = max(0, open_braces - close_braces)

        # Track the last full line for precise interruption recovery
        lines = ctx.buffer.split("\n")
        if len(lines) > 1:
            # -2 because the last element is the currently forming line after the final \n
            ctx.last_complete_line = lines[-2].strip()

        is_complete = not ctx.incomplete_code_block and ctx.unclosed_braces == 0
        return is_complete

    def flag_interrupted(self, session_id: str):
        """Marks a session as interrupted (e.g. network drop, token limit)."""
        ctx = self._get_context(session_id)
        ctx.is_interrupted = True
        logger.warning(f"Session {session_id} generation marked as interrupted.")

    def get_recovery_prompt(self, session_id: str) -> Optional[str]:
        """
        Builds an automatic recovery instruction if the previous state
        was left interrupted or syntactically incomplete.
        """
        ctx = self._get_context(session_id)
        
        # If generation was completed perfectly, no recovery needed
        if not ctx.is_interrupted and not ctx.incomplete_code_block and ctx.unclosed_braces == 0:
            return None

        prompt_parts = ["[SYSTEM RECOVERY]: The previous generation was interrupted or incomplete."]
        
        if ctx.current_state == ResponseState.CODING_SOLVER:
            prompt_parts.append("Maintain CODE SOLVER MODE. Do NOT restart reasoning or explanation.")
            
            if ctx.last_complete_line:
                prompt_parts.append(f"Resume exactly from the last complete line: `{ctx.last_complete_line}`.")
            else:
                prompt_parts.append("Resume code generation exactly where it left off.")
                
            if ctx.incomplete_code_block:
                prompt_parts.append("Ensure the markdown code block (```) is properly closed.")
                
            if ctx.unclosed_braces > 0:
                prompt_parts.append(f"Ensure {ctx.unclosed_braces} pending open braces ({{) are closed.")
                
            prompt_parts.append("Complete the final code immediately.")
        else:
            prompt_parts.append("Please seamlessly continue the previous response without apologizing.")

        return " ".join(prompt_parts)

    def finalize_response(self, session_id: str):
        """Clears the generation buffer and unlocks the state machine after success."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session {session_id} finalized and state cleared.")
