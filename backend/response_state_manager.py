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
import re
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
    detected_language: str = ""
    last_complete_line: str = ""


class CompetitiveProgrammingValidator:
    """
    Advanced validation for competitive programming code blocks.
    Tracks markdown state, bracket matching, and language-specific syntax rules.
    """
    @staticmethod
    def validate(buffer: str) -> dict:
        # 1. Markdown validation & Language Detection
        code_blocks = re.findall(r"```(\w*)", buffer)
        incomplete_code_block = len(code_blocks) % 2 != 0
        
        detected_language = ""
        if code_blocks and incomplete_code_block:
            detected_language = code_blocks[-1].lower()

        # 2. Bracket Matching
        open_braces = buffer.count("{")
        close_braces = buffer.count("}")
        unclosed_braces = max(0, open_braces - close_braces)

        # 3. Python Specific Check (Indentation/Block check)
        is_python_incomplete = False
        if detected_language == "python" or detected_language == "py":
            lines = buffer.split("\n")
            if lines:
                last_line = lines[-1].rstrip()
                # If the last line ends with a colon, we are definitely expecting an indented block
                if last_line.endswith(":"):
                    is_python_incomplete = True

        return {
            "incomplete_code_block": incomplete_code_block,
            "detected_language": detected_language,
            "unclosed_braces": unclosed_braces,
            "is_python_incomplete": is_python_incomplete
        }


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
        
        # Validate using the CP Validator
        validation = CompetitiveProgrammingValidator.validate(ctx.buffer)
        ctx.incomplete_code_block = validation["incomplete_code_block"]
        ctx.unclosed_braces = validation["unclosed_braces"]
        ctx.detected_language = validation["detected_language"]
        is_python_incomplete = validation["is_python_incomplete"]

        # Track the last full line for precise interruption recovery
        lines = ctx.buffer.split("\n")
        if len(lines) > 1:
            # -2 because the last element is the currently forming line after the final \n
            ctx.last_complete_line = lines[-2].strip()

        # Is the code functionally complete?
        is_complete = not ctx.incomplete_code_block and ctx.unclosed_braces == 0 and not is_python_incomplete
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
                
            if ctx.unclosed_braces > 0 and ctx.detected_language not in ["python", "py"]:
                prompt_parts.append(f"Ensure {ctx.unclosed_braces} pending open braces ({{) are closed.")
                
            if ctx.detected_language in ["python", "py"]:
                prompt_parts.append("Maintain proper Python indentation for the incomplete block.")
                
            prompt_parts.append("Complete the final code immediately.")
        else:
            prompt_parts.append("Please seamlessly continue the previous response without apologizing.")

        return " ".join(prompt_parts)

    def finalize_response(self, session_id: str):
        """Clears the generation buffer and unlocks the state machine after success."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session {session_id} finalized and state cleared.")
