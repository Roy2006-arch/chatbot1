"""
reasoning_pipeline.py
=====================
Internal reasoning and response-refinement middleware.

Pipeline (all internal — never surfaced to the user):
  Step 1 — Intent Classification : categorise user intent
  Step 2 — Problem Decomposition : break into sub-tasks / reasoning steps
  Step 3 — Prompt Augmentation   : enrich the raw prompt before inference
  Step 4 — Response Validation   : score draft against quality thresholds
  Step 5 — Response Refinement   : rewrite poor drafts for tone & clarity

Usage (inside main.py):
    from reasoning_pipeline import ReasoningPipeline
    pipeline = ReasoningPipeline()
    augmented_prompt = pipeline.prepare_prompt(user_message, raw_context)
    final_response  = pipeline.refine(user_message, draft_response)
"""

from __future__ import annotations

import re
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("reasoning_pipeline")
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTrace:
    """Holds every internal step for debugging/logging (never sent to user)."""
    intent:          str = ""
    intent_category: str = "general"
    coding_sub_intent: str = ""
    is_coding_challenge: bool = False
    steps:           list[str] = field(default_factory=list)
    draft_issues:    list[str] = field(default_factory=list)
    refinement_applied: bool = False
    latency_ms:      float = 0.0


# ---------------------------------------------------------------------------
# Step 1 — Intent Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Rule-based intent classifier.
    Categories: factual | coding | math | opinion | clarification |
                instruction | greeting | complaint | general
    """

    _PATTERNS: list[tuple[str, str]] = [
        (r"\b(hello|hi|hey|good morning|good evening|howdy|greetings|sup|yo)\b", "greeting"),
        # debugging must come before coding so "fix"/"bug" route correctly
        (r"\b(debug|fix the bug|traceback|stack trace|exception|why is this failing|not working as expected|diagnose|runtime error|syntax error)\b", "debugging"),
        (r"\b(brainstorm|ideas for|come up with|suggest ideas|think of|what are some ways|alternatives|possibilities|generate ideas|creative ideas)\b", "brainstorming"),
        (r"\b(how to|steps to|guide|tutorial|explain|walk me through|steps for|procedure)\b", "instruction"),
        (r"\b(write|code|implement|function|class|script|python|javascript|java|cpp|c#|html|css)\b", "coding"),
        (r"\b(calculate|solve|equation|math|integral|derivative|sum of|probability|maths|formula)\b", "math"),
        (r"\b(what is|who is|when did|where is|define|meaning of|tell me about|details on|info on|fact check|history of)\b", "factual"),
        (r"\b(should i|do you think|opinion|better|worse|recommend|suggest)\b", "opinion"),
        (r"\b(clarify|what do you mean|can you explain|elaborate|more detail)\b", "clarification"),
        (r"\b(not working|broken|wrong|incorrect|bad|terrible|frustrated|annoyed)\b", "complaint"),
    ]

    def classify(self, message: str) -> tuple[str, str]:
        """Return (intent_summary, category_label)."""
        
        # 1. Advanced Code Detection: Check for markdown code blocks or regex signatures
        has_code = False
        if "```" in message:
            has_code = True
        else:
            # Look for common language signatures (Python, JS, Java, C++, SQL, etc.)
            code_signatures = [
                r"\bdef \w+\(", r"\bclass \w+:", r"\bimport \w+",    # Python
                r"\bfunction\s*\(?", r"\bconst \w+\s*=", r"=>",       # JavaScript
                r"\bpublic static void", r"\bSystem\.out",            # Java
                r"#include\s*<", r"\bstd::",                          # C++
                r"SELECT .* FROM", r"UPDATE .* SET"                   # SQL
            ]
            has_code = any(re.search(p, message) for p in code_signatures)

        # 2. Standard Intent Matching
        msg_lower = message.lower()
        for pattern, category in self._PATTERNS:
            if re.search(pattern, msg_lower):
                # If code is present but intent doesn't match coding/debugging inherently, upgrade it
                if has_code and category not in ["coding", "debugging"]:
                    return "User is asking a question or requesting a fix for the provided code.", "debugging"
                    
                intent_summary = f"User is making a {category} request."
                return intent_summary, category
                
        # 3. Fallbacks
        if has_code:
            return "User provided a raw code snippet. They likely want it reviewed or explained.", "coding"

        return "User is making a general request.", "general"


# ---------------------------------------------------------------------------
# Step 2 — Problem Decomposer
# ---------------------------------------------------------------------------

class ProblemDecomposer:
    """
    Breaks a user message into reasoning sub-steps based on intent category.
    These steps are used to construct a richer prompt prefix.
    """

    _TEMPLATES: dict[str, list[str]] = {
        "factual": [
            "Recall relevant facts and definitions.",
            "Cross-check for potential ambiguity in the question.",
            "Provide a clear, direct answer with supporting context.",
        ],
        "coding:direct_solution": [
            "Identify this as a coding challenge and lock into CODE SOLVER MODE.",
            "Provide a short logic summary (maximum 3 lines).",
            "Write the complete, final working code.",
            "Optionally provide a complexity analysis.",
            "Strictly follow the structure: Logic Summary -> Complete Code -> Complexity Analysis."
        ],
        "coding:debugging": [
            "Identify the root cause of the error or unexpected behavior.",
            "Explain why it happens in plain terms.",
            "Provide a concrete fix with corrected code if applicable.",
            "Mention how to avoid the issue in the future."
        ],
        "coding:explanation": [
            "Break down the provided code step by step.",
            "Explain what each part does and how it contributes to the overall logic.",
            "Avoid rewriting the code unless a simplification helps understanding."
        ],
        "coding:optimization": [
            "Analyze the current time and space complexity.",
            "Identify bottlenecks in the code.",
            "Provide an optimized version with improved complexity.",
            "Explain the improvements clearly."
        ],
        "coding:conversion": [
            "Identify the source language and the target language.",
            "Translate the logic idiomatically to the target language.",
            "Note any language-specific differences or caveats."
        ],
        "coding:algorithm": [
            "Explain the theoretical concept behind the algorithm.",
            "Walk through an example of how it works.",
            "Discuss its typical use cases and complexities."
        ],
        "coding": [
            "Understand the programming task and language constraints.",
            "Plan the solution approach (algorithm / data structure).",
            "Write clean, commented code.",
            "Add usage examples or edge-case notes.",
        ],
        "math": [
            "Identify the mathematical concept and required operations.",
            "Lay out the step-by-step solution method.",
            "Compute and verify the result.",
            "Explain the reasoning so the user can learn from it.",
        ],
        "instruction": [
            "Break the task into ordered, actionable steps.",
            "Highlight prerequisites or caveats up front.",
            "Provide each step with enough detail to follow independently.",
        ],
        "opinion": [
            "Acknowledge this is a subjective or preference-based question.",
            "Present balanced perspectives where applicable.",
            "Offer a concrete recommendation with clear reasoning.",
        ],
        "clarification": [
            "Identify which part of the previous response was unclear.",
            "Rephrase the explanation using simpler language or an analogy.",
            "Invite follow-up if still unclear.",
        ],
        "complaint": [
            "Acknowledge the user's frustration empathetically.",
            "Diagnose what went wrong without being defensive.",
            "Propose a clear corrective action or alternative.",
        ],
        "debugging": [
            "Identify the root cause of the error or unexpected behavior.",
            "Explain why it happens in plain terms.",
            "Provide a concrete fix with corrected code if applicable.",
            "Mention how to avoid the issue in the future.",
        ],
        "brainstorming": [
            "Generate a range of distinct, creative ideas relevant to the topic.",
            "Keep each idea concise — a sentence or two max.",
            "Prioritize variety and usefulness over quantity.",
        ],
        "greeting": [
            "Respond naturally and personably in one short sentence.",
            "Do NOT offer a numbered list of options or ask how to assist in a robotic way.",
        ],
        "general": [
            "Understand the core of the user's request.",
            "Provide a helpful, well-structured response.",
            "Offer to elaborate or clarify if needed.",
        ],
    }

    def decompose(self, category: str, sub_intent: str = "", is_challenge: bool = False) -> list[str]:
        if category in ["coding", "debugging"]:
            if is_challenge:
                return self._TEMPLATES.get("coding:direct_solution", self._TEMPLATES["coding"])
            if sub_intent:
                key = f"coding:{sub_intent}"
                if key in self._TEMPLATES:
                    return self._TEMPLATES[key]
        return self._TEMPLATES.get(category, self._TEMPLATES["general"])


# ---------------------------------------------------------------------------
# Step 3 — Prompt Augmenter
# ---------------------------------------------------------------------------

class PromptAugmenter:
    """
    Injects a structured reasoning prefix into the raw context prompt.
    The model is instructed to follow the internal reasoning chain
    before generating its visible reply.
    """

    def augment_messages(
        self,
        messages: list[dict[str, str]],
        intent_summary: str,
        reasoning_steps: list[str],
    ) -> list[dict[str, str]]:
        """
        Injects a structured reasoning prefix into the system message.
        """
        steps_text = "\n".join(
            f"  {i+1}. {step}" for i, step in enumerate(reasoning_steps)
        )
        reasoning_guidance = (
            f"\n\n[Reasoning Context]\n"
            f"Goal: {intent_summary}\n"
            f"Strategy: {', '.join(reasoning_steps)}\n"
        )
        
        # Copy to avoid mutation
        new_messages = [m.copy() for m in messages]
        if new_messages and new_messages[0]["role"] == "system":
            new_messages[0]["content"] += reasoning_guidance
            
        return new_messages


# ---------------------------------------------------------------------------
# Step 4 — Response Validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool
    issues: list[str]
    score:  float        # 0.0 – 1.0


class ResponseValidator:
    """
    Lightweight rule-based validator.  Flags responses that are:
      - Too short (likely incomplete)
      - Contain role-leakage tokens (user:/assistant:/system:)
      - Highly repetitive (hallucination proxy)
      - Contain profanity or offensive language stubs
      - Missing punctuation entirely
    """

    MIN_WORDS = 8

    def validate(self, response: str, intent_category: str) -> ValidationResult:
        issues: list[str] = []
        score = 1.0
        words = response.split()

        if intent_category != "greeting" and len(words) < self.MIN_WORDS:
            issues.append(f"Response too short ({len(words)} words, min {self.MIN_WORDS}).")
            score -= 0.35

        if re.search(r'\b(user:|assistant:|system:)\b', response, re.IGNORECASE):
            issues.append("Role-leakage tokens detected (user:/assistant:/system:).")
            score -= 0.30

        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.35:
            issues.append(f"High repetition detected (unique-word ratio: {unique_ratio:.2f}).")
            score -= 0.25

        if not re.search(r'[.!?]', response):
            issues.append("No sentence-ending punctuation found.")
            score -= 0.10

        # For coding intent: penalise if no code block found or if incomplete
        if "coding" in intent_category or intent_category == "coding":
            if "```" not in response:
                issues.append("Coding response missing code block (```).")
                score -= 0.15
            else:
                if response.count("```") % 2 != 0:
                    issues.append("Markdown code block is not properly closed.")
                    score -= 0.40
                if response.count("{") != response.count("}"):
                    issues.append("Unclosed braces detected in code.")
                    score -= 0.30

        score = round(max(0.0, min(1.0, score)), 4)
        return ValidationResult(passed=score >= 0.55, issues=issues, score=score)


# ---------------------------------------------------------------------------
# Step 5 — Response Rewriter / Refiner
# ---------------------------------------------------------------------------

class ResponseRewriter:
    """
    Post-processes the model's raw draft to improve:
      - Tone     : remove curt or cold phrasing
      - Clarity  : strip role-leakage artefacts
      - Structure: enforce sentence capitalisation & spacing
    """

    # Curt openers that signal an impersonal tone
    _CURT_OPENERS = [
        (r'^(The answer is:?)', "Here's the answer:"),
        (r'^(No\.?\s)', "I'm sorry, but no. "),
        (r'^(Yes\.?\s)', "Yes — "),
        (r'^(I don\'t know\.?)', "I'm not entirely sure, but I can try to help."),
        (r'^(Okay\.?\s)', "Sure — "),
    ]

    # Rule 4: Robotic phrases explicitly banned by the behavioral prompt
    _ROBOTIC_PHRASES = [
        (r'As an AI(?: assistant|language model)?[,.]?\s*', ''),
        (r'I would be delighted to[,.]?\s*', ''),
        (r'How may I assist your[^?]*\??\s*', ''),
        (r'Certainly(?:! Here are several options\.\.\.|[,!])\s*', ''),
        (r'Based on your query[,.]?\s*', ''),
        (r'Of course[,!]\s*', ''),
        (r'Absolutely[,!]\s*', ''),
        (r'Great question[,!]\s*', ''),
        (r'(?i)Here is the code:?\s*', ''),
        (r'(?i)Let\'s understand .*? first\s*', ''),
    ]

    # Role-leak artefacts to strip
    _LEAK_PATTERNS = [
        r'\buser:\s*', r'\bassistant:\s*', r'\bsystem:\s*',
    ]

    def rewrite(self, response: str, intent_category: str, issues: list[str]) -> str:
        if not issues:
            # Still run basic cleanup even when no issues flagged
            return self._basic_cleanup(response)

        result = response.strip()

        # --- Rule 4: Strip robotic phrases (always applied) ---
        for pattern, replacement in self._ROBOTIC_PHRASES:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # --- Strip role leakage ---
        for pattern in self._LEAK_PATTERNS:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # --- Fix curt openers ---
        for pattern, replacement in self._CURT_OPENERS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE, count=1)

        # --- Capitalise first letter ---
        if result and not result[0].isupper():
            result = result[0].upper() + result[1:]

        # --- Ensure trailing punctuation ---
        if result and result[-1] not in '.!?':
            result += '.'

        # --- Add empathetic header for complaints ---
        if intent_category == "complaint" and not result.lower().startswith("i understand"):
            result = "I understand this isn't working as expected — let's fix it. " + result

        # --- Normalize excess whitespace ---
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {2,}', ' ', result)

        return result.strip()

    def _basic_cleanup(self, response: str) -> str:
        result = response.strip()
        # Rule 4: Always strip robotic phrases even in passing responses
        for pattern, replacement in self._ROBOTIC_PHRASES:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {2,}', ' ', result)
        if result and not result[0].isupper():
            result = result[0].upper() + result[1:]
        return result


# ---------------------------------------------------------------------------
# Public Facade — ReasoningPipeline
# ---------------------------------------------------------------------------

class ReasoningPipeline:
    """
    Orchestrates all internal reasoning steps.

    Usage in main.py:
        pipeline = ReasoningPipeline()

        # Before inference:
        augmented_prompt = pipeline.prepare_prompt(user_message, raw_context)

        # After inference (draft collected):
        final_response, trace = pipeline.refine(user_message, draft_response)
    """

    def __init__(self, refinement_threshold: float = 0.55):
        self.classifier    = IntentClassifier()
        self.decomposer    = ProblemDecomposer()
        self.augmenter     = PromptAugmenter()
        self.validator     = ResponseValidator()
        self.rewriter      = ResponseRewriter()
        self.threshold     = refinement_threshold
        self._last_trace: Optional[ReasoningTrace] = None

    # ---- Phase A: Pre-inference -------------------------------------------

    def prepare_messages(self, user_message: str, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], ReasoningTrace]:
        """
        Augments the message list with internal reasoning instructions.
        Returns (augmented_messages, trace).
        """
        t0 = time.perf_counter()
        trace = ReasoningTrace()

        intent_summary, category = self.classifier.classify(user_message)
        trace.intent          = intent_summary
        trace.intent_category = category

        if category in ["coding", "debugging"]:
            from coding_intent import CodingIntentClassifier
            coding_classifier = CodingIntentClassifier()
            coding_intent = coding_classifier.classify(user_message)
            if coding_intent.is_coding:
                trace.coding_sub_intent = coding_intent.sub_intent
                trace.is_coding_challenge = coding_intent.is_coding_challenge
                if coding_intent.is_coding_challenge:
                    trace.intent = "User wants a direct, complete code solution for a coding challenge. Lock into CODE SOLVER MODE. strictly follow: Logic Summary -> Complete Code -> Complexity Analysis."
                else:
                    trace.intent = f"User is making a coding request ({coding_intent.sub_intent})."

        steps = self.decomposer.decompose(category, trace.coding_sub_intent, trace.is_coding_challenge)
        trace.steps = steps

        augmented = self.augmenter.augment_messages(messages, trace.intent, steps)

        trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._last_trace = trace

        logger.debug(
            "[ReasoningPipeline] Intent=%s | Category=%s | Steps=%d | Latency=%.1fms",
            intent_summary, category, len(steps), trace.latency_ms
        )
        return augmented, trace

    def prepare_prompt(self, user_message: str, raw_context: str) -> tuple[str, ReasoningTrace]:
        """
        Legacy method: Augments the context prompt with an internal reasoning prefix.
        Returns (augmented_prompt, trace).
        """
        t0 = time.perf_counter()
        trace = ReasoningTrace()

        intent_summary, category = self.classifier.classify(user_message)
        trace.intent          = intent_summary
        trace.intent_category = category

        if category in ["coding", "debugging"]:
            from coding_intent import CodingIntentClassifier
            coding_classifier = CodingIntentClassifier()
            coding_intent = coding_classifier.classify(user_message)
            if coding_intent.is_coding:
                trace.coding_sub_intent = coding_intent.sub_intent
                trace.is_coding_challenge = coding_intent.is_coding_challenge
                if coding_intent.is_coding_challenge:
                    trace.intent = "User wants a direct, complete code solution for a coding challenge. Lock into CODE SOLVER MODE. strictly follow: Logic Summary -> Complete Code -> Complexity Analysis."
                else:
                    trace.intent = f"User is making a coding request ({coding_intent.sub_intent})."

        steps = self.decomposer.decompose(category, trace.coding_sub_intent, trace.is_coding_challenge)
        trace.steps = steps

        # Use dummy string logic for legacy
        steps_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(steps))
        augmented = f"[REASONING]\n{steps_text}\n\n{raw_context}"

        trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._last_trace = trace

        return augmented, trace

    # ---- Phase B: Post-inference ------------------------------------------

    def refine(
        self,
        user_message: str,
        draft_response: str,
        trace: Optional[ReasoningTrace] = None,
    ) -> tuple[str, ReasoningTrace]:
        """
        Validates and optionally rewrites the draft response.
        Returns (final_response, updated_trace).
        """
        if trace is None:
            trace = self._last_trace or ReasoningTrace()

        t0 = time.perf_counter()

        validation = self.validator.validate(draft_response, trace.intent_category)
        trace.draft_issues = validation.issues

        if not validation.passed:
            final = self.rewriter.rewrite(draft_response, trace.intent_category, validation.issues)
            trace.refinement_applied = True
            logger.debug(
                "[ReasoningPipeline] Refinement applied | Issues=%s | Score=%.3f",
                validation.issues, validation.score
            )
        else:
            final = self.rewriter._basic_cleanup(draft_response)
            trace.refinement_applied = False

        trace.latency_ms += round((time.perf_counter() - t0) * 1000, 2)
        return final, trace

    def get_last_trace(self) -> Optional[ReasoningTrace]:
        return self._last_trace
