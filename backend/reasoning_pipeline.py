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
    
    # NEW: Stage 3 — Internal Planning fields
    reasoning_strategy: str = "step_by_step"
    response_depth:     str = "intermediate"  # beginner | intermediate | advanced
    response_mode:      str = "normal"        # normal | locked_solver
    output_structure:   str = "default"
    
    steps:           list[str] = field(default_factory=list)
    draft_issues:    list[str] = field(default_factory=list)
    refinement_applied: bool = False
    latency_ms:      float = 0.0


# ---------------------------------------------------------------------------
# Stage 1 — Intent Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Rule-based intent classifier.
    Categories: coding_problem | debugging | explanation | architecture | 
                optimization | casual_chat | document_query
    """

    _PATTERNS: list[tuple[str, str]] = [
        (r"\b(hello|hi|hey|good morning|good evening|howdy|greetings|sup|yo)\b", "casual_chat"),
        (r"\b(debug|fix the bug|traceback|stack trace|exception|why is this failing|not working as expected|diagnose|runtime error|syntax error)\b", "debugging"),
        (r"\b(architecture|design|structure|system design|scalability|blueprint|infrastructure)\b", "architecture"),
        (r"\b(optimize|make it faster|performance|efficient|bottleneck|refactor)\b", "optimization"),
        (r"\b(explain|how does|what is|define|clarify|elaborate|walk me through)\b", "explanation"),
        (r"\b(solve|implement|write code|code for|function|class|program|challenge|problem)\b", "coding_problem"),
        (r"\b(document|pdf|file|read this|search in|find in document)\b", "document_query"),
    ]

    def classify(self, message: str) -> tuple[str, str]:
        """Return (intent_summary, category_label)."""
        
        # 1. Advanced Code Detection: Check for markdown code blocks or regex signatures
        has_code = False
        if "```" in message:
            has_code = True
        else:
            code_signatures = [
                r"\bdef \w+\(", r"\bclass \w+:", r"\bimport \w+",    # Python
                r"\bfunction\s*\(?", r"\bconst \w+\s*=", r"=>",       # JavaScript
                r"\bpublic static void", r"\bSystem\.out",            # Java
                r"#include\s*<", r"\bstd::",                          # C++
                r"SELECT .* FROM", r"UPDATE .* SET"                   # SQL
            ]
            has_code = any(re.search(p, message, re.IGNORECASE) for p in code_signatures)

        # 2. Standard Intent Matching
        msg_lower = message.lower()
        for pattern, category in self._PATTERNS:
            if re.search(pattern, msg_lower):
                if has_code and category not in ["coding_problem", "debugging", "explanation", "optimization"]:
                    return "User provided code and is requesting assistance.", "debugging"
                    
                intent_summary = f"User is making a {category} request."
                return intent_summary, category
                
        # 3. Fallbacks
        if has_code:
            return "User provided a code snippet. Assumed coding challenge or implementation task.", "coding_problem"

        return "User is making a general request.", "general"


# ---------------------------------------------------------------------------
# Stage 3 — Internal Planner (was ProblemDecomposer)
# ---------------------------------------------------------------------------

class InternalPlanner:
    """
    Identifies required output structure, estimates response depth,
    determines reasoning strategy, and determines response mode.
    """

    _DEPTH_INDICATORS = {
        "beginner": r"\b(beginner|basic|simple|eli5|easy|start|introduction|learn)\b",
        "advanced": r"\b(advanced|complex|deep dive|professional|expert|senior|optimized|high performance)\b"
    }

    _TEMPLATES: dict[str, list[str]] = {
        "coding_problem": [
            "Identify this as a coding challenge and lock into CODE SOLVER MODE.",
            "Identify required output structure: Approach -> Code -> Complexity.",
            "Determine reasoning strategy: Algorithmic decomposition and complexity analysis.",
            "Generate structured draft response with complete compilable code.",
            "Strictly follow the MANDATORY RESPONSE FORMAT."
        ],
        "debugging": [
            "Provide a concrete fix with corrected code.",
            "Validate syntax internally.",
        ],
        "explanation": [
            "Break down the concept or code step by step.",
            "Improve clarity and technical precision.",
            "Avoid rewriting code unless necessary.",
        ],
        "architecture": [
            "Identify high-level components and their interactions.",
            "Evaluate scalability and maintainability.",
            "Provide structured design suggestions.",
        ],
        "optimization": [
            "Identify performance bottlenecks.",
            "Suggest algorithmic or structural improvements.",
            "Explain the trade-offs clearly.",
        ],
        "casual_chat": [
            "Respond naturally and personably.",
            "Keep it brief and helpful.",
        ],
        "document_query": [
            "Retrieve relevant context from documents.",
            "Ground claims in retrieved context.",
            "Summarize findings accurately.",
        ],
        "general": [
            "Understand the core request.",
            "Identify reasoning strategy.",
            "Provide a structured, complete response.",
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

    def plan(self, message: str, category: str, is_challenge: bool = False) -> tuple[ReasoningTrace, list[str]]:
        """
        Creates a reasoning trace and determines the steps needed to solve the problem.
        """
        trace = ReasoningTrace(intent_category=category, is_coding_challenge=is_challenge)
        
        # 1. Determine Depth
        msg_lower = message.lower()
        for depth, pattern in self._DEPTH_INDICATORS.items():
            if re.search(pattern, msg_lower):
                trace.response_depth = depth
                break
        
        # 2. Determine Strategy & Mode
        if category == "coding_problem" or is_challenge:
            trace.response_mode = "locked_solver"
            trace.reasoning_strategy = "algorithmic_decomposition"
            trace.output_structure = "Approach -> Code -> Complexity"
        elif category == "debugging":
            trace.reasoning_strategy = "root_cause_analysis"
            trace.output_structure = "Diagnosis -> Fix -> Verification"
        elif category == "explanation":
            trace.reasoning_strategy = "pedagogical_breakdown"
        else:
            trace.reasoning_strategy = "step_by_step"

        # 3. Get Template Steps
        steps = self.decompose(category, is_challenge=is_challenge)
        
        return trace, steps


# ---------------------------------------------------------------------------
# Stage 4 — Prompt Augmenter
# ---------------------------------------------------------------------------

class PromptAugmenter:
    """
    Injects a structured reasoning prefix and Stage 3 planning into the messages.
    """

    def augment_messages(
        self,
        messages: list[dict[str, str]],
        trace: ReasoningTrace,
    ) -> list[dict[str, str]]:
        """
        Injects a structured reasoning prefix into the system message.
        """
        planning_block = (
            f"\n\n[INTERNAL PLANNING]\n"
            f"Intent: {trace.intent_category}\n"
            f"Strategy: {trace.reasoning_strategy}\n"
            f"Mode: {trace.response_mode}\n"
            f"Depth: {trace.response_depth}\n"
            f"Required Structure: {trace.output_structure}\n"
            f"Reasoning Steps:\n" + "\n".join(f"  - {s}" for s in trace.steps) +
            f"\n\n[INSTRUCTION]\n"
            f"You MUST first perform hidden internal reasoning inside <thought> tags. "
            f"Analyze the problem, break it into subproblems, verify logical consistency, "
            f"and then provide your final polished response OUTSIDE the tags."
        )
        
        # Copy to avoid mutation
        new_messages = [m.copy() for m in messages]
        if new_messages and new_messages[0]["role"] == "system":
            new_messages[0]["content"] += planning_block
            
        return new_messages


# ---------------------------------------------------------------------------
# Stage 5 — Response Validator (Self-Evaluation)
# ---------------------------------------------------------------------------

# Obsolete ResponseValidator removed in favor of ResponseValidationEngine


# ---------------------------------------------------------------------------
# Stage 6 — Response Refiner
# ---------------------------------------------------------------------------

class ResponseRewriter:
    """
    Improves clarity, conciseness, readability, and technical precision.
    Automatically repairs detected issues from Stage 5.
    """

    _ROBOTIC_PHRASES = [
        (r'As an AI(?: assistant|language model)?[,.]?\s*', ''),
        (r'I would be delighted to[,.]?\s*', ''),
        (r'Certainly(?:! Here are several options\.\.\.|[,!])\s*', ''),
        (r'Based on your query[,.]?\s*', ''),
        (r'Of course[,!]\s*', ''),
        (r'Absolutely[,!]\s*', ''),
        (r'Great question[,!]\s*', ''),
    ]

    def rewrite(self, response: str, trace: ReasoningTrace, issues: list[str]) -> str:
        result = response.strip()

        # 1. Technical Precision: Strip robotic intros
        for pattern, replacement in self._ROBOTIC_PHRASES:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # 2. Repair Structural Issues
        if "Incomplete markdown code block" in " ".join(issues):
            if result.count("```") % 2 != 0:
                result += "\n```"
        
        # 3. Tone & Clarity Refinement
        if trace.response_depth == "beginner":
            # Potentially simplify (in a real scenario, this might involve another LLM pass,
            # but here we apply rule-based cleanup)
            pass
        elif trace.response_depth == "advanced":
            # Ensure technical precision
            pass

        # 4. Cleanup
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {2,}', ' ', result)
        
        if result and not result[0].isupper() and trace.response_mode != "locked_solver":
            result = result[0].upper() + result[1:]

        return result.strip()

    def _basic_cleanup(self, response: str) -> str:
        result = response.strip()
        for pattern, replacement in self._ROBOTIC_PHRASES:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {2,}', ' ', result)
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

    def __init__(self, refinement_threshold: float = 0.60):
        self.classifier    = IntentClassifier()
        self.planner       = InternalPlanner()
        self.augmenter     = PromptAugmenter()
        self.rewriter      = ResponseRewriter()
        from backend.response_ranker import ResponseRanker
        from backend.dsa_expert import DSAExpert, ExecutionTracer
        from backend.hallucination_guard import HallucinationGuard
        from backend.validation_engine import ResponseValidationEngine
        self.ranker        = ResponseRanker()
        self.dsa_expert    = DSAExpert()
        self.tracer        = ExecutionTracer()
        self.guard         = HallucinationGuard()
        self.validator     = ResponseValidationEngine()
        self.threshold     = refinement_threshold
        self._last_trace: Optional[ReasoningTrace] = None

    # ---- Phase A: Pre-inference -------------------------------------------

    def prepare_messages(self, user_message: str, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], ReasoningTrace]:
        """
        Orchestrates Stages 1-4.
        Returns (augmented_messages, trace).
        """
        t0 = time.perf_counter()

        # Stage 1: Intent Classification
        intent_summary, category = self.classifier.classify(user_message)
        
        is_challenge = False
        coding_sub_intent = ""
        if category in ["coding_problem", "debugging"]:
            from backend.coding_intent import CodingIntentClassifier
            coding_classifier = CodingIntentClassifier()
            coding_intent = coding_classifier.classify(user_message)
            if coding_intent.is_coding:
                coding_sub_intent = coding_intent.sub_intent
                is_challenge = coding_intent.is_coding_challenge

        # Stage 3: Internal Planning
        trace, steps = self.planner.plan(user_message, category, is_challenge)
        trace.intent = intent_summary
        trace.coding_sub_intent = coding_sub_intent
        
        # SPECIALIZATION: DSA Pattern Detection
        if category == "coding_problem":
            pattern = self.dsa_expert.detect_pattern(user_message)
            if pattern:
                trace.reasoning_strategy = f"dsa_pattern:{pattern.name.lower().replace(' ', '_')}"
                edge_cases = self.dsa_expert.analyze_edge_cases(pattern)
                optimization = self.dsa_expert.get_optimization_strategy(pattern)
                
                steps.insert(1, f"Pattern Detected: {pattern.name}")
                steps.append(f"Edge Cases to Handle: {', '.join(edge_cases[:3])}")
                steps.append(f"Optimization Target: {optimization}")
                steps.append("Simulation: Perform a mental symbolic trace with sample input to verify logic.")
                logger.info("[ReasoningPipeline] DSA Pattern Detected: %s", pattern.name)

        trace.steps = steps

        # Stage 4: Prompt Augmentation
        augmented = self.augmenter.augment_messages(messages, trace)

        trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._last_trace = trace

        logger.info(
            "[ReasoningPipeline] Intent=%s | Depth=%s | Mode=%s | Latency=%.1fms",
            category, trace.response_depth, trace.response_mode, trace.latency_ms
        )
        return augmented, trace

    def prepare_prompt(self, user_message: str, raw_context: str) -> tuple[str, ReasoningTrace]:
        """
        Legacy method updated to use the new planning logic.
        """
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user_message}]
        augmented_messages, trace = self.prepare_messages(user_message, messages)
        
        # Extract the planning block from the augmented system message
        planning_block = augmented_messages[0]["content"].split("[INTERNAL PLANNING]")[1]
        augmented_prompt = f"[REASONING]\n[INTERNAL PLANNING]{planning_block}\n\n{raw_context}"
        
        return augmented_prompt, trace

    # ---- Phase B: Post-inference ------------------------------------------

    def refine(
        self,
        user_message: str,
        draft_response: str,
        trace: Optional[ReasoningTrace] = None,
        candidates: Optional[list[str]] = None,
        context: str = "",
        allow_regeneration: bool = True
    ) -> tuple[str, ReasoningTrace]:
        """
        Orchestrates Stage 5 (Evaluation) and Stage 6 (Refinement).
        Uses the Self-Correction Engine for automated repairs.
        """
        if trace is None:
            trace = self._last_trace or ReasoningTrace()

        t0 = time.perf_counter()

        # Step 5a: Ranking (if multiple candidates exist)
        if candidates and len(candidates) > 1:
            draft_response = self.ranker.rank(candidates, user_message, context, trace.steps)

        # Step 5b: Advanced Validation Diagnostic
        report = self.validator.validate(
            draft_response, 
            context=context, 
            expected_mode=trace.response_mode
        )
        trace.draft_issues = [i.message for i in report.issues]

        # Step 5c: Hallucination Guard (Grounded Verification)
        # Still using HallucinationGuard for deep context checking
        h_report = self.guard.evaluate(draft_response, context)
        if not h_report.is_grounded or h_report.confidence_score < 0.5:
            logger.warning("[ReasoningPipeline] Low confidence detected: %.3f", h_report.confidence_score)
            draft_response = self.guard.handle_uncertainty(draft_response, h_report)
            trace.draft_issues.extend(h_report.contradictions)

        # Step 5d: Check for Regeneration Necessity
        if report.needs_regeneration and allow_regeneration:
            logger.warning("[ReasoningPipeline] Critical issues detected. Triggering regeneration/refinement.")
            trace.refinement_applied = True
            # In a real system, we might call an LLM here to REGENERATE.
            # For now, we signal it in the trace.
        
        # Step 5e: Extract Hidden Reasoning
        internal_reasoning, clean_response = self.extract_final_answer(draft_response)
        if internal_reasoning:
            logger.info("[ReasoningPipeline] Extracted internal reasoning (%d chars)", len(internal_reasoning))
            draft_response = clean_response

        # Step 6: Final Refinement & Repair
        if not report.is_valid:
            final = self.validator.repair(draft_response, report)
            final = self.rewriter.rewrite(final, trace, [i.message for i in report.issues])
            trace.refinement_applied = True
            logger.info(
                "[ReasoningPipeline] Advanced repair applied | Issues=%d",
                len(report.issues)
            )
        else:
            final = self.rewriter._basic_cleanup(draft_response)
            trace.refinement_applied = False

        trace.latency_ms += round((time.perf_counter() - t0) * 1000, 2)
        return final, trace

    def get_last_trace(self) -> Optional[ReasoningTrace]:
        return self._last_trace

    @staticmethod
    def extract_final_answer(response: str) -> tuple[str, str]:
        """
        Splits the response into (internal_reasoning, final_answer).
        The reasoning is assumed to be inside <thought> tags.
        """
        thought_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            reasoning = thought_match.group(1).strip()
            final_answer = response.replace(thought_match.group(0), "").strip()
            return reasoning, final_answer
        
        # Fallback if tags are missing or malformed
        if "<thought>" in response.lower():
            parts = re.split(r"</thought>", response, flags=re.IGNORECASE)
            if len(parts) > 1:
                reasoning = parts[0].replace("<thought>", "").strip()
                final_answer = parts[1].strip()
                return reasoning, final_answer
                
        return "", response.strip()
