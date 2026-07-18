from __future__ import annotations

import re
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.response_ranker import ResponseRanker
from backend.dsa_expert import DSAExpert, ExecutionTracer
from backend.hallucination_guard import HallucinationGuard
from backend.validation_engine import ResponseValidationEngine
from backend.url_verifier import URLVerifier, ResponseURLReport
from backend.refinement_middleware import ResponseRefinementMiddleware

logger = logging.getLogger("reasoning_pipeline")
logger.setLevel(logging.DEBUG)


@dataclass
class ReasoningTrace:
    intent: str = ""
    intent_category: str = "general"
    coding_sub_intent: str = ""
    is_coding_challenge: bool = False
    reasoning_strategy: str = "step_by_step"
    response_depth: str = "intermediate"
    response_mode: str = "normal"
    output_structure: str = "default"
    steps: list[str] = field(default_factory=list)
    draft_issues: list[str] = field(default_factory=list)
    refinement_applied: bool = False
    latency_ms: float = 0.0


class IntentClassifier:
    _PATTERNS: list[tuple[str, str]] = [
        (r"\b(hello|hi|hey|good morning|good evening|howdy|greetings|sup|yo)\b", "casual_chat"),
        (r"\b(debug|fix the bug|traceback|stack trace|exception|why is this failing|not working as expected|diagnose|runtime error|syntax error)\b", "debugging"),
        (r"\b(architecture|design|structure|system design|scalability|blueprint|infrastructure)\b", "architecture"),
        (r"\b(optimize|make it faster|performance|efficient|bottleneck|refactor)\b", "optimization"),
        (r"\b(explain|how does|what is|define|clarify|elaborate|walk me through)\b", "explanation"),
        (r"\b(solve|implement|write code|code for|function|class|program|challenge|problem)\b", "coding_problem"),
        (r"\b(document|pdf|file|read this|search in|find in document)\b", "document_query"),
        (r"\b(what time is it|current time|today'?s date|what'?s the date|current date|what day is|time in |timestamp)\b", "realtime_query"),
    ]

    def classify(self, message: str) -> tuple[str, str]:
        has_code = False
        if "```" in message:
            has_code = True
        else:
            code_signatures = [
                r"\bdef \w+\(", r"\bclass \w+:", r"\bimport \w+",
                r"\bfunction\s*\(?", r"\bconst \w+\s*=", r"=>",
                r"\bpublic static void", r"\bSystem\.out",
                r"#include\s*<", r"\bstd::",
                r"SELECT .* FROM", r"UPDATE .* SET",
            ]
            has_code = any(re.search(p, message, re.IGNORECASE) for p in code_signatures)

        msg_lower = message.lower()
        for pattern, category in self._PATTERNS:
            if re.search(pattern, msg_lower):
                if has_code and category not in ["coding_problem", "debugging", "explanation", "optimization"]:
                    return "User provided code and is requesting assistance.", "debugging"
                return f"User is making a {category} request.", category

        if has_code:
            return "User provided a code snippet. Assumed coding challenge.", "coding_problem"

        return "User is making a general request.", "general"


class InternalPlanner:
    _DEPTH_INDICATORS = {
        "beginner": r"\b(beginner|basic|simple|eli5|easy|start|introduction|learn)\b",
        "advanced": r"\b(advanced|complex|deep dive|professional|expert|senior|optimized|high performance)\b",
    }

    _TEMPLATES: dict[str, list[str]] = {
        "coding_problem": [
            "Identify this as a coding challenge and lock into CODE SOLVER MODE.",
            "Identify required output structure: Approach -> Code -> Complexity.",
            "Determine reasoning strategy: Algorithmic decomposition and complexity analysis.",
            "Generate structured draft response with complete compilable code.",
            "Strictly follow the MANDATORY RESPONSE FORMAT.",
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
            "Respond in 1-2 short sentences.",
            "Be conversational, not formal.",
            "No letters, no closings, no sign-offs.",
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
        return self._TEMPLATES.get(category, self._TEMPLATES["general"])

    def plan(self, message: str, category: str, is_challenge: bool = False) -> tuple[ReasoningTrace, list[str]]:
        trace = ReasoningTrace(intent_category=category, is_coding_challenge=is_challenge)

        msg_lower = message.lower()
        for depth, pattern in self._DEPTH_INDICATORS.items():
            if re.search(pattern, msg_lower):
                trace.response_depth = depth
                break

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

        steps = self.decompose(category, is_challenge=is_challenge)
        return trace, steps


class PromptAugmenter:
    def augment_messages(
        self,
        messages: list[dict[str, str]],
        trace: ReasoningTrace,
    ) -> list[dict[str, str]]:
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

        new_messages = [m.copy() for m in messages]
        if new_messages and new_messages[0]["role"] == "system":
            new_messages[0]["content"] += planning_block

        return new_messages


class ResponseRewriter:
    _ROBOTIC_PHRASES = [
        (re.compile(r'As an AI(?: assistant|language model)?[,.]?\s*', re.IGNORECASE), ''),
        (re.compile(r'I would be delighted to[,.]?\s*', re.IGNORECASE), ''),
        (re.compile(r'Certainly(?:! Here are several options\.\.\.|[,!])\s*', re.IGNORECASE), ''),
        (re.compile(r'Based on your query[,.]?\s*', re.IGNORECASE), ''),
        (re.compile(r'Of course[,!]\s*', re.IGNORECASE), ''),
        (re.compile(r'Absolutely[,!]\s*', re.IGNORECASE), ''),
        (re.compile(r'Great question[,!]\s*', re.IGNORECASE), ''),
        (re.compile(r'(?:I am|I\'m) (?:an|the) AI(?: assistant| language model)?[,.]?\s*', re.IGNORECASE), ''),
        (re.compile(r'(?:Thank you|Thanks) for (?:your |reaching out|contacting|the )(?:question|message|query|inquiry)[!.]?\s*', re.IGNORECASE), ''),
        (re.compile(r'(?:I hope|Hope) (?:this|that) (?:helps|answers your question|clarifies things|is what you were looking for)[!.]?\s*', re.IGNORECASE), ''),
        (re.compile(r'(?:Please )?(?:feel free|don\'t hesitate) to (?:ask|reach out|let me know)[^.]*[!.]?\s*', re.IGNORECASE), ''),
        (re.compile(r'(?:It\'s|It is) a (?:pleasure|great) to (?:assist|help)[^.]*[!.]?\s*', re.IGNORECASE), ''),
        (re.compile(r'(?:Greetings|Salutations)[!.]?\s*', re.IGNORECASE), ''),
    ]

    def rewrite(self, response: str, trace: ReasoningTrace, issues: list[str]) -> str:
        result = response.strip()
        if not result:
            return result

        for pattern, replacement in self._ROBOTIC_PHRASES:
            result = pattern.sub(replacement, result)

        if "Incomplete markdown code block" in " ".join(issues):
            if result.count("```") % 2 != 0:
                result += "\n```"

        if trace.response_depth == "beginner":
            pass
        elif trace.response_depth == "advanced":
            pass

        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {2,}', ' ', result)

        if result and not result[0].isupper() and trace.response_mode != "locked_solver":
            result = result[0].upper() + result[1:]

        return result.strip()

    def _basic_cleanup(self, response: str) -> str:
        result = response.strip()
        if not result:
            return result
        for pattern, replacement in self._ROBOTIC_PHRASES:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {2,}', ' ', result)
        return result


class ReasoningPipeline:
    _coding_classifier = None

    def __init__(self, refinement_threshold: float = 0.60):
        self.classifier = IntentClassifier()
        self.planner = InternalPlanner()
        self.augmenter = PromptAugmenter()
        self.rewriter = ResponseRewriter()
        self.ranker = ResponseRanker()
        self.dsa_expert = DSAExpert()
        self.tracer = ExecutionTracer()
        self.guard = HallucinationGuard()
        self.validator = ResponseValidationEngine()
        self.url_verifier = URLVerifier()
        self.refinement_middleware = ResponseRefinementMiddleware()
        self.threshold = refinement_threshold
        self._last_trace: Optional[ReasoningTrace] = None

    @classmethod
    def _get_coding_classifier(cls):
        if cls._coding_classifier is None:
            from backend.coding_intent import CodingIntentClassifier
            cls._coding_classifier = CodingIntentClassifier()
        return cls._coding_classifier

    def prepare_messages(self, user_message: str, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], ReasoningTrace]:
        t0 = time.perf_counter()

        intent_summary, category = self.classifier.classify(user_message)

        is_challenge = False
        coding_sub_intent = ""
        if category in ["coding_problem", "debugging"]:
            coding_classifier = self._get_coding_classifier()
            coding_intent = coding_classifier.classify(user_message)
            if coding_intent.is_coding:
                coding_sub_intent = coding_intent.sub_intent
                is_challenge = coding_intent.is_coding_challenge

        trace, steps = self.planner.plan(user_message, category, is_challenge)
        trace.intent = intent_summary
        trace.coding_sub_intent = coding_sub_intent

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

        augmented = self.augmenter.augment_messages(messages, trace)

        trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._last_trace = trace
        # Also set request-scoped context variable for thread-safe audit logging
        try:
            from backend.response_middleware import request_trace_var
            request_trace_var.set(trace)
        except ImportError:
            pass

        logger.info(
            "[ReasoningPipeline] Intent=%s | Depth=%s | Mode=%s | Latency=%.1fms",
            category, trace.response_depth, trace.response_mode, trace.latency_ms,
        )
        return augmented, trace

    def prepare_prompt(self, user_message: str, raw_context: str) -> tuple[str, ReasoningTrace]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]
        augmented_messages, trace = self.prepare_messages(user_message, messages)
        planning_block = augmented_messages[0]["content"].split("[INTERNAL PLANNING]")[1]
        augmented_prompt = f"[REASONING]\n[INTERNAL PLANNING]{planning_block}\n\n{raw_context}"
        return augmented_prompt, trace

    def refine(
        self,
        user_message: str,
        draft_response: str,
        trace: Optional[ReasoningTrace] = None,
        candidates: Optional[list[str]] = None,
        context: str = "",
        allow_regeneration: bool = True,
    ) -> tuple[str, ReasoningTrace]:
        if trace is None:
            trace = self._last_trace or ReasoningTrace()

        t0 = time.perf_counter()

        if candidates and len(candidates) > 1:
            draft_response = self.ranker.rank(candidates, user_message, context, trace.steps)

        if self.url_verifier.has_any_urls(draft_response):
            trace.steps.append("URL verification: checking links in response")

        report = self.validator.validate(
            draft_response,
            context=context,
            expected_mode=trace.response_mode,
        )
        trace.draft_issues = [i.message for i in report.issues]

        h_report = self.guard.evaluate(draft_response, context)
        if not h_report.is_grounded or h_report.confidence_score < 0.5:
            logger.warning("[ReasoningPipeline] Low confidence: %.3f", h_report.confidence_score)
            draft_response = self.guard.handle_uncertainty(draft_response, h_report, trace.intent_category)
            trace.draft_issues.extend(h_report.contradictions)
        if h_report.url_issues:
            trace.draft_issues.extend(h_report.url_issues)
            logger.warning("[ReasoningPipeline] URL issues: %s", h_report.url_issues)

        if report.needs_regeneration and allow_regeneration:
            logger.warning("[ReasoningPipeline] Critical issues — regeneration signaled.")
            trace.refinement_applied = True

        internal_reasoning, clean_response = self.extract_final_answer(draft_response)
        if internal_reasoning:
            logger.info("[ReasoningPipeline] Extracted reasoning (%d chars)", len(internal_reasoning))
            draft_response = clean_response

        if self.url_verifier.has_any_urls(draft_response):
            if h_report.url_issues:
                cleaned, url_warnings = self.url_verifier.sanitize_response(draft_response, ResponseURLReport(urls=[]))
                for w in url_warnings:
                    trace.draft_issues.append(w)
                draft_response = cleaned
                trace.refinement_applied = True

        if not report.is_valid:
            final = self.validator.repair(draft_response, report)
            final = self.rewriter.rewrite(final, trace, [i.message for i in report.issues])
            trace.refinement_applied = True
            logger.info("[ReasoningPipeline] Repair applied | Issues=%d", len(report.issues))
        else:
            final = self.rewriter._basic_cleanup(draft_response)
            trace.refinement_applied = False

        final = self.refinement_middleware.refine_response(user_message, final)

        trace.latency_ms += round((time.perf_counter() - t0) * 1000, 2)
        return final, trace

    def get_last_trace(self) -> Optional[ReasoningTrace]:
        return self._last_trace

    @staticmethod
    def extract_final_answer(response: str) -> tuple[str, str]:
        thought_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            reasoning = thought_match.group(1).strip()
            final_answer = response.replace(thought_match.group(0), "").strip()
            return reasoning, final_answer

        if "<thought>" in response.lower():
            parts = re.split(r"</thought>", response, flags=re.IGNORECASE)
            if len(parts) > 1:
                reasoning = parts[0].replace("<thought>", "").strip()
                final_answer = parts[1].strip()
                return reasoning, final_answer

        return "", response.strip()
