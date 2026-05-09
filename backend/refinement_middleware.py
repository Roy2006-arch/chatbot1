"""
refinement_middleware.py
=======================
8-step pipeline + ResponseScorer + ASGI middleware for response quality.

Pipeline:
1. Detect user intent
2. Estimate ideal response length
3. Generate draft response (intercepted from LLM)
4. Remove repetitive text
5. Remove filler sentences
6. Shorten overlong responses
7. Improve clarity
8. Return polished answer
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("refinement_middleware")


# ─────────────────────────────────────────────────────────
#  ResponseScorer — scores responses on quality metrics
# ─────────────────────────────────────────────────────────

@dataclass
class ResponseScore:
    conciseness: float = 1.0
    naturalness: float = 1.0
    informativeness: float = 1.0
    overall: float = 1.0
    issues: List[str] = field(default_factory=list)


class ResponseScorer:
    ROBOTIC_PATTERNS = re.compile(
        r'(?i)'
        r'(?:'
        r'as an ai(?: assistant| language model)?[,.]?\s*|'
        r'i am an ai[,.]?\s*|'
        r'i\'?m an ai(?: assistant)?[,.]?\s*|'
        r'i would be delighted[,.]?\s*|'
        r'i can certainly help with that[!.]?\s*|'
        r'certainly[!.]\s*(?:here (?:are|is))|'
        r'of course[,!]\s*|'
        r'absolutely[,!]\s*|'
        r'great question[,!]\s*|'
        r'let me know if you need anything else[!.]?\s*|'
        r'i hope this helps[!.]?\s*|'
        r'feel free to ask[^.]*[!.]?\s*|'
        r'(?:thank you|thanks) for (?:your |the )?(?:question|message|query)[!.]?\s*|'
        r'greetings[!.]?\s*|'
        r'salutations[!.]?\s*'
        r')'
    )

    FILLER_PATTERNS = re.compile(
        r'(?i)'
        r'(?:'
        r'in (?:order )?to (?:answer|address) your (?:question|query|concern)|'
        r'it is (?:important|worth) (?:noting|mentioning) that|'
        r"it(?:'?s)? (?:important|worth) to (?:note|mention) that|"
        r'as (?:previously|mentioned|stated) (?:above|before|earlier)|'
        r'at the end of the day|'
        r'when it comes to|'
        r'in (?:my |our )?(?:opinion|view|experience)|'
        r'i (?:think|believe|would say) that|'
        r'basically[,.]|'
        r'essentially[,.]|'
        r'honestly[,.]|'
        r'literally[,.]|'
        r'simply put[,.]|'
        r'needless to say|'
        r'it goes without saying|'
        r'as a matter of fact|'
        r'let me (?:explain|break down|elaborate|clarify)|'
        r'if that makes sense|'
        r'does that (?:answer|help|clarify)|'
        r'i hope (?:this|that) (?:helps|answers|clarifies)|'
        r'please let me know if|'
        r'don\'?t hesitate to|'
        r'with that (?:said|in mind)|'
        r'all things considered|'
        r'more often than not|'
        r'in the (?:end|final analysis)'
        r')'
    )

    GREETING_SENTENCES = re.compile(
        r'(?i)^(?:'
        r'(?:hello|hi|hey|greetings|salutations)[!.]?\s*$|'
        r'(?:good|great) (?:morning|afternoon|evening)[!.]?\s*$|'
        r'(?:how (?:are you|is it going|are things))[!.]?\s*$|'
        r'(?:nice|great|good) to (?:meet you|see you|hear from you)[!.]?\s*$'
        r')'
    )

    def score(self, text: str, intent: str) -> ResponseScore:
        score = ResponseScore()
        issues = []

        # --- Conciseness: penalize verbosity relative to intent ---
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text)) or 1

        ideal_lengths = {"greeting": 10, "factual": 60, "coding": 300, "opinion": 80, "general": 60}
        ideal = ideal_lengths.get(intent, 60)

        if word_count > ideal * 2:
            ratio = max(0.0, 1.0 - (word_count - ideal) / (ideal * 3))
            score.conciseness = round(ratio, 3)
            issues.append(f"overlength: {word_count} words (ideal {ideal})")
        elif word_count < 3 and intent not in ("greeting",):
            score.conciseness = 0.3
            issues.append("too_short")
        else:
            score.conciseness = min(1.0, ideal / max(word_count, 1))

        # --- Naturalness: penalize robotic phrases ---
        robotic_matches = self.ROBOTIC_PATTERNS.findall(text)
        if robotic_matches:
            penalty = min(1.0, len(robotic_matches) * 0.25)
            score.naturalness = round(max(0.0, 1.0 - penalty), 3)
            issues.append(f"robotic_phrases: {len(robotic_matches)}")
        else:
            score.naturalness = 1.0

        # --- Informativeness: penalize filler and greeting-only responses ---
        filler_matches = self.FILLER_PATTERNS.findall(text)
        filler_penalty = min(1.0, len(filler_matches) * 0.15)
        score.informativeness = round(max(0.0, 1.0 - filler_penalty), 3)
        if filler_matches:
            issues.append(f"filler_sentences: {len(filler_matches)}")

        if intent == "greeting" and word_count > 15:
            score.informativeness = max(0.0, score.informativeness - 0.3)
            issues.append("greeting_too_long")

        # Check if response is mostly greeting sentences when user asked something else
        if intent != "greeting":
            lines = [l.strip() for l in text.split('.') if l.strip()]
            greeting_lines = sum(1 for l in lines if self.GREETING_SENTENCES.match(l.strip()))
            if len(lines) > 0 and greeting_lines / len(lines) > 0.3:
                score.informativeness = max(0.0, score.informativeness - 0.2)
                issues.append("excess_greeting_in_response")

        # --- Overall ---
        score.overall = round(
            0.35 * score.conciseness +
            0.35 * score.naturalness +
            0.30 * score.informativeness,
            3,
        )
        score.issues = issues
        return score


# ─────────────────────────────────────────────────────────
#  ResponseRefinementMiddleware — 8-step pipeline
# ─────────────────────────────────────────────────────────

class ResponseRefinementMiddleware:
    def __init__(self):
        self.scorer = ResponseScorer()

        self.intent_rules = {
            "greeting": {"max_sentences": 1, "max_words": 12},
            "factual": {"max_sentences": 4, "max_words": 80},
            "coding": {"max_sentences": 15, "max_words": 400},
            "opinion": {"max_sentences": 5, "max_words": 100},
            "general": {"max_sentences": 4, "max_words": 80},
        }

        self.robotic_patters = ResponseScorer.ROBOTIC_PATTERNS

    # ── Step 1: Detect user intent ──
    def _detect_intent(self, prompt: str) -> str:
        p = prompt.lower().strip()
        if re.match(r'^(hi|hello|hey|heyy|howdy|sup|yo|good morning|good evening|good afternoon|morning|evening)[\s!?.]*$', p):
            return "greeting"
        if re.search(r'\b(hi|hello|hey|how are you|how\'?s it going|what\'?s up)\b', p):
            return "greeting"
        if re.search(r'\b(debug|bug|error|exception|traceback|not working|fix|broken|crash|fail|issue)\b', p):
            return "debugging"
        if re.search(r'\b(code|function|implement|write|program|script|algorithm|class|def |import |function\s*\()', p):
            return "coding"
        if re.search(r'\b(architecture|design pattern|system design|scalab|microservice|distributed)\b', p):
            return "architecture"
        if re.search(r'\b(explain|what is|define|how does|describe|clarify|elaborate|walk me through)\b', p):
            return "factual"
        if re.search(r'\b(optimize|faster|performance|slow|efficient|refactor|bottleneck)\b', p):
            return "optimization"
        if re.search(r'\b(think|opinion|recommend|should|suggest|best|prefer)\b', p):
            return "opinion"
        return "general"

    # ── Step 2: Estimate ideal response length ──
    def _estimate_length(self, intent: str) -> dict:
        return self.intent_rules.get(intent, self.intent_rules["general"])

    # ── Step 4: Remove repetitive content ──
    def _remove_repetitive(self, text: str) -> str:
        if not text.strip():
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        seen = []
        unique = []

        for s in sentences:
            normalized = re.sub(r'[^a-z0-9]', '', s.lower().strip())
            if not normalized or len(normalized) < 5:
                unique.append(s)
                continue
            is_dup = False
            for prev in seen:
                if self._similarity(normalized, prev) > 0.75:
                    is_dup = True
                    break
            if not is_dup:
                seen.append(normalized)
                unique.append(s)

        return " ".join(unique)

    def _similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        longer, shorter = (a, b) if len(a) > len(b) else (b, a)
        if len(longer) < 5:
            return 1.0 if a == b else 0.0
        matches = sum(1 for i in range(len(shorter)) if shorter[i] == longer[i])
        return matches / len(longer)

    # ── Step 5: Remove filler sentences ──
    def _remove_filler(self, sentences: List[str], intent: str) -> List[str]:
        cleaned = []
        for s in sentences:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            if intent == "greeting" and len(s_stripped.split()) > 15:
                continue
            if ResponseScorer.FILLER_PATTERNS.search(s_stripped):
                continue
            cleaned.append(s_stripped)
        return cleaned

    # ── Step 6: Shorten overlong responses ──
    def _shorten(self, text: str, intent: str) -> str:
        limits = self._estimate_length(intent)
        word_limit = limits["max_words"]
        sentence_limit = limits["max_sentences"]

        word_count = len(text.split())
        if word_count <= word_limit:
            return text

        # For coding intents, preserve code blocks while trimming prose
        if intent == "coding":
            return self._shorten_coding(text, word_limit)

        # For greetings, hard truncate
        if intent == "greeting":
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return sentences[0] if sentences else text[:word_limit]

        # General shortening: keep first N sentences within word limit
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        result = []
        word_count = 0
        for s in sentences[:sentence_limit]:
            sw = len(s.split())
            if word_count + sw > word_limit:
                break
            result.append(s)
            word_count += sw

        return " ".join(result) if result else text[:word_limit]

    def _shorten_coding(self, text: str, word_limit: int) -> str:
        blocks = re.split(r'(```[\s\S]*?```)', text)
        code_chars = 0
        prose_chars = 0
        prose_parts = []
        code_parts = []

        for block in blocks:
            if block.startswith("```"):
                code_parts.append(block)
                code_chars += len(block)
            else:
                prose_parts.append(block)
                prose_chars += len(block)

        if not prose_parts:
            return text

        # Trim prose to fit within limit
        total_words = len(text.split())
        if total_words <= word_limit:
            return text

        excess = total_words - word_limit
        prose_text = "".join(prose_parts)
        prose_words = prose_text.split()
        if not prose_words:
            return text

        keep_ratio = max(0.3, 1.0 - (excess / len(prose_words)))
        keep_count = max(1, int(len(prose_words) * keep_ratio))
        trimmed_prose = " ".join(prose_words[:keep_count])

        # Reconstruct
        result = trimmed_prose
        for cp in code_parts:
            result += "\n\n" + cp

        return result.strip()

    # ── Step 7: Improve clarity ──
    def _improve_clarity(self, text: str) -> str:
        if not text.strip():
            return text

        result = text

        # Remove robotic phrases
        result = self.robotic_patters.sub("", result)

        # Fix leading punctuation from removals
        result = re.sub(r'^\s*[,.;:!?\s]+', '', result)

        # Clean multiple spaces
        result = re.sub(r' {2,}', ' ', result)

        # Clean excessive newlines (preserve code blocks)
        code_blocks = re.findall(r'```.*?```', result, re.DOTALL)
        placeholders = {}
        for i, cb in enumerate(code_blocks):
            key = f"__CODEBLOCK_{i}__"
            placeholders[key] = cb
            result = result.replace(cb, key)

        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r'[ \t]+\n', '\n', result)

        for key, cb in placeholders.items():
            result = result.replace(key, cb)

        result = result.strip()

        # Ensure first char is uppercase
        if result and result[0].islower():
            result = result[0].upper() + result[1:]

        return result

    # ── Main entry point: 8-step pipeline ──
    def refine_response(self, user_prompt: str, draft_response: str) -> str:
        if not draft_response.strip():
            return draft_response

        # Step 1: Detect user intent
        intent = self._detect_intent(user_prompt)

        # Step 2: Estimate ideal response length (used in step 6)
        limits = self._estimate_length(intent)

        # Step 3: Draft response provided as input (intercepted)

        # Step 4: Remove repetitive text
        no_reps = self._remove_repetitive(draft_response)

        # Step 5: Remove filler sentences
        sentences = re.split(r'(?<=[.!?])\s+', no_reps.strip())
        focused = self._remove_filler(sentences, intent)
        focused_text = " ".join(focused)

        # Step 6: Shorten overlong responses
        shortened = self._shorten(focused_text, intent)

        # Step 7: Improve clarity
        polished = self._improve_clarity(shortened)

        # Score the result
        score = self.scorer.score(polished, intent)
        if score.overall < 0.4:
            logger.warning(
                "[Refinement] Low score %.3f for intent=%s: %s",
                score.overall, intent, score.issues,
            )

        # Step 8: Return polished answer
        return polished


# ─────────────────────────────────────────────────────────
#  ResponseRefinementASGIMiddleware — FastAPI ASGI middleware
# ─────────────────────────────────────────────────────────

CHAT_PATHS = {"/chat/stream", "/chat"}


class ResponseRefinementASGIMiddleware:
    def __init__(self, app: Callable):
        self.app = app
        self._refiner = ResponseRefinementMiddleware()
        self._scorer = ResponseScorer()

    async def __call__(self, scope: dict, receive: Callable, send: Callable):
        if scope["type"] != "http" or scope.get("path") not in CHAT_PATHS:
            await self.app(scope, receive, send)
            return

        async def send_intercept(message: dict):
            if message["type"] == "http.response.start":
                await send(message)
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                more = message.get("more_body", False)
                if more:
                    await send(message)
                else:
                    modified = await self._refine_last_chunk(chunk)
                    await send({
                        "type": "http.response.body",
                        "body": modified,
                        "more_body": False,
                    })

        await self.app(scope, receive, send_intercept)

    async def _refine_last_chunk(self, chunk: bytes) -> bytes:
        if not chunk:
            return chunk

        try:
            body_str = chunk.decode("utf-8")
        except UnicodeDecodeError:
            return chunk

        if not (b"data: " in chunk[:1024]):
            return chunk

        full_text = ""
        refined_already = False
        done = False

        for line in body_str.split("\n"):
            if line.startswith("data: "):
                data = line[6:].strip()
                if data == "[DONE]":
                    done = True
                else:
                    try:
                        parsed = json.loads(data)
                        if "full" in parsed and parsed.get("refined"):
                            refined_already = True
                        elif "content" in parsed:
                            full_text += parsed["content"]
                    except json.JSONDecodeError:
                        pass

        if done and full_text and not refined_already:
            refined = self._refiner.refine_response("", full_text)
            if refined != full_text and refined.strip():
                event = json.dumps({"content": "", "refined": True, "full": refined})
                chunk += f"\ndata: {event}\n\n".encode("utf-8")

        return chunk


# ── Convenience function for scoring ──
def score_response(text: str, intent: Optional[str] = None) -> dict:
    scorer = ResponseScorer()
    if not intent:
        refiner = ResponseRefinementMiddleware()
        intent = refiner._detect_intent(text)
    score = scorer.score(text, intent)
    return {
        "conciseness": score.conciseness,
        "naturalness": score.naturalness,
        "informativeness": score.informativeness,
        "overall": score.overall,
        "issues": score.issues,
    }
