import json
import logging
import re
from typing import Callable, List, Optional

logger = logging.getLogger("compression_middleware")

GREETING_KEYWORDS = re.compile(
    r'^(hi|hello|hey|heyy|howdy|sup|yo|good morning|good evening|good afternoon|'
    r'morning|evening|what\'?s up)[\s!?.]*$', re.IGNORECASE
)

SIMPLE_QUESTION_KEYWORDS = re.compile(
    r'\b(what is|who is|where is|when is|define|explain|what time|'
    r'how (old|tall|big|far|many|much|long)|'
    r'why (is|are|do|does|did)|tell me about)\b', re.IGNORECASE
)

CODING_KEYWORDS = re.compile(
    r'\b(code|function|implement|write|program|script|algorithm|class|'
    r'def |import |function\s*\(|leetcode|hackerrank|'
    r'sample input|sample output|constraints)\b', re.IGNORECASE
)

CORPORATE_PATTERNS = re.compile(
    r'(?i)'
    r'(?:'
    r'subject\s*:|'
    r'dear\s+(?:user|customer|sir|madam|team|support|\[)|'
    r'welcome to (?:our|the|my)|'
    r'(?:i|we) am? (?:delighted|pleased|excited|happy|thrilled)|'
    r'thank you for (?:your |the )?(?:interest|patience|inquiry|message|email|choosing|contacting|reaching)|'
    r'we value your|'
    r'we appreciate your|'
    r'please do not hesitate|'
    r'(?:please )?feel free to (?:reach out|ask|contact|let me know)|'
    r'contact us (?:today|at|via)|'
    r'best regards|'
    r'kind regards|'
    r'warm regards|'
    r'sincerely|'
    r'yours truly|'
    r'with sincere gratitude|'
    r'look(?:ing)? forward to hearing from you|'
    r'i (?:would|will) (?:be happy|be glad|love) to (?:assist|help|answer)|'
    r'i am here to (?:assist|help|support)|'
    r'(?:my name is|i am|i\'?m)\s*(?:called\s*)?\[|'
    r'\[your name\](?:[\s]*\[)?|'
    r'\[company name\]|'
    r'\[user name\]|'
    r'\[chatbot name\]|'
    r'\[your chatbot|'
    r'\[recipient|'
    r'\[sender|'
    r'\[email|'
    r'\[phone|'
    r'\[address\]|'
    r'\[website\]|'
    r'\[date\]|'
    r'\[insert|'
    r'as a part of my services|'
    r'my name is \[|\bhere are some examples\b|'
    r'please let me know if (?:you|there)|'
    r'i look forward to (?:hearing|assisting|helping|working)'
    r')'
)

FILLER_PATTERNS = re.compile(
    r'(?i)'
    r'(?:'
    r'as an ai (?:assistant|language model)|'
    r'i am an ai|'
    r'i\'?m an ai|'
    r'i would be delighted|'
    r'i can certainly help|'
    r'certainly[!.]?\s*(?:here (?:are|is))|'
    r'of course[,!]|'
    r'absolutely[,!]|'
    r'great question[,!]|'
    r'let me know if you (?:need|have|would like)|'
    r'i hope (?:this|that) (?:helps|answers|clarifies|is helpful)|'
    r'feel free to (?:ask|reach out)|'
    r'don\'?t hesitate|'
    r'in (?:order )?to (?:answer|address) your|'
    r'it is (?:important|worth) (?:noting|mentioning|considering)|'
    r'it\'?s (?:important|worth) to (?:note|mention|consider)|'
    r'as previously (?:mentioned|stated|noted)|'
    r'with that (?:said|in mind)|'
    r'at the end of the day|'
    r'when it comes to|'
    r'i (?:think|believe) that|'
    r'basically[,.]|'
    r'essentially[,.]|'
    r'needless to say|'
    r'it goes without saying|'
    r'i (?:can|will) (?:provide|offer|give) you|'
    r'i (?:can|will) help you (?:with|to|in)|'
    r'if you (?:have|need) any (?:questions|further|other)|'
    r'do(?: n?ot)? hesitate to|'
    r'if there\'?s anything (?:else|more)|'
    r'i would be (?:happy|glad|pleased) to (?:assist|help|answer)|'
    r'it (?:has been|was) a (?:pleasure|great) (?:to|helping)|'
    r'thank you for choosing|'
    r'as a (?:helpful|friendly|knowledgeable|caring) (?:ai|assistant|chatbot)|'
    r'i am (?:here|available) (?:to|if you)|'
    r'please (?:let me know|don\'?t hesitate|reach out|feel free)'
    r')'
)

INTRO_PATTERNS = re.compile(
    r'(?i)^(?:'
    r'sure[!.]?\s*|'
    r'sure thing[!.]?\s*|'
    r'okay[,!.]?\s*|'
    r'ok[,!.]?\s*|'
    r'here (?:is|are|\'?s|goes)[!.]?\s*|'
    r'here\'?s (?:the |what |how |my )|'
    r'let me (?:explain|show|give|provide|help|answer|break down)[^,]*[,.]?\s*|'
    r'i would (?:say|recommend|suggest) that\s*|'
    r'i (?:think|believe) (?:the |that |this )|'
    r'the answer is[!.:]?\s*|'
    r'the solution is[!.:]?\s*|'
    r'in short[,.]?\s*|'
    r'in brief[,.]?\s*|'
    r'to answer your question[,.]?\s*|'
    r'to (?:put it )?simply[,.]?\s*'
    r')'
)

REASONING_CONTEXT_BLOCK = re.compile(
    r'\[Reasoning Context\].*?(?:\n\n|\Z)', re.DOTALL | re.IGNORECASE
)

PLACEHOLDER_LINE = re.compile(
    r'^.*?\[(?:your name|company name|user name|chatbot name|recipient|'
    r'sender|email|phone|address|website|date|insert|your chatbot).*?\].*?$',
    re.IGNORECASE | re.MULTILINE,
)

EMAIL_SIGNATURE_LINE = re.compile(
    r'^(?:best regards|kind regards|warm regards|sincerely|yours truly|'
    r'thanks?(?:\s+(?:and\s+)?regards)?|cheers|thank you)[,.]?\s*$',
    re.IGNORECASE | re.MULTILINE,
)

SUBJECT_LINE = re.compile(r'^subject\s*:.*$', re.IGNORECASE | re.MULTILINE)

CHAT_PATHS = {"/chat/stream", "/chat"}


def detect_intent(message: str) -> str:
    msg = message.strip().lower()
    if GREETING_KEYWORDS.match(msg):
        return "greeting"
    if CODING_KEYWORDS.search(msg):
        return "coding"
    if SIMPLE_QUESTION_KEYWORDS.search(msg):
        return "simple_question"
    if len(msg.split()) <= 5:
        return "simple_question"
    return "technical"


def estimate_limits(intent: str) -> dict:
    limits = {
        "greeting": {"max_words": 12, "max_sentences": 1, "preserve_code": False},
        "simple_question": {"max_words": 80, "max_sentences": 3, "preserve_code": False},
        "coding": {"max_words": 400, "max_sentences": 30, "preserve_code": True},
        "technical": {"max_words": 300, "max_sentences": 10, "preserve_code": False},
    }
    return limits.get(intent, limits["technical"])


def _corporate_bloat_score(text: str) -> float:
    corporate_matches = len(CORPORATE_PATTERNS.findall(text))
    filler_matches = len(FILLER_PATTERNS.findall(text))
    placeholder_lines = len(PLACEHOLDER_LINE.findall(text))
    sig_lines = len(EMAIL_SIGNATURE_LINE.findall(text))
    has_subject = bool(SUBJECT_LINE.search(text))
    has_reasoning = bool(REASONING_CONTEXT_BLOCK.search(text))

    score = 0.0
    score += corporate_matches * 0.15
    score += filler_matches * 0.1
    score += placeholder_lines * 0.3
    score += sig_lines * 0.2
    if has_subject:
        score += 0.5
    if has_reasoning:
        score += 0.6

    return score


class ResponseCompressor:
    def detect_intent(self, message: str) -> str:
        return detect_intent(message)

    def estimate_limits(self, intent: str) -> dict:
        return estimate_limits(intent)

    def compress(self, text: str, message: str) -> str:
        if not text.strip():
            return text

        intent = self.detect_intent(message)
        limits = self.estimate_limits(intent)

        result = text

        result = self._remove_reasoning_context(result)
        result = self._remove_subject_line(result)
        result = self._remove_placeholder_lines(result)
        result = self._remove_email_signatures(result)

        bloat = _corporate_bloat_score(result)
        if bloat > 0.4:
            logger.info(
                "[Compression] High bloat score %.2f — forcing strict truncation", bloat
            )
            limits = self._bloat_limits(intent)

        result = self._remove_corporate_wording(result)
        result = self._remove_filler(result)
        result = self._remove_repetition(result)
        result = self._remove_intros(result)
        result = self._cleanup_whitespace(result)
        result = self._truncate_by_limits(result, limits, intent)
        result = self._enforce_code_first(result, intent)

        return result.strip()

    def _bloat_limits(self, intent: str) -> dict:
        if intent == "greeting":
            return {"max_words": 12, "max_sentences": 1, "preserve_code": False}
        return {"max_words": 60, "max_sentences": 2, "preserve_code": False}

    def _remove_reasoning_context(self, text: str) -> str:
        return REASONING_CONTEXT_BLOCK.sub("", text)

    def _remove_subject_line(self, text: str) -> str:
        return SUBJECT_LINE.sub("", text)

    def _remove_placeholder_lines(self, text: str) -> str:
        return PLACEHOLDER_LINE.sub("", text)

    def _remove_email_signatures(self, text: str) -> str:
        lines = text.split("\n")
        cleaned = []
        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            if EMAIL_SIGNATURE_LINE.match(line.strip()):
                skip_next = True
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def _remove_corporate_wording(self, text: str) -> str:
        return CORPORATE_PATTERNS.sub("", text)

    def _remove_filler(self, text: str) -> str:
        return FILLER_PATTERNS.sub("", text)

    def _remove_repetition(self, text: str) -> str:
        if not text.strip():
            return text
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        seen = set()
        unique = []
        for s in sentences:
            normalized = re.sub(r'[^a-z0-9]', '', s.lower().strip())
            if not normalized or len(normalized) < 8:
                unique.append(s)
                continue
            if normalized not in seen:
                seen.add(normalized)
                unique.append(s)
        return " ".join(unique)

    def _remove_intros(self, text: str) -> str:
        result = text
        while True:
            match = INTRO_PATTERNS.match(result)
            if not match:
                break
            result = result[match.end():].lstrip()
        return result

    def _cleanup_whitespace(self, text: str) -> str:
        result = text
        result = re.sub(r' {2,}', ' ', result)
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r'[ \t]+\n', '\n', result)
        return result.strip()

    def _truncate_by_limits(self, text: str, limits: dict, intent: str) -> str:
        max_words = limits["max_words"]
        max_sentences = limits["max_sentences"]

        word_count = len(text.split())
        if word_count <= max_words:
            return text

        if intent == "greeting":
            words = text.split()
            return " ".join(words[:max_words])

        if intent == "coding" and limits.get("preserve_code"):
            return self._truncate_coding(text, max_words)

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        result = []
        wc = 0
        sc = 0
        for s in sentences:
            sw = len(s.split())
            if wc + sw > max_words or sc >= max_sentences:
                break
            result.append(s)
            wc += sw
            sc += 1
        return " ".join(result) if result else " ".join(text.split()[:max_words])

    def _truncate_coding(self, text: str, max_words: int) -> str:
        blocks = re.split(r'(```[\s\S]*?```)', text)
        prose_parts = []
        code_parts = []
        for block in blocks:
            if block.startswith("```"):
                code_parts.append(block)
            else:
                prose_parts.append(block)

        if not prose_parts:
            return text

        total_words = len(text.split())
        if total_words <= max_words:
            return text

        excess = total_words - max_words
        prose_text = "".join(prose_parts)
        prose_words = prose_text.split()
        if not prose_words:
            return text

        keep_ratio = max(0.3, 1.0 - (excess / len(prose_words)))
        keep_count = max(1, int(len(prose_words) * keep_ratio))
        trimmed_prose = " ".join(prose_words[:keep_count])

        result = trimmed_prose
        for cp in code_parts:
            result += "\n\n" + cp
        return result.strip()

    def _enforce_code_first(self, text: str, intent: str) -> str:
        if intent != "coding":
            return text
        blocks = re.findall(r'```[\s\S]*?```', text)
        if not blocks:
            return text
        code_blocks = "\n\n".join(blocks)
        non_code = re.sub(r'```[\s\S]*?```', '', text).strip()
        if non_code:
            return code_blocks + "\n\n" + non_code
        return code_blocks


class StrictCompressionMiddleware:
    """ASGI middleware that compresses LLM responses based on intent.

    1. Intercepts request body to get user message
    2. Intercepts response body to get full response text
    3. Detects intent and enforces hard word limits
    4. Removes filler, repetition, corporate wording, intros
    5. Detects corporate bloat and forces strict truncation
    6. Emits compressed version as SSE event
    """

    def __init__(self, app: Callable):
        self.app = app
        self.compressor = ResponseCompressor()

    async def __call__(self, scope: dict, receive: Callable, send: Callable):
        if scope["type"] != "http" or scope.get("path") not in CHAT_PATHS:
            await self.app(scope, receive, send)
            return

        request_body_chunks = []
        user_message = ""

        async def receive_intercept() -> dict:
            nonlocal user_message
            msg = await receive()
            if msg["type"] == "http.request":
                chunk = msg.get("body", b"")
                request_body_chunks.append(chunk)
                if not msg.get("more_body", False):
                    raw = b"".join(request_body_chunks)
                    user_message = self._extract_message_from_request(raw)
                    logger.debug(
                        "[Compression] Extracted user message: %.80s", user_message
                    )
            return msg

        response_chunks = []
        response_started = False
        response_status = 200
        response_headers = []

        async def send_intercept(message: dict):
            nonlocal response_started, response_status, response_headers
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
                response_headers = message["headers"]
            elif message["type"] == "http.response.body":
                response_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    compressed = self._process_response(
                        b"".join(response_chunks), user_message
                    )
                    await send({
                        "type": "http.response.start",
                        "status": response_status,
                        "headers": response_headers,
                    })
                    await send({
                        "type": "http.response.body",
                        "body": compressed,
                        "more_body": False,
                    })

        await self.app(scope, receive_intercept, send_intercept)

    def _extract_message_from_request(self, raw: bytes) -> str:
        try:
            data = json.loads(raw.decode("utf-8"))
            return data.get("message", "")
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
            return ""

    def _process_response(self, raw: bytes, user_message: str) -> bytes:
        if not raw or not user_message:
            if not user_message:
                logger.warning("[Compression] No user message extracted, skipping")
            return raw

        try:
            body_str = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw

        is_sse = b"text/event-stream" in raw[:1024] or b"data: " in raw[:1024]
        if is_sse:
            return self._compress_sse(body_str, user_message)
        return self._compress_json(body_str, user_message)

    def _compress_sse(self, body: str, user_message: str) -> bytes:
        lines = body.split("\n")
        full_text = ""
        refined_text = None
        done = False
        compressed_already = False

        for line in lines:
            if line.startswith("data: "):
                data = line[6:].strip()
                if data == "[DONE]":
                    done = True
                else:
                    try:
                        parsed = json.loads(data)
                        if parsed.get("compressed"):
                            compressed_already = True
                            break
                        if parsed.get("refined") and "full" in parsed:
                            refined_text = parsed["full"]
                        if "content" in parsed:
                            full_text += parsed["content"]
                    except json.JSONDecodeError:
                        pass

        if done and full_text and not compressed_already:
            target = refined_text if refined_text else full_text
            original_wc = len(target.split())
            compressed = self.compressor.compress(target, user_message)
            compressed_wc = len(compressed.split())
            if compressed != target and compressed.strip():
                logger.info(
                    "[Compression] %d→%d words for intent=%s | bloat=%.2f",
                    original_wc, compressed_wc,
                    self.compressor.detect_intent(user_message),
                    _corporate_bloat_score(target),
                )
                compressed_event = (
                    f"data: {json.dumps({'content': '', 'compressed': True, 'full': compressed})}\n\n"
                )
                body += "\n" + compressed_event
            else:
                logger.debug(
                    "[Compression] No change (%d words) for intent=%s",
                    original_wc, self.compressor.detect_intent(user_message),
                )

        return body.encode("utf-8")

    def _compress_json(self, body: str, user_message: str) -> bytes:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and "response" in parsed:
                original = parsed["response"]
                compressed = self.compressor.compress(original, user_message)
                if compressed != original:
                    parsed["response"] = compressed
                    parsed["compressed"] = True
                    logger.info(
                        "[Compression] JSON: %d→%d words",
                        len(original.split()), len(compressed.split()),
                    )
                    return json.dumps(parsed).encode("utf-8")
        except (json.JSONDecodeError, TypeError):
            pass
        return body.encode("utf-8")
