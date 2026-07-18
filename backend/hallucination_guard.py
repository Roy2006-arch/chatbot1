import asyncio
import re
import logging
from typing import List, Optional
from dataclasses import dataclass

from backend.url_verifier import URLVerifier

logger = logging.getLogger("hallucination_guard")


@dataclass
class HallucinationReport:
    confidence_score: float
    is_grounded: bool
    contradictions: List[str]
    uncertainty_detected: bool
    url_issues: List[str] = None

    def __post_init__(self):
        if self.url_issues is None:
            self.url_issues = []


class HallucinationGuard:
    def __init__(self, confidence_threshold: float = 0.70):
        self.threshold = confidence_threshold
        self.url_verifier = URLVerifier()

    def evaluate(self, response: str, context: str) -> HallucinationReport:
        if not context:
            report = self._internal_consistency_check(response)
            self._check_response_urls(response, report)
            return report

        grounding_score = self._calculate_grounding_score(response, context)
        contradictions = self._detect_contradictions(response, context)
        uncertainty = self._detect_uncertainty(response)
        url_issues = self._check_response_urls_sync(response)

        penalty = 0.4 if contradictions else 0.0
        if re.search(r'\b(user:|assistant:|system:)\b', response, re.IGNORECASE):
            penalty += 0.3
        if url_issues:
            penalty += 0.2

        confidence = max(0.0, grounding_score - penalty)

        return HallucinationReport(
            confidence_score=confidence,
            is_grounded=grounding_score > 0.5 and not contradictions,
            contradictions=contradictions,
            uncertainty_detected=uncertainty or confidence < 0.5,
            url_issues=url_issues,
        )

    def _calculate_grounding_score(self, response: str, context: str) -> float:
        context_tokens = set(re.findall(r'\w+', context.lower()))
        res_tokens = set(re.findall(r'\w+', response.lower()))

        STOP_WORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'that', 'this', 'these', 'those', 'and', 'but', 'or', 'nor',
            'not', 'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
            'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
            'if', 'when', 'while', 'where', 'how', 'what', 'which', 'who',
            'whom', 'whose', 'why', 'it', 'its', 'i', 'me', 'my', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
            'their', 'about', 'up', 'also', 'here', 'there', 'now', 'still',
            'well', 'back', 'even', 'way', 'much', 'make', 'like', 'get',
            'know', 'take', 'come', 'think', 'see', 'want', 'look', 'use',
            'find', 'give', 'tell', 'work', 'call', 'try', 'ask', 'need',
            'feel', 'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin',
            'seem', 'help', 'show', 'hear', 'play', 'run', 'move', 'live',
            'believe', 'bring', 'happen', 'write', 'provide', 'sit', 'stand',
            'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn',
            'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create',
            'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk',
            'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy',
            'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall',
            'cut', 'reach', 'kill', 'remain', 'suggest', 'raise', 'pass', 'sell',
            'require', 'report', 'decide', 'pull', 'return', 'explain', 'hope',
            'develop', 'carry', 'break', 'receive', 'agree', 'support', 'hold',
            'produce', 'eat', 'apply', 'cover', 'choose', 'start', 'point',
            'type', 'also', 'very', 'often', 'however', 'too', 'usually',
            'really', 'already', 'since', 'long', 'around', 'sure', 'yet',
            'code', 'function', 'class', 'return', 'def', 'import', 'use',
            'data', 'value', 'string', 'number', 'list', 'array', 'item',
            'element', 'result', 'example', 'problem', 'solution', 'case',
        }

        meaningful_res = {t for t in res_tokens if len(t) > 3 and t not in STOP_WORDS}
        if not meaningful_res:
            return 1.0

        supported = meaningful_res.intersection(context_tokens)
        base_score = len(supported) / len(meaningful_res)

        key_phrase_bonus = 0.0
        context_lower = context.lower()
        response_lower = response.lower()
        key_phrases = re.findall(r'\b\w+(?:\s+\w+){2,5}\b', context_lower)
        matched_phrases = sum(1 for phrase in key_phrases if phrase in response_lower)
        if key_phrases:
            key_phrase_bonus = min(0.15, matched_phrases / len(key_phrases) * 0.15)

        return min(1.0, base_score + key_phrase_bonus)

    def _detect_contradictions(self, response: str, context: str) -> List[str]:
        contradictions = []
        context_sentences = re.split(r'[.!?]\s+', context.lower())
        response_sentences = re.split(r'[.!?]\s+', response.lower())

        for res_sent in response_sentences:
            if " not " in res_sent or " no " in res_sent or " never " in res_sent:
                positive_sent = res_sent.replace(" not ", " ").replace(" no ", " ").replace(" never ", " ")
                words = set(re.findall(r'\w+', positive_sent))
                for ctx_sent in context_sentences:
                    ctx_words = set(re.findall(r'\w+', ctx_sent))
                    if len(words.intersection(ctx_words)) > (len(words) * 0.8):
                        contradictions.append(f"Potential contradiction: '{res_sent}' may contradict context.")

        return contradictions

    def _detect_uncertainty(self, response: str) -> bool:
        uncertainty_markers = [
            r"\bi'm not sure\b", r"\bperhaps\b", r"\bmaybe\b",
            r"\bi think\b", r"\bpossibly\b",
        ]
        return any(re.search(marker, response.lower()) for marker in uncertainty_markers)

    def _internal_consistency_check(self, response: str) -> HallucinationReport:
        confidence = 0.9
        if re.search(r'\b(user:|assistant:|system:)\b', response, re.IGNORECASE):
            confidence = 0.4
        return HallucinationReport(confidence, True, [], False)

    def _check_response_urls(self, response: str, report: HallucinationReport) -> None:
        urls = self.url_verifier.extract_urls(response)
        for url in urls:
            if not self.url_verifier.validate_format(url):
                report.url_issues.append(f"Invalid URL format: {url}")
                report.uncertainty_detected = True
                report.confidence_score = max(0.0, report.confidence_score - 0.15)

    def _check_response_urls_sync(self, response: str) -> List[str]:
        issues = []
        urls = self.url_verifier.extract_urls(response)
        for url in urls:
            if not self.url_verifier.validate_format(url):
                issues.append(f"Invalid URL format: {url}")
            elif not self.url_verifier.check_whitelist(url):
                issues.append(f"Unverifiable URL (not in whitelist): {url}")
        return issues

    def handle_uncertainty(self, response: str, report: HallucinationReport, intent_category: str = "") -> str:
        if report.confidence_score >= 0.4:
            return response

        if intent_category in ("coding_problem", "debugging", "optimization"):
            return response

        return "I am currently uncertain about this specific detail based on the available information. " + response
