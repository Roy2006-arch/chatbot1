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

        meaningful_res = {t for t in res_tokens if len(t) > 3}
        if not meaningful_res:
            return 1.0

        supported = meaningful_res.intersection(context_tokens)
        return len(supported) / len(meaningful_res)

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
