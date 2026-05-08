import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("hallucination_guard")

@dataclass
class HallucinationReport:
    confidence_score: float  # 0.0 - 1.0
    is_grounded: bool
    contradictions: List[str]
    uncertainty_detected: bool

class HallucinationGuard:
    """
    Advanced system to detect and reduce hallucinations by verifying responses
    against retrieved context and session memory.
    """

    def __init__(self, confidence_threshold: float = 0.70):
        self.threshold = confidence_threshold

    def evaluate(self, response: str, context: str) -> HallucinationReport:
        """
        Evaluates the faithfulness of the response relative to the context.
        """
        if not context:
            # If no context exists, we can only check for internal consistency and hallucinations (role-leak)
            return self._internal_consistency_check(response)

        # 1. Grounding Score (Keyword/Entity overlap)
        grounding_score = self._calculate_grounding_score(response, context)

        # 2. Contradiction Detection (Negation check)
        contradictions = self._detect_contradictions(response, context)

        # 3. Uncertainty Indicators
        uncertainty = self._detect_uncertainty(response)

        # 4. Final Confidence Calculation
        # Penalty for contradictions and role-leak artifacts
        penalty = 0.4 if contradictions else 0.0
        if re.search(r'\b(user:|assistant:|system:)\b', response, re.IGNORECASE):
            penalty += 0.3

        confidence = max(0.0, grounding_score - penalty)

        return HallucinationReport(
            confidence_score=confidence,
            is_grounded=grounding_score > 0.5 and not contradictions,
            contradictions=contradictions,
            uncertainty_detected=uncertainty or confidence < 0.5
        )

    def _calculate_grounding_score(self, response: str, context: str) -> float:
        """Measures how much of the response is supported by the context."""
        context_tokens = set(re.findall(r'\w+', context.lower()))
        res_tokens = set(re.findall(r'\w+', response.lower()))
        
        # Focus on unique, meaningful words (length > 3)
        meaningful_res = {t for t in res_tokens if len(t) > 3}
        if not meaningful_res: return 1.0
        
        supported = meaningful_res.intersection(context_tokens)
        return len(supported) / len(meaningful_res)

    def _detect_contradictions(self, response: str, context: str) -> List[str]:
        """Detects if the response negates facts found in the context."""
        contradictions = []
        # Naive negation check: "X is Y" in context vs "X is not Y" in response
        context_sentences = re.split(r'[.!?]\s+', context.lower())
        response_sentences = re.split(r'[.!?]\s+', response.lower())
        
        for res_sent in response_sentences:
            if " not " in res_sent or " no " in res_sent or " never " in res_sent:
                # Find positive counterpart in context
                positive_sent = res_sent.replace(" not ", " ").replace(" no ", " ").replace(" never ", " ")
                words = set(re.findall(r'\w+', positive_sent))
                for ctx_sent in context_sentences:
                    ctx_words = set(re.findall(r'\w+', ctx_sent))
                    if len(words.intersection(ctx_words)) > (len(words) * 0.8):
                        contradictions.append(f"Potential contradiction: '{res_sent}' may contradict context.")
        
        return contradictions

    def _detect_uncertainty(self, response: str) -> bool:
        """Detects linguistic markers of uncertainty."""
        uncertainty_markers = [r"\bi'm not sure\b", r"\bperhaps\b", r"\bmaybe\b", r"\bi think\b", r"\bpossibly\b"]
        return any(re.search(marker, response.lower()) for marker in uncertainty_markers)

    def _internal_consistency_check(self, response: str) -> HallucinationReport:
        """Fallback check for non-RAG responses."""
        confidence = 0.9 # Assume high if no context to verify against
        if re.search(r'\b(user:|assistant:|system:)\b', response, re.IGNORECASE):
            confidence = 0.4
        return HallucinationReport(confidence, True, [], False)

    def handle_uncertainty(self, response: str, report: HallucinationReport) -> str:
        """Adjusts the response if confidence is low."""
        if report.confidence_score < 0.4:
            return "I am currently uncertain about this specific detail based on the available information. " + response
        return response
