import re
import ast
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("response_ranker")

@dataclass
class CandidateResponse:
    text: str
    scores: Dict[str, float] = None
    total_score: float = 0.0

class ResponseRanker:
    """
    Advanced ranking engine for multi-candidate response evaluation.
    Scoring metrics: Correctness, Coherence, Completeness, Code Validity, Relevance.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "correctness": 0.30,
            "code_validity": 0.30,
            "completeness": 0.15,
            "relevance": 0.15,
            "coherence": 0.10
        }

    def score_candidate(self, candidate: str, prompt: str, context: str, planning_steps: List[str]) -> CandidateResponse:
        scores = {
            "correctness": self._score_correctness(candidate, context),
            "code_validity": self._score_code_validity(candidate),
            "completeness": self._score_completeness(candidate, planning_steps),
            "relevance": self._score_relevance(candidate, prompt),
            "coherence": self._score_coherence(candidate)
        }
        
        # Calculate weighted total
        total = sum(scores[m] * self.weights[m] for m in scores)
        
        # Penalty: If code exists but is invalid, slash total score
        if "```" in candidate and scores["code_validity"] < 0.5:
            total *= 0.5

        return CandidateResponse(text=candidate, scores=scores, total_score=total)

    def _score_correctness(self, text: str, context: str) -> float:
        """Scores based on grounding in retrieved context (overlap/fact-check)."""
        if not context: return 1.0 # No context to contradict
        
        # Basic overlap scoring
        context_words = set(re.findall(r'\w+', context.lower()))
        text_words = set(re.findall(r'\w+', text.lower()))
        
        if not context_words: return 1.0
        overlap = len(text_words.intersection(context_words)) / len(text_words) if text_words else 0
        return min(1.0, overlap * 2.0) # Boosted overlap score

    def _score_code_validity(self, text: str) -> float:
        """Strict coding-aware scoring using markdown and AST parsing."""
        if "```" not in text: return 1.0 # No code is "valid" code
        
        code_blocks = re.findall(r"```(?:python|py)?\n(.*?)\n```", text, re.DOTALL)
        if not code_blocks: return 0.2 # Incomplete block
        
        valid_blocks = 0
        for block in code_blocks:
            try:
                ast.parse(block)
                valid_blocks += 1
            except SyntaxError:
                continue
        
        return valid_blocks / len(code_blocks)

    def _score_completeness(self, text: str, steps: List[str]) -> float:
        """Measures percentage of internal planning steps fulfilled."""
        if not steps: return 1.0
        
        fulfilled = 0
        text_lower = text.lower()
        for step in steps:
            # Extract keywords from step
            keywords = [w for w in re.findall(r'\w+', step.lower()) if len(w) > 4]
            if any(k in text_lower for k in keywords):
                fulfilled += 1
        
        return fulfilled / len(steps)

    def _score_relevance(self, text: str, prompt: str) -> float:
        """Measures semantic alignment with user prompt."""
        prompt_keywords = set(re.findall(r'\w+', prompt.lower()))
        text_keywords = set(re.findall(r'\w+', text.lower()))
        
        if not prompt_keywords: return 1.0
        overlap = len(text_keywords.intersection(prompt_keywords)) / len(prompt_keywords)
        return min(1.0, overlap * 3.0)

    def _score_coherence(self, text: str) -> float:
        """Scores structural coherence and lack of repetition."""
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) < 2: return 1.0
        
        unique_sentences = set(s.strip().lower() for s in sentences)
        repetition_penalty = len(unique_sentences) / len(sentences)
        
        # Hallucination check: role leak tokens
        hallucination_penalty = 1.0
        if re.search(r'\b(user:|assistant:|system:)\b', text, re.IGNORECASE):
            hallucination_penalty = 0.3
            
        return repetition_penalty * hallucination_penalty

    def rank(self, candidates: List[str], prompt: str, context: str, steps: List[str]) -> str:
        """Ranks multiple candidates and returns the best one."""
        if not candidates: return ""
        
        scored_candidates = [
            self.score_candidate(c, prompt, context, steps) 
            for c in candidates
        ]
        
        # Sort by total score descending
        ranked = sorted(scored_candidates, key=lambda x: x.total_score, reverse=True)
        
        best = ranked[0]
        logger.info(f"Selected best candidate with score {best.total_score:.3f}. Metrics: {best.scores}")
        return best.text
