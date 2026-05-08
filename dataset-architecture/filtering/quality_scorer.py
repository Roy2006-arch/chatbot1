import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FilterResult, FilterIssue, Severity
from .hallucination_detector import HallucinationDetector
from .repetition_detector import RepetitionDetector
from .reasoning_validator import ReasoningValidator
from .toxicity_filter import ToxicityFilter


DEFAULT_DIMENSIONS = {
    "relevance": 0.15,
    "factual_correctness": 0.20,
    "completeness": 0.10,
    "clarity": 0.05,
    "safety": 0.15,
    "instruction_following": 0.05,
    "non_repetitiveness": 0.10,
    "reasoning_quality": 0.10,
    "hallucination_free": 0.10,
}


class EnhancedQualityScorer:
    def __init__(self, dimensions: Optional[Dict[str, float]] = None, config: Optional[Dict] = None):
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.config = config or {}
        self.hallucination_detector = HallucinationDetector(self.config.get("hallucination", {}))
        self.repetition_detector = RepetitionDetector(self.config.get("repetition", {}))
        self.reasoning_validator = ReasoningValidator(self.config.get("reasoning", {}))
        self.toxicity_filter = ToxicityFilter(self.config.get("toxicity", {}))
        self.stats = {"scored": 0, "failed": 0}

    def score_text(self, instruction: str, output: str, input_text: str = "") -> FilterResult:
        all_results = []
        dimension_scores = {}

        relevance_result = self._score_relevance(instruction, output)
        dimension_scores["relevance"] = relevance_result.score
        all_results.extend(relevance_result.issues)

        correctness_result = self._score_correctness(output)
        dimension_scores["factual_correctness"] = correctness_result.score
        all_results.extend(correctness_result.issues)

        completeness_result = self._score_completeness(instruction, output)
        dimension_scores["completeness"] = completeness_result.score
        all_results.extend(completeness_result.issues)

        clarity_result = self._score_clarity(output)
        dimension_scores["clarity"] = clarity_result.score
        all_results.extend(clarity_result.issues)

        safety_result = self.toxicity_filter.check(output)
        dimension_scores["safety"] = safety_result.score
        all_results.extend(safety_result.issues)

        instruction_following_result = self._score_instruction_following(instruction, output)
        dimension_scores["instruction_following"] = instruction_following_result.score
        all_results.extend(instruction_following_result.issues)

        repetition_result = self.repetition_detector.check(output)
        dimension_scores["non_repetitiveness"] = repetition_result.score
        all_results.extend(repetition_result.issues)

        reasoning_result = self.reasoning_validator.check(output)
        dimension_scores["reasoning_quality"] = reasoning_result.score
        all_results.extend(reasoning_result.issues)

        hallucination_result = self.hallucination_detector.check(output)
        dimension_scores["hallucination_free"] = hallucination_result.score
        all_results.extend(hallucination_result.issues)

        composite = sum(
            dimension_scores.get(dim, 0.5) * weight
            for dim, weight in self.dimensions.items()
        )

        self.stats["scored"] += 1

        return FilterResult(
            passed=composite >= 0.5,
            score=composite,
            issues=all_results,
            dimension_scores=dimension_scores,
            metadata={"composite_score": composite, "dimension_count": len(dimension_scores)},
        )

    def score_batch(
        self, instructions: List[str], outputs: List[str], inputs: Optional[List[str]] = None,
        num_workers: int = 8,
    ) -> List[FilterResult]:
        inputs = inputs or [""] * len(instructions)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self.score_text, inst, out, inp): i
                for i, (inst, out, inp) in enumerate(zip(instructions, outputs, inputs))
            }
            results = [None] * len(instructions)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    self.stats["failed"] += 1
                    results[idx] = FilterResult(passed=False, score=0.0)
            return results

    def _score_relevance(self, instruction: str, output: str) -> FilterResult:
        inst_words = set(self._tokenize(instruction))
        out_words = set(self._tokenize(output))
        if not inst_words or not out_words:
            return FilterResult(passed=False, score=0.3, dimension_scores={"relevance": 0.3})

        overlap = len(inst_words & out_words)
        coverage = overlap / max(len(inst_words), 1)
        score = min(1.0, 0.3 + coverage * 0.7)

        issues = []
        if score < 0.3:
            issues.append(FilterIssue(
                code="SCORE_LOW_RELEVANCE",
                message=f"Low relevance score ({score:.2f}): output shares little vocabulary with instruction",
                severity=Severity.MEDIUM,
                dimension="relevance",
            ))

        return FilterResult(passed=score >= 0.3, score=score, issues=issues, dimension_scores={"relevance": score})

    def _score_correctness(self, output: str) -> FilterResult:
        score = 0.55
        output_lower = output.lower()
        issues = []

        positive_signals = [
            r"\b(?:def |class |function |import |return|print|console\.log|const |let |var )",
            r"\b(?:therefore|hence|thus|because|since|as a result)\b",
            r"\b(?:correct|accurate|verified|confirmed|valid|proven)\b",
            r"```\w*\n",
            r"\b(?:solution|approach|method|algorithm|implementation|technique)\b",
            r"\b(?:demonstrate|prove|verify|validate|confirm)\b",
        ]
        negative_signals = [
            r"\b(?:maybe|perhaps|i think|i believe|not sure|possibly|might be)\b",
            r"\b(?:incorrect|wrong|mistake|error|bug|flawed)\b",
            r"\b(?:i don't know|i'm not sure|i cannot determine|i am not sure)\b",
            r"\b(?:sorry,? i|apologize|unfortunately)\b",
        ]

        for pattern in positive_signals:
            if re.search(pattern, output_lower):
                score = min(1.0, score + 0.05)

        for pattern in negative_signals:
            if re.search(pattern, output_lower):
                score = max(0.0, score - 0.1)
                issues.append(FilterIssue(
                    code="SCORE_NEGATIVE_SIGNAL",
                    message=f"Negative correctness signal: '{pattern}'",
                    severity=Severity.LOW,
                    dimension="factual_correctness",
                ))

        return FilterResult(passed=score >= 0.4, score=score, issues=issues, dimension_scores={"factual_correctness": score})

    def _score_completeness(self, instruction: str, output: str) -> FilterResult:
        inst_len = len(instruction)
        out_len = len(output)

        if inst_len == 0:
            return FilterResult(passed=True, score=0.5, dimension_scores={"completeness": 0.5})

        ratio = out_len / max(inst_len, 1)
        issues = []

        if ratio < 0.3:
            score = max(0.1, ratio * 1.5)
            issues.append(FilterIssue(
                code="SCORE_TOO_SHORT",
                message=f"Response too short relative to instruction (ratio: {ratio:.2f})",
                severity=Severity.MEDIUM,
                dimension="completeness",
            ))
        elif ratio < 1.0:
            score = 0.5 + ratio * 0.3
        elif ratio < 5.0:
            score = 0.8
        else:
            score = min(1.0, 0.8 + (ratio - 5.0) * 0.01)

        return FilterResult(passed=score >= 0.4, score=score, issues=issues, dimension_scores={"completeness": score})

    def _score_clarity(self, output: str) -> FilterResult:
        if not output:
            return FilterResult(passed=False, score=0.0, issues=[
                FilterIssue(code="SCORE_EMPTY", message="Empty output", severity=Severity.HIGH, dimension="clarity"),
            ])

        score = 0.6
        issues = []
        sentences = re.split(r'[.!?]+', output)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            avg_len = sum(len(s) for s in sentences) / len(sentences)
            if 20 <= avg_len <= 100:
                score += 0.1
            elif avg_len > 200:
                score -= 0.1
                issues.append(FilterIssue(
                    code="SCORE_LONG_SENTENCES",
                    message=f"Average sentence length {avg_len:.0f} exceeds 200 chars",
                    severity=Severity.LOW,
                    dimension="clarity",
                ))

        structure_signals = [
            r'\n\d+\.\s', r'\n-\s', r'\n\*\s', r'```', r'\n\n',
            r'\b(first|second|finally|overall|in conclusion)\b',
            r'\n#{1,6}\s',
        ]
        for pattern in structure_signals:
            if re.search(pattern, output):
                score = min(1.0, score + 0.05)

        return FilterResult(passed=score >= 0.3, score=min(1.0, max(0.0, score)), issues=issues,
                           dimension_scores={"clarity": score})

    def _score_instruction_following(self, instruction: str, output: str) -> FilterResult:
        inst_lower = instruction.lower()
        out_lower = output.lower()
        score = 0.6
        issues = []

        constraint_patterns = [
            (r"\b(in |as |using )?\w+ (only|exactly|precisely)\b", 0.1),
            (r"\b(explain|describe|list|enumerate|summarize|outline)\b", 0.05),
            (r"\b(step by step|step-by-step)\b", 0.1),
            (r"\b(short|concise|brief|detailed|comprehensive|elaborate)\b", 0.05),
            (r"\b(code|program|function|script|implementation|algorithm)\b", 0.05),
            (r"\b(example|instance|sample|demonstration)\b", 0.05),
            (r"\b(format|structure|organize|arrange)\b", 0.05),
        ]

        for pattern, delta in constraint_patterns:
            if re.search(pattern, inst_lower):
                if re.search(pattern, out_lower):
                    score += delta
                else:
                    score -= delta * 0.5
                    issues.append(FilterIssue(
                        code="SCORE_CONSTRAINT_MISS",
                        message=f"Instruction constraint '{pattern}' not reflected in output",
                        severity=Severity.LOW,
                        dimension="instruction_following",
                    ))

        return FilterResult(passed=score >= 0.3, score=max(0.0, min(1.0, score)), issues=issues,
                           dimension_scores={"instruction_following": score})

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def get_stats(self) -> Dict:
        return self.stats
