import re
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FilterResult, FilterIssue, Severity

FILLER_WORDS = {
    "um", "uh", "ah", "er", "like", "you know", "i mean", "well",
    "actually", "basically", "honestly", "literally", "so", "anyway",
    "right", "okay", "ok", "see", "listen", "look",
}


class RepetitionDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ngram_size = self.config.get("ngram_size", 4)
        self.ngram_threshold = self.config.get("ngram_threshold", 0.3)
        self.sentence_similarity_threshold = self.config.get("sentence_similarity_threshold", 0.85)
        self.max_filler_ratio = self.config.get("max_filler_ratio", 0.05)
        self.stats = {"checked": 0, "flagged_repetitive": 0}

    def check(self, text: str) -> FilterResult:
        self.stats["checked"] += 1
        issues: List[FilterIssue] = []
        dim_scores = {}

        if not text.strip():
            return FilterResult(passed=False, score=0.0, issues=[
                FilterIssue(code="REPETITION_EMPTY", message="Empty text", severity=Severity.HIGH, dimension="repetition"),
            ])

        ngram_score = self._check_ngram_repetition(text)
        dim_scores["ngram_repetition"] = 1.0 - ngram_score
        if ngram_score > self.ngram_threshold:
            issues.append(FilterIssue(
                code="REPETITION_NGRAM",
                message=f"High n-gram repetition score ({ngram_score:.2f})",
                severity=Severity.MEDIUM,
                dimension="repetition",
                details={"ngram_score": ngram_score, "threshold": self.ngram_threshold},
            ))

        sentence_score = self._check_sentence_repetition(text)
        dim_scores["sentence_repetition"] = 1.0 - sentence_score
        if sentence_score > self.sentence_similarity_threshold:
            issues.append(FilterIssue(
                code="REPETITION_SENTENCE",
                message="Near-identical sentences detected",
                severity=Severity.MEDIUM,
                dimension="repetition",
                details={"sentence_similarity": sentence_score},
            ))

        filler_score = self._check_filler_words(text)
        dim_scores["filler_words"] = 1.0 - filler_score
        if filler_score > self.max_filler_ratio:
            issues.append(FilterIssue(
                code="REPETITION_FILLERS",
                message=f"Excessive filler words ({filler_score:.2%})",
                severity=Severity.LOW,
                dimension="repetition",
                details={"filler_ratio": filler_score, "threshold": self.max_filler_ratio},
            ))

        if self.config.get("chunk_repetition_check", True):
            chunk_score = self._check_chunk_repetition(text)
            dim_scores["chunk_repetition"] = 1.0 - chunk_score
            if chunk_score > self.ngram_threshold:
                issues.append(FilterIssue(
                    code="REPETITION_CHUNK",
                    message="Repeated chunks of text detected",
                    severity=Severity.HIGH,
                    dimension="repetition",
                    details={"chunk_score": chunk_score},
                ))

        composite = sum(dim_scores.values()) / max(len(dim_scores), 1)

        critical_issues = [i for i in issues if i.severity in (Severity.HIGH, Severity.CRITICAL)]
        passed = len(critical_issues) == 0
        if not passed:
            self.stats["flagged_repetitive"] += 1

        return FilterResult(
            passed=passed,
            score=composite,
            issues=issues,
            dimension_scores=dim_scores,
            metadata={"ngram_rep_ratio": ngram_score, "filler_ratio": filler_score},
        )

    def check_batch(self, texts: List[str], num_workers: int = 8) -> List[FilterResult]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.check, t) for t in texts]
            return [f.result() for f in as_completed(futures)]

    def _check_ngram_repetition(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < self.ngram_size * 2:
            return 0.0

        ngrams = [" ".join(words[i:i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        if total == 0:
            return 0.0
        return 1.0 - (unique / total)

    def _check_sentence_repetition(self, text: str) -> float:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        if len(sentences) < 2:
            return 0.0

        similar_pairs = 0
        total_pairs = 0
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                total_pairs += 1
                sim = self._sentence_similarity(sentences[i], sentences[j])
                if sim > self.sentence_similarity_threshold:
                    similar_pairs += 1

        return similar_pairs / max(total_pairs, 1)

    def _sentence_similarity(self, s1: str, s2: str) -> float:
        words1 = set(s1.split())
        words2 = set(s2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def _check_filler_words(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        filler_count = sum(1 for w in words if w in FILLER_WORDS)
        return filler_count / len(words)

    def _check_chunk_repetition(self, text: str) -> float:
        chunks = re.split(r'\n{2,}', text)
        chunks = [c.strip() for c in chunks if len(c.strip()) > 50]
        if len(chunks) < 3:
            return 0.0

        normalized = [re.sub(r'\s+', ' ', c).lower() for c in chunks]
        seen = set()
        repeats = 0
        for c in normalized:
            prefix = c[:100]
            if prefix in seen:
                repeats += 1
            seen.add(prefix)

        return repeats / max(len(chunks), 1)

    def get_stats(self) -> Dict:
        return self.stats
