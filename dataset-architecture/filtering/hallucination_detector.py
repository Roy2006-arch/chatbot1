import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FilterResult, FilterIssue, Severity

HEDGING_PATTERNS = [
    r"\bmight\b", r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b",
    r"\bi think\b", r"\bi believe\b", r"\bi suppose\b", r"\bi guess\b",
    r"\bseems like\b", r"\bit seems\b", r"\bas far as i know\b",
    r"\bi'm not sure\b", r"\bi don't know\b", r"\bnot entirely sure\b",
    r"\bcould be\b", r"\bconceivably\b", r"\barguably\b",
    r"\bmore or less\b", r"\bbasically\b", r"\bkind of\b", r"\bsort of\b",
    r"\bin my opinion\b", r"\bfrom what i understand\b",
    r"\bi would say\b", r"\bpotentially\b", r"\bpresumably\b",
]

VAGUENESS_PATTERNS = [
    r"\bet cetera\b", r"\betc\.?\b", r"\band more\b", r"\band so on\b",
    r"\bsomething like\b", r"\bthings like\b", r"\bamong others\b",
    r"\band others\b", r"\bwhatever\b", r"\bthings\b",
]

UNVERIFIABLE_CLAIMS = [
    r"(?i)research shows?\b",
    r"(?i)studies (show|indicate|suggest|prove|demonstrate)\b",
    r"(?i)experts (say|believe|agree|claim)\b",
    r"(?i)scientists (say|believe|have proven|have shown)\b",
    r"(?i)according to (research|studies|experts|scientists)\b",
    r"(?i)it is (widely |well )?(known|accepted|believed|understood)\b",
    r"(?i)it has been (proven|shown|demonstrated)\b",
    r"(?i)it is a (known|proven|established) fact\b",
    r"(?i)data (shows|suggests|indicates|reveals)\b",
    r"(?i)evidence (shows|suggests|indicates|points to)\b",
]

TEMPLATE_LEAKAGE = [
    r"\{\{[a-z_]+\}\}",
    r"<\|[a-z_]+\|>",
    r"\[/[a-z_]+\]",
    r"\{\s*\{?\s*(prompt|question|input|instruction|query|user)",
    r"\{\s*\}?\s*(response|output|answer|completion|assistant)",
]

OVERCONFIDENT_PATTERNS = [
    r"\balways\b", r"\bnever\b", r"\bdefinitely\b", r"\babsolutely\b",
    r"\bwithout (any )?doubt\b", r"\bwithout question\b",
    r"\bundoubtedly\b", r"\bindisputably\b", r"\birrefutably\b",
    r"\b100%|100 percent\b", r"\bguaranteed?\b",
]

CONTRADICTION_MARKERS = [
    (r"\bhowever\b", r"\bthis contradicts\b"),
    (r"\bon the one hand\b", r"\bon the other hand\b"),
]


class HallucinationDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.hedging_threshold = self.config.get("hedging_threshold", 0.3)
        self.vagueness_threshold = self.config.get("vagueness_threshold", 0.4)
        self.max_unverifiable = self.config.get("max_unverifiable_claims", 2)
        self.stats = {"checked": 0, "flagged": 0}

    def check(self, text: str) -> FilterResult:
        self.stats["checked"] += 1
        issues: List[FilterIssue] = []
        dim_scores = {}
        text_lower = text.lower()

        hedging_score = self._score_hedging(text_lower)
        dim_scores["hedging"] = 1.0 - hedging_score
        if hedging_score > self.hedging_threshold:
            issues.append(FilterIssue(
                code="HALLUCINATION_HEDGING",
                message=f"High hedging score ({hedging_score:.2f}): excessive uncertain language",
                severity=Severity.MEDIUM,
                dimension="hallucination",
                details={"hedging_score": hedging_score, "threshold": self.hedging_threshold},
            ))

        vagueness_score = self._score_vagueness(text_lower)
        dim_scores["vagueness"] = 1.0 - vagueness_score
        if vagueness_score > self.vagueness_threshold:
            issues.append(FilterIssue(
                code="HALLUCINATION_VAGUENESS",
                message=f"High vagueness score ({vagueness_score:.2f})",
                severity=Severity.MEDIUM,
                dimension="hallucination",
                details={"vagueness_score": vagueness_score, "threshold": self.vagueness_threshold},
            ))

        unverifiable_count = self._count_unverifiable(text_lower)
        dim_scores["unverifiable_claims"] = 1.0 - min(1.0, unverifiable_count / max(self.max_unverifiable, 1))
        if unverifiable_count > self.max_unverifiable:
            issues.append(FilterIssue(
                code="HALLUCINATION_UNVERIFIABLE",
                message=f"Contains {unverifiable_count} unverifiable claims (max {self.max_unverifiable})",
                severity=Severity.HIGH,
                dimension="hallucination",
                details={"unverifiable_count": unverifiable_count, "max_allowed": self.max_unverifiable},
            ))

        template_issues = self._check_template_leakage(text)
        issues.extend(template_issues)
        dim_scores["template_leakage"] = 1.0 - len(template_issues) * 0.25

        if self.config.get("contradiction_check", True):
            contradiction_score = self._check_contradictions(text_lower)
            dim_scores["contradictions"] = 1.0 - contradiction_score
            if contradiction_score > 0:
                issues.append(FilterIssue(
                    code="HALLUCINATION_CONTRADICTION",
                    message=f"Possible internal contradiction detected",
                    severity=Severity.HIGH,
                    dimension="hallucination",
                    details={"contradiction_score": contradiction_score},
                ))

        if self.config.get("confidence_mismatch_check", True):
            mismatch_score = self._check_confidence_mismatch(text_lower)
            dim_scores["confidence_mismatch"] = 1.0 - mismatch_score

        composite = sum(dim_scores.values()) / max(len(dim_scores), 1) if dim_scores else 1.0

        passed = len([i for i in issues if i.severity in (Severity.HIGH, Severity.CRITICAL)]) == 0
        if not passed:
            self.stats["flagged"] += 1

        return FilterResult(
            passed=passed,
            score=composite,
            issues=issues,
            dimension_scores=dim_scores,
            metadata={"hedging_score": hedging_score, "vagueness_score": vagueness_score},
        )

    def check_batch(self, texts: List[str], num_workers: int = 8) -> List[FilterResult]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.check, t) for t in texts]
            return [f.result() for f in as_completed(futures)]

    def _score_hedging(self, text: str) -> float:
        matches = sum(1 for p in HEDGING_PATTERNS if re.search(p, text))
        words = len(text.split())
        if words == 0:
            return 0.0
        return min(1.0, matches / max(words * 0.02, 1))

    def _score_vagueness(self, text: str) -> float:
        matches = sum(1 for p in VAGUENESS_PATTERNS if re.search(p, text))
        words = len(text.split())
        if words == 0:
            return 0.0
        return min(1.0, matches / max(words * 0.01, 1))

    def _count_unverifiable(self, text: str) -> int:
        return sum(1 for p in UNVERIFIABLE_CLAIMS if re.search(p, text))

    def _check_template_leakage(self, text: str) -> List[FilterIssue]:
        issues = []
        for pattern in TEMPLATE_LEAKAGE:
            matches = re.findall(pattern, text)
            if matches:
                issues.append(FilterIssue(
                    code="HALLUCINATION_TEMPLATE_LEAK",
                    message=f"Template placeholder leaked into output: {matches[0][:30]}",
                    severity=Severity.HIGH,
                    dimension="hallucination",
                    details={"matches": list(set(m[:30] for m in matches))},
                ))
        return issues

    def _check_contradictions(self, text: str) -> float:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        contradiction_count = 0
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if self._is_contradictory(sentences[i], sentences[j]):
                    contradiction_count += 1
        return min(1.0, contradiction_count / max(len(sentences), 1))

    def _is_contradictory(self, s1: str, s2: str) -> bool:
        negation_words = {"not", "no", "never", "neither", "nor", "none", "cannot", "can't", "don't", "doesn't"}
        s1_words = set(s1.lower().split())
        s2_words = set(s2.lower().split())
        common = s1_words & s2_words
        if len(common) < 3:
            return False
        has_neg1 = bool(common & negation_words)
        has_neg2 = bool(s2_words & negation_words)
        return has_neg1 != has_neg2 and len(common - negation_words) >= 3

    def _check_confidence_mismatch(self, text: str) -> float:
        hedging = self._score_hedging(text)
        overconfident = sum(1 for p in OVERCONFIDENT_PATTERNS if re.search(p, text))
        words = len(text.split())
        overconfident_score = min(1.0, overconfident / max(words * 0.01, 1))
        if hedging > 0.2 and overconfident_score > 0.2:
            return min(1.0, (hedging + overconfident_score) / 2)
        return 0.0

    def get_stats(self) -> Dict:
        return self.stats
