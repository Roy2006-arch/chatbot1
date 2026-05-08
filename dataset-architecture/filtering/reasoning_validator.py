import re
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FilterResult, FilterIssue, Severity

CONCLUSION_SIGNALS = [
    r"\bin conclusion\b", r"\bto summarize\b", r"\bin summary\b",
    r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bso\b",
    r"\bin short\b", r"\boverall\b", r"\bfinally\b",
    r"\bthe answer is\b", r"\bthe result is\b",
    r"\bto sum up\b", r"\bin closing\b", r"\bto conclude\b",
]

STEP_LABELS = [
    r"\bfirst\b", r"\bsecond\b", r"\bthird\b",
    r"\bstep \d+\b", r"\b\d+\.\s",
    r"\bfirstly\b", r"\bsecondly\b", r"\bthirdly\b",
    r"\bnext\b", r"\bthen\b", r"\bafter that\b",
    r"\bfinally\b", r"\blastly\b",
]

COT_OPENERS = [
    r"\blet'?s (think|reason|break|work|solve|approach|consider)\b",
    r"\bfirst,?\s+(let|we|i)\b",
    r"\bhere'?s (how|the|a)\b",
    r"\bto (solve|find|determine|calculate|figure)\b",
    r"\bthe (approach|solution|method|way)\b",
]

TRUNCATION_PATTERNS = [
    ("<", ">"),
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
    ("<answer>", "</answer>"),
]


class ReasoningValidator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_steps = self.config.get("min_steps", 2)
        self.conclusion_required = self.config.get("conclusion_required", True)
        self.truncation_check = self.config.get("truncation_check", True)
        self.min_cot_length = self.config.get("min_cot_length", 50)
        self.step_label_required = self.config.get("step_label_required", True)
        self.stats = {"checked": 0, "flagged_truncated": 0, "flagged_incomplete": 0}

    def check(self, text: str) -> FilterResult:
        self.stats["checked"] += 1
        issues: List[FilterIssue] = []
        dim_scores = {}

        if not text.strip():
            return FilterResult(passed=False, score=0.0, issues=[
                FilterIssue(code="REASONING_EMPTY", message="Empty text", severity=Severity.HIGH, dimension="reasoning"),
            ])

        truncation_issues = self._check_truncation(text)
        issues.extend(truncation_issues)
        dim_scores["not_truncated"] = 1.0 - len(truncation_issues) * 0.33

        has_conclusion, conclusion_score = self._check_conclusion(text)
        dim_scores["has_conclusion"] = conclusion_score
        if self.conclusion_required and not has_conclusion:
            issues.append(FilterIssue(
                code="REASONING_NO_CONCLUSION",
                message="Response lacks a conclusion or final answer",
                severity=Severity.MEDIUM,
                dimension="reasoning",
            ))

        step_info = self._count_reasoning_steps(text)
        dim_scores["step_count"] = min(1.0, step_info["count"] / max(self.min_steps, 1))
        if step_info["count"] < self.min_steps:
            issues.append(FilterIssue(
                code="REASONING_TOO_FEW_STEPS",
                message=f"Only {step_info['count']} reasoning steps found (min {self.min_steps})",
                severity=Severity.MEDIUM,
                dimension="reasoning",
                details={"steps_found": step_info["count"], "min_required": self.min_steps},
            ))

        has_cot, cot_score = self._check_chain_of_thought(text)
        dim_scores["has_cot"] = cot_score

        if self.step_label_required and not step_info["has_labels"]:
            issues.append(FilterIssue(
                code="REASONING_NO_STEP_LABELS",
                message="Reasoning steps lack clear ordering labels",
                severity=Severity.LOW,
                dimension="reasoning",
            ))

        coherence_score = self._check_coherence(text)
        dim_scores["coherence"] = coherence_score
        if coherence_score < 0.3:
            issues.append(FilterIssue(
                code="REASONING_INCOHERENT",
                message="Reasoning appears incoherent or disconnected",
                severity=Severity.HIGH,
                dimension="reasoning",
                details={"coherence_score": coherence_score},
            ))

        composite = sum(dim_scores.values()) / max(len(dim_scores), 1)

        critical = [i for i in issues if i.severity in (Severity.HIGH, Severity.CRITICAL)]
        passed = len(critical) == 0
        if not passed:
            self.stats["flagged_truncated"] += 1
            if any(i.code == "REASONING_TOO_FEW_STEPS" for i in issues):
                self.stats["flagged_incomplete"] += 1

        return FilterResult(
            passed=passed,
            score=composite,
            issues=issues,
            dimension_scores=dim_scores,
            metadata={"steps": step_info, "has_conclusion": has_conclusion},
        )

    def check_batch(self, texts: List[str], num_workers: int = 8) -> List[FilterResult]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.check, t) for t in texts]
            return [f.result() for f in as_completed(futures)]

    def _check_truncation(self, text: str) -> List[FilterIssue]:
        issues = []
        if not text.strip():
            return issues

        text_stripped = text.strip()
        last_char = text_stripped[-1]
        if last_char not in ".!?)]}>\"'`":
            issues.append(FilterIssue(
                code="REASONING_ABRUPT_END",
                message=f"Response ends abruptly (last char: '{last_char}')",
                severity=Severity.MEDIUM,
                dimension="reasoning",
            ))

        if self.truncation_check:
            fence_count = text.count("```")
            if fence_count % 2 != 0:
                issues.append(FilterIssue(
                    code="REASONING_UNCLOSED_TAG",
                    message=f"Unclosed code fence (odd number of ```: {fence_count})",
                    severity=Severity.HIGH,
                    dimension="reasoning",
                    details={"pattern": "```", "opens": fence_count, "closes": 0},
                ))

            for open_pat, close_pat in TRUNCATION_PATTERNS:
                opens = text.count(open_pat)
                closes = text.count(close_pat)
                if opens > closes:
                    issues.append(FilterIssue(
                        code="REASONING_UNCLOSED_TAG",
                        message=f"Unclosed '{open_pat}' (found {opens} opens vs {closes} closes)",
                        severity=Severity.HIGH,
                        dimension="reasoning",
                        details={"pattern": open_pat, "opens": opens, "closes": closes},
                    ))

        return issues

    def _check_conclusion(self, text: str) -> Tuple[bool, float]:
        text_lower = text.lower()
        matches = sum(1 for p in CONCLUSION_SIGNALS if re.search(p, text_lower))
        score = min(1.0, matches * 0.25)
        return matches > 0, score

    def _count_reasoning_steps(self, text: str) -> Dict:
        text_lower = text.lower()
        labels = [p for p in STEP_LABELS if re.search(p, text_lower)]
        count = len(labels)

        step_numbers = re.findall(r'(?:^|\n)\s*(\d+)[.)]\s', text)
        count = max(count, len(step_numbers))

        code_blocks = re.findall(r'```\w*\n.*?```', text, re.DOTALL)
        code_block_count = len(code_blocks)

        return {
            "count": count,
            "has_labels": count > 0,
            "labels_found": labels[:10],
            "code_blocks": code_block_count,
        }

    def _check_chain_of_thought(self, text: str) -> Tuple[bool, float]:
        text_lower = text.lower()
        has_thought_tags = bool(re.search(r'<thought>.*?</thought>', text, re.DOTALL))
        has_cot_phrases = sum(1 for p in COT_OPENERS if re.search(p, text_lower))

        if has_thought_tags:
            return True, 1.0
        if has_cot_phrases >= 2:
            return True, 0.8
        if has_cot_phrases >= 1:
            return True, 0.5
        return False, 0.0

    def _check_coherence(self, text: str) -> float:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) < 2:
            return 0.5

        transitions = 0
        transition_words = {"however", "therefore", "furthermore", "moreover", "nevertheless",
                            "consequently", "additionally", "meanwhile", "subsequently",
                            "first", "second", "third", "then", "next", "finally", "lastly"}
        for s in sentences:
            words = set(re.findall(r'\b\w+\b', s.lower()))
            if words & transition_words:
                transitions += 1

        return min(1.0, transitions / max(len(sentences) * 0.3, 1))

    def get_stats(self) -> Dict:
        return self.stats
