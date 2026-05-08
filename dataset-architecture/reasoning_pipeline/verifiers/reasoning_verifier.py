import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..schema import ReasoningExample, ReasoningStep, ReasoningType, ReasoningTask


class ReasoningVerifier:
    def __init__(self):
        self.stats = {"verified": 0, "valid": 0, "invalid": 0}

    def verify(self, example: ReasoningExample) -> Tuple[bool, List[str]]:
        issues = []

        step_valid = self._verify_steps(example, issues)
        logical_valid = self._verify_logical_flow(example, issues)
        completeness_valid = self._verify_completeness(example, issues)
        consistency_valid = self._verify_consistency(example, issues)

        all_valid = all([step_valid, logical_valid, completeness_valid, consistency_valid])
        example.metadata["verification_result"] = {
            "valid": all_valid,
            "issues": issues,
            "step_valid": step_valid,
            "logical_valid": logical_valid,
            "completeness_valid": completeness_valid,
            "consistency_valid": consistency_valid,
        }

        self.stats["verified"] += 1
        if all_valid:
            self.stats["valid"] += 1
        else:
            self.stats["invalid"] += 1

        return all_valid, issues

    def verify_batch(self, examples: List[ReasoningExample], num_workers: int = 8) -> List[ReasoningExample]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.verify, ex): ex for ex in examples}
            valid = []
            for future in as_completed(futures):
                ex = futures[future]
                is_valid, issues = future.result()
                ex.metadata["verification_issues"] = issues
                if is_valid:
                    valid.append(ex)
        return valid

    def _verify_steps(self, example: ReasoningExample, issues: List[str]) -> bool:
        if not example.reasoning_steps:
            issues.append("No reasoning steps provided")
            return False

        valid = True
        for i, step in enumerate(example.reasoning_steps):
            if step.index != i + 1:
                issues.append(f"Step {i+1}: index mismatch (expected {i+1}, got {step.index})")
                valid = False
            if not step.content or len(step.content) < 5:
                issues.append(f"Step {i+1}: content too short or empty")
                valid = False

        return valid

    def _verify_logical_flow(self, example: ReasoningExample, issues: List[str]) -> bool:
        if len(example.reasoning_steps) < 2:
            return True

        valid = True
        for i in range(1, len(example.reasoning_steps)):
            prev = example.reasoning_steps[i - 1]
            curr = example.reasoning_steps[i]
            if not curr.justification or len(curr.justification) < 3:
                issues.append(f"Step {i+1}: missing justification for transition from step {i}")
                valid = False

        return valid

    def _verify_completeness(self, example: ReasoningExample, issues: List[str]) -> bool:
        valid = True
        if not example.question or len(example.question) < 5:
            issues.append("Question is too short or empty")
            valid = False
        if not example.final_answer or len(example.final_answer) < 3:
            issues.append("Final answer is missing or too short")
            valid = False
        if not example.domain:
            issues.append("Domain not specified")
            valid = False
        return valid

    def _verify_consistency(self, example: ReasoningExample, issues: List[str]) -> bool:
        if not example.reasoning_steps:
            return True

        valid = True
        contradictions = []

        for i, step in enumerate(example.reasoning_steps):
            for j in range(i + 1, len(example.reasoning_steps)):
                if self._detect_contradiction(step, example.reasoning_steps[j]):
                    contradictions.append((i, j))

        if contradictions:
            for i, j in contradictions:
                issues.append(f"Potential contradiction between step {i+1} and step {j+1}")
            valid = False

        return valid

    def _detect_contradiction(self, step_a: ReasoningStep, step_b: ReasoningStep) -> bool:
        negation_patterns = [
            (r"\bnot\b", r"\bnot\b"),
            (r"\bcannot\b", r"\bcan\b"),
            (r"\bnever\b", r"\balways\b"),
            (r"\bimpossible\b", r"\bpossible\b"),
        ]
        text_a = step_a.content.lower()
        text_b = step_b.content.lower()

        for pos_pat, neg_pat in negation_patterns:
            has_pos_a = bool(re.search(pos_pat, text_a))
            has_neg_a = bool(re.search(neg_pat, text_a))
            has_pos_b = bool(re.search(pos_pat, text_b))
            has_neg_b = bool(re.search(neg_pat, text_b))

            if (has_pos_a and has_neg_b) or (has_neg_a and has_pos_b):
                return True

        return False

    def analyze_reasoning_quality(self, example: ReasoningExample) -> Dict[str, float]:
        scores = {}
        scores["step_count"] = min(1.0, len(example.reasoning_steps) / 10.0)
        scores["has_justification"] = sum(1 for s in example.reasoning_steps if len(s.justification) > 5) / max(len(example.reasoning_steps), 1)
        scores["has_alternatives"] = min(1.0, sum(len(s.alternatives) for s in example.reasoning_steps) / 3.0)
        scores["depth"] = min(1.0, sum(len(s.sub_steps) for s in example.reasoning_steps) / 5.0)
        scores["verification_detail"] = min(1.0, len(example.verification) / 100.0)
        scores["composite"] = sum(scores.values()) / max(len(scores), 1)
        return scores

    def verify_contradiction_pair(self, premise: str, conclusion_a: str, conclusion_b: str) -> Dict:
        tokens_a = set(conclusion_a.lower().split())
        tokens_b = set(conclusion_b.lower().split())

        overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
        contradictory_keywords = ["not", "never", "no", "cannot", "impossible", "false", "wrong"]

        contradiction_score = 0
        for kw in contradictory_keywords:
            if (kw in conclusion_a.lower()) != (kw in conclusion_b.lower()):
                contradiction_score += 0.2

        return {
            "text_overlap": round(overlap, 3),
            "contradiction_score": min(1.0, contradiction_score),
            "likely_contradiction": contradiction_score > 0.3 or (overlap > 0.5 and contradiction_score > 0.1),
        }

    def get_stats(self) -> Dict:
        return self.stats


class LogicalFallacyDetector:
    FALLACY_PATTERNS = {
        "ad_hominem": r"(?i)you'?re (wrong|stupid|ignorant)|you don'?t understand",
        "straw_man": r"(?i)so you'?re saying|what you really mean is|you think that",
        "false_dilemma": r"(?i)either.*or.*(?:no other|only two)|there (?:is|are) only two options",
        "circular_reasoning": r"(?i)because.*because|is true because it'?s true|proves itself",
        "hasty_generalization": r"(?i)all.*are|every.*always|never.*any",
        "post_hoc": r"(?i)after.*therefore.*caused|since.*happened.*must have caused",
        "slippery_slope": r"(?i)if we allow.*then.*next|first.*then.*eventually",
        "appeal_to_authority": r"(?i)because.*said so|experts agree|studies show \(no citation\)",
        "appeal_to_emotion": r"(?i)think of the children|if you cared|how can you live with",
        "red_herring": r"(?i)that'?s irrelevant but|what about|but the real issue is",
    }

    def detect(self, text: str) -> List[Dict]:
        found = []
        for fallacy, pattern in self.FALLACY_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found.append({"fallacy": fallacy, "count": len(matches), "severity": len(matches) * 0.25})
        return found

    def get_stats(self) -> Dict:
        return {}
