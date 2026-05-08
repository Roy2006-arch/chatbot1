import json
import os
import re
import random
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schema import ModelEvalResult, EvalCase


DEFAULT_EVAL_CASES = [
    EvalCase(prompt="What is 2+2?", expected_keywords=["4", "four"], category="math", difficulty=1),
    EvalCase(prompt="What is the capital of France?", expected_keywords=["Paris"], category="factual", difficulty=1),
    EvalCase(prompt="Explain what a variable is in programming.", expected_keywords=["variable", "store", "value"], category="code", difficulty=1),
    EvalCase(prompt="Write a function that adds two numbers.", expected_keywords=["def", "return", "add"], category="code", difficulty=2),
    EvalCase(prompt="What is the difference between TCP and UDP?", expected_keywords=["TCP", "UDP", "connection", "reliable"], category="technical", difficulty=3),
    EvalCase(prompt="Explain how garbage collection works.", expected_keywords=["garbage", "memory", "collect", "automatic"], category="technical", difficulty=3),
    EvalCase(prompt="What is the time complexity of binary search?", expected_keywords=["O(log n)", "logarithmic"], category="code", difficulty=2),
    EvalCase(prompt="Describe the water cycle.", expected_keywords=["evaporation", "condensation", "precipitation"], category="factual", difficulty=1),
    EvalCase(prompt="What is an API?", expected_keywords=["API", "interface", "application"], category="technical", difficulty=1),
    EvalCase(prompt="Explain recursion with an example.", expected_keywords=["recursion", "function", "calls", "base"], category="code", difficulty=2),
    EvalCase(prompt="How does a database index work?", expected_keywords=["index", "search", "query", "B-tree"], category="technical", difficulty=3),
    EvalCase(prompt="What is the difference between HTTP and HTTPS?", expected_keywords=["HTTP", "HTTPS", "SSL", "TLS", "encrypt"], category="technical", difficulty=2),
    EvalCase(prompt="Explain what Docker is.", expected_keywords=["Docker", "container", "image", "isolate"], category="technical", difficulty=2),
    EvalCase(prompt="What is a linked list?", expected_keywords=["linked", "list", "node", "pointer"], category="code", difficulty=1),
    EvalCase(prompt="Why is Python interpreted?", expected_keywords=["interpreted", "compiled", "runtime", "execution"], category="code", difficulty=2),
]


class ModelEvaluator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.score_threshold = self.config.get("score_threshold", 0.55)
        self.min_eval_cases = self.config.get("min_eval_cases", 10)
        self.stats = {"evaluations": 0, "cases_run": 0}

    def _score_accuracy(self, response: str, expected_keywords: List[str]) -> float:
        if not expected_keywords:
            return None
        response_lower = response.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        return found / len(expected_keywords)

    def _score_relevance(self, prompt: str, response: str) -> float:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb_p = model.encode(prompt, normalize_embeddings=True)
            emb_r = model.encode(response, normalize_embeddings=True)
            return float(emb_p @ emb_r)
        except ImportError:
            prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
            resp_words = set(re.findall(r'\b\w+\b', response.lower()))
            if not prompt_words:
                return 0.5
            overlap = len(prompt_words & resp_words)
            return min(1.0, overlap / len(prompt_words))

    def _score_coherence(self, response: str) -> float:
        if not response:
            return 0.0
        words = response.split()
        if len(words) < 5:
            return 0.3
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        has_punctuation = bool(re.search(r'[.!?]', response))
        score = 0.4
        if unique_ratio > 0.4:
            score += 0.2
        if has_punctuation:
            score += 0.2
        if len(words) >= 20:
            score += 0.1
        if len(words) <= 3:
            score -= 0.2
        return min(1.0, max(0.0, score))

    def evaluate_response(
        self, prompt: str, response: str, expected_keywords: Optional[List[str]] = None
    ) -> Dict[str, float]:
        accuracy = self._score_accuracy(response, expected_keywords or [])
        relevance = self._score_relevance(prompt, response)
        coherence = self._score_coherence(response)

        if accuracy is not None:
            composite = accuracy * 0.40 + relevance * 0.35 + coherence * 0.25
        else:
            composite = relevance * 0.55 + coherence * 0.45

        return {
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "relevance": round(relevance, 4),
            "coherence": round(coherence, 4),
            "composite": round(composite, 4),
        }

    def evaluate_case(self, case: EvalCase, response: str) -> Dict:
        scores = self.evaluate_response(case.prompt, response, case.expected_keywords)
        scores["passed"] = scores["composite"] >= self.score_threshold
        scores["category"] = case.category
        scores["difficulty"] = case.difficulty
        return scores

    def evaluate_model(
        self, model_name: str, response_fn, cases: Optional[List[EvalCase]] = None
    ) -> ModelEvalResult:
        if not self.enabled:
            return ModelEvalResult(model_name=model_name, run_id="", timestamp="")

        cases = cases or DEFAULT_EVAL_CASES
        run_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.utcnow().isoformat()

        results = []
        per_category: Dict[str, Dict] = {}
        grades: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        failures: Dict[str, int] = {}

        for case in cases:
            response = response_fn(case.prompt)
            eval_result = self.evaluate_case(case, response)
            results.append(eval_result)
            self.stats["cases_run"] += 1

            cat = case.category
            if cat not in per_category:
                per_category[cat] = {"count": 0, "total_composite": 0.0, "passed": 0}
            per_category[cat]["count"] += 1
            per_category[cat]["total_composite"] += eval_result["composite"]
            if eval_result["passed"]:
                per_category[cat]["passed"] += 1

            grade = self._grade(eval_result["composite"])
            grades[grade] = grades.get(grade, 0) + 1

            if not eval_result["passed"]:
                for kw in case.expected_keywords:
                    if kw.lower() not in response.lower():
                        failures[f"missing_{kw.lower().replace(' ', '_')}"] = failures.get(f"missing_{kw.lower().replace(' ', '_')}", 0) + 1

        total = len(results)
        avg_acc = sum(r.get("accuracy", 0) or 0 for r in results) / max(total, 1)
        avg_rel = sum(r["relevance"] for r in results) / max(total, 1)
        avg_coh = sum(r["coherence"] for r in results) / max(total, 1)
        avg_comp = sum(r["composite"] for r in results) / max(total, 1)
        passed = sum(1 for r in results if r["passed"])

        for cat in per_category:
            d = per_category[cat]
            d["avg_composite"] = round(d["total_composite"] / max(d["count"], 1), 4)
            d["pass_rate"] = round(d["passed"] / max(d["count"], 1), 4)

        result = ModelEvalResult(
            model_name=model_name,
            run_id=run_id,
            timestamp=timestamp,
            total_cases=total,
            avg_accuracy=round(avg_acc, 4),
            avg_relevance=round(avg_rel, 4),
            avg_coherence=round(avg_coh, 4),
            avg_composite=round(avg_comp, 4),
            pass_rate=round(passed / max(total, 1), 4),
            grade_distribution=grades,
            failure_breakdown=dict(sorted(failures.items(), key=lambda x: -x[1])[:10]),
            per_category=per_category,
        )

        self.stats["evaluations"] += 1
        return result

    def compare_models(
        self, before: ModelEvalResult, after: ModelEvalResult
    ) -> Dict[str, Any]:
        metrics = ["avg_accuracy", "avg_relevance", "avg_coherence", "avg_composite", "pass_rate"]
        comparison = {}
        for metric in metrics:
            b = getattr(before, metric, 0)
            a = getattr(after, metric, 0)
            comparison[metric] = {
                "before": b,
                "after": a,
                "delta": round(a - b, 4),
                "improved": a > b,
            }

        primary = self.config.get("comparison_metric", "avg_composite")
        improvement = comparison.get(primary, {}).get("delta", 0)

        grade_changes = {}
        for grade in ["A", "B", "C", "D", "F"]:
            b = before.grade_distribution.get(grade, 0)
            a = after.grade_distribution.get(grade, 0)
            grade_changes[grade] = {"before": b, "after": a, "delta": a - b}

        return {
            "comparison": comparison,
            "grade_changes": grade_changes,
            "score_improvement": round(improvement, 4),
            "overall_improved": improvement > 0,
        }

    def _grade(self, score: float) -> str:
        if score >= 0.85:
            return "A"
        elif score >= 0.70:
            return "B"
        elif score >= 0.55:
            return "C"
        elif score >= 0.40:
            return "D"
        return "F"

    def get_stats(self) -> Dict:
        return self.stats
