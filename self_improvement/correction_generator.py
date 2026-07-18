import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .schema import CorrectionRecord, SelfImprovementExample, ExampleSource, CorrectionMethod


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feedback", "feedback.db")


class CorrectionGenerator:
    def __init__(self, config: Optional[Dict] = None, db_path: str = DB_PATH):
        self.config = config or {}
        self.db_path = db_path
        self.enabled = self.config.get("enabled", True)
        self.min_quality_threshold = self.config.get("min_quality_threshold", 0.6)
        self.max_score_before = self.config.get("max_score_before", 0.6)
        self.stats = {"loaded": 0, "generated": 0, "failed": 0}

    def load_failed_queries(self, limit: int = 500, unresolved_only: bool = True) -> List[Dict]:
        if not os.path.exists(self.db_path):
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = """
                SELECT fq.*, COUNT(fb.id) as downvote_count
                FROM failed_queries fq
                LEFT JOIN feedback fb ON fb.conv_id = fq.conv_id AND fb.vote = -1
                WHERE 1=1
            """
            if unresolved_only:
                query += " AND (fq.resolved IS NULL OR fq.resolved = 0)"
            query += " GROUP BY fq.id ORDER BY fq.occurrence_count DESC, fq.id DESC"
            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query)
            rows = [dict(r) for r in cursor.fetchall()]
            self.stats["loaded"] = len(rows)
            return rows
        finally:
            conn.close()

    def generate_correction(self, item: Dict) -> Optional[CorrectionRecord]:
        prompt = item.get("prompt", "")
        original = item.get("response", "")

        if not prompt or not original:
            self.stats["failed"] += 1
            return None

        correction = self._heuristic_fix(prompt, original)
        if not correction:
            correction = self._template_fix(prompt, original)
        if not correction:
            self.stats["failed"] += 1
            return None

        score_before = item.get("composite_score", 0.0) or 0.0
        try:
            score_before = float(score_before)
        except (ValueError, TypeError):
            score_before = 0.0

        record = CorrectionRecord(
            failed_query_id=item.get("id", 0),
            prompt=prompt,
            original_response=original,
            corrected_response=correction,
            correction_method=self._detect_method(correction),
            score_before=score_before,
            score_after=self._quick_score(correction),
            created_at=datetime.utcnow().isoformat(),
            metadata={
                "occurrence_count": item.get("occurrence_count", 1),
                "failure_reasons": json.loads(item.get("failure_reasons", "[]")) if isinstance(item.get("failure_reasons"), str) else item.get("failure_reasons", []),
            },
        )
        self.stats["generated"] += 1
        return record

    def generate_batch(
        self, items: List[Dict], num_workers: int = 4
    ) -> List[CorrectionRecord]:
        if not self.enabled:
            return []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.generate_correction, item) for item in items]
            results = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
            return results

    def _heuristic_fix(self, prompt: str, original: str) -> Optional[str]:
        """Apply safe regex-based cleanup to strip refusals and hedging.

        Does NOT fall back to template generation — if the residual text
        after cleanup is too short or still starts with a refusal, we
        return None so the item gets flagged for human/LLM review instead
        of injecting boilerplate into the training set.
        """
        refusal_patterns = [
            (r"(?i)i'?m (just |only )?an ai", "I"),
            (r"(?i)as an ai (assistant|language model)", ""),
            (r"(?i)i'?m sorry,? (i|i'?m|i am)", ""),
            (r"(?i)i cannot (assist|help|answer|provide)", "Let me"),
            (r"(?i)i am (unable|not able).*", ""),
            (r"(?i)as a large language model", ""),
            (r"(?i)i am an ai language model", ""),
            (r"(?i)unfortunately,? (i|cannot)", ""),
        ]
        fixed = original
        applied = False
        for pattern, replacement in refusal_patterns:
            if re.search(pattern, fixed):
                fixed = re.sub(pattern, replacement, fixed)
                applied = True

        hedging_patterns = [
            r"\bi think\b", r"\bi believe\b", r"\bi suppose\b",
            r"\bnot sure\b", r"\bi don't know\b",
        ]
        for pattern in hedging_patterns:
            if re.search(pattern, fixed):
                fixed = re.sub(pattern, "", fixed)
                applied = True

        # If the response still starts with a refusal after cleanup,
        # reject it entirely — do NOT substitute a template.
        if re.search(r"(?i)^(i don't know|i'm not sure|i cannot|sorry)", fixed.strip()):
            return None

        fixed = re.sub(r'\s+', ' ', fixed).strip()
        if not fixed.endswith(('.', '!', '?')):
            fixed += "."
        fixed = fixed[0].upper() + fixed[1:] if fixed else fixed

        if applied and len(fixed) > 10:
            return fixed
        return None

    def _template_fix(self, prompt: str, original: str) -> Optional[str]:
        """Template-based correction — DISABLED.

        Previously this injected generic boilerplate paragraphs like
        'The core idea is straightforward...' which polluted training data.
        Items that reach this point should be flagged for human/LLM review.
        """
        return None

    def _generate_constructive_response(self, prompt: str) -> Optional[str]:
        """Constructive response generation — DISABLED.

        Previously returned robotic filler text that poisoned DPO/SFT datasets.
        Returns None so items are flagged for human or teacher-model review.
        """
        return None

    def _detect_method(self, correction: str) -> CorrectionMethod:
        return CorrectionMethod.AUTO_GENERATED

    def _quick_score(self, text: str) -> float:
        if not text or len(text) < 10:
            return 0.0
        words = len(text.split())
        if words < 5:
            return 0.2
        has_structure = bool(re.search(r'\n\d+\.\s|\n-\s|\n\*\s', text))
        has_period = text.rstrip()[-1:] in (".", "!", "?")
        base = 0.4
        if words >= 20:
            base += 0.2
        if has_structure:
            base += 0.2
        if has_period:
            base += 0.1
        return min(1.0, base)

    def to_examples(self, records: List[CorrectionRecord]) -> List[SelfImprovementExample]:
        examples = []
        for r in records:
            example = SelfImprovementExample(
                prompt=r.prompt,
                original_response=r.original_response,
                corrected_response=r.corrected_response,
                source=ExampleSource.CORRECTION,
                quality_score=r.score_after or self._quick_score(r.corrected_response),
                failure_reasons=r.validator_issues,
                correction_method=r.correction_method,
                metadata={"failed_query_id": r.failed_query_id, "score_before": r.score_before},
            )
            examples.append(example)
        return examples

    def get_stats(self) -> Dict:
        return self.stats
