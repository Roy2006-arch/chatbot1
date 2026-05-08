import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .schema import SelfImprovementExample, ExampleSource, CorrectionMethod

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feedback", "feedback.db")


class QualityCurator:
    def __init__(self, config: Optional[Dict] = None, db_path: str = DB_PATH):
        self.config = config or {}
        self.db_path = db_path
        self.enabled = self.config.get("enabled", True)
        self.min_quality_score = self.config.get("min_quality_score", 0.8)
        self.min_response_length = self.config.get("min_response_length", 20)
        self.max_examples_per_category = self.config.get("max_examples_per_category", 50)
        self.stats = {"scanned": 0, "extracted": 0, "filtered_out": 0}

    def load_high_quality_conversations(self, limit: int = 1000) -> List[Dict]:
        if not os.path.exists(self.db_path):
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("""
                SELECT c.*, fb.vote as user_vote
                FROM conversations c
                LEFT JOIN feedback fb ON fb.conv_id = c.conv_id AND fb.turn_index = c.turn_index
                WHERE c.role = 'assistant'
                AND c.content IS NOT NULL
                AND LENGTH(c.content) >= ?
                ORDER BY c.composite_score DESC, c.timestamp_utc DESC
                LIMIT ?
            """, (self.min_response_length, limit))
            rows = [dict(r) for r in cursor.fetchall()]
            self.stats["scanned"] = len(rows)
            return rows
        finally:
            conn.close()

    def load_by_quality_threshold(self, threshold: float = 0.8, limit: int = 500) -> List[Dict]:
        if not os.path.exists(self.db_path):
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("""
                SELECT c.*, fq.id as failed_query_id
                FROM conversations c
                LEFT JOIN failed_queries fq ON fq.conv_id = c.conv_id
                WHERE c.role = 'assistant'
                AND c.composite_score >= ?
                AND c.content IS NOT NULL
                ORDER BY c.composite_score DESC
                LIMIT ?
            """, (threshold, limit))
            rows = [dict(r) for r in cursor.fetchall()]
            return rows
        finally:
            conn.close()

    def score_response_quality(self, prompt: str, response: str) -> Dict[str, float]:
        scores = {}
        total_words = len(response.split())
        prompt_words = len(prompt.split())

        if total_words == 0:
            return {"length": 0.0, "structure": 0.0, "completeness": 0.0, "composite": 0.0}

        length_score = min(1.0, total_words / 50)
        scores["length"] = length_score

        structure_score = 0.0
        if re.search(r'\n\d+\.\s|\n-\s|\n\*\s', response):
            structure_score += 0.4
        if re.search(r'```', response):
            structure_score += 0.3
        if re.search(r'\n#{1,3}\s', response):
            structure_score += 0.3
        scores["structure"] = structure_score

        has_intro = bool(re.search(r'(?i)^(here|let|the|this|to|there)', response.strip()))
        has_conclusion = bool(re.search(r'(?i)(in conclusion|to summarize|overall|finally|in short)', response))
        completeness = 0.3
        if has_intro:
            completeness += 0.3
        if has_conclusion:
            completeness += 0.2
        if total_words >= prompt_words * 2:
            completeness += 0.2
        scores["completeness"] = min(1.0, completeness)

        composite = sum(scores.values()) / len(scores)
        scores["composite"] = composite
        return scores

    def curate(self, limit: int = 500) -> List[SelfImprovementExample]:
        if not self.enabled:
            return []

        records = self.load_by_quality_threshold(self.min_quality_score, limit)
        examples = []
        category_counts: Dict[str, int] = {}

        for record in records:
            prompt = self._get_prompt_for_turn(record.get("conv_id", ""), record.get("turn_index", 0))
            response = record.get("content", "")

            if len(response) < self.min_response_length:
                self.stats["filtered_out"] += 1
                continue

            scores = self.score_response_quality(prompt, response)
            category = record.get("failure_reasons", "") or "general"
            if isinstance(category, str):
                try:
                    reasons = json.loads(category)
                    category = reasons[0] if reasons else "general"
                except (json.JSONDecodeError, IndexError):
                    category = "general"

            if category_counts.get(category, 0) >= self.max_examples_per_category:
                self.stats["filtered_out"] += 1
                continue

            example = SelfImprovementExample(
                prompt=prompt,
                original_response="",
                corrected_response=response,
                source=ExampleSource.HIGH_QUALITY,
                category=category,
                quality_score=scores["composite"],
                metadata={
                    "conv_id": record.get("conv_id", ""),
                    "turn_index": record.get("turn_index", 0),
                    "quality_scores": scores,
                    "auto_score": record.get("composite_score", 0),
                    "grade": record.get("grade", ""),
                },
            )
            examples.append(example)
            category_counts[category] = category_counts.get(category, 0) + 1
            self.stats["extracted"] += 1

        examples.sort(key=lambda x: x.quality_score, reverse=True)
        return examples

    def _get_prompt_for_turn(self, conv_id: str, turn_index: int) -> str:
        if not os.path.exists(self.db_path) or not conv_id:
            return ""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT content FROM conversations WHERE conv_id = ? AND role = 'user' AND turn_index = ?",
                (conv_id, max(0, turn_index - 1)),
            )
            row = cursor.fetchone()
            return row[0] if row else ""
        except Exception:
            return ""
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        return self.stats
