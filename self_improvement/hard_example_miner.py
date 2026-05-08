import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict

from .schema import HardExample, SelfImprovementExample, ExampleSource

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feedback", "feedback.db")


class HardExampleMiner:
    def __init__(self, config: Optional[Dict] = None, db_path: str = DB_PATH):
        self.config = config or {}
        self.db_path = db_path
        self.enabled = self.config.get("enabled", True)
        self.min_occurrence_count = self.config.get("min_occurrence_count", 2)
        self.cluster_threshold = self.config.get("cluster_threshold", 0.75)
        self.max_examples_per_cluster = self.config.get("max_examples_per_cluster", 10)
        self.stats = {"loaded": 0, "clustered": 0, "mined": 0}

    def load_failed_queries(self, limit: int = 1000) -> List[Dict]:
        if not os.path.exists(self.db_path):
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("""
                SELECT * FROM failed_queries
                WHERE occurrence_count >= ?
                ORDER BY occurrence_count DESC, composite_score ASC
                LIMIT ?
            """, (self.min_occurrence_count, limit))
            rows = [dict(r) for r in cursor.fetchall()]
            self.stats["loaded"] = len(rows)
            return rows
        finally:
            conn.close()

    def load_all_failed(self, limit: int = 2000) -> List[Dict]:
        if not os.path.exists(self.db_path):
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("""
                SELECT * FROM failed_queries
                ORDER BY occurrence_count DESC
                LIMIT ?
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def _embed(self, text: str) -> Optional[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.config.get("embedding_model", "all-MiniLM-L6-v2"))
            return model.encode(text, normalize_embeddings=True).tolist()
        except ImportError:
            return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return dot

    def _token_similarity(self, text1: str, text2: str) -> float:
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def mine(self) -> List[HardExample]:
        if not self.enabled:
            return []

        records = self.load_all_failed()
        if not records:
            return []

        texts = [r.get("prompt", "") or "" for r in records]

        embeddings = []
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.config.get("embedding_model", "all-MiniLM-L6-v2"))
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except ImportError:
            pass

        clusters = self._cluster_by_failure_type(records, embeddings if len(embeddings) > 0 else None)

        hard_examples = []
        seen_prompts: Set[str] = set()
        for cluster_id, members in clusters.items():
            members.sort(key=lambda x: (x["occurrence_count"], -(x.get("composite_score", 0) or 0)), reverse=True)
            for member in members[:self.max_examples_per_cluster]:
                prompt = member.get("prompt", "")
                if prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)

                failure_reasons_str = member.get("failure_reasons", "[]")
                if isinstance(failure_reasons_str, str):
                    try:
                        reasons = json.loads(failure_reasons_str)
                    except (json.JSONDecodeError, TypeError):
                        reasons = [str(failure_reasons_str)]
                else:
                    reasons = list(failure_reasons_str) if failure_reasons_str else []

                score = member.get("composite_score", 0.0)
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0.0

                he = HardExample(
                    prompt=prompt,
                    response=member.get("response", ""),
                    category=self._categorize_prompt(prompt),
                    difficulty=min(5, max(1, member.get("occurrence_count", 1))),
                    failure_reasons=reasons[:3],
                    occurrence_count=member.get("occurrence_count", 1),
                    cluster_id=cluster_id,
                    quality_score=max(0.0, 1.0 - score),
                )
                hard_examples.append(he)

        self.stats["mined"] = len(hard_examples)
        return hard_examples

    def _cluster_by_failure_type(
        self, records: List[Dict], embeddings: Any = None
    ) -> Dict[int, List[Dict]]:
        clusters: Dict[int, List[Dict]] = {}
        assigned = [False] * len(records)
        cluster_id = 0

        for i, record in enumerate(records):
            if assigned[i]:
                continue
            clusters[cluster_id] = [record]
            assigned[i] = True
            text_i = record.get("prompt", "") or ""
            failure_i = record.get("failure_reasons", "[]") or "[]"

            for j in range(i + 1, len(records)):
                if assigned[j]:
                    continue
                text_j = records[j].get("prompt", "") or ""
                failure_j = records[j].get("failure_reasons", "[]") or "[]"

                sim = self._token_similarity(text_i, text_j)
                same_failure = self._failure_types_overlap(
                    json.loads(failure_i) if isinstance(failure_i, str) else failure_i,
                    json.loads(failure_j) if isinstance(failure_j, str) else failure_j,
                )

                if sim > self.cluster_threshold or same_failure:
                    clusters[cluster_id].append(records[j])
                    assigned[j] = True

            cluster_id += 1

        self.stats["clustered"] = cluster_id
        return clusters

    def _failure_types_overlap(self, a: List, b: List) -> bool:
        a_set = set(str(x).lower().strip() for x in a) if isinstance(a, list) else {str(a).lower()}
        b_set = set(str(x).lower().strip() for x in b) if isinstance(b, list) else {str(b).lower()}
        return len(a_set & b_set) > 0

    def _categorize_prompt(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        categories = {
            "code": ["code", "function", "program", "implement", "debug", "error", "bug", "syntax"],
            "reasoning": ["why", "explain", "reason", "analyze", "compare", "contrast", "difference"],
            "math": ["calculate", "solve", "equation", "math", "number", "sum", "value"],
            "technical": ["how to", "setup", "configure", "install", "deploy", "architecture"],
            "factual": ["what is", "who is", "when did", "where is", "definition"],
        }
        for cat, keywords in categories.items():
            if any(k in prompt_lower for k in keywords):
                return cat
        return "general"

    def to_examples(self, hard_examples: List[HardExample]) -> List[SelfImprovementExample]:
        return [
            SelfImprovementExample(
                prompt=he.prompt,
                original_response=he.response,
                source=ExampleSource.HARD_EXAMPLE,
                category=he.category,
                difficulty=he.difficulty,
                quality_score=he.quality_score,
                failure_reasons=he.failure_reasons,
                metadata={"cluster_id": he.cluster_id, "occurrence_count": he.occurrence_count},
            )
            for he in hard_examples
        ]

    def compute_statistics(self, hard_examples: List[HardExample]) -> Dict:
        if not hard_examples:
            return {}
        category_counts = Counter(he.category for he in hard_examples)
        reason_counts: Counter = Counter()
        for he in hard_examples:
            for reason in he.failure_reasons:
                reason_counts[reason] += 1
        return {
            "total_mined": len(hard_examples),
            "unique_clusters": len(set(he.cluster_id for he in hard_examples)),
            "by_category": dict(category_counts.most_common()),
            "top_failure_reasons": dict(reason_counts.most_common(10)),
            "avg_difficulty": sum(he.difficulty for he in hard_examples) / max(len(hard_examples), 1),
        }

    def get_stats(self) -> Dict:
        return self.stats
