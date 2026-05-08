from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FilterResult, FilterIssue, Severity


class SemanticSimilarityFilter:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model_name = self.config.get("model", "all-MiniLM-L6-v2")
        self.threshold = self.config.get("threshold", 0.85)
        self.batch_size = self.config.get("batch_size", 64)
        self.use_gpu = self.config.get("use_gpu", False)
        self._model = None
        self.stats = {"checked": 0, "duplicates_found": 0, "clusters": 0}

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                if self.use_gpu:
                    try:
                        self._model = self._model.to("cuda")
                    except Exception:
                        pass
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Install with: pip install sentence-transformers"
                )
        return self._model

    def compute_similarity(self, text1: str, text2: str) -> float:
        model = self._get_model()
        emb1 = model.encode(text1, normalize_embeddings=True)
        emb2 = model.encode(text2, normalize_embeddings=True)
        return float(emb1 @ emb2)

    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def find_near_duplicates(self, texts: List[str]) -> List[Tuple[int, int, float]]:
        if len(texts) < 2:
            return []

        model = self._get_model()
        embeddings = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)

        import numpy as np
        sim_matrix = embeddings @ embeddings.T
        pairs = []
        seen = set()
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if sim_matrix[i][j] >= self.threshold:
                    pairs.append((i, j, float(sim_matrix[i][j])))
                    seen.add(i)
                    seen.add(j)

        self.stats["duplicates_found"] = len(pairs)
        self.stats["clusters"] = len(set(i for i, _, _ in pairs))
        return pairs

    def cluster_duplicates(self, texts: List[str]) -> List[List[int]]:
        pairs = self.find_near_duplicates(texts)
        if not pairs:
            return []

        adj = {i: set() for i in range(len(texts))}
        for i, j, _ in pairs:
            adj[i].add(j)
            adj[j].add(i)

        visited = set()
        clusters = []
        for i in range(len(texts)):
            if i not in visited and adj.get(i):
                cluster = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        cluster.append(node)
                        stack.extend(adj.get(node, set()) - visited)
                if len(cluster) > 1:
                    clusters.append(cluster)

        return clusters

    def check(self, text: str, reference_texts: List[str]) -> FilterResult:
        self.stats["checked"] += 1
        issues: List[FilterIssue] = []

        if not reference_texts:
            return FilterResult(passed=True, score=1.0)

        all_texts = [text] + reference_texts
        model = self._get_model()
        embeddings = model.encode(all_texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)

        import numpy as np
        text_emb = embeddings[0:1]
        ref_embs = embeddings[1:]

        sims = (text_emb @ ref_embs.T)[0]
        max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0

        similar_indices = np.where(sims >= self.threshold)[0]
        duplicate_count = len(similar_indices)

        dim_scores = {
            "semantic_uniqueness": 1.0 - min(1.0, max_sim),
            "no_duplicates": 1.0 if duplicate_count == 0 else 1.0 - min(1.0, duplicate_count * 0.25),
        }

        if duplicate_count > 0:
            issues.append(FilterIssue(
                code="SEMANTIC_DUPLICATE",
                message=f"Semantically similar to {duplicate_count} existing text(s) (max sim: {max_sim:.3f})",
                severity=Severity.MEDIUM,
                dimension="semantic",
                details={"max_similarity": float(max_sim), "duplicate_count": int(duplicate_count)},
            ))

        composite = sum(dim_scores.values()) / len(dim_scores)
        passed = duplicate_count == 0
        if duplicate_count > 0:
            self.stats["duplicates_found"] += 1

        return FilterResult(
            passed=passed,
            score=composite,
            issues=issues,
            dimension_scores=dim_scores,
            metadata={"max_similarity": float(max_sim), "duplicate_count": int(duplicate_count)},
        )

    def check_batch_pairwise(self, texts: List[str], num_workers: int = 8) -> Dict[int, List[int]]:
        pairs = self.find_near_duplicates(texts)
        dup_map: Dict[int, List[int]] = {}
        for i, j, _ in pairs:
            dup_map.setdefault(i, []).append(j)
            dup_map.setdefault(j, []).append(i)
        return dup_map

    def get_stats(self) -> Dict:
        return self.stats
