import hashlib
import json
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from ..pipeline.ingestion import DatasetExample


class Deduplicator:
    def __init__(self, config_path: str = "config/quality.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        dedup_config = config.get("deduplication", {})
        self.threshold = dedup_config.get("threshold", 0.85)
        self.ngram_size = dedup_config.get("ngram_size", 5)
        self.num_permutations = dedup_config.get("num_permutations", 128)
        self.use_exact_dedup = dedup_config.get("use_exact_dedup", True)
        self.use_semantic_dedup = dedup_config.get("use_semantic_dedup", False)
        self.stats = {"exact_removed": 0, "near_removed": 0, "total_removed": 0, "kept": 0}

    def deduplicate(self, examples: List[DatasetExample], num_workers: int = 8) -> List[DatasetExample]:
        if self.use_exact_dedup:
            examples, removed = self._exact_dedup(examples)
            self.stats["exact_removed"] = removed

        if self.use_semantic_dedup and len(examples) > 1:
            examples = self._minhash_dedup(examples, num_workers)

        self.stats["kept"] = len(examples)
        return examples

    def _exact_dedup(self, examples: List[DatasetExample]) -> Tuple[List[DatasetExample], int]:
        seen: Set[str] = set()
        unique = []
        removed = 0

        for ex in examples:
            key = hashlib.sha256(
                f"{ex.instruction}|{ex.input}|{ex.output}".encode()
            ).hexdigest()

            if key not in seen:
                seen.add(key)
                unique.append(ex)
            else:
                removed += 1

        self.stats["total_removed"] += removed
        return unique, removed

    def _minhash_dedup(self, examples: List[DatasetExample], num_workers: int) -> List[DatasetExample]:
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            return examples

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_permutations)
        minhashes = {}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self._compute_minhash, ex, i): (ex, i)
                for i, ex in enumerate(examples)
            }
            for future in as_completed(future_to_idx):
                ex, idx = future_to_idx[future]
                mh = future.result()
                if mh is not None:
                    minhashes[idx] = mh
                    lsh.insert(f"ex_{idx}", mh)

        seen = set()
        unique = []
        for idx in sorted(minhashes.keys()):
            if idx in seen:
                continue
            result = lsh.query(minhashes[idx])
            similar = [int(r.split("_")[1]) for r in result if r.startswith("ex_")]
            kept = False
            for s in sorted(similar, key=lambda x: examples[x].quality_score, reverse=True):
                if s not in seen:
                    if not kept:
                        unique.append(examples[s])
                        kept = True
                    seen.add(s)

        removed = len(examples) - len(unique)
        self.stats["near_removed"] += removed
        self.stats["total_removed"] += removed
        return unique

    def _compute_minhash(self, example: DatasetExample, idx: int):
        try:
            from datasketch import MinHash
            text = f"{example.instruction} {example.input} {example.output}"
            tokens = self._tokenize(text)
            if len(tokens) < self.ngram_size:
                return None
            mh = MinHash(num_perm=self.num_permutations)
            ngrams = self._get_ngrams(tokens)
            for ng in ngrams:
                mh.update(ng.encode("utf-8"))
            return mh
        except Exception:
            return None

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        return [" ".join(tokens[i:i + self.ngram_size])
                for i in range(len(tokens) - self.ngram_size + 1)]

    def get_stats(self) -> Dict:
        return self.stats
