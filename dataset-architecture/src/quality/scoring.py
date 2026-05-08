import re
import math
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from ..pipeline.ingestion import DatasetExample


class QualityScorer:
    def __init__(self, config_path: str = "config/quality.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.dimensions = config["scoring"]["dimensions"]
        self.composite_threshold = config["scoring"].get("composite_threshold", 0.65)
        self.stats = {"scored": 0, "failed": 0}

    def score(self, examples: List[DatasetExample], num_workers: int = 8) -> List[DatasetExample]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._score_single, ex) for ex in examples]
            results = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _score_single(self, example: DatasetExample) -> Optional[DatasetExample]:
        try:
            scores = {}
            for dim_name, dim_config in self.dimensions.items():
                score = getattr(self, f"_score_{dim_name}", self._score_default)(example)
                scores[dim_name] = max(0.0, min(1.0, score))

            composite = sum(
                scores[dim] * self.dimensions[dim]["weight"]
                for dim in self.dimensions
            )

            example.quality_score = round(composite, 4)
            example.metadata["quality_scores"] = scores
            example.metadata["quality_dimensions"] = scores
            self.stats["scored"] += 1
            return example
        except Exception:
            self.stats["failed"] += 1
            return example

    def _score_relevance(self, example: DatasetExample) -> float:
        instruction_words = set(self._tokenize(example.instruction.lower()))
        output_words = set(self._tokenize(example.output.lower()))
        if not instruction_words or not output_words:
            return 0.4
        overlap = len(instruction_words & output_words)
        coverage = overlap / max(len(instruction_words), 1)
        return min(1.0, 0.3 + coverage * 0.7)

    def _score_correctness(self, example: DatasetExample) -> float:
        score = 0.55
        output = example.output.lower()

        positive_signals = [
            r"\b(?:def |class |function |import |return|print|console\.log|const |let |var )",
            r"\b(?:therefore|hence|thus|because|since|as a result)\b",
            r"\b(?:correct|accurate|verified|confirmed|valid)\b",
            r"```\w*\n",
            r"\b(?:solution|approach|method|algorithm|implementation)\b",
        ]
        negative_signals = [
            r"\b(?:maybe|perhaps|i think|i believe|not sure|possibly)\b",
            r"\b(?:incorrect|wrong|mistake|error|bug)\b.*\?",
            r"\b(?:i don't know|i'm not sure|i cannot determine)\b",
        ]

        for pattern in positive_signals:
            if re.search(pattern, output):
                score = min(1.0, score + 0.1)

        for pattern in negative_signals:
            if re.search(pattern, output):
                score = max(0.0, score - 0.15)

        return score

    def _score_completeness(self, example: DatasetExample) -> float:
        inst_len = len(example.instruction)
        out_len = len(example.output)

        if inst_len == 0:
            return 0.5

        ratio = out_len / max(inst_len, 1)
        if ratio < 0.3:
            return max(0.1, ratio * 1.5)
        elif ratio < 1.0:
            return 0.5 + ratio * 0.3
        elif ratio < 5.0:
            return 0.8
        else:
            return min(1.0, 0.8 + (ratio - 5.0) * 0.01)

    def _score_clarity(self, example: DatasetExample) -> float:
        output = example.output
        if not output:
            return 0.0

        score = 0.6
        sentences = re.split(r'[.!?]+', output)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_len = sum(len(s) for s in sentences) / len(sentences)
            if 20 <= avg_len <= 100:
                score += 0.1
            elif avg_len > 200:
                score -= 0.1

        structure_signals = [
            r'\n\d+\.\s', r'\n-\s', r'\n\*\s', r'```', r'\n\n',
            r'\b(first|second|finally|overall|in conclusion)\b',
            r'\n#{1,6}\s',
        ]
        for pattern in structure_signals:
            if re.search(pattern, output):
                score = min(1.0, score + 0.05)

        return min(1.0, max(0.0, score))

    def _score_safety(self, example: DatasetExample) -> float:
        output = example.output.lower()
        score = 1.0

        harmful_patterns = [
            r"(?i)hate (speech|content)",
            r"(?i)discriminat",
            r"(?i)violent|violence",
            r"(?i)explicit|nsfw",
            r"(?i)illegal activity",
        ]
        for pattern in harmful_patterns:
            if re.search(pattern, output):
                score -= 0.3

        return max(0.0, score)

    def _score_instruction_following(self, example: DatasetExample) -> float:
        instruction = example.instruction.lower()
        output = example.output.lower()
        score = 0.6

        constraint_patterns = [
            (r"\b(in |as |using )?\w+ (only|exactly|precisely)\b", 0.1),
            (r"\b(explain|describe|list|enumerate|summarize)\b", 0.05),
            (r"\b(step by step|step-by-step)\b", 0.1),
            (r"\b(short|concise|brief|detailed|comprehensive)\b", 0.05),
            (r"\b(code|program|function|script|implementation)\b", 0.05),
            (r"\b(example|instance|sample)\b", 0.05),
            (r"\b(format|structure|organize)\b", 0.05),
        ]

        for pattern, delta in constraint_patterns:
            if re.search(pattern, instruction):
                if re.search(pattern, output):
                    score += delta
                else:
                    score -= delta * 0.5

        return max(0.0, min(1.0, score))

    def _score_default(self, example: DatasetExample) -> float:
        return 0.5

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def score_pair(self, example: DatasetExample, reference_output: str) -> float:
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, example.output, reference_output).ratio()
        return similarity

    def get_stats(self) -> Dict:
        return self.stats


class HardExampleMiner:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def mine_hard_examples(self, examples: List[DatasetExample]) -> List[DatasetExample]:
        hard = [ex for ex in examples if ex.quality_score <= self.threshold]
        hard.sort(key=lambda x: x.quality_score)
        return hard

    def mine_edge_cases(self, examples: List[DatasetExample]) -> Dict[str, List[DatasetExample]]:
        categories = {}
        for ex in examples:
            cat = ex.category or "uncategorized"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ex)

        edge_cases = {}
        for cat, cat_examples in categories.items():
            cat_examples.sort(key=lambda x: x.quality_score)
            edge_cases[cat] = cat_examples[:max(10, len(cat_examples) // 20)]

        return edge_cases
