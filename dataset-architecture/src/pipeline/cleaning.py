import re
import json
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ingestion import DatasetExample


class DataCleaner:
    REFUSAL_PATTERNS = [
        r"(?i)i'?m (just |only )?an ai",
        r"(?i)as an ai (assistant|language model)",
        r"(?i)i cannot (assist|help|answer|provide|complete)",
        r"(?i)i'?m (sorry|sorry, but) (i|i'?m|i am)",
        r"(?i)as a large language model",
        r"(?i)i am an ai language model",
        r"(?i)unfortunately, i cannot",
        r"(?i)i apologize, but i cannot",
        r"(?i)it is not (possible|appropriate)",
        r"(?i)i must (decline|refuse)",
    ]

    TEMPLATE_HALLUCINATION_PATTERNS = [
        r"(?i){\s*\{?\s*(prompt|question|input|instruction|query)",
        r"(?i)}{\s*\}?\s*(response|output|answer|completion)",
        r"\{\{[a-z_]+\}\}",
        r"<\|[a-z_]+\|>",
    ]

    UNSAFE_PATTERNS = [
        r"(?i)how to (hack|crack|exploit|bypass)",
        r"(?i)instructions for (making|creating|synthesizing) (drugs|weapons|explosives)",
        r"(?i)child (porn|abuse|exploitation)",
    ]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.stats = {
            "removed_empty": 0,
            "removed_too_short": 0,
            "removed_too_long": 0,
            "removed_refusal": 0,
            "removed_template_hallucination": 0,
            "removed_unsafe": 0,
            "total_removed": 0,
        }

    def clean(self, examples: List[DatasetExample], num_workers: int = 8) -> List[DatasetExample]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._filter_single, ex) for ex in examples]
            results = []
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        return results

    def _filter_single(self, example: DatasetExample) -> Optional[DatasetExample]:
        if self._is_empty(example):
            self.stats["removed_empty"] += 1
            self.stats["total_removed"] += 1
            return None

        if self._is_too_short(example):
            self.stats["removed_too_short"] += 1
            self.stats["total_removed"] += 1
            return None

        if self._is_too_long(example):
            self.stats["removed_too_long"] += 1
            self.stats["total_removed"] += 1
            return None

        if self._is_refusal(example):
            self.stats["removed_refusal"] += 1
            self.stats["total_removed"] += 1
            return None

        if self._is_template_hallucination(example):
            self.stats["removed_template_hallucination"] += 1
            self.stats["total_removed"] += 1
            return None

        if self._is_unsafe(example):
            self.stats["removed_unsafe"] += 1
            self.stats["total_removed"] += 1
            return None

        return example

    def _is_empty(self, example: DatasetExample) -> bool:
        return not example.output.strip()

    def _is_too_short(self, example: DatasetExample) -> bool:
        return (
            len(example.instruction.strip()) < 5
            or len(example.output.strip()) < 1
        )

    def _is_too_long(self, example: DatasetExample) -> bool:
        return len(example.output) > 32768 or len(example.instruction) > 4096

    def _is_refusal(self, example: DatasetExample) -> bool:
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, example.output):
                return True
        return False

    def _is_template_hallucination(self, example: DatasetExample) -> bool:
        for pattern in self.TEMPLATE_HALLUCINATION_PATTERNS:
            if re.search(pattern, example.output) or re.search(pattern, example.instruction):
                return True
        return False

    def _is_unsafe(self, example: DatasetExample) -> bool:
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, example.instruction) or re.search(pattern, example.output):
                return True
        return False

    def sanitize_text(self, text: str) -> str:
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '[URL]', text)
        text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[EMAIL]', text)
        return text

    def get_stats(self) -> Dict:
        return self.stats


class PIIRemover:
    PII_PATTERNS = {
        "email": r'[\w.-]+@[\w.-]+\.\w+',
        "phone": r'\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "ip": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "api_key": r'(?i)(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?key)\s*[:=]\s*["\']?[\w-]{16,}["\']?',
        "token": r'(?i)(?:bearer\s+)[\w-]+\.[\w-]+\.[\w-]+',
    }

    @classmethod
    def remove_pii(cls, text: str, replacement: str = "[REDACTED]") -> str:
        for pii_type, pattern in cls.PII_PATTERNS.items():
            text = re.sub(pattern, replacement, text)
        return text
