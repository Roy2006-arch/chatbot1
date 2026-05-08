import json
import re
from typing import Dict, List, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ingestion import DatasetExample


class Preprocessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.stats = {"normalized": 0, "converted_encoding": 0, "standardized": 0}

    def process(self, examples: List[DatasetExample], num_workers: int = 8) -> List[DatasetExample]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._normalize_single, ex) for ex in examples]
            results = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _normalize_single(self, example: DatasetExample) -> Optional[DatasetExample]:
        try:
            example.instruction = self._clean_text(example.instruction)
            example.input = self._clean_text(example.input)
            example.output = self._clean_text(example.output)
            example = self._standardize_format(example)
            self.stats["normalized"] += 1
            return example
        except Exception:
            return None

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\u0000", "", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()
        return text

    def _standardize_format(self, example: DatasetExample) -> DatasetExample:
        if not example.instruction.endswith(("?", ".", "!", ":", "```")):
            example.instruction = example.instruction.strip() + "\n\n"
        if example.input and not example.input.startswith("\n"):
            example.input = "\n" + example.input.strip()
        return example

    def normalize_chat_format(self, example: DatasetExample) -> DatasetExample:
        conv = example.metadata.get("conversation", [])
        if conv and isinstance(conv, list):
            messages = []
            for msg in conv:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    messages.append({"role": role, "content": self._clean_text(content)})
            if messages:
                example.metadata["chat_messages"] = messages
        return example

    def extract_code_blocks(self, text: str) -> List[Dict]:
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [{"language": lang or "", "code": code.strip()} for lang, code in matches]

    def get_stats(self) -> Dict:
        return self.stats


class FormatStandardizer:
    FIELD_MAPPINGS = {
        "prompt": "instruction",
        "question": "instruction",
        "query": "instruction",
        "problem": "instruction",
        "messages": "conversation",
        "response": "output",
        "answer": "output",
        "completion": "output",
        "target": "output",
        "solution": "output",
        "context": "input",
        "source_code": "input",
    }

    @classmethod
    def standardize(cls, data: Dict) -> Dict:
        standardized = {"instruction": "", "input": "", "output": "", "metadata": {}}
        for key, value in data.items():
            mapped = cls.FIELD_MAPPINGS.get(key, key)
            if mapped in standardized:
                standardized[mapped] = cls._convert_value(value)
            else:
                standardized["metadata"][key] = value
        return standardized

    @staticmethod
    def _convert_value(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)
