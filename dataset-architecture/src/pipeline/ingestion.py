import json
import os
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import pandas as pd


@dataclass
class DatasetExample:
    instruction: str
    output: str
    input: str = ""
    category: str = ""
    source: str = ""
    difficulty: int = 1
    quality_score: float = 0.0
    metadata: Optional[Dict] = None
    id: str = ""

    def __post_init__(self):
        if not self.id:
            content = f"{self.instruction}{self.input}{self.output}"
            self.id = hashlib.sha256(content.encode()).hexdigest()[:16]
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetExample":
        return cls(
            instruction=data.get("instruction", ""),
            input=data.get("input", ""),
            output=data.get("output", ""),
            category=data.get("category", ""),
            source=data.get("source", ""),
            difficulty=data.get("difficulty", 1),
            quality_score=data.get("quality_score", 0.0),
            metadata=data.get("metadata", {}),
            id=data.get("id", ""),
        )


class DataIngestor:
    SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".parquet", ".tsv"}

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.stats = {"total_loaded": 0, "by_source": {}, "errors": []}

    def ingest_from_path(self, path: str, category: str = "") -> Generator[DatasetExample, None, None]:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path_obj.is_file():
            yield from self._ingest_file(path_obj, category)
        elif path_obj.is_dir():
            for file_path in path_obj.rglob("*"):
                if file_path.suffix in self.SUPPORTED_EXTENSIONS:
                    yield from self._ingest_file(file_path, category)

    def _ingest_file(self, file_path: Path, category: str) -> Generator[DatasetExample, None, None]:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".json":
                yield from self._ingest_json(file_path, category)
            elif suffix == ".jsonl":
                yield from self._ingest_jsonl(file_path, category)
            elif suffix == ".csv":
                yield from self._ingest_csv(file_path, category)
            elif suffix == ".tsv":
                yield from self._ingest_csv(file_path, category, delimiter="\t")
            elif suffix == ".parquet":
                yield from self._ingest_parquet(file_path, category)
        except Exception as e:
            self.stats["errors"].append({"file": str(file_path), "error": str(e)})

    def _ingest_json(self, file_path: Path, category: str) -> Generator[DatasetExample, None, None]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield self._normalize_item(item, category, str(file_path))
        elif isinstance(data, dict):
            yield self._normalize_item(data, category, str(file_path))

    def _ingest_jsonl(self, file_path: Path, category: str) -> Generator[DatasetExample, None, None]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        yield self._normalize_item(item, category, str(file_path))
                    except json.JSONDecodeError:
                        continue

    def _ingest_csv(self, file_path: Path, category: str, delimiter: str = ",") -> Generator[DatasetExample, None, None]:
        df = pd.read_csv(file_path, delimiter=delimiter)
        for _, row in df.iterrows():
            yield self._normalize_item(row.to_dict(), category, str(file_path))

    def _ingest_parquet(self, file_path: Path, category: str) -> Generator[DatasetExample, None, None]:
        df = pd.read_parquet(file_path)
        for _, row in df.iterrows():
            yield self._normalize_item(row.to_dict(), category, str(file_path))

    def _normalize_item(self, item: Dict, category: str, source: str) -> DatasetExample:
        normalized = {
            "instruction": self._extract_field(item, ["instruction", "prompt", "question", "query", "problem", "messages[-1].content"]),
            "input": self._extract_field(item, ["input", "context", "code", "source", "messages", ""]),
            "output": self._extract_field(item, ["output", "response", "answer", "completion", "target", "solution"]),
            "category": category or item.get("category", ""),
            "source": source,
            "difficulty": item.get("difficulty", 1),
            "quality_score": item.get("quality_score", 0.0),
            "metadata": {
                k: v for k, v in item.items()
                if k not in ["instruction", "input", "output", "category", "difficulty", "quality_score"]
            },
        }

        if isinstance(normalized["input"], list):
            normalized["input"] = json.dumps(normalized["input"])

        if not normalized["instruction"] and "messages" in item:
            messages = item["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                normalized["instruction"] = messages[0].get("content", "") if isinstance(messages[0], dict) else str(messages[0])
                normalized["output"] = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
                normalized["metadata"]["conversation"] = messages

        return DatasetExample(**normalized)

    def _extract_field(self, item: Dict, candidates: List[str]) -> str:
        for field in candidates:
            if field in item and item[field] is not None:
                value = item[field]
                if isinstance(value, str):
                    return value
                if isinstance(value, (list, dict)):
                    return json.dumps(value)
                return str(value)
            if "." in field:
                parts = field.split(".")
                if parts[0] == "messages" and len(parts) > 2:
                    idx = int(parts[1].strip("[]"))
                    key = parts[2]
                    if "messages" in item and isinstance(item["messages"], list):
                        if idx < len(item["messages"]) and isinstance(item["messages"][idx], dict):
                            return item["messages"][idx].get(key, "")
        return ""

    def load_huggingface_dataset(self, path: str, split: str = "train", max_samples: Optional[int] = None) -> List[DatasetExample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        dataset = load_dataset(path, split=split, streaming=max_samples is not None)
        if max_samples:
            dataset = dataset.take(max_samples)

        examples = []
        for item in dataset:
            examples.append(self._normalize_item(item, "", path))
        return examples

    def get_stats(self) -> Dict:
        return self.stats


class HuggingFaceLoader:
    def __init__(self, config: Dict):
        self.config = config

    def load_competitive_programming(self) -> Generator[DatasetExample, None, None]:
        datasets_config = self.config["datasets"]["competitive_programming"]
        ingestor = DataIngestor()
        for ds in datasets_config:
            if ds["source"] == "huggingface":
                yield from ingestor.load_huggingface_dataset(
                    ds["path"], max_samples=ds.get("max_samples")
                )

    def load_conversational(self) -> Generator[DatasetExample, None, None]:
        datasets_config = self.config["datasets"]["conversational_ai"]
        ingestor = DataIngestor()
        for ds in datasets_config:
            if ds["source"] == "huggingface":
                yield from ingestor.load_huggingface_dataset(
                    ds["path"], max_samples=ds.get("max_samples")
                )

    def load_reasoning(self) -> Generator[DatasetExample, None, None]:
        datasets_config = self.config["datasets"]["general_reasoning"]
        ingestor = DataIngestor()
        for ds in datasets_config:
            if ds["source"] == "huggingface":
                yield from ingestor.load_huggingface_dataset(
                    ds["path"], max_samples=ds.get("max_samples")
                )

    def load_math(self) -> Generator[DatasetExample, None, None]:
        datasets_config = self.config["datasets"]["math_logic"]
        ingestor = DataIngestor()
        for ds in datasets_config:
            if ds["source"] == "huggingface":
                yield from ingestor.load_huggingface_dataset(
                    ds["path"], max_samples=ds.get("max_samples")
                )
