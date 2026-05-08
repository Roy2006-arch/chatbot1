import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import yaml

from .ingestion import DatasetExample


class DataExporter:
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.export_config = self.config.get("export", {})
        self.split_config = self.export_config.get("train_val_test_split", {"train": 0.85, "validation": 0.10, "test": 0.05})

    def export(self, examples: List[DatasetExample], output_dir: str, format: str = "jsonl", split: bool = True):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if split:
            train, val, test = self._split(examples)
            splits = {"train": train, "validation": val, "test": test}
        else:
            splits = {"full": examples}

        for split_name, split_data in splits.items():
            self._write_split(split_data, output_path, split_name, format)

        return output_path

    def _split(self, examples: List[DatasetExample]) -> Tuple[List[DatasetExample], List[DatasetExample], List[DatasetExample]]:
        indices = list(range(len(examples)))
        random.shuffle(indices)

        n = len(examples)
        train_end = int(n * self.split_config["train"])
        val_end = train_end + int(n * self.split_config["validation"])

        train = [examples[i] for i in indices[:train_end]]
        val = [examples[i] for i in indices[train_end:val_end]]
        test = [examples[i] for i in indices[val_end:]]

        return train, val, test

    def _write_split(self, data: List[DatasetExample], output_dir: Path, split_name: str, format: str):
        if format == "jsonl":
            self._write_jsonl(data, output_dir, split_name)
        elif format == "json":
            self._write_json(data, output_dir, split_name)
        else:
            self._write_jsonl(data, output_dir, split_name)

    def _write_jsonl(self, data: List[DatasetExample], output_dir: Path, split_name: str):
        filepath = output_dir / f"{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

    def _write_json(self, data: List[DatasetExample], output_dir: Path, split_name: str):
        filepath = output_dir / f"{split_name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([ex.to_dict() for ex in data], f, ensure_ascii=False, indent=2)

    def export_for_training(self, examples: List[DatasetExample], output_dir: str, framework: str = "transformers"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train, val, test = self._split(examples)

        if framework == "transformers":
            self._export_transformers(train, val, test, output_path)
        elif framework == "axolotl":
            self._export_axolotl(train, val, test, output_path)
        elif framework == "openai":
            self._export_openai(train, val, test, output_path)
        else:
            self._write_jsonl(train, output_path, "train")
            self._write_jsonl(val, output_path, "validation")
            self._write_jsonl(test, output_path, "test")

        return output_path

    def _export_transformers(self, train: List[DatasetExample], val: List[DatasetExample], test: List[DatasetExample], output_dir: Path):
        for split_name, data in [("train", train), ("validation", val), ("test", test)]:
            filepath = output_dir / f"{split_name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for ex in data:
                    record = {
                        "instruction": ex.instruction,
                        "input": ex.input,
                        "output": ex.output,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _export_axolotl(self, train: List[DatasetExample], val: List[DatasetExample], test: List[DatasetExample], output_dir: Path):
        for split_name, data in [("train", train), ("validation", val), ("test", test)]:
            filepath = output_dir / f"{split_name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for ex in data:
                    text = f"### Instruction:\n{ex.instruction}\n\n"
                    if ex.input:
                        text += f"### Input:\n{ex.input}\n\n"
                    text += f"### Response:\n{ex.output}"
                    record = {"text": text}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _export_openai(self, train: List[DatasetExample], val: List[DatasetExample], test: List[DatasetExample], output_dir: Path):
        for split_name, data in [("train", train), ("validation", val), ("test", test)]:
            filepath = output_dir / f"{split_name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for ex in data:
                    messages = [
                        {"role": "user", "content": ex.instruction + ("\n" + ex.input if ex.input else "")},
                        {"role": "assistant", "content": ex.output},
                    ]
                    record = {"messages": messages}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def export_metadata(self, examples: List[DatasetExample], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        categories = {}
        for ex in examples:
            cat = ex.category or "uncategorized"
            if cat not in categories:
                categories[cat] = {"count": 0, "total_score": 0.0, "difficulty_sum": 0}
            categories[cat]["count"] += 1
            categories[cat]["total_score"] += ex.quality_score
            categories[cat]["difficulty_sum"] += ex.difficulty

        for cat in categories:
            c = categories[cat]
            c["avg_quality"] = round(c["total_score"] / c["count"], 3) if c["count"] > 0 else 0
            c["avg_difficulty"] = round(c["difficulty_sum"] / c["count"], 2) if c["count"] > 0 else 0
            del c["total_score"]
            del c["difficulty_sum"]

        metadata = {
            "total_examples": len(examples),
            "categories": categories,
            "exported_at": datetime.utcnow().isoformat(),
            "avg_quality": round(sum(ex.quality_score for ex in examples) / len(examples), 3) if examples else 0,
        }

        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata
