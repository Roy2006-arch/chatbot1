"""
training/dataset_bridge.py
--------------------------
Unified dataset loader that merges data from 4 sources:
  1. Self-improvement corrections (exported JSONL)
  2. Multimodal pipeline outputs
  3. Debug edge-cases / hard examples
  4. Filtering pipeline outputs

Outputs a standardized DatasetDict with train/eval/test splits,
curriculum stage tags (1-4), and source provenance tracking.
"""

import json
import logging
import os
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger("chatbot.dataset_bridge")

STAGE_LABELS = {1: "basic", 2: "intermediate", 3: "advanced", 4: "expert"}


@dataclass
class DatasetExample:
    prompt: str
    response: str
    source: str
    stage: int = 1
    quality_score: float = 0.0
    category: str = ""
    difficulty: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "source": self.source,
            "stage": self.stage,
            "quality_score": self.quality_score,
            "category": self.category,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }


@dataclass
class DatasetDict:
    train: List[DatasetExample]
    val: List[DatasetExample]
    test: List[DatasetExample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "train_size": len(self.train),
            "val_size": len(self.val),
            "test_size": len(self.test),
            "sources": self.metadata.get("sources", {}),
            "stage_distribution": self.metadata.get("stage_distribution", {}),
        }


def _load_self_improvement_data(
    data_dir: str = "self_improvement_data",
) -> List[DatasetExample]:
    """Load self-improvement corrections from exported JSONL files."""
    examples = []

    patterns = [
        os.path.join(data_dir, "train_combined.jsonl"),
        os.path.join(data_dir, "curriculum", "train_combined.jsonl"),
    ]

    for pattern in patterns:
        if not os.path.exists(pattern):
            continue
        with open(pattern, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    prompt = record.get("instruction") or record.get("prompt", "")
                    response = record.get("output") or record.get("chosen", "")
                    if not prompt or not response:
                        continue

                    quality = record.get("quality_score", 0.5)
                    category = record.get("category", "")
                    difficulty = record.get("difficulty", 1)
                    stage = min(max(int(difficulty), 1), 4)

                    ex = DatasetExample(
                        prompt=prompt,
                        response=response,
                        source="self_improvement",
                        stage=stage,
                        quality_score=float(quality),
                        category=category,
                        difficulty=difficulty,
                        metadata={"export_source": record.get("source", "correction")},
                    )
                    examples.append(ex)
                except (json.JSONDecodeError, ValueError):
                    continue

    log.info("[DatasetBridge] Loaded %d self-improvement examples", len(examples))
    return examples


def _load_multimodal_data(
    data_dir: str = "multimodal_data/output",
) -> List[DatasetExample]:
    """Load multimodal pipeline exported examples."""
    import glob
    examples = []

    patterns = [
        os.path.join(data_dir, "train.jsonl"),
        os.path.join(data_dir, "conversation*.jsonl"),
    ]

    for pattern in patterns:
        for fp in sorted(glob.glob(pattern)):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())

                        if "conversations" in record:
                            convs = record["conversations"]
                            user_parts = [c["value"] for c in convs if c.get("from") in ("human", "user")]
                            asst_parts = [c["value"] for c in convs if c.get("from") in ("gpt", "assistant")]
                            for u, a in zip(user_parts, asst_parts):
                                examples.append(DatasetExample(
                                    prompt=u,
                                    response=a,
                                    source="multimodal",
                                    stage=2,
                                    quality_score=0.7,
                                    category="multimodal",
                                    metadata={"format": "llava", "image": record.get("image", "")},
                                ))

                        elif "messages" in record:
                            msgs = record["messages"]
                            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                            if user and asst:
                                examples.append(DatasetExample(
                                    prompt=user,
                                    response=asst,
                                    source="multimodal",
                                    stage=2,
                                    quality_score=record.get("quality_score", 0.7),
                                    category="multimodal",
                                    metadata={"format": "openai"},
                                ))

                        elif "prompt" in record or "instruction" in record:
                            prompt = record.get("prompt", "") or record.get("instruction", "")
                            response = record.get("response", "") or record.get("output", "")
                            if prompt and response:
                                examples.append(DatasetExample(
                                    prompt=prompt,
                                    response=response,
                                    source="multimodal",
                                    stage=2,
                                    quality_score=float(record.get("quality_score", 0.7)),
                                    category=record.get("category", "multimodal"),
                                    metadata={"format": "generic"},
                                ))
                    except (json.JSONDecodeError, ValueError):
                        continue

    log.info("[DatasetBridge] Loaded %d multimodal examples", len(examples))
    return examples


def _load_filtering_data(
    data_dir: str = "filtering_data",
) -> List[DatasetExample]:
    """Load filtering pipeline outputs."""
    examples = []

    filtering_path = os.path.join(data_dir, "filtered_dataset.jsonl")
    if os.path.exists(filtering_path):
        with open(filtering_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    prompt = record.get("prompt", "") or record.get("instruction", "")
                    response = record.get("response", "") or record.get("output", "")
                    quality = record.get("quality_score") or record.get("composite_score", 0.5)
                    if prompt and response:
                        examples.append(DatasetExample(
                            prompt=prompt,
                            response=response,
                            source="filtering",
                            stage=1,
                            quality_score=float(quality) if quality else 0.5,
                            category=record.get("category", ""),
                            difficulty=record.get("difficulty", 1),
                            metadata={"filter_reasons": record.get("failure_reasons", [])},
                        ))
                except (json.JSONDecodeError, ValueError):
                    continue

    log.info("[DatasetBridge] Loaded %d filtering examples", len(examples))
    return examples


def _load_debug_edge_cases(
    data_dir: str = "debug_data",
) -> List[DatasetExample]:
    """Load manually-crafted debug edge cases."""
    examples = []

    debug_path = os.path.join(data_dir, "edge_cases.jsonl")
    if os.path.exists(debug_path):
        with open(debug_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    prompt = record.get("prompt", "") or record.get("instruction", "")
                    response = record.get("response", "") or record.get("output", "")
                    if prompt and response:
                        stage = record.get("stage", 4)
                        examples.append(DatasetExample(
                            prompt=prompt,
                            response=response,
                            source="debug",
                            stage=int(stage),
                            quality_score=float(record.get("quality_score", 1.0)),
                            category=record.get("category", "edge_case"),
                            difficulty=record.get("difficulty", 4),
                            metadata={"debug_type": record.get("type", "edge_case")},
                        ))
                except (json.JSONDecodeError, ValueError):
                    continue

    log.info("[DatasetBridge] Loaded %d debug edge cases", len(examples))
    return examples


def _compute_stage(ex: DatasetExample) -> int:
    """Assign a curriculum stage based on source and quality."""
    if ex.source == "debug":
        return max(ex.stage, 3)
    if ex.source == "multimodal":
        return max(ex.stage, 2)
    if ex.source == "self_improvement":
        return ex.stage
    return 1


def _deduplicate(examples: List[DatasetExample]) -> List[DatasetExample]:
    """Remove duplicate prompts (keep highest quality)."""
    seen: Dict[str, DatasetExample] = {}
    for ex in examples:
        key = ex.prompt[:100]
        if key not in seen or ex.quality_score > seen[key].quality_score:
            seen[key] = ex
    return list(seen.values())


def build_dataset(
    *,
    self_improvement_dir: str = "self_improvement_data",
    multimodal_dir: str = "multimodal_data/output",
    filtering_dir: str = "filtering_data",
    debug_dir: str = "debug_data",
    train_split: float = 0.8,
    val_split: float = 0.1,
    max_examples: int = 5000,
    curriculum_order: bool = True,
    seed: int = 42,
) -> DatasetDict:
    """
    Build a unified dataset from all 4 sources.

    Returns a DatasetDict with train/val/test splits, stage-tagged.
    """
    all_examples: List[DatasetExample] = []
    all_examples.extend(_load_self_improvement_data(self_improvement_dir))
    all_examples.extend(_load_multimodal_data(multimodal_dir))
    all_examples.extend(_load_filtering_data(filtering_dir))
    all_examples.extend(_load_debug_edge_cases(debug_dir))

    for ex in all_examples:
        ex.stage = _compute_stage(ex)

    all_examples = _deduplicate(all_examples)

    if curriculum_order:
        all_examples.sort(key=lambda x: (x.stage, -x.quality_score))

    if len(all_examples) > max_examples:
        all_examples.sort(key=lambda x: (0 if x.source == "debug" else 1, -x.quality_score))
        all_examples = all_examples[:max_examples]
        if curriculum_order:
            all_examples.sort(key=lambda x: (x.stage, -x.quality_score))

    rng = random.Random(seed)
    rng.shuffle(all_examples)

    n = len(all_examples)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train = all_examples[:n_train]
    val = all_examples[n_train:n_train + n_val]
    test = all_examples[n_train + n_val:]

    if curriculum_order:
        train.sort(key=lambda x: (x.stage, -x.quality_score))

    stage_dist = {}
    for s in range(1, 5):
        stage_dist[STAGE_LABELS[s]] = sum(1 for ex in all_examples if ex.stage == s)

    source_dist = {}
    for ex in all_examples:
        source_dist[ex.source] = source_dist.get(ex.source, 0) + 1

    dataset = DatasetDict(
        train=train,
        val=val,
        test=test,
        metadata={
            "total": n,
            "sources": source_dist,
            "stage_distribution": stage_dist,
            "train_split": train_split,
            "val_split": val_split,
            "built_at": datetime.utcnow().isoformat(),
        },
    )

    log.info(
        "[DatasetBridge] Built dataset: train=%d val=%d test=%d sources=%s stages=%s",
        len(train), len(val), len(test), source_dist, stage_dist,
    )
    return dataset


def export_dataset(
    dataset: DatasetDict,
    output_dir: str,
    formats: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Export dataset splits to JSONL files for training.

    Formats:
      - "transformers": {"instruction": ..., "output": ...}
      - "openai": {"messages": [...]}
    """
    formats = formats or ["transformers"]
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    for fmt in formats:
        for split_name, split_data in [
            ("train", dataset.train),
            ("val", dataset.val),
            ("test", dataset.test),
        ]:
            path = os.path.join(output_dir, f"{split_name}_{fmt}.jsonl")

            if fmt == "transformers":
                with open(path, "w", encoding="utf-8") as f:
                    for ex in split_data:
                        f.write(json.dumps({
                            "instruction": ex.prompt,
                            "output": ex.response,
                            "source": ex.source,
                            "stage": ex.stage,
                            "quality_score": ex.quality_score,
                            "category": ex.category,
                            "difficulty": ex.difficulty,
                        }) + "\n")

            elif fmt == "openai":
                with open(path, "w", encoding="utf-8") as f:
                    for ex in split_data:
                        f.write(json.dumps({
                            "messages": [
                                {"role": "user", "content": ex.prompt},
                                {"role": "assistant", "content": ex.response},
                            ],
                            "source": ex.source,
                            "stage": ex.stage,
                        }) + "\n")

            paths[f"{split_name}_{fmt}"] = path

    for stage in range(1, 5):
        stage_examples = [ex for ex in dataset.train if ex.stage == stage]
        if stage_examples:
            path = os.path.join(output_dir, f"train_stage{stage}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for ex in stage_examples:
                    f.write(json.dumps({
                        "instruction": ex.prompt,
                        "output": ex.response,
                        "source": ex.source,
                        "quality_score": ex.quality_score,
                        "category": ex.category,
                    }) + "\n")
            paths[f"train_stage{stage}"] = path

    meta_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(dataset.to_dict(), f, indent=2)
    paths["metadata"] = meta_path

    log.info("[DatasetBridge] Exported dataset to %s (%d files)", output_dir, len(paths))
    return paths
