import json
import random
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from ..schema import BuggyExample, BugCategory, Language, DEBUG_DATASET_CONFIG


class DebugFormatConverter:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.stats = {"converted": 0, "by_category": {}}

    INSTRUCTION_TEMPLATES = {
        "debug": "Fix the bug in the following code. Explain what was wrong and provide the corrected version.",
        "explain_error": "What error does this code produce? Explain the root cause and how to fix it.",
        "classify_bug": "Identify the type of bug in this code and suggest a fix strategy.",
        "improve": "The following code has a {category} bug. Find and fix it, then explain your changes.",
        "multiple_choice": "Which of the following best describes the bug in this code?\nA) {option_a}\nB) {option_b}\nC) {option_c}",
    }

    def convert_to_instructions(
        self,
        examples: List[BuggyExample],
        include_errors: bool = True,
    ) -> List[Dict]:
        records = []

        for ex in examples:
            records.append(self._make_debug_record(ex))
            records.append(self._make_explain_record(ex))
            if include_errors and ex.error_info:
                records.append(self._make_error_record(ex))

            self.stats["converted"] += 1
            cat_name = ex.category.value
            self.stats["by_category"][cat_name] = self.stats["by_category"].get(cat_name, 0) + 1

        return records

    def _make_debug_record(self, example: BuggyExample) -> Dict:
        lang = example.language.value
        code = example.buggy_code.code
        fixed = example.corrected_code.code

        instruction = self.INSTRUCTION_TEMPLATES["debug"]
        if example.difficulty >= 3:
            instruction = f"This is a challenging bug ({example.category.value.replace('_', ' ')}). {instruction}"

        intro = f"The bug is a **{example.category.value.replace('_', ' ')}** error."
        if example.error_info:
            intro += f"\n\n**Error:** {example.error_info.message}"

        explanation = f"{intro}\n\n## Root Cause Analysis\n{example.explanation}\n\n## Fix Strategy\n{example.fix_strategy}\n\n## Corrected Code\n```{lang}\n{fixed}\n```"

        if example.tags:
            explanation += f"\n\n**Tags:** {', '.join(example.tags)}"

        return {
            "instruction": instruction,
            "input": f"## Language\n{lang}\n\n## Buggy Code\n```{lang}\n{code}\n```",
            "output": explanation,
            "metadata": {
                "type": "debug",
                "language": lang,
                "category": example.category.value,
                "severity": example.severity.value,
                "difficulty": example.difficulty,
                "tags": example.tags,
            },
        }

    def _make_explain_record(self, example: BuggyExample) -> Dict:
        lang = example.language.value
        code = example.buggy_code.code
        fixed = example.corrected_code.code

        return {
            "instruction": self.INSTRUCTION_TEMPLATES["explain_error"],
            "input": f"```{lang}\n{code}\n```",
            "output": (
                f"## Error Analysis\n\n"
                f"**Bug Type:** {example.category.value.replace('_', ' ').title()}\n"
                f"**Severity:** {example.severity.name}\n\n"
                f"### What's Wrong\n{example.explanation}\n\n"
                f"### The Fix\n{example.fix_strategy}\n\n"
                f"### Corrected Version\n```{lang}\n{fixed}\n```"
            ),
            "metadata": {
                "type": "explain_error",
                "language": lang,
                "category": example.category.value,
                "difficulty": example.difficulty,
            },
        }

    def _make_error_record(self, example: BuggyExample) -> Dict:
        lang = example.language.value
        code = example.buggy_code.code
        err = example.error_info

        return {
            "instruction": f"Fix this {err.error_type.value} error in {lang}.",
            "input": (
                f"## Error\n{err.message}\n"
                + (f"**Line {err.line_number}:** " if err.line_number else "")
                + (f"\n```\n{err.stack_trace}\n```" if err.stack_trace else "")
                + f"\n\n## Code\n```{lang}\n{code}\n```"
            ),
            "output": (
                f"## Error Resolution\n\n"
                f"### Root Cause\n{example.explanation}\n\n"
                f"### Fix\n{example.fix_strategy}\n\n"
                f"### Corrected\n```{lang}\n{example.corrected_code.code}\n```"
            ),
            "metadata": {
                "type": "error_fix",
                "language": lang,
                "error_type": err.error_type.value,
                "category": example.category.value,
            },
        }

    def export_jsonl(self, records: List[Dict], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def export_for_finetuning(
        self,
        records: List[Dict],
        output_dir: str,
        framework: str = "transformers",
        split_ratio: tuple = (0.85, 0.10, 0.05),
    ):
        indices = list(range(len(records)))
        random.shuffle(indices)

        n = len(records)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        splits = {
            "train": [records[i] for i in indices[:train_end]],
            "validation": [records[i] for i in indices[train_end:val_end]],
            "test": [records[i] for i in indices[val_end:]],
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_records in splits.items():
            filepath = output_path / f"{split_name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for record in split_records:
                    if framework == "transformers":
                        out = {
                            "instruction": record["instruction"],
                            "input": record["input"],
                            "output": record["output"],
                        }
                    elif framework == "axolotl":
                        text = f"### Instruction:\n{record['instruction']}\n\n"
                        if record.get("input"):
                            text += f"### Input:\n{record['input']}\n\n"
                        text += f"### Response:\n{record['output']}"
                        out = {"text": text}
                    elif framework == "openai":
                        content = record["instruction"]
                        if record.get("input"):
                            content += "\n\n" + record["input"]
                        out = {
                            "messages": [
                                {"role": "user", "content": content},
                                {"role": "assistant", "content": record["output"]},
                            ]
                        }
                    else:
                        out = record
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

        type_dist = {}
        for r in records:
            t = r.get("metadata", {}).get("type", "unknown")
            type_dist[t] = type_dist.get(t, 0) + 1

        metadata = {
            "total": len(records),
            "splits": {k: len(v) for k, v in splits.items()},
            "framework": framework,
            "type_distribution": type_dist,
            "category_distribution": self.stats["by_category"],
            "generated_at": datetime.utcnow().isoformat(),
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def get_stats(self) -> Dict:
        return self.stats
