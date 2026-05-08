import json
import random
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from ..schema import ReasoningExample, ContradictionPair, ReasoningTask, REASONING_TEMPLATES


class ReasoningFormatConverter:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.stats = {"converted": 0, "by_task": {}}

    def convert_to_instructions(
        self,
        examples: List[ReasoningExample],
        include_cot: bool = True,
        include_verification: bool = True,
    ) -> List[Dict]:
        records = []

        for ex in examples:
            record = self._build_record(ex, include_cot, include_verification)
            if record:
                records.append(record)
                task_name = ex.reasoning_task.value
                self.stats["by_task"][task_name] = self.stats["by_task"].get(task_name, 0) + 1

        self.stats["converted"] = len(records)
        return records

    def _build_record(self, ex: ReasoningExample, include_cot: bool, include_verification: bool) -> Optional[Dict]:
        template = REASONING_TEMPLATES.get(ex.reasoning_task, {})

        instruction = template.get("instruction", "Solve this reasoning problem step by step.")
        cot_output = self._build_cot_output(ex) if include_cot else ex.final_answer

        if include_verification and ex.verification:
            cot_output += f"\n\n**Verification:** {ex.verification}"

        if ex.common_errors:
            cot_output += f"\n\n**Common Errors to Avoid:**\n"
            for err in ex.common_errors:
                cot_output += f"- {err}\n"

        if ex.alternative_approaches:
            cot_output += f"\n**Alternative Approaches:**\n"
            for alt in ex.alternative_approaches[:2]:
                cot_output += f"- {alt}\n"

        return {
            "instruction": instruction,
            "input": self._build_input(ex),
            "output": cot_output,
            "metadata": {
                "reasoning_type": ex.reasoning_type.value,
                "reasoning_task": ex.reasoning_task.value,
                "difficulty": ex.difficulty.value,
                "domain": ex.domain,
                "tags": ex.tags,
            },
        }

    def _build_input(self, ex: ReasoningExample) -> str:
        parts = []
        if ex.question:
            parts.append(f"## Question\n{ex.question}")
        if ex.context:
            parts.append(f"\n## Context\n{ex.context}")
        if ex.wrong_answer:
            parts.append(f"\n## Flawed Reasoning\n{ex.wrong_answer}")
        return "\n".join(parts)

    def _build_cot_output(self, ex: ReasoningExample) -> str:
        if not ex.reasoning_steps:
            return ex.final_answer

        parts = ["Let me work through this step by step.\n"]
        for i, step in enumerate(ex.reasoning_steps):
            step_text = f"**Step {i+1}:** {step.content}"
            if step.justification:
                step_text += f"\n*Why:* {step.justification}"
            if step.alternatives:
                step_text += f"\n*Alternative:* {step.alternatives[0]}"
            parts.append(step_text)

        parts.append(f"\n**Final Answer:** {ex.final_answer}")

        if ex.verification:
            parts.append(f"\n**Verification:** {ex.verification}")

        return "\n\n".join(parts)

    def convert_contradiction_pairs(self, pairs: List[ContradictionPair]) -> List[Dict]:
        records = []
        for pair in pairs:
            verdict_a = "consistent" if pair.consistent_a else "contradictory"
            verdict_b = "consistent" if pair.consistent_b else "contradictory"

            record = {
                "instruction": "Determine whether the following conclusions are logically consistent or contradictory given the premise.",
                "input": f"## Premise\n{pair.premise}\n\n## Conclusion A\n{pair.conclusion_a}\n\n## Conclusion B\n{pair.conclusion_b}",
                "output": f"## Analysis\n{pair.explanation}\n\n## Verdict\nConclusion A is **{verdict_a}** with the premise.\nConclusion B is **{verdict_b}** with the premise.",
                "metadata": {
                    "type": "contradiction_detection",
                    "consistent_a": pair.consistent_a,
                    "consistent_b": pair.consistent_b,
                    "difficulty": pair.difficulty.value,
                },
            }
            records.append(record)
        return records

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
            rt = r.get("metadata", {}).get("reasoning_task", "unknown")
            type_dist[rt] = type_dist.get(rt, 0) + 1

        metadata = {
            "total": len(records),
            "splits": {k: len(v) for k, v in splits.items()},
            "framework": framework,
            "reasoning_type_distribution": type_dist,
            "generated_at": datetime.utcnow().isoformat(),
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def get_stats(self) -> Dict:
        return self.stats
