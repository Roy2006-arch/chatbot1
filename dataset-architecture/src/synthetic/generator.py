import json
import random
from typing import Dict, List, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import yaml

from ..pipeline.ingestion import DatasetExample
from .templates import (
    get_template_for_category,
    get_difficulty_instruction,
    COT_PROMPT,
    ERROR_TYPES,
)


class SyntheticDataGenerator:
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        synthetic_config = self.config.get("synthetic", {}).get("generation", {})
        self.enabled = synthetic_config.get("enabled", False)
        self.max_per_category = synthetic_config.get("max_synthetic_per_category", 100000)
        self.model = synthetic_config.get("model", "gpt-4o-mini")
        self.temperature = synthetic_config.get("temperature", 0.7)
        self.seed = self.config.get("pipeline", {}).get("seed", 42)
        self.stats = {"generated": 0, "by_category": {}}

    def generate(self, category: str, count: int = 1000, difficulty_range: range = range(1, 6)) -> List[DatasetExample]:
        templates = get_template_for_category(category)
        if not templates:
            return []

        examples = []
        random.seed(self.seed + hash(category) % (2**31))

        count = min(count, self.max_per_category)

        for i in range(count):
            template = random.choice(templates)
            difficulty = random.choice(difficulty_range)
            example = self._fill_template(template, category, difficulty)
            if example:
                examples.append(example)

        self.stats["generated"] += len(examples)
        self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + len(examples)

        return examples

    def _fill_template(self, template: Dict, category: str, difficulty: int) -> Optional[DatasetExample]:
        try:
            instruction = template.get("instruction", "")
            input_template = template.get("template", "")
            output_template = template.get("output_template", "")

            placeholder_values = self._generate_placeholder_values(category, difficulty)

            filled_input = self._fill_placeholders(input_template, placeholder_values) if input_template else ""
            filled_output = self._fill_placeholders(output_template, placeholder_values) if output_template else placeholder_values.get("final_answer", "")

            if difficulty >= 3 and category in ("general_reasoning", "math_logic"):
                instruction = f"{instruction}\n\n{COT_PROMPT}"

            difficulty_instruction = get_difficulty_instruction(difficulty)
            instruction = f"{instruction}\n\n{difficulty_instruction}"

            return DatasetExample(
                instruction=instruction,
                input=filled_input,
                output=filled_output,
                category=category,
                source="synthetic_generation",
                difficulty=difficulty,
                metadata={
                    "synthetic": True,
                    "template_type": category,
                    "generation_params": {
                        "model": self.model,
                        "temperature": self.temperature,
                    },
                    "generated_at": datetime.utcnow().isoformat(),
                },
            )
        except Exception:
            return None

    def _generate_placeholder_values(self, category: str, difficulty: int) -> Dict:
        values = {
            "language": random.choice(["python", "javascript", "typescript", "java", "cpp", "rust", "go"]),
            "difficulty": str(difficulty),
            "error_type": random.choice(ERROR_TYPES),
            "buggy_code": "# Buggy placeholder code",
            "fixed_code": "# Fixed placeholder code",
            "time_complexity": random.choice(["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(2^n)"]),
            "space_complexity": random.choice(["O(1)", "O(n)", "O(n²)", "O(log n)"]),
        }

        if category == "competitive_programming":
            values.update({
                "problem_statement": f"Given an array of integers, find the optimal solution for a level-{difficulty} problem.",
                "constraints": "1 ≤ n ≤ 10^5\n-10^9 ≤ arr[i] ≤ 10^9",
                "examples": "Input: [1, 2, 3, 4, 5]\nOutput: 15",
                "approach": f"Level-{difficulty} optimal approach description.",
                "code": "# Solution implementation",
            })

        elif category == "debugging":
            values.update({
                "root_cause": f"Root cause analysis for {random.choice(ERROR_TYPES)}.",
                "explanation": "Step-by-step explanation of the bug and fix.",
            })

        elif category == "system_design":
            values.update({
                "system_description": f"a scalable {random.choice(['chat', 'e-commerce', 'streaming', 'social media', 'payment'])} platform",
                "requirements": "Handle 1M DAU, 99.9% uptime, <200ms latency",
                "scale": f"{random.choice([1000, 10000, 100000, 1000000])} requests/second",
                "system_name": f"{random.choice(['Distributed', 'Scalable', 'Real-time', 'Cloud-native'])} {random.choice(['Platform', 'Service', 'System', 'Engine'])}",
                "architecture": "High-level architecture description.",
                "detailed_design": "Detailed component design.",
                "tradeoffs": "Key trade-offs considered.",
                "scaling": "Scaling strategies and bottlenecks.",
            })

        elif category in ("general_reasoning", "math_logic"):
            values.update({
                "problem": f"Level-{difficulty} reasoning problem description.",
                "reasoning_steps": "\n".join([f"{i+1}. Step {i+1} of the reasoning process." for i in range(3 + difficulty)]),
                "final_answer": f"The correct answer for level-{difficulty} problem.",
            })

        return values

    def _fill_placeholders(self, template: str, values: Dict) -> str:
        result = template
        for key, value in values.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    def generate_chain_of_thought(self, problem: str, category: str) -> DatasetExample:
        return DatasetExample(
            instruction=problem,
            input="",
            output=f"Let me work through this step by step.\n\n[Reasoning steps placeholder]\n\n**Answer:** [Final answer placeholder]",
            category=category,
            source="synthetic_cot",
            difficulty=3,
            metadata={
                "synthetic": True,
                "cot": True,
                "generated_at": datetime.utcnow().isoformat(),
            },
        )

    def get_stats(self) -> Dict:
        return self.stats
