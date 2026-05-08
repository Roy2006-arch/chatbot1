from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from ..schema import ReasoningTask, ReasoningType, Difficulty, REASONING_TEMPLATES


@dataclass
class AnnotationGuideline:
    task: ReasoningTask
    required_elements: List[str]
    quality_criteria: List[str]
    examples_count: int
    min_steps: int
    max_steps: int
    verification_required: bool


ANNOTATION_GUIDELINES = {
    ReasoningTask.MULTI_STEP: AnnotationGuideline(
        task=ReasoningTask.MULTI_STEP,
        required_elements=["question", "reasoning_steps", "final_answer", "verification"],
        quality_criteria=[
            "Each step must advance the reasoning",
            "Steps must be logically connected",
            "Final answer must follow from the steps",
            "No leaps in logic - each inference must be explicit",
        ],
        examples_count=100_000,
        min_steps=3,
        max_steps=12,
        verification_required=True,
    ),
    ReasoningTask.LOGICAL_DEDUCTION: AnnotationGuideline(
        task=ReasoningTask.LOGICAL_DEDUCTION,
        required_elements=["premises", "inference_rules", "conclusion", "validity_check"],
        quality_criteria=[
            "Premises must be clearly stated",
            "Inference rules must be named (modus ponens, etc.)",
            "Conclusion must follow validly from premises",
            "Fallacies must be explicitly identified and avoided",
        ],
        examples_count=60_000,
        min_steps=2,
        max_steps=8,
        verification_required=True,
    ),
    ReasoningTask.CONTRADICTION_DETECTION: AnnotationGuideline(
        task=ReasoningTask.CONTRADICTION_DETECTION,
        required_elements=["statement_a", "statement_b", "analysis", "verdict"],
        quality_criteria=[
            "Contradiction must be explicit and unambiguous",
            "Analysis must explain why statements conflict",
            "Non-contradictory examples should also be included",
            "Self-referential paradoxes are valid",
        ],
        examples_count=40_000,
        min_steps=2,
        max_steps=5,
        verification_required=True,
    ),
    ReasoningTask.DECOMPOSITION: AnnotationGuideline(
        task=ReasoningTask.DECOMPOSITION,
        required_elements=["complex_problem", "sub_problems", "sub_solutions", "integrated_solution"],
        quality_criteria=[
            "Decomposition must be exhaustive (no missing parts)",
            "Sub-problems must be independently solvable",
            "Integration must show how sub-solutions combine",
            "Decomposition strategy must be justified",
        ],
        examples_count=40_000,
        min_steps=3,
        max_steps=10,
        verification_required=True,
    ),
    ReasoningTask.PLANNING: AnnotationGuideline(
        task=ReasoningTask.PLANNING,
        required_elements=["goal", "constraints", "plan_steps", "dependencies", "verification"],
        quality_criteria=[
            "Plan must address all constraints",
            "Dependencies between steps must be identified",
            "Resource requirements must be estimated",
            "Contingency plans for failure modes",
        ],
        examples_count=30_000,
        min_steps=3,
        max_steps=10,
        verification_required=True,
    ),
    ReasoningTask.DEBUGGING_REASONING: AnnotationGuideline(
        task=ReasoningTask.DEBUGGING_REASONING,
        required_elements=["problem", "flawed_reasoning", "error_analysis", "correct_reasoning"],
        quality_criteria=[
            "Error must be precisely localized",
            "Root cause analysis must be provided",
            "Correct reasoning must fix all identified issues",
            "Common variants of the error should be mentioned",
        ],
        examples_count=20_000,
        min_steps=3,
        max_steps=8,
        verification_required=True,
    ),
}


class AnnotationStrategy:
    def __init__(self):
        self.guidelines = ANNOTATION_GUIDELINES

    def get_guideline(self, task: ReasoningTask) -> Optional[AnnotationGuideline]:
        return self.guidelines.get(task)

    def get_all_guidelines(self) -> Dict[str, AnnotationGuideline]:
        return {k.value: v for k, v in self.guidelines.items()}

    def build_annotation_prompt(self, task: ReasoningTask) -> str:
        guideline = self.get_guideline(task)
        if not guideline:
            return ""

        template = REASONING_TEMPLATES.get(task, {})
        prompt = f"""# Annotation Task: {task.value}

## Required Elements
{self._format_list(guideline.required_elements)}

## Quality Criteria
{self._format_list(guideline.quality_criteria)}

## Format

```json
{{
  "question": "...",
  "context": "...",
  "reasoning_steps": [
    {{"index": 1, "content": "...", "reasoning_type": "...", "justification": "..."}},
    ...
  ],
  "final_answer": "...",
  "verification": "...",
  "common_errors": []
}}
```

## Instruction Template
{template.get('instruction', '')}

## Requirements
- Minimum {guideline.min_steps} steps, maximum {guideline.max_steps} steps
- Each step must include justification
- Verification is {'required' if guideline.verification_required else 'optional'}
"""
        return prompt

    def _format_list(self, items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    def quality_check(self, example_dict: Dict, task: ReasoningTask) -> List[str]:
        issues = []
        guideline = self.get_guideline(task)
        if not guideline:
            return ["No guideline for task"]

        for element in guideline.required_elements:
            if element not in example_dict or not example_dict[element]:
                issues.append(f"Missing required element: {element}")

        steps = example_dict.get("reasoning_steps", [])
        if len(steps) < guideline.min_steps:
            issues.append(f"Too few steps: {len(steps)} < {guideline.min_steps}")
        if len(steps) > guideline.max_steps:
            issues.append(f"Too many steps: {len(steps)} > {guideline.max_steps}")

        return issues
