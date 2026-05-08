from typing import Dict, List, Optional


COMPETITIVE_PROGRAMMING_TEMPLATES = [
    {
        "instruction": "Solve the following competitive programming problem optimally.",
        "template": "Problem: {problem_statement}\n\nConstraints:\n{constraints}\n\nExample:\n{examples}",
        "output_template": "## Approach\n\n{approach}\n\n## Complexity Analysis\n- Time Complexity: {time_complexity}\n- Space Complexity: {space_complexity}\n\n## Solution\n\n```{language}\n{code}\n```",
    }
]

ALGORITHM_TEMPLATES = [
    {
        "instruction": "Explain the {algorithm_name} algorithm and provide a {language} implementation.",
        "template": "",
        "output_template": "# {algorithm_name}\n\n## Overview\n{overview}\n\n## How It Works\n\n{explanation}\n\n## Implementation\n\n```{language}\n{code}\n```\n\n## Time & Space Complexity\n- Time: {time_complexity}\n- Space: {space_complexity}\n\n## When to Use\n{use_cases}",
    }
]

DEBUGGING_TEMPLATES = [
    {
        "instruction": "Debug the following code and explain the fix.",
        "template": "```{language}\n{buggy_code}\n```\n\nError: {error_message}",
        "output_template": "## Bug Analysis\n\n{root_cause}\n\n## The Fix\n\n```{language}\n{fixed_code}\n```\n\n## Explanation\n\n{explanation}",
    }
]

SYSTEM_DESIGN_TEMPLATES = [
    {
        "instruction": "Design a system for {system_description}.",
        "template": "Requirements:\n{requirements}\n\nExpected scale: {scale}",
        "output_template": "# System Design: {system_name}\n\n## Requirements\n{requirements_detail}\n\n## High-Level Architecture\n\n{architecture}\n\n## Detailed Design\n\n{detailed_design}\n\n## Trade-offs\n\n{tradeoffs}\n\n## Scaling Considerations\n\n{scaling}",
    }
]

REASONING_TEMPLATES = [
    {
        "instruction": "Solve the following reasoning problem step by step.",
        "template": "{problem}",
        "output_template": "Let me reason through this step by step.\n\n{reasoning_steps}\n\nTherefore, the answer is: {final_answer}",
    }
]

CONVERSATIONAL_TEMPLATES = [
    {
        "instruction": "{user_query}",
        "template": "",
        "output_template": "{assistant_response}",
    }
]

TOOL_USAGE_TEMPLATES = [
    {
        "instruction": "How do I {task_description} using {tool_name}?",
        "template": "",
        "output_template": "## Using {tool_name} for {task_description}\n\n```bash\n{command}\n```\n\n### Explanation\n{explanation}\n\n### Common Options\n{options}",
    }
]

SYNTHETIC_TEMPLATES_BY_CATEGORY = {
    "competitive_programming": COMPETITIVE_PROGRAMMING_TEMPLATES,
    "algorithms_dsa": ALGORITHM_TEMPLATES,
    "debugging": DEBUGGING_TEMPLATES,
    "system_design": SYSTEM_DESIGN_TEMPLATES,
    "general_reasoning": REASONING_TEMPLATES,
    "conversational_ai": CONVERSATIONAL_TEMPLATES,
    "tool_usage": TOOL_USAGE_TEMPLATES,
}


DIFFICULTY_PROMPTS = {
    1: "Provide a simple, beginner-friendly explanation suitable for someone new to programming.",
    2: "Provide a clear explanation with basic examples for someone with introductory knowledge.",
    3: "Provide a detailed explanation with moderate complexity and real-world examples.",
    4: "Provide an in-depth explanation covering edge cases, optimizations, and advanced concepts.",
    5: "Provide an expert-level explanation with formal analysis, advanced optimizations, and novel insights.",
}

COT_PROMPT = "Think through this step-by-step before giving the final answer."

ERROR_TYPES = [
    "syntax_error",
    "logical_error",
    "off_by_one",
    "null_pointer",
    "type_error",
    "race_condition",
    "memory_leak",
    "infinite_loop",
    "edge_case",
    "performance_bug",
]


def get_template_for_category(category: str) -> Optional[List[Dict]]:
    return SYNTHETIC_TEMPLATES_BY_CATEGORY.get(category)


def get_difficulty_instruction(level: int) -> str:
    return DIFFICULTY_PROMPTS.get(level, DIFFICULTY_PROMPTS[3])
