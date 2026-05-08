import json
import random
from typing import Dict, List, Optional, Generator
from pathlib import Path

from ..schema import Problem, CPInstructionExample, Solution, Language, DifficultyLevel, INSTRUCTION_TEMPLATES


class CPFormatConverter:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.stats = {"converted": 0, "by_type": {}}

    def convert_to_instructions(
        self,
        problems: List[Problem],
        include_debugging: bool = True,
        include_analysis: bool = True,
        multi_language: bool = True,
        chain_of_thought: bool = True,
    ) -> List[CPInstructionExample]:
        examples = []

        for problem in problems:
            examples.append(self._make_solve_example(problem, chain_of_thought))
            examples.append(self._make_explain_example(problem, chain_of_thought))

            if include_debugging:
                debugs = self._make_debug_examples(problem)
                examples.extend(debugs)

            if include_analysis:
                examples.append(self._make_complexity_example(problem))
                examples.append(self._make_edge_case_example(problem))

            if multi_language and len(problem.solutions) >= 2:
                examples.append(self._make_multilang_example(problem))

            examples.append(self._make_pattern_example(problem))

        examples = [ex for ex in examples if ex is not None]

        self.stats["converted"] += len(examples)
        return examples

    def _format_problem_context(self, problem: Problem) -> str:
        lines = [f"## Problem: {problem.title}"]
        if problem.problem_statement:
            lines.append(f"\n{problem.problem_statement}")
        if problem.constraints:
            lines.append("\n### Constraints")
            for c in problem.constraints:
                lines.append(f"- {c}")
        if problem.sample_test_cases:
            lines.append("\n### Sample Test Cases")
            for i, tc in enumerate(problem.sample_test_cases[:3]):
                lines.append(f"\n**Example {i+1}:**")
                lines.append(f"Input: `{tc.input}`")
                lines.append(f"Output: `{tc.expected_output}`")
                if tc.explanation:
                    lines.append(f"Explanation: {tc.explanation}")
        return "\n".join(lines)

    def _format_optimized_solution(self, problem: Problem, lang: Language = Language.PYTHON) -> str:
        sol = problem.solutions.get(lang.value)
        if not sol:
            sol = next(iter(problem.solutions.values()), None)
        if not sol:
            return "# No solution available"

        parts = [f"### {lang.value.title()} Solution\n"]
        parts.append(f"**Approach:** {sol.approach}\n")
        parts.append(f"**Time Complexity:** {sol.time_complexity}")
        parts.append(f"**Space Complexity:** {sol.space_complexity}\n")
        parts.append(f"```{lang.value}\n{sol.code}\n```")
        return "\n".join(parts)

    def _make_solve_example(self, problem: Problem, cot: bool = True) -> CPInstructionExample:
        instruction = INSTRUCTION_TEMPLATES["solve"]
        if cot:
            instruction += "\n\nThink step-by-step and explain your reasoning before writing code."

        context = self._format_problem_context(problem)
        solution = self._format_optimized_solution(problem)

        if cot:
            cot_steps = self._generate_cot(problem)
            output = f"{cot_steps}\n\n{solution}"
        else:
            output = solution

        return CPInstructionExample(
            instruction=instruction,
            input=context,
            output=output,
            problem=problem,
            metadata={"type": "solve", "platform": problem.platform.value, "difficulty": problem.difficulty.value},
        )

    def _make_explain_example(self, problem: Problem, cot: bool = True) -> CPInstructionExample:
        instruction = INSTRUCTION_TEMPLATES["explain"]
        context = self._format_problem_context(problem)
        sol = next(iter(problem.solutions.values()), None)

        explanation_parts = [f"## Approach: {sol.approach if sol else 'Solution'}\n"]

        patterns = [p.value for p in problem.dsa_patterns]
        explanation_parts.append(f"**DSA Patterns:** {', '.join(patterns)}\n")

        if sol:
            explanation_parts.append("### Step-by-Step Explanation\n")
            explanation_parts.append(f"1. **Problem Analysis:** {self._analyze_problem(problem)}")
            explanation_parts.append(f"2. **Pattern Recognition:** This problem uses {patterns[0] if patterns else 'standard techniques'} because...")
            explanation_parts.append(f"3. **Algorithm Design:** {sol.approach}")
            explanation_parts.append(f"4. **Implementation Details:** Key considerations for the solution.")

        if problem.complexity_analysis:
            explanation_parts.append(f"\n### Complexity\n{problem.complexity_analysis}")
        elif sol:
            explanation_parts.append(f"\n### Complexity\n- Time: {sol.time_complexity}\n- Space: {sol.space_complexity}")

        output = "\n".join(explanation_parts)
        return CPInstructionExample(
            instruction=instruction,
            input=context,
            output=output,
            problem=problem,
            metadata={"type": "explain", "platform": problem.platform.value, "difficulty": problem.difficulty.value},
        )

    def _make_debug_examples(self, problem: Problem) -> List[CPInstructionExample]:
        examples = []
        debug_examples = problem.debugging_examples or []
        sol = next(iter(problem.solutions.values()), None)

        if not debug_examples and not sol:
            return []

        bug_types = [
            {"type": "off_by_one", "desc": "off-by-one error in loop condition"},
            {"type": "edge_case", "desc": "missing empty input check"},
            {"type": "logic_flip", "desc": "inverted comparison operator"},
        ]

        for bug in bug_types[:2]:
            instruction = INSTRUCTION_TEMPLATES["debug"].format(
                current_complexity=sol.time_complexity if sol else "O(n)",
                target_complexity=sol.time_complexity if sol else "O(n)",
            )
            context = self._format_problem_context(problem)

            buggy_code = self._inject_bug(sol.code if sol else "", bug["type"]) if sol else f"# Buggy code with {bug['desc']}"
            output = (
                f"## Bug Analysis\n\n"
                f"**Bug Type:** {bug['desc']}\n\n"
                f"### Root Cause\n"
                f"The {bug['desc']} causes incorrect output when...\n\n"
                f"### The Fix\n"
                f"```python\n{sol.code if sol else '# Fixed code here'}\n```\n\n"
                f"### Prevention\n"
                f"Always check boundary conditions and edge cases."
            )

            debug_input = f"{context}\n\n### Buggy Code\n```python\n{buggy_code}\n```"

            examples.append(CPInstructionExample(
                instruction=instruction,
                input=debug_input,
                output=output,
                problem=problem,
                metadata={"type": "debug", "bug_type": bug["type"]},
            ))

        return examples

    def _make_complexity_example(self, problem: Problem) -> CPInstructionExample:
        instruction = INSTRUCTION_TEMPLATES["complexity"]
        context = self._format_problem_context(problem)
        sol = next(iter(problem.solutions.values()), None)

        output = "## Complexity Analysis\n\n"
        if sol:
            time_explanation = self._explain_complexity(sol.time_complexity)
            space_explanation = self._explain_complexity(sol.space_complexity)
            output += (
                f"### Time Complexity: {sol.time_complexity}\n"
                f"{time_explanation}\n\n"
                f"### Space Complexity: {sol.space_complexity}\n"
                f"{space_explanation}\n\n"
                f"### Trade-offs\n"
                f"- Can we reduce time at the cost of more space?\n"
                f"- Is there a brute-force alternative with better space?\n"
                f"- What's the best conceivable runtime for this problem?\n"
            )

        return CPInstructionExample(
            instruction=instruction,
            input=context,
            output=output,
            problem=problem,
            metadata={"type": "complexity_analysis"},
        )

    def _make_edge_case_example(self, problem: Problem) -> CPInstructionExample:
        instruction = INSTRUCTION_TEMPLATES["edge_cases"]
        context = self._format_problem_context(problem)

        edge_analysis = "## Edge Case Analysis\n\n"
        patterns = [p.value for p in problem.dsa_patterns]

        if "array" in patterns:
            edge_analysis += (
                "### Array Edge Cases\n"
                "- **Empty array**: `[]` → should return default value\n"
                "- **Single element**: `[x]` → loop boundaries matter\n"
                "- **All identical elements**: `[5,5,5,5]` → stability of algorithm\n"
                "- **Already sorted/reversed**: `[1,2,3]` vs `[3,2,1]`\n"
                "- **Negative numbers**: `[-1,-2,3]` → sign handling\n"
                "- **Integer overflow**: `[2147483647, -2147483648]`\n"
            )
        if "string" in patterns:
            edge_analysis += (
                "\n### String Edge Cases\n"
                "- **Empty string**: `\"\"`\n"
                "- **Single character**: `\"a\"`\n"
                "- **All same characters**: `\"aaaa\"`\n"
                "- **Unicode/UTF-8**: non-ASCII characters\n"
                "- **Case sensitivity**: `\"a\"` vs `\"A\"`\n"
            )
        if "tree" in patterns:
            edge_analysis += (
                "\n### Tree Edge Cases\n"
                "- **Null root**: null input\n"
                "- **Single node**: root only\n"
                "- **Skewed tree**: all nodes on one side\n"
                "- **Complete binary tree**: full tree structure\n"
            )

        edge_analysis += (
            "\n### General Tips\n"
            "- Always validate input before processing\n"
            "- Consider constraints: `n ≤ 10^5` means O(n) or O(n log n) required\n"
            "- Test with minimum and maximum constraint values\n"
        )

        return CPInstructionExample(
            instruction=instruction,
            input=context,
            output=edge_analysis,
            problem=problem,
            metadata={"type": "edge_cases"},
        )

    def _make_multilang_example(self, problem: Problem) -> CPInstructionExample:
        instruction = INSTRUCTION_TEMPLATES["multi_lang"]
        context = self._format_problem_context(problem)

        output_parts = ["## Multi-Language Solutions\n"]
        for lang in [Language.PYTHON, Language.JAVA, Language.CPP]:
            sol = problem.solutions.get(lang.value)
            if sol:
                output_parts.append(f"\n### {lang.value.title()}\n")
                output_parts.append(f"```{lang.value}\n{sol.code}\n```")

        return CPInstructionExample(
            instruction=instruction,
            input=context,
            output="\n".join(output_parts),
            problem=problem,
            metadata={"type": "multilang", "languages": list(problem.solutions.keys())},
        )

    def _make_pattern_example(self, problem: Problem) -> CPInstructionExample:
        instruction = INSTRUCTION_TEMPLATES["pattern"]
        context = self._format_problem_context(problem)
        patterns = [p.value for p in problem.dsa_patterns]

        output = f"## DSA Pattern Analysis\n\n"
        output += f"**Identified Patterns:** {', '.join(patterns)}\n\n"

        for pat in problem.dsa_patterns:
            output += f"### {pat.value.replace('_', ' ').title()}\n"
            from ..classifiers.dsa_classifier import DSAClassifier
            classifier = DSAClassifier()
            output += f"{classifier.explain_pattern(pat)}\n"
            output += f"**Company Frequency:** {classifier.get_company_frequency(pat)}\n\n"

        return CPInstructionExample(
            instruction=instruction,
            input=context,
            output=output,
            problem=problem,
            metadata={"type": "pattern_analysis", "patterns": patterns},
        )

    def _generate_cot(self, problem: Problem) -> str:
        patterns = [p.value for p in problem.dsa_patterns]
        sol = next(iter(problem.solutions.values()), None)

        steps = ["## Step-by-Step Reasoning\n"]

        steps.append(f"**Step 1: Analyze the Problem**")
        steps.append(f"- We need to: {problem.title}")
        steps.append(f"- Input/Output format identified")
        steps.append(f"- Key constraints to consider")

        steps.append(f"\n**Step 2: Identify the Pattern**")
        steps.append(f"- This problem falls under: {', '.join(patterns)}")
        steps.append(f"- Why: The problem structure matches the signature of these patterns")

        if sol:
            steps.append(f"\n**Step 3: Design the Algorithm**")
            steps.append(f"- Approach: {sol.approach}")
            steps.append(f"- Key data structures needed")
            steps.append(f"- Algorithm outline:")

        steps.append(f"\n**Step 4: Complexity Analysis**")
        if sol:
            steps.append(f"- Time: {sol.time_complexity}")
            steps.append(f"- Space: {sol.space_complexity}")
        steps.append(f"- Can we do better? Consider trade-offs")

        steps.append(f"\n**Step 5: Edge Cases**")
        steps.append(f"- Empty input: handled")
        steps.append(f"- Single element: handled")
        steps.append(f"- Boundary values: within constraints")

        return "\n".join(steps)

    def _analyze_problem(self, problem: Problem) -> str:
        patterns = [p.value for p in problem.dsa_patterns]
        return f"This is a {problem.platform.value} problem requiring {', '.join(patterns)} techniques."

    def _inject_bug(self, code: str, bug_type: str) -> str:
        import re
        lines = code.split("\n")

        if bug_type == "off_by_one":
            for i, line in enumerate(lines):
                if "<=" in line and ("for" in line or "while" in line):
                    lines[i] = line.replace("<=", "<")
                    break
                elif "range(" in line:
                    lines[i] = re.sub(r"range\((\w+)\)", r"range(\1 + 1)", line)
                    break
                elif "range(" in line:
                    lines[i] = re.sub(r"range\((\w+)\)", r"range(\1 - 1)", line)
                    break

        elif bug_type == "edge_case":
            for i, line in enumerate(lines):
                if "def " in line:
                    indent = " " * (len(line) - len(line.lstrip()) + 4)
                    lines.insert(i + 1, f"{indent}# BUG: missing null/empty check")
                    break

        elif bug_type == "logic_flip":
            for i, line in enumerate(lines):
                if "if " in line and ">" in line:
                    lines[i] = line.replace(">", "<")
                    break
                elif "if " in line and "<" in line:
                    lines[i] = line.replace("<", ">")
                    break

        return "\n".join(lines)

    def _explain_complexity(self, complexity: str) -> str:
        explanations = {
            "O(1)": "Constant time - operation takes the same time regardless of input size.",
            "O(log n)": "Logarithmic - grows slowly. Typical of binary search and balanced tree operations.",
            "O(n)": "Linear - grows proportionally to input. Single pass through data.",
            "O(n log n)": "Linearithmic - typical of efficient sorting (merge sort, heap sort).",
            "O(n^2)": "Quadratic - nested loops over data. Acceptable only for small inputs (n < 1000).",
            "O(n^3)": "Cubic - triple nested loops. Rarely acceptable (n < 100).",
            "O(2^n)": "Exponential - grows extremely fast. Only for n < 20-30.",
            "O(n!)": "Factorial - worst case. Only for n < 10-12.",
        }
        for key, explanation in explanations.items():
            if key.lower() in complexity.lower():
                return explanation
        return f"{complexity} - custom complexity analysis based on the algorithm structure."

    def export_jsonl(self, examples: List[CPInstructionExample], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                    "category": "competitive_programming",
                    "metadata": ex.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def export_for_finetuning(
        self,
        examples: List[CPInstructionExample],
        output_dir: str,
        framework: str = "transformers",
        split_ratio: tuple = (0.85, 0.10, 0.05),
    ):
        import random
        random.seed(self.seed)

        indices = list(range(len(examples)))
        random.shuffle(indices)

        n = len(examples)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        splits = {
            "train": [examples[i] for i in indices[:train_end]],
            "validation": [examples[i] for i in indices[train_end:val_end]],
            "test": [examples[i] for i in indices[val_end:]],
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_examples in splits.items():
            filepath = output_path / f"{split_name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for ex in split_examples:
                    if framework == "transformers":
                        record = {
                            "instruction": ex.instruction,
                            "input": ex.input,
                            "output": ex.output,
                        }
                    elif framework == "axolotl":
                        text = f"### Instruction:\n{ex.instruction}\n\n"
                        if ex.input:
                            text += f"### Input:\n{ex.input}\n\n"
                        text += f"### Response:\n{ex.output}"
                        record = {"text": text}
                    elif framework == "openai":
                        record = {
                            "messages": [
                                {"role": "user", "content": ex.instruction + ("\n" + ex.input if ex.input else "")},
                                {"role": "assistant", "content": ex.output},
                            ]
                        }
                    else:
                        record = ex.to_dict()
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        metadata = {
            "total": len(examples),
            "splits": {k: len(v) for k, v in splits.items()},
            "framework": framework,
            "types": {},
        }
        for ex in examples:
            t = ex.metadata.get("type", "unknown")
            metadata["types"][t] = metadata["types"].get(t, 0) + 1

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.stats["by_type"] = metadata["types"]

    def get_stats(self) -> Dict:
        return self.stats
