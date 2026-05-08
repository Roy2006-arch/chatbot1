import ast
import re
import sys
import subprocess
import tempfile
import os
import time
import signal
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from ..schema import TestCase, Solution, Language, Problem


@dataclass
class ValidationResult:
    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    runtime_ms: float = 0.0
    peak_memory_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    failed_details: List[Dict] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_tests / max(self.total_tests, 1)


class CodeValidator:
    TIMEOUT_SECONDS = 10
    MAX_OUTPUT_SIZE = 10000

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.stats = {"validated": 0, "passed": 0, "failed": 0}

    def validate_solution(self, problem: Problem, language: Language = Language.PYTHON) -> ValidationResult:
        solution = problem.solutions.get(language.value)
        if not solution:
            return ValidationResult(passed=False, errors=[f"No solution for {language.value}"])

        all_tests = (
            problem.sample_test_cases
            + problem.hidden_test_cases
            + problem.edge_test_cases
        )

        if not all_tests:
            return ValidationResult(passed=True, total_tests=0)

        result = ValidationResult(total_tests=len(all_tests))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for i, tc in enumerate(all_tests):
                future = executor.submit(self._run_test, solution, tc, language)
                futures[future] = (i, tc)

            for future in as_completed(futures):
                i, tc = futures[future]
                try:
                    test_result = future.result(timeout=self.timeout + 2)
                    if test_result["passed"]:
                        result.passed_tests += 1
                    else:
                        result.failed_tests += 1
                        result.failed_details.append({
                            "test_index": i,
                            "input": tc.input[:200],
                            "expected": tc.expected_output[:200],
                            "actual": test_result.get("actual", "")[:200],
                            "error": test_result.get("error", ""),
                            "is_edge_case": tc.is_edge_case,
                        })
                    result.runtime_ms += test_result.get("runtime_ms", 0)
                except Exception as e:
                    result.failed_tests += 1
                    result.failed_details.append({
                        "test_index": i,
                        "input": tc.input[:200],
                        "error": str(e),
                    })

        result.passed = result.failed_tests == 0
        result.runtime_ms = result.runtime_ms / max(len(all_tests), 1)

        self.stats["validated"] += 1
        if result.passed:
            self.stats["passed"] += 1
        else:
            self.stats["failed"] += 1

        return result

    def _run_test(self, solution: Solution, test_case: TestCase, language: Language) -> Dict:
        start = time.time()
        try:
            if language == Language.PYTHON:
                output = self._run_python(solution.code, test_case.input)
            elif language == Language.JAVASCRIPT:
                output = self._run_javascript(solution.code, test_case.input)
            elif language == Language.CPP:
                output = self._run_cpp(solution.code, test_case.input)
            elif language == Language.JAVA:
                output = self._run_java(solution.code, test_case.input)
            else:
                output = self._run_python(solution.code, test_case.input)

            runtime = (time.time() - start) * 1000
            passed = output.strip() == test_case.expected_output.strip()

            return {
                "passed": passed,
                "runtime_ms": runtime,
                "actual": output.strip(),
                "error": "",
            }
        except Exception as e:
            runtime = (time.time() - start) * 1000
            return {
                "passed": False,
                "runtime_ms": runtime,
                "actual": "",
                "error": str(e),
            }

    def _run_python(self, code: str, input_data: str) -> str:
        wrapped = self._wrap_python_code(code, input_data)
        try:
            exec_globals = {"__builtins__": __builtins__}
            output_catcher = OutputCapture()
            exec_globals["print"] = output_catcher.capture

            exec(wrapped, exec_globals)
            return output_catcher.get_output()
        except Exception as e:
            try:
                result = subprocess.run(
                    [sys.executable, "-c", wrapped],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode == 0:
                    return result.stdout
                return f"Error: {result.stderr[:500]}"
            except subprocess.TimeoutExpired:
                return "Error: Timeout"
            except Exception as e2:
                return f"Error: {str(e2)[:500]}"

    def _wrap_python_code(self, code: str, input_data: str) -> str:
        input_lines = input_data.strip().split("\n") if input_data else []
        input_repr = json.dumps(input_lines)

        wrapper = f"""
import sys
from io import StringIO

_input_data = {input_repr}
_input_iter = iter(_input_data)

def input(prompt=""):
    try:
        return next(_input_iter)
    except StopIteration:
        return ""

sys.stdin = StringIO("\\n".join(_input_data))

{code}
"""
        return wrapper

    def _run_javascript(self, code: str, input_data: str) -> str:
        try:
            result = subprocess.run(
                ["node", "-e", code],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode == 0:
                return result.stdout
            return f"Error: {result.stderr[:500]}"
        except FileNotFoundError:
            return "Error: Node.js not found"
        except subprocess.TimeoutExpired:
            return "Error: Timeout"

    def _run_cpp(self, code: str, input_data: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "solution.cpp"
            exe = Path(tmp) / "solution.exe"
            src.write_text(code)
            try:
                compile_result = subprocess.run(
                    ["g++", "-std=c++17", "-O2", str(src), "-o", str(exe)],
                    capture_output=True, text=True, timeout=30,
                )
                if compile_result.returncode != 0:
                    return f"Compile Error: {compile_result.stderr[:500]}"

                run_result = subprocess.run(
                    [str(exe)], input=input_data,
                    capture_output=True, text=True, timeout=self.timeout,
                )
                if run_result.returncode == 0:
                    return run_result.stdout
                return f"Runtime Error: {run_result.stderr[:500]}"
            except FileNotFoundError:
                return "Error: g++ not found"
            except subprocess.TimeoutExpired:
                return "Error: Timeout"

    def _run_java(self, code: str, input_data: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            class_name = self._extract_java_class(code)
            src = Path(tmp) / f"{class_name}.java"
            src.write_text(code)
            try:
                compile_result = subprocess.run(
                    ["javac", str(src)],
                    capture_output=True, text=True, timeout=30,
                )
                if compile_result.returncode != 0:
                    return f"Compile Error: {compile_result.stderr[:500]}"

                run_result = subprocess.run(
                    ["java", "-cp", tmp, class_name],
                    input=input_data,
                    capture_output=True, text=True, timeout=self.timeout,
                )
                if run_result.returncode == 0:
                    return run_result.stdout
                return f"Runtime Error: {run_result.stderr[:500]}"
            except FileNotFoundError:
                return "Error: Java not found"
            except subprocess.TimeoutExpired:
                return "Error: Timeout"

    def _extract_java_class(self, code: str) -> str:
        match = re.search(r"public\s+class\s+(\w+)", code)
        return match.group(1) if match else "Solution"

    def validate_syntax(self, code: str, language: Language) -> Tuple[bool, Optional[str]]:
        if language == Language.PYTHON:
            try:
                ast.parse(code)
                return True, None
            except SyntaxError as e:
                return False, str(e)
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            try:
                subprocess.run(["node", "--check", "-e", code],
                             capture_output=True, text=True, timeout=5)
                return True, None
            except subprocess.TimeoutExpired:
                return True, None
            except Exception as e:
                return False, str(e)
        elif language == Language.JAVA:
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    class_name = self._extract_java_class(code)
                    src = Path(tmp) / f"{class_name}.java"
                    src.write_text(code)
                    result = subprocess.run(["javac", str(src)],
                                          capture_output=True, text=True, timeout=15)
                    if result.returncode == 0:
                        return True, None
                    return False, result.stderr[:500]
                except FileNotFoundError:
                    return True, None
                except Exception as e:
                    return False, str(e)
        elif language == Language.CPP:
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    src = Path(tmp) / "check.cpp"
                    src.write_text(code)
                    result = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", str(src)],
                                          capture_output=True, text=True, timeout=15)
                    if result.returncode == 0:
                        return True, None
                    return False, result.stderr[:500]
                except FileNotFoundError:
                    return True, None
                except Exception as e:
                    return False, str(e)
        return True, None

    def analyze_complexity(self, code: str, language: Language = Language.PYTHON) -> Dict:
        analysis = {
            "loops": 0,
            "nested_loops": 0,
            "recursion": False,
            "estimated_time": "O(1)",
            "estimated_space": "O(1)",
            "has_recursion": False,
            "data_structures": [],
        }

        recursion_patterns = [
            r"def\s+\w+.*:\s*\n\s*(?:return\s+.*\w+\s*\(|if\s+.*:\s*\n\s+\w+\s*\()",
            r"function\s+\w+.*\{[\s\S]*?\1\s*\(",
        ]
        for p in recursion_patterns:
            if re.search(p, code):
                analysis["recursion"] = True
                analysis["has_recursion"] = True
                break

        loop_patterns = [
            (r"for\s+\w+\s+in\s+\w+", 1),
            (r"while\s+\w+", 1),
            (r"for\s*\(.*;.*;.*\)", 1),
            (r"while\s*\(.*\)", 1),
        ]
        total_loops = 0
        for pat, weight in loop_patterns:
            count = len(re.findall(pat, code))
            total_loops += count * weight
        analysis["loops"] = total_loops

        nested = re.findall(r"(for|while).*\n\s*(for|while)", code)
        analysis["nested_loops"] = len(nested)

        ds_patterns = [
            (r"\[\]", "list"), (r"\{\}", "hash_map"), (r"set\(\)", "set"),
            (r"heapq|PriorityQueue|priority_queue", "heap"),
            (r"deque|collections\.deque", "deque"),
            (r"stack|Stack<", "stack"),
            (r"queue|Queue<", "queue"),
            (r"defaultdict", "hash_map"),
            (r"TreeNode|tree\s*<|BinaryTree", "tree"),
            (r"Graph|graph|adjacency", "graph"),
        ]
        for pat, name in ds_patterns:
            if re.search(pat, code):
                analysis["data_structures"].append(name)

        for _ in range(total_loops):
            for _ in range(analysis["nested_loops"]):
                pass

        if analysis["nested_loops"] >= 2:
            analysis["estimated_time"] = "O(n^2)"
        elif analysis["nested_loops"] == 1:
            analysis["estimated_time"] = "O(n log n)" if "heap" in analysis["data_structures"] else "O(n^2)"
        elif total_loops >= 1:
            analysis["estimated_time"] = "O(n)"
        if analysis["recursion"] and not total_loops:
            analysis["estimated_time"] = "O(2^n)"

        return analysis


class OutputCapture:
    def __init__(self):
        self._output = []

    def capture(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        self._output.append(text)

    def get_output(self) -> str:
        return "\n".join(self._output)


class TestCaseGenerator:
    def __init__(self):
        self.stats = {"generated": 0}

    def generate_edge_cases(self, problem: Problem) -> List[TestCase]:
        edge_cases = []

        if "array" in [p.value for p in problem.dsa_patterns]:
            edge_cases.append(TestCase(
                input="[]",
                expected_output=self._infer_empty_output(problem),
                explanation="Empty array edge case",
                is_edge_case=True,
                tags=["empty"],
            ))
            edge_cases.append(TestCase(
                input="[1]",
                expected_output="1",
                explanation="Single element array",
                is_edge_case=True,
                tags=["single_element"],
            ))
            edge_cases.append(TestCase(
                input="[2147483647, -2147483648]",
                expected_output=self._infer_boundary_output(problem),
                explanation="Integer boundary values",
                is_edge_case=True,
                tags=["boundary"],
            ))

        if "string" in [p.value for p in problem.dsa_patterns]:
            edge_cases.append(TestCase(
                input='""',
                expected_output=self._infer_empty_output(problem),
                explanation="Empty string edge case",
                is_edge_case=True,
                tags=["empty"],
            ))
            edge_cases.append(TestCase(
                input='"a"',
                expected_output="a",
                explanation="Single character string",
                is_edge_case=True,
                tags=["single_element"],
            ))

        if "tree" in [p.value for p in problem.dsa_patterns]:
            edge_cases.append(TestCase(
                input="null",
                expected_output=self._infer_empty_output(problem),
                explanation="Null tree edge case",
                is_edge_case=True,
                tags=["null"],
            ))

        self.stats["generated"] += len(edge_cases)
        return edge_cases

    def _infer_empty_output(self, problem: Problem) -> str:
        return "0"

    def _infer_boundary_output(self, problem: Problem) -> str:
        return "-2147483648"

    def generate_debugging_examples(self, problem: Problem, count: int = 3) -> List[Dict]:
        debugging_examples = []

        bug_templates = [
            {
                "type": "off_by_one",
                "description": "Loop off-by-one error in boundary condition",
                "fix_hint": "Change '<=' to '<' or vice versa in the loop condition",
            },
            {
                "type": "missing_edge_case",
                "description": "Missing handling for empty input",
                "fix_hint": "Add a check for empty/null input at the start",
            },
            {
                "type": "wrong_comparison",
                "description": "Incorrect comparison operator in condition",
                "fix_hint": "Review the comparison direction in the if statement",
            },
            {
                "type": "uninitialized_variable",
                "description": "Variable used before initialization",
                "fix_hint": "Initialize the variable with a default value before use",
            },
            {
                "type": "type_error",
                "description": "Type mismatch between expected and actual value",
                "fix_hint": "Add type conversion or check the expected type",
            },
            {
                "type": "infinite_loop",
                "description": "Missing termination condition in loop",
                "fix_hint": "Ensure the loop variable is updated in each iteration",
            },
            {
                "type": "memory_overflow",
                "description": "List index out of range or buffer overflow",
                "fix_hint": "Add bounds checking before accessing array elements",
            },
            {
                "type": "logic_flip",
                "description": "Condition logic is inverted",
                "fix_hint": "Negate the condition or swap if/else branches",
            },
        ]

        import random
        selected_bugs = random.sample(bug_templates, min(count, len(bug_templates)))
        for bug in selected_bugs:
            debugging_examples.append({
                "type": bug["type"],
                "description": bug["description"],
                "fix_hint": bug["fix_hint"],
                "worst_case_complexity": "O(n^2) due to the bug",
                "correct_complexity": "O(n) after fix",
            })

        return debugging_examples

    def get_stats(self) -> Dict:
        return self.stats
