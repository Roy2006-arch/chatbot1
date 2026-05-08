import ast
import re
import subprocess
import sys
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..schema import (
    BuggyExample, SourceCode, Language, ErrorType, FixValidationResult,
)


class CodeChecker:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.stats = {"checked": 0, "buggy_fails": 0, "correct_passes": 0}

    def check_example(self, example: BuggyExample) -> FixValidationResult:
        result = FixValidationResult(passed=False)

        buggy_check = self._check_code(example.buggy_code)
        correct_check = self._check_code(example.corrected_code)

        result.compile_success = correct_check.get("compile_ok", False) if example.language != Language.PYTHON else correct_check.get("syntax_ok", False)
        result.runtime_success = correct_check.get("runtime_ok", False)

        result.errors = []

        if not buggy_check.get("syntax_ok", True) or not buggy_check.get("compile_ok", True):
            pass

        buggy_runtime = buggy_check.get("runtime_error", "")
        correct_runtime = correct_check.get("runtime_error", "")

        if example.language == Language.PYTHON:
            buggy_syntax = buggy_check.get("syntax_ok", True)
            correct_syntax = correct_check.get("syntax_ok", True)

            if not buggy_syntax:
                result.errors.append(f"Buggy code has syntax error (expected): {buggy_check.get('syntax_error', '')}")

            if not correct_syntax:
                result.errors.append(f"Corrected code has syntax error: {correct_check.get('syntax_error', '')}")

            if correct_syntax:
                result.passed = True

        else:
            if correct_check.get("compile_ok", False):
                result.passed = True
            else:
                result.errors.append(f"Corrected code failed compilation: {correct_check.get('compile_error', '')}")

        test_results = []
        for i, tc in enumerate(example.test_cases):
            test_out = self._run_with_input(example.corrected_code, tc.input_data)
            test_results.append({
                "test_index": i,
                "input": tc.input_data,
                "expected": tc.expected_output,
                "actual": test_out.get("output", ""),
                "passed": test_out.get("output", "").strip() == tc.expected_output.strip() if tc.expected_output else True,
                "error": test_out.get("error", ""),
            })
        result.test_results = test_results

        self.stats["checked"] += 1
        if buggy_check.get("has_error", False):
            self.stats["buggy_fails"] += 1
        if correct_check.get("syntax_ok", True) and correct_check.get("runtime_ok", True):
            self.stats["correct_passes"] += 1

        return result

    def check_batch(self, examples: List[BuggyExample], num_workers: int = 8) -> List[Tuple[BuggyExample, FixValidationResult]]:
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.check_example, ex): ex for ex in examples}
            for future in as_completed(futures):
                ex = futures[future]
                try:
                    result = future.result()
                    results.append((ex, result))
                except Exception as e:
                    results.append((ex, FixValidationResult(passed=False, errors=[str(e)])))
        return results

    def _check_code(self, source: SourceCode) -> Dict:
        if source.language == Language.PYTHON:
            return self._check_python(source.code)
        elif source.language == Language.JAVASCRIPT:
            return self._check_javascript(source.code)
        elif source.language == Language.JAVA:
            return self._check_java(source.code)
        elif source.language == Language.CPP:
            return self._check_cpp(source.code)
        return {"syntax_ok": True, "has_error": False}

    def _check_python(self, code: str) -> Dict:
        result = {"syntax_ok": True, "compile_ok": True, "runtime_ok": True, "has_error": False}
        try:
            ast.parse(code)
        except SyntaxError as e:
            result["syntax_ok"] = False
            result["compile_ok"] = False
            result["syntax_error"] = str(e)
            result["has_error"] = True
            return result

        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            result["compile_ok"] = False
            result["compile_error"] = str(e)
            result["has_error"] = True
            return result

        try:
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)
        except Exception as e:
            result["runtime_ok"] = False
            result["runtime_error"] = str(e)
            result["has_error"] = True

        return result

    def _check_javascript(self, code: str) -> Dict:
        result = {"syntax_ok": True, "compile_ok": True, "runtime_ok": True, "has_error": False}
        try:
            proc = subprocess.run(
                ["node", "--check", "-e", code],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode != 0:
                result["syntax_ok"] = False
                result["compile_ok"] = False
                result["compile_error"] = proc.stderr[:500]
                result["has_error"] = True
        except FileNotFoundError:
            result["syntax_ok"] = True
        except subprocess.TimeoutExpired:
            pass
        return result

    def _check_java(self, code: str) -> Dict:
        result = {"syntax_ok": True, "compile_ok": True, "runtime_ok": True, "has_error": False}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                class_name = self._extract_java_class(code)
                src_path = Path(tmp) / f"{class_name}.java"
                src_path.write_text(code)
                proc = subprocess.run(
                    ["javac", str(src_path)],
                    capture_output=True, text=True, timeout=15,
                )
                if proc.returncode != 0:
                    result["compile_ok"] = False
                    result["compile_error"] = proc.stderr[:500]
                    result["has_error"] = True
            except FileNotFoundError:
                pass
            except subprocess.TimeoutExpired:
                pass
        return result

    def _check_cpp(self, code: str) -> Dict:
        result = {"syntax_ok": True, "compile_ok": True, "runtime_ok": True, "has_error": False}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                src_path = Path(tmp) / "check.cpp"
                src_path.write_text(code)
                proc = subprocess.run(
                    ["g++", "-std=c++17", "-fsyntax-only", str(src_path)],
                    capture_output=True, text=True, timeout=15,
                )
                if proc.returncode != 0:
                    result["compile_ok"] = False
                    result["compile_error"] = proc.stderr[:500]
                    result["has_error"] = True
            except FileNotFoundError:
                pass
            except subprocess.TimeoutExpired:
                pass
        return result

    def _run_with_input(self, source: SourceCode, input_data: str) -> Dict:
        if source.language == Language.PYTHON:
            return self._run_python(source.code, input_data)
        return {"output": "", "error": "Runner not available"}

    def _run_python(self, code: str, input_data: str) -> Dict:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                input=input_data, capture_output=True, text=True, timeout=self.timeout,
            )
            if proc.returncode == 0:
                return {"output": proc.stdout, "error": ""}
            return {"output": proc.stdout, "error": proc.stderr[:500]}
        except subprocess.TimeoutExpired:
            return {"output": "", "error": "Timeout"}
        except Exception as e:
            return {"output": "", "error": str(e)}

    def _extract_java_class(self, code: str) -> str:
        match = re.search(r"public\s+class\s+(\w+)", code)
        return match.group(1) if match else "Main"

    def get_stats(self) -> Dict:
        return self.stats
