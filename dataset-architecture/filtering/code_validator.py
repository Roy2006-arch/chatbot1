import ast
import re
import subprocess
import tempfile
import os
import textwrap
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FilterResult, FilterIssue, Severity

CODE_SMELL_PATTERNS = {
    "magic_number": r"\b\d{4,}\b",
    "hardcoded_credential": (
        r"(?i)(password|passwd|pwd|secret|api[_-]?key|apikey)\s*[:=]\s*[\"'][^\"']+[\"']"
    ),
    "todo_comment": r"#\s*(TODO|FIXME|HACK|XXX|BUG|WORKAROUND)",
    "commented_code": r"#\s*(def |class |import |for |while |if |return |print )",
    "overly_long_line": r"^.{120,}$",
    "bare_except": r"except\s*:",
    "duplicate_code": r"(\n\s*.+\n)\1{2,}",
}

LANGUAGE_EXTENSIONS = {
    "python": ".py", "javascript": ".js", "typescript": ".ts",
    "java": ".java", "cpp": ".cpp", "c": ".c", "csharp": ".cs",
    "rust": ".rs", "go": ".go", "ruby": ".rb", "php": ".php",
    "swift": ".swift", "kotlin": ".kt", "scala": ".scala",
}


class AdvancedCodeValidator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_code_length = self.config.get("max_code_length", 10000)
        self.enable_ast_check = self.config.get("enable_ast_check", True)
        self.enable_bracket_check = self.config.get("enable_bracket_check", True)
        self.enable_test_execution = self.config.get("enable_test_execution", False)
        self.supported_languages = self.config.get("supported_languages", [
            "python", "javascript", "typescript", "java", "cpp", "rust", "go",
        ])
        self.stats = {"checked": 0, "valid": 0, "invalid": 0}

    def check(self, code: str, language: str = "python") -> FilterResult:
        self.stats["checked"] += 1
        issues: List[FilterIssue] = []
        dim_scores = {}

        if not code.strip():
            return FilterResult(passed=False, score=0.0, issues=[
                FilterIssue(code="CODE_EMPTY", message="Empty code block", severity=Severity.HIGH, dimension="code"),
            ])

        if len(code) > self.max_code_length:
            issues.append(FilterIssue(
                code="CODE_TOO_LONG",
                message=f"Code exceeds max length ({len(code)} > {self.max_code_length})",
                severity=Severity.MEDIUM,
                dimension="code",
            ))

        syntax_valid, syntax_error = self._check_syntax(code, language)
        dim_scores["syntax"] = 1.0 if syntax_valid else 0.0
        if not syntax_valid:
            issues.append(FilterIssue(
                code="CODE_SYNTAX_ERROR",
                message=f"Syntax error: {syntax_error}",
                severity=Severity.HIGH,
                dimension="code",
                details={"error": syntax_error, "language": language},
            ))

        if self.enable_bracket_check and language != "python":
            bracket_valid, bracket_error = self._check_brackets(code)
            dim_scores["brackets"] = 1.0 if bracket_valid else 0.0
            if not bracket_valid:
                issues.append(FilterIssue(
                    code="CODE_BRACKET_ERROR",
                    message=f"Bracket mismatch: {bracket_error}",
                    severity=Severity.HIGH,
                    dimension="code",
                    details={"error": bracket_error},
                ))
        else:
            dim_scores["brackets"] = 1.0

        smell_issues = self._check_code_smells(code, language)
        issues.extend(smell_issues)
        dim_scores["no_smells"] = 1.0 - len([s for s in smell_issues if s.severity == Severity.MEDIUM]) * 0.1

        if self.enable_test_execution and language == "python":
            exec_valid, exec_error = self._try_execute(code)
            dim_scores["executable"] = 1.0 if exec_valid else 0.3
            if not exec_valid and exec_error:
                issues.append(FilterIssue(
                    code="CODE_EXECUTION_ERROR",
                    message=f"Runtime error: {exec_error[:200]}",
                    severity=Severity.MEDIUM,
                    dimension="code",
                    details={"error": exec_error[:200]},
                ))
        else:
            dim_scores["executable"] = 1.0

        dim_scores["length"] = min(1.0, len(code) / max(self.max_code_length, 1))

        composite = sum(dim_scores.values()) / max(len(dim_scores), 1)
        critical = [i for i in issues if i.severity in (Severity.HIGH, Severity.CRITICAL)]
        passed = len(critical) == 0
        if passed:
            self.stats["valid"] += 1
        else:
            self.stats["invalid"] += 1

        return FilterResult(
            passed=passed,
            score=composite,
            issues=issues,
            dimension_scores=dim_scores,
            metadata={"language": language, "syntax_valid": syntax_valid, "line_count": code.count("\n") + 1},
        )

    def check_batch(self, code_pairs: List[Tuple[str, str]], num_workers: int = 8) -> List[FilterResult]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.check, code, lang) for code, lang in code_pairs]
            return [f.result() for f in as_completed(futures)]

    def _check_syntax(self, code: str, language: str) -> Tuple[bool, Optional[str]]:
        language = language.lower()
        if language == "python":
            return self._validate_python(code)
        elif language in ("javascript", "js", "typescript", "ts"):
            return self._check_brackets(code)
        elif language in ("java", "cpp", "c", "csharp", "cs", "rust", "go", "kotlin", "scala"):
            return self._check_brackets(code)
        elif language in ("ruby", "php", "perl", "swift"):
            return self._check_brackets(code)
        return True, None

    def _validate_python(self, code: str) -> Tuple[bool, Optional[str]]:
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def _check_brackets(self, code: str) -> Tuple[bool, Optional[str]]:
        stack = []
        pairs = {"{": "}", "[": "]", "(": ")"}
        in_string = False
        string_char = None
        i = 0

        while i < len(code):
            ch = code[i]
            if ch == "\\" and in_string:
                i += 2
                continue
            if ch in ("'", '"', "`") and not in_string:
                in_string = True
                string_char = ch
            elif ch == string_char and in_string:
                in_string = False
                string_char = None
            elif not in_string:
                if ch in pairs:
                    stack.append(pairs[ch])
                elif ch in pairs.values():
                    if not stack:
                        return False, f"Unmatched closing bracket '{ch}' at position {i}"
                    expected = stack.pop()
                    if ch != expected:
                        return False, f"Mismatched bracket: expected '{expected}', got '{ch}' at position {i}"
            i += 1

        if stack:
            return False, f"Unclosed brackets: {len(stack)} remaining"
        return True, None

    def _check_code_smells(self, code: str, language: str) -> List[FilterIssue]:
        issues = []
        for smell_name, pattern in CODE_SMELL_PATTERNS.items():
            matches = re.findall(pattern, code, re.MULTILINE)
            if matches:
                severity = Severity.LOW if smell_name in ("magic_number", "overly_long_line") else Severity.MEDIUM
                issues.append(FilterIssue(
                    code=f"CODE_SMELL_{smell_name.upper()}",
                    message=f"Code smell: {smell_name.replace('_', ' ')} ({len(matches)} occurrence(s))",
                    severity=severity,
                    dimension="code",
                    details={"smell": smell_name, "count": len(matches), "examples": matches[:3]},
                ))
        return issues

    def _try_execute(self, code: str) -> Tuple[bool, Optional[str]]:
        try:
            restricted = {
                "__builtins__": {
                    "print": print, "range": range, "len": len, "int": int,
                    "str": str, "list": list, "dict": dict, "set": set,
                    "tuple": tuple, "float": float, "bool": bool,
                    "True": True, "False": False, "None": None,
                    "abs": abs, "max": max, "min": min, "sum": sum,
                    "sorted": sorted, "reversed": reversed,
                    "enumerate": enumerate, "zip": zip, "map": map,
                    "filter": filter, "any": any, "all": all,
                    "isinstance": isinstance, "hasattr": hasattr,
                    "getattr": getattr, "type": type, "range": range,
                }
            }
            compile(code, "<filtering>", "exec")
            exec(code, restricted)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def detect_language(self, code: str) -> str:
        patterns = [
            (r'import\s+java\.|public\s+(class|static|void|interface|enum)', "java"),
            (r'#include\s*<|int\s+main\s*\(|std::|cout|cin', "cpp"),
            (r'fn\s+\w+|let\s+mut|pub\s+fn|impl\s+|->\s*\w+', "rust"),
            (r'package\s+\w+|import\s+\w+|func\s+\w+|defer\s+', "go"),
            (r'using System;|namespace\s+\w+|class\s+\w+\s*:', "csharp"),
            (r'function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+|console\.|=>', "javascript"),
            (r'def\s+\w+\s*\(|if\s+__name__\s*==|print\s*\(|class\s+\w+\s*:', "python"),
            (r'SELECT\s+|FROM\s+|WHERE\s+|JOIN\s+|INSERT\s+|UPDATE\s+', "sql"),
            (r'require\s+|module\.exports|exports\.', "javascript"),
            (r'def\s+\w+|end\b', "ruby"),
            (r'<\?php', "php"),
            (r'import\s+Swift|func\s+\w+|var\s+\w+:\s*', "swift"),
            (r'fun\s+\w+|val\s+\w+|var\s+\w+', "kotlin"),
        ]
        score: Dict[str, int] = {}
        for pattern, lang in patterns:
            if re.search(pattern, code):
                score[lang] = score.get(lang, 0) + 1
        best = max(score, key=score.get) if score else "unknown"
        return best

    def validate_code_blocks(self, text: str) -> List[FilterResult]:
        blocks = re.findall(r"```(\w+)?\n(.*?)```", text, re.DOTALL)
        results = []
        for lang, code in blocks:
            lang = lang.strip() if lang else "unknown"
            code = code.strip()
            if code:
                results.append(self.check(code, lang))
        return results

    def get_stats(self) -> Dict:
        return self.stats
